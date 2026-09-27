"""Dialogue manager: multi-turn state, slot filling, fallback handling."""
import random

from src.nlu import predict_intent, extract_entities

# Intents that need a slot before a useful answer can be given.
REQUIRED_SLOTS = {
    "fertilizer": ["crop"],
    "crop_disease": ["crop"],
}
SLOT_QUESTIONS = {
    "crop": "Which crop is this for?",
}
FALLBACK_RESPONSES = [
    "I'm not sure I understood. I can help with crop advice, irrigation, "
    "pest control, fertilizer, weather, market prices, soil health, or crop "
    "disease diagnosis — could you rephrase?",
    "Sorry, I didn't catch that. Try asking about a crop, irrigation, pests, "
    "fertilizer, or market prices.",
]


def new_session():
    """Fresh conversation state."""
    return {"slots": {}, "pending": None, "turns": 0, "greeted": False}


class DialogueManager:
    """Stateful multi-turn dialogue over the NLU predictions.

    - Confidence below `threshold` -> fallback response (never a wild guess).
    - Intents in REQUIRED_SLOTS ask a follow-up question when the slot is
      missing, and remember what they were waiting for (`pending`).
    - Response templates may contain {crop}; filled from entities/slots.
    """

    def __init__(self, vec, clf, responses, threshold=0.35, seed=42):
        self.vec = vec
        self.clf = clf
        self.responses = responses
        self.threshold = threshold
        self.rng = random.Random(seed)

    def _pick(self, tag):
        return self.rng.choice(self.responses[tag])

    def _fill(self, template, session, entities):
        crop = entities.get("crop") or session["slots"].get("crop") or "your crop"
        return template.replace("{crop}", crop)

    def respond(self, message, session):
        """Return (reply, info) and mutate session in place."""
        session["turns"] += 1
        entities = extract_entities(message)
        intent, confidence = predict_intent(message, self.vec, self.clf)

        # 1) We were waiting for a slot value: try to fill it first.
        if session["pending"]:
            pending_intent, slot = session["pending"]
            value = entities.get("crop") if slot == "crop" else None
            if value:
                session["slots"][slot] = value
                session["pending"] = None
                reply = self._fill(self._pick(pending_intent), session, entities)
                return reply, {"intent": pending_intent, "confidence": confidence,
                               "slot_filled": {slot: value}}
            # Still no value: ask once more, then answer generically.
            session["pending"] = None
            reply = self._fill(self._pick(pending_intent), session, entities)
            return reply, {"intent": pending_intent, "confidence": confidence,
                           "slot_filled": None}

        # 2) Low confidence -> honest fallback, never a guess.
        if confidence < self.threshold:
            return self.rng.choice(FALLBACK_RESPONSES), {
                "intent": None, "confidence": confidence, "fallback": True}

        # 3) Slot filling for intents that need more info.
        for slot in REQUIRED_SLOTS.get(intent, []):
            if not session["slots"].get(slot) and not entities.get(slot):
                session["pending"] = (intent, slot)
                return SLOT_QUESTIONS[slot], {
                    "intent": intent, "confidence": confidence,
                    "awaiting_slot": slot}

        # 4) Normal answer, remembering any entities for later turns.
        if entities.get("crop"):
            session["slots"]["crop"] = entities["crop"]
        if intent == "goodbye":
            session.update(new_session())
        reply = self._fill(self._pick(intent), session, entities)
        return reply, {"intent": intent, "confidence": confidence}
