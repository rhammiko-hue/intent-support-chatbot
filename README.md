# Intent Support Chatbot (Agriculture)

An intent-based conversational support chatbot for agriculture. It
classifies the user's intent (TF-IDF + logistic regression), extracts
entities (crop names, quantities with units) with regex/keyword rules,
manages multi-turn dialogue with slot filling, and falls back honestly
when confidence is low. Served via a Flask REST API plus a minimal web
chat UI. **Fully offline.**

## Architecture

```
                        ┌──────────────┐
                        │ intents.json │  12 intents × 10–14 hand-written
                        └──────┬───────┘  example utterances (148 total)
                               │ train.py
                               ▼
              ┌─────────────────────────────────┐
              │  TF-IDF (1–2 grams) +           │  model/nlu.joblib
              │  LogisticRegression (C=10)      │  (vectorizer, classifier,
              └──────────────┬──────────────────┘   responses)
                               │
User message ──┬───────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│ NLU (src/nlu.py)                                         │
│  • predict_intent → (intent, confidence)                  │
│  • extract_entities → crop names, quantities+units (regex)│
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│ DialogueManager (src/dialogue.py) — per-session state    │
│  1. pending slot? → try to fill it from this message     │
│  2. confidence < 0.35 → honest fallback (never a guess)  │
│  3. intent needs a slot (e.g. fertilizer→crop)? → ask    │
│  4. else answer; remember entities for later turns       │
└──────────────────────────┬───────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │ demo.py (CLI)  │  app.py (Flask)
              │                │  POST /chat  +  web UI (static/)
              └────────────────────────┘
```

## Install & run

```bash
pip install -r requirements.txt   # scikit-learn, flask, numpy
python train.py                   # train on intents.json → model/nlu.joblib
python demo.py                    # CLI demo
python eval.py                    # train/test split evaluation
python app.py                     # API + UI on http://127.0.0.1:5000
```

API example:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "which fertilizer should I use?", "session_id": "s1"}'
# → {"reply": "Which crop is this for?", "intent": "fertilizer", "confidence": 0.787}
```

## Intents (12)

greeting, goodbye, thanks, help, crop_advice, irrigation, pest_control,
fertilizer, weather, market_price, soil_health, crop_disease.
`fertilizer` and `crop_disease` require a crop slot — the bot asks a
follow-up question when it's missing.

## Example conversation (real run)

```
Bot: Hello! I'm your farm assistant. Ask me about crops, irrigation,
     pests, fertilizer, weather, or market prices.

You: hello
Bot: Hello! I'm your farm assistant. ...          [intent=greeting, conf=0.76]

You: which fertilizer should I use?
Bot: Which crop is this for?                      [intent=fertilizer, conf=0.79]

You: wheat
Bot: For wheat, get a soil test first for precise doses. A common starting
     reference for cereals is about 120:60:40 kg NPK per hectare, split
     between a basal dose and top-dressing. ...   [slot filled: crop=wheat]

You: my tomato plants have yellow leaves
Bot: Common tomato issues: yellow leaves often mean nitrogen deficiency or
     overwatering; spots suggest fungal infection ... [intent=crop_disease, conf=0.84]

You: what is the price of wheat today
Bot: I can't see live mandi prices offline — check your nearest APMC mandi
     board ...                                    [intent=market_price, conf=0.76]

You: blargh xyzzy
Bot: I'm not sure I understood. I can help with crop advice, irrigation, pest
     control, ... could you rephrase?            [fallback, conf=0.11]

You: thanks
Bot: You're welcome! Anything else about your farm? [intent=thanks, conf=0.85]

You: bye
Bot: Goodbye! Good luck with your crops.          [intent=goodbye, conf=0.85]
```

## Evaluation (measured, not estimated)

`python eval.py` — stratified 75/25 train/test split (`random_state=42`),
measured on 2026-09-27:

| metric | score |
|---|---|
| overall intent accuracy | **28/37 = 0.757** |
| train accuracy (sanity check, full data) | 1.000 |

Per-intent accuracy: crop_advice 1.00, crop_disease 1.00, greeting 1.00,
soil_health 1.00, weather 1.00, market_price 1.00, fertilizer 0.75,
help 0.67, irrigation 0.67, thanks 0.50, goodbye 0.33, pest_control 0.25.

Honest misclassifications (the main weakness): `pest_control` is confused
with `crop_disease` ("pests are attacking my crop", "termite in field",
"bollworm in cotton") because both intents share vocabulary about sick
plants; short utterances ("thx", "that's all", "goodbye") are brittle with
only ~10 training examples per intent. More, and more distinctive, training
utterances for the confused pairs is the clear next step.

## Limitations (interview-honest)

- **Small hand-written corpus.** 148 utterances; accuracy 0.757 reflects
  that. This is a baseline, not a production NLU.
- **Template responses, not generation.** Answers are hand-written templates
  with `{crop}` filled in — no LLM involved.
- **Simple entity extraction.** Regex/keyword matching over a fixed crop
  list; won't catch misspellings or crops outside the list.
- **No live data.** Weather and market-price intents honestly say they can't
  fetch live data offline; wiring real APIs is future work.
- **In-memory sessions.** The Flask app keeps conversation state in a dict —
  it resets on restart and isn't multi-process safe.
- Confidence threshold (0.35) was picked after checking the model's
  confidence separation (in-domain ~0.7–0.85, gibberish ~0.1–0.2); it is a
  heuristic, not a tuned value.

## Files

```
intent-support-chatbot/
├── intents.json        # 12 intents, 148 example utterances, responses
├── src/
│   ├── data.py         # load intents.json
│   ├── nlu.py          # TF-IDF + LogReg, entity extraction
│   └── dialogue.py     # DialogueManager: state, slots, fallback
├── train.py            # train → model/nlu.joblib
├── eval.py             # stratified eval with per-intent breakdown
├── demo.py             # CLI demo
├── app.py              # Flask API (POST /chat) + serves UI
├── static/index.html    # minimal web chat UI
├── requirements.txt
└── README.md
```
