"""Load the hand-written intent corpus."""
import json


def load_intents(path):
    """Return (texts, labels, responses) from intents.json.

    texts: every example utterance; labels: its intent tag;
    responses: {tag: [response templates]}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts, labels, responses = [], [], {}
    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(tag)
    return texts, labels, responses
