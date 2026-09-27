#!/usr/bin/env python3
"""Train the intent classifier on the full corpus and save it.

Usage:
    python train.py
Saves model/nlu.joblib (TF-IDF vectorizer, classifier, responses, tags).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_intents
from src.nlu import train_classifier

BASE = Path(__file__).resolve().parent


def main():
    import joblib

    texts, labels, responses = load_intents(BASE / "intents.json")
    print(f"Training on {len(texts)} utterances across {len(set(labels))} intents...")
    vec, clf = train_classifier(texts, labels)
    train_acc = clf.score(vec.transform(texts), labels)
    print(f"train accuracy (sanity check): {train_acc:.3f}")

    (BASE / "model").mkdir(exist_ok=True)
    joblib.dump(
        {"vec": vec, "clf": clf, "responses": responses, "tags": sorted(set(labels))},
        BASE / "model" / "nlu.joblib",
    )
    print("saved model/nlu.joblib")


if __name__ == "__main__":
    main()
