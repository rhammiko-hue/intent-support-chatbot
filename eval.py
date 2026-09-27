#!/usr/bin/env python3
"""Evaluation: stratified train/test split, honest numbers.

Reports overall accuracy plus per-intent accuracy and prints misclassified
examples so weaknesses are visible, not hidden.

Usage:
    python eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_intents
from src.nlu import train_classifier

BASE = Path(__file__).resolve().parent


def main():
    from sklearn.model_selection import train_test_split

    texts, labels, _ = load_intents(BASE / "intents.json")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"train: {len(X_train)} utterances | test: {len(X_test)} utterances")
    vec, clf = train_classifier(X_train, y_train)

    preds = clf.predict(vec.transform(X_test))
    correct = sum(p == t for p, t in zip(preds, y_test))
    print(f"\noverall accuracy: {correct}/{len(y_test)} = {correct / len(y_test):.3f}")

    # Per-intent accuracy.
    from collections import defaultdict

    stats = defaultdict(lambda: [0, 0])  # tag -> [correct, total]
    for t, p in zip(y_test, preds):
        stats[t][1] += 1
        stats[t][0] += p == t
    print("\nper-intent accuracy:")
    for tag in sorted(stats):
        c, n = stats[tag]
        print(f"  {tag:<14} {c}/{n} = {c / n:.2f}")

    # Misclassified examples, shown honestly.
    print("\nmisclassified examples:")
    shown = 0
    for text, t, p in zip(X_test, y_test, preds):
        if t != p:
            print(f"  '{text}' -> predicted '{p}', true '{t}'")
            shown += 1
            if shown >= 8:
                break
    if shown == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
