#!/usr/bin/env python3
"""CLI demo of the intent chatbot (uses the trained model).

Usage:
    python train.py   # first, to create model/nlu.joblib
    python demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dialogue import DialogueManager, new_session

BASE = Path(__file__).resolve().parent


def main():
    import joblib

    model_path = BASE / "model" / "nlu.joblib"
    if not model_path.exists():
        print("No trained model found. Run `python train.py` first.")
        sys.exit(1)
    bundle = joblib.load(model_path)
    dm = DialogueManager(bundle["vec"], bundle["clf"], bundle["responses"])
    session = new_session()

    print("Farm support chatbot (type 'quit' to exit).")
    print("Bot: Hello! I'm your farm assistant. Ask me about crops, irrigation,")
    print("     pests, fertilizer, weather, or market prices.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Good luck with your crops.")
            break
        if msg.lower() in {"quit", "exit", "q"}:
            print("Bot: Goodbye! Good luck with your crops.")
            break
        if not msg:
            continue
        reply, info = dm.respond(msg, session)
        print(f"Bot: {reply}")
        intent = info.get("intent") or "fallback"
        print(f"     [intent={intent} confidence={info.get('confidence', 0):.2f}]")


if __name__ == "__main__":
    main()
