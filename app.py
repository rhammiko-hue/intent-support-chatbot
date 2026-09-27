#!/usr/bin/env python3
"""Flask REST API + web UI for the intent chatbot.

Endpoints:
    GET  /            -> chat UI (static/index.html)
    POST /chat        -> {"message": str, "session_id": str (optional)}
                         returns {"reply", "intent", "confidence"}

Usage:
    python train.py   # once, to create model/nlu.joblib
    python app.py     # serves on http://127.0.0.1:5000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent

from flask import Flask, request, jsonify, send_from_directory
from src.dialogue import DialogueManager, new_session

app = Flask(__name__, static_folder="static")

_bundle = None
_manager = None
_sessions = {}


def get_manager():
    """Load the model once, on first request."""
    global _bundle, _manager
    if _manager is None:
        import joblib

        model_path = BASE / "model" / "nlu.joblib"
        if not model_path.exists():
            raise RuntimeError("Run `python train.py` first: model/nlu.joblib missing.")
        _bundle = joblib.load(model_path)
        _manager = DialogueManager(
            _bundle["vec"], _bundle["clf"], _bundle["responses"]
        )
    return _manager


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or "default"
    if not message:
        return jsonify({"error": "empty message"}), 400
    manager = get_manager()
    session = _sessions.setdefault(session_id, new_session())
    reply, info = manager.respond(message, session)
    return jsonify({
        "reply": reply,
        "intent": info.get("intent"),
        "confidence": round(info.get("confidence", 0.0), 3),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
