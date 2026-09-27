"""Intent classifier: TF-IDF features + logistic regression.

Why this model: with ~150 short training utterances, a linear model on
TF-IDF is fast, robust, and far less data-hungry than a neural net.
Character/word n-grams (1,2) capture phrases like "how much" or "per acre".
"""
import re

CROPS = [
    "rice", "wheat", "maize", "cotton", "sugarcane", "tomato", "potato",
    "onion", "soybean", "groundnut", "chili", "chilli", "banana", "mustard",
    "paddy",
]

# number + unit, e.g. "2 acres", "50 kg", "120mm"
QUANTITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(acres?|hectares?|kg|kgs|litres?|liters?|mm|days?|weeks?|tonnes?)",
    re.IGNORECASE,
)


def train_classifier(texts, labels):
    """Fit TF-IDF vectorizer + logistic regression. Returns (vec, clf)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vec.fit_transform(texts)
    # C=10: lighter regularization than the default C=1. With only ~150
    # training utterances across 12 intents, C=1 underfits and leaves
    # predict_proba nearly flat (top class ~0.25 even on training data),
    # which breaks confidence-threshold fallback. C=10 gives in-domain
    # confidences ~0.7-0.85 while out-of-domain gibberish stays ~0.1-0.2,
    # so a 0.35 fallback threshold cleanly separates them.
    clf = LogisticRegression(max_iter=1000, C=10.0)
    clf.fit(X, labels)
    return vec, clf


def predict_intent(text, vec, clf):
    """Return (intent_tag, confidence) for one utterance."""
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    best = int(proba.argmax())
    return clf.classes_[best], float(proba[best])


def extract_entities(text):
    """Regex/keyword entity extraction: crop names and quantities with units.

    Returns {"crop": str|None, "quantities": [(value, unit), ...]}.
    """
    lowered = text.lower()
    crop = next((c for c in CROPS if re.search(rf"\b{re.escape(c)}\b", lowered)), None)
    quantities = [(m.group(1), m.group(2).lower()) for m in QUANTITY_RE.finditer(text)]
    return {"crop": crop, "quantities": quantities}
