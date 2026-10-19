"""Vocal delivery classifier (optional model #2). Loads models/vocal_delivery.joblib if trained."""
import os
import json
import hashlib
import joblib
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vocal_delivery.joblib")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vocal_delivery_report.json")
FEATURES_SRC = os.path.join(os.path.dirname(__file__), "audio_features.py")


def extractor_hash():
    """Fingerprint of audio_features.py. Saved at training time and checked here, so the
    model is never fed features computed differently from the ones it was trained on."""
    with open(FEATURES_SRC, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _stale_model_reason():
    try:
        with open(REPORT_PATH) as f:
            trained_with = json.load(f).get("feature_extractor_hash")
    except (OSError, ValueError):
        trained_with = None
    if trained_with != extractor_hash():
        return ("modules/audio_features.py has changed since the vocal delivery model was trained, "
                "so its predictions would be wrong. Retrain: python training/train_vocal_delivery.py")
    return None
_m = None
last_error = None   # Human-readable reason when predict() returns None

def available():
    """Checks whether the trained vocal delivery model pipeline exists on disk."""
    return os.path.exists(MODEL_PATH)

def predict(audio_path):
    """
    Extracts features from the given audio file and predicts delivery classification.
    
    Returns:
        dict: {"label": 'confident'|'nervous'|'flat', "probs": {'confident': float, 'nervous': float, 'flat': float}}
        None: If prediction fails (error reason stored in last_error).
    """
    global _m, last_error
    last_error = None

    if not available():
        last_error = (
            f"Model file not found at {os.path.abspath(MODEL_PATH)}. "
            "Run training/train_vocal_delivery.py (and start the server from the project root)."
        )
        return None

    stale = _stale_model_reason()
    if stale:
        last_error = stale
        return None

    try:
        from .audio_features import extract

        # Lazily load model pipeline into memory
        if _m is None:
            _m = joblib.load(MODEL_PATH)

        # Extract features (uses normalized peak gain and safeguards)
        x = extract(audio_path).reshape(1, -1)

        if not x.any():
            last_error = "Audio too short or silent - speak for at least a second or two."
            return None

        # Predict class probabilities using the loaded sklearn Pipeline (handles scaling + classification)
        p = _m.predict_proba(x)[0]

        # Extract class labels (Pipeline forwards `classes_` from the estimator)
        classes = getattr(_m, "classes_", ["confident", "flat", "nervous"])

        # Construct normalized, lowercased probability mapping
        probs = {str(c).lower(): round(float(v), 3) for c, v in zip(classes, p)}
        predicted_label = str(classes[p.argmax()]).lower()

        return {
            "label": predicted_label,
            "probs": probs
        }

    except Exception as e:
        last_error = f"Could not read/analyze the audio ({type(e).__name__}: {e})"
        print(f"[vocal_delivery] Prediction failed: {last_error}")
        return None