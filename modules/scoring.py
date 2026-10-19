"""Final scoring (Team). Weighted rubric (always available) blended with the trained
communication-score predictor (when models/communication_score.joblib exists)."""

import os
import joblib
import pandas as pd
from .features import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "communication_score.joblib")
WEIGHTS = {"accuracy": 0.35, "relevance": 0.20, "completeness": 0.20, "fluency": 0.15, "delivery": 0.10}
_m = None

def rubric_score(semantic, drift_pct, concept_cov, fluency, delivery=None):
    """
    Computes weighted rubric score (0-100 scale) with fallbacks for missing values.
    """
    # Safe float conversions with fallbacks
    safe_sem = float(semantic) if semantic is not None else 0.0
    safe_drift = float(drift_pct) if drift_pct is not None else 0.0
    safe_cov = float(concept_cov) if concept_cov is not None else 0.0
    safe_fluency = float(fluency) if fluency is not None else 50.0  # neutral fallback
    safe_delivery = float(delivery) if delivery is not None else safe_fluency

    parts = {
        "accuracy": max(0.0, min(100.0, safe_sem)),
        "relevance": max(0.0, min(100.0, 100.0 - safe_drift)),
        "completeness": max(0.0, min(100.0, safe_cov)),
        "fluency": max(0.0, min(100.0, safe_fluency)),
        "delivery": max(0.0, min(100.0, safe_delivery))
    }

    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return round(total, 1), {k: round(v, 1) for k, v in parts.items()}

def ml_score(feature_dict):
    """
    Predicts communication score using trained Model #1 pipeline.
    """
    global _m
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        if _m is None:
            _m = joblib.load(MODEL_PATH)

        # Enforce explicit numeric types across columns to prevent object-dtype issues in sklearn
        row = [float(feature_dict.get(f, 0.0)) for f in FEATURES]
        x = pd.DataFrame([row], columns=FEATURES, dtype=float)

        pred = _m.predict(x)[0]
        return float(max(0.0, min(100.0, pred)))

    except Exception as e:
        print(f"[scoring] ml_score prediction failed: {e}")
        return None

def final_score(rubric, ml):
    if ml is None:
        return round(float(rubric), 1)
    return round(0.5 * float(rubric) + 0.5 * float(ml), 1)

def grade(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Fair"
    return "Needs work"