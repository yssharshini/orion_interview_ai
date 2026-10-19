"""Builds the feature vector used by the communication score predictor (model #1)."""
import numpy as np

FEATURES = [
    "semantic", "drift_pct", "concept_cov", "wpm", "filler_rate",
    "pause_ratio", "type_token_ratio", "word_count", "vocal_confident_prob"
]

def build(semantic, drift_pct, concept_cov, flu=None, vocal_conf=0.5):
    """
    Safely builds a dictionary of all required features for model #1.
    """
    flu = flu or {}
    
    return {
        "semantic": float(semantic) if semantic is not None else 0.0,
        "drift_pct": float(drift_pct) if drift_pct is not None else 0.0,
        "concept_cov": float(concept_cov) if concept_cov is not None else 0.0,
        "wpm": float(flu.get("wpm", 0.0)),
        "filler_rate": float(flu.get("filler_rate", 0.0)),
        "pause_ratio": float(flu.get("pause_ratio", 0.0)),
        "type_token_ratio": float(flu.get("type_token_ratio", 0.0)),
        "word_count": int(flu.get("word_count", 0)),
        "vocal_confident_prob": float(vocal_conf) if vocal_conf is not None else 0.5
    }

def to_vector(feature_dict):
    """
    Converts a feature dict into a 2D numpy array with columns ordered by FEATURES.
    Use this before feeding features into your model.predict().
    """
    return np.array([[feature_dict[k] for k in FEATURES]], dtype=float)