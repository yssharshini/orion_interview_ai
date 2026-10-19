"""Topic drift (Anuraag): per-sentence cosine distance from the question + ideal answer."""
import numpy as np
from .text_utils import sentences
from .embeddings import embed, cosine

def topic_drift(question, ideal, candidate, threshold=0.20):
    sents = sentences(candidate)
    if not sents:
        return {"drift_pct": 100.0, "off_topic_sentences": [], "per_sentence": []}
    vecs = embed([question + " " + ideal] + sents)
    ref, sv = vecs[0], vecs[1:]
    sims = [max(0.0, cosine(ref, v)) for v in sv]
    off = [s for s, x in zip(sents, sims) if x < threshold]
    return {
        "drift_pct": round(100 * len(off) / len(sents), 1),
        "off_topic_sentences": off,
        "per_sentence": [{"sentence": s, "relevance": round(x, 3)} for s, x in zip(sents, sims)],
    }
