"""Semantic accuracy (Harshini): similarity of candidate answer to the ideal answer, 0-100."""
from .embeddings import similarity

def semantic_accuracy(candidate, ideal):
    if not candidate.strip():
        return 0.0
    s = similarity(candidate, ideal)
    # rescale: raw cosine of ~0.2 is 'unrelated', ~0.8 is 'near paraphrase'
    return round(100 * min(1.0, max(0.0, (s - 0.15) / 0.65)), 1)
