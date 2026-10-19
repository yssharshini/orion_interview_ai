"""Missing concepts + recommendations (Srijan)."""
from .text_utils import tokenize
from .embeddings import embed, cosine

def concept_coverage(candidate, concepts, use_semantic=True, sem_threshold=0.55):
    """concepts: list of {"name": str, "keywords": [str,...]}. A concept is covered if any
    keyword appears in the answer, or (optionally) a candidate sentence is semantically close."""
    text = " " + " ".join(tokenize(candidate)) + " "
    covered, missing = [], []
    for c in concepts:
        hit = any((" " + k.lower() + " ") in text or k.lower() in text for k in c["keywords"])
        if not hit and use_semantic and candidate.strip():
            vs = embed([c["name"] + " " + " ".join(c["keywords"])] + [candidate])
            hit = cosine(vs[0], vs[1]) >= sem_threshold
        (covered if hit else missing).append(c["name"])
    pct = round(100 * len(covered) / len(concepts), 1) if concepts else 100.0
    return {"coverage_pct": pct, "covered": covered, "missing": missing}

def recommend(missing, resources):
    out = []
    for m in missing:
        r = resources.get(m)
        out.append({"concept": m, "study": r or f"Review notes on '{m}' and practise explaining it aloud."})
    return out
