"""Embedding backend. Uses Sentence-Transformers when installed (and model is cached/downloadable),
otherwise falls back to TF-IDF so the whole project still runs offline."""
import numpy as np
from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def _st_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None

def backend_name():
    return "sentence-transformers" if _st_model() is not None else "tfidf-fallback"

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0

def embed(texts):
    """Return a list of vectors for the given texts (consistent within one call)."""
    m = _st_model()
    if m is not None:
        return list(m.encode(texts, normalize_embeddings=True))
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
    return list(vec.transform(texts).toarray())

def similarity(a, b):
    v = embed([a, b])
    return max(0.0, cosine(v[0], v[1]))
