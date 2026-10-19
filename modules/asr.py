"""Speech-to-text (Rishika). Whisper if installed; otherwise returns None so the API
can accept a typed transcript instead (useful for demos / testing)."""
from functools import lru_cache

@lru_cache(maxsize=1)
def _whisper(size="base"):
    try:
        import whisper
        return whisper.load_model(size)
    except Exception:
        return None

def transcribe(audio_path):
    """Returns dict(text, segments=[{start,end,text}], duration) or None if ASR unavailable."""
    m = _whisper()
    if m is None:
        return None
    r = m.transcribe(audio_path, word_timestamps=True, fp16=False)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in r["segments"]]
    dur = segs[-1]["end"] if segs else 0.0
    return {"text": r["text"].strip(), "segments": segs, "duration": dur}
