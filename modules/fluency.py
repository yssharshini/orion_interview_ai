"""Fluency & pauses (deterministic, no ML)."""
import re
from .text_utils import tokenize, FILLERS

def fluency_metrics(transcript, segments=None, duration=None, response_latency=None):
    words = tokenize(transcript)
    n = len(words)
    low = " " + " ".join(words) + " "
    filler_count = sum(len(re.findall(r"(?<!\w)" + re.escape(f) + r"(?!\w)", low)) for f in FILLERS)
    # duration: from ASR segments, else estimate at 130 wpm
    dur = duration or (segments[-1]["end"] if segments else None) or max(1.0, n / 130 * 60)
    wpm = n / (dur / 60) if dur else 0
    long_pauses, pause_time = 0, 0.0
    if segments and len(segments) > 1:
        for a, b in zip(segments, segments[1:]):
            gap = b["start"] - a["end"]
            if gap > 0.7:
                long_pauses += 1
                pause_time += gap
    ttr = len(set(words)) / n if n else 0
    return {
        "word_count": n, "duration_s": round(dur, 1), "wpm": round(wpm, 1),
        "filler_count": filler_count, "filler_rate": round(filler_count / n, 3) if n else 0,
        "long_pauses": long_pauses, "pause_ratio": round(pause_time / dur, 3) if dur else 0,
        "type_token_ratio": round(ttr, 3), "response_latency_s": response_latency,
    }

def fluency_score(m):
    """0-100. Ideal pace 110-160 wpm, few fillers, few long pauses."""
    s = 100.0
    w = m["wpm"]
    if w < 110: s -= min(30, (110 - w) * 0.6)
    elif w > 160: s -= min(30, (w - 160) * 0.6)
    s -= min(30, m["filler_rate"] * 300)
    s -= min(20, m["long_pauses"] * 4)
    if m["word_count"] < 15: s -= 15
    return round(max(0.0, s), 1)
