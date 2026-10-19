"""Backend & API (Sidharth). Run: uvicorn app.main:app --reload"""
import json, os, tempfile, random, subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import db
from .pipeline import evaluate
from modules import asr, embeddings

ROOT = os.path.join(os.path.dirname(__file__), "..")
BANK = json.load(open(os.path.join(ROOT, "data", "question_bank.json")))
QS = {q["id"]: q for q in BANK["questions"]}
db.init(BANK)

app = FastAPI(title="Orion - AI Interview Trainer")

def convert_to_wav(input_path: str) -> str:
    """Converts browser WebM/OGG audio to standard 16kHz mono WAV for librosa & Whisper."""
    wav_path = input_path + ".wav"
    try:
        # Use ffmpeg CLI to convert any input format to 16kHz mono WAV
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return wav_path
    except Exception as e:
        print(f"[app/main] ffmpeg conversion warning ({e}), falling back to raw file.")
        return input_path

@app.get("/api/health")
def health():
    return {"ok": True, "embedding_backend": embeddings.backend_name(), "asr_available": asr._whisper() is not None}

@app.get("/api/categories")
def categories():
    return sorted({q["category"] for q in QS.values()})

@app.post("/api/session")
def start_session(category: str = "all", n: int = 5):
    pool = [q for q in QS.values() if category in ("all", q["category"])]
    if not pool:
        raise HTTPException(404, "No questions in that category")
    picked = random.sample(pool, min(n, len(pool)))
    sid = db.new_session(category)
    return {"session_id": sid, "questions": [{"id": q["id"], "category": q["category"], "question": q["question"]} for q in picked]}

@app.post("/api/answer")
async def answer(session_id: int = Form(...), question_id: int = Form(...), transcript: str = Form(""),
                 latency: float = Form(None), audio: UploadFile = File(None)):
    """Send audio (transcribed by Whisper) and/or a typed/browser-STT transcript."""
    q = QS.get(question_id)
    if not q:
        raise HTTPException(404, "Unknown question")
    
    segments = duration = audio_path = wav_path = None

    if audio is not None:
        suffix = os.path.splitext(audio.filename or "a.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await audio.read())
            audio_path = f.name

        # Convert browser WebM to clean 16kHz Mono WAV before feature extraction
        wav_path = convert_to_wav(audio_path)

        # DEBUG: keep a copy of each recording so the vocal model can be checked against real input.
        # Delete this block (and data/debug_recordings/) once vocal delivery is sorted out.
        try:
            import shutil, time
            dbg = os.path.join(ROOT, "data", "debug_recordings")
            os.makedirs(dbg, exist_ok=True)
            ext = ".wav" if wav_path != audio_path else suffix
            shutil.copy(wav_path, os.path.join(dbg, f"{int(time.time())}_q{question_id}{ext}"))
        except Exception as e:
            print(f"[app/main] could not save debug recording: {e}")

        if not transcript.strip():
            t = asr.transcribe(wav_path)
            if t:
                transcript, segments, duration = t["text"], t["segments"], t["duration"]

    if not transcript.strip():
        raise HTTPException(422, "No transcript: install Whisper or send the 'transcript' field")

    try:
        # Pass the converted WAV path to evaluate and vocal delivery classifier
        r = evaluate(q, transcript, segments, duration, latency, wav_path or audio_path, BANK["resources"])
    finally:
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        if wav_path and os.path.exists(wav_path) and wav_path != audio_path:
            os.unlink(wav_path)

    r["transcript"] = transcript
    db.save_answer(session_id, question_id, transcript, r)
    return r

@app.get("/api/report/{session_id}")
def report(session_id: int):
    rows = db.session_report(session_id)
    if not rows:
        raise HTTPException(404, "No answers for that session")
    n = len(rows)
    avg = lambda k: round(sum(r[k] for r in rows) / n, 1)
    missing = {}
    for r in rows:
        for m in json.loads(r["report_json"])["concepts"]["missing"]:
            missing[m] = missing.get(m, 0) + 1
    return {
        "session_id": session_id, "answers": n, "avg_final": avg("final_score"), "avg_semantic": avg("semantic"),
        "avg_drift_pct": avg("drift_pct"), "avg_concept_cov": avg("concept_cov"), "avg_fluency": avg("fluency"),
        "top_missing_concepts": sorted(missing.items(), key=lambda x: -x[1])[:5],
        "per_question": [{"question_id": r["question_id"], "final": r["final_score"], "created_at": r["created_at"]} for r in rows]
    }

app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))