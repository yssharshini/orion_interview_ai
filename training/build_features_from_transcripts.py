"""REAL data path for model #1.
Input CSV columns: question_id, transcript, human_score   (human_score 0-100 from 2+ raters, averaged)
Optional columns: duration_s, audio_path (adds vocal features if vocal model is trained)
Output: data/scored_answers.csv with the FEATURES + human_score columns."""
import sys, os, json, argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import semantic, drift, concepts, fluency, features, vocal_delivery
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", default="data/scored_answers.csv")
a = ap.parse_args()
bank = {q["id"]: q for q in json.load(open("data/question_bank.json"))["questions"]}
rows = []
for r in pd.read_csv(a.inp).to_dict("records"):
    q = bank[int(r["question_id"])]; t = str(r["transcript"])
    sem = semantic.semantic_accuracy(t, q["ideal_answer"])
    dr = drift.topic_drift(q["question"], q["ideal_answer"], t)["drift_pct"]
    cov = concepts.concept_coverage(t, q["concepts"])["coverage_pct"]
    fm = fluency.fluency_metrics(t, duration=r.get("duration_s") if pd.notna(r.get("duration_s", float("nan"))) else None)
    vd = vocal_delivery.predict(r["audio_path"]) if "audio_path" in r and pd.notna(r["audio_path"]) else None
    f = features.build(sem, dr, cov, fm, vd["probs"].get("confident", .5) if vd else .5)
    f["human_score"] = r["human_score"]; rows.append(f)
pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out, len(rows))
