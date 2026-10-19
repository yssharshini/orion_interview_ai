"""Glues all modules together: one answer in -> full evaluation report out."""
from modules import semantic, drift, concepts, fluency, scoring, features, vocal_delivery

def evaluate(question, transcript, segments=None, duration=None, latency=None, audio_path=None, resources=None):
    sem = semantic.semantic_accuracy(transcript, question["ideal_answer"])
    dr = drift.topic_drift(question["question"], question["ideal_answer"], transcript)
    cov = concepts.concept_coverage(transcript, question["concepts"])
    fm = fluency.fluency_metrics(transcript, segments, duration, latency)
    fs = fluency.fluency_score(fm)
    vd, vnote = None, None
    if not audio_path:
        vnote = "no audio received - press Record, speak, press Stop (or just Submit), so audio is sent"
    else:
        vd = vocal_delivery.predict(audio_path)
        if vd is None:
            vnote = vocal_delivery.last_error
    vconf = vd["probs"].get("confident", 0.5) if vd else 0.5   # model #1 was trained on P(confident); keep as-is
    # Delivery rubric: calm/"flat" speech is not a fault, so it earns half credit; only "nervous" earns none.
    # (Scoring P(confident) alone gave calm, steady answers ~30/100.)
    delivery_score = (round(100 * (vd["probs"].get("confident", 0.0) + 0.5 * vd["probs"].get("flat", 0.0)), 1)
                      if vd else None)
    rubric, parts = scoring.rubric_score(sem, dr["drift_pct"], cov["coverage_pct"], fs, delivery_score)
    ml = scoring.ml_score(features.build(sem, dr["drift_pct"], cov["coverage_pct"], fm, vconf))
    final = scoring.final_score(rubric, ml)
    return {
        "question_id": question["id"], "semantic_accuracy": sem, "topic_drift": dr, "concepts": cov,
        "fluency": fm, "fluency_score": fs, "vocal_delivery": vd, "vocal_delivery_note": vnote, "rubric_parts": parts,
        "rubric_score": rubric, "ml_score": ml, "final_score": final, "grade": scoring.grade(final),
        "recommendations": concepts.recommend(cov["missing"], resources or {}),
        "ideal_answer": question["ideal_answer"],
    }