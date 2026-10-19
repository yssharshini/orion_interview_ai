import os, sys, tempfile
os.environ["ORION_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)
GOOD = ("Overfitting is when a model memorizes the noise in the training data so it performs well on training data but poorly on unseen validation data. "
        "To prevent it you can use regularization like L2 or dropout, early stopping, cross-validation and more data.")
BAD = "Um, so, like, I really like pizza and the weekend was good, you know, basically uh yeah."

def test_flow():
    assert c.get("/api/health").json()["ok"]
    s = c.post("/api/session?category=Machine Learning&n=2").json()
    sid = s["session_id"]
    g = c.post("/api/answer", data={"session_id": sid, "question_id": 1, "transcript": GOOD}).json()
    b = c.post("/api/answer", data={"session_id": sid, "question_id": 1, "transcript": BAD}).json()
    assert g["final_score"] > b["final_score"] + 15, (g["final_score"], b["final_score"])
    assert b["topic_drift"]["drift_pct"] > g["topic_drift"]["drift_pct"]
    assert len(b["concepts"]["missing"]) > len(g["concepts"]["missing"])
    assert b["fluency"]["filler_count"] >= 4
    r = c.get(f"/api/report/{sid}").json()
    assert r["answers"] == 2
    print("good", g["final_score"], "bad", b["final_score"])
