"""Database (Samyuktha). SQLite via stdlib; schema is portable to PostgreSQL/MySQL."""
import sqlite3, json, os, time
DB_PATH = os.environ.get("ORION_DB", os.path.join(os.path.dirname(__file__), "..", "orion.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, created_at REAL);
CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY, category TEXT, question TEXT, ideal_answer TEXT, concepts_json TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, category TEXT, started_at REAL, finished_at REAL,
    FOREIGN KEY(user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS answers(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, question_id INTEGER, transcript TEXT,
    semantic REAL, drift_pct REAL, concept_cov REAL, fluency REAL, delivery_label TEXT,
    rubric_score REAL, ml_score REAL, final_score REAL, report_json TEXT, created_at REAL,
    FOREIGN KEY(session_id) REFERENCES sessions(id), FOREIGN KEY(question_id) REFERENCES questions(id));
"""

def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init(bank):
    with conn() as c:
        c.executescript(SCHEMA)
        for q in bank["questions"]:
            c.execute("INSERT OR REPLACE INTO questions VALUES(?,?,?,?,?)",
                      (q["id"], q["category"], q["question"], q["ideal_answer"], json.dumps(q["concepts"])))

def new_session(category="all", user_id=None):
    with conn() as c:
        cur = c.execute("INSERT INTO sessions(user_id,category,started_at) VALUES(?,?,?)", (user_id, category, time.time()))
        return cur.lastrowid

def save_answer(session_id, qid, transcript, r):
    with conn() as c:
        c.execute("""INSERT INTO answers(session_id,question_id,transcript,semantic,drift_pct,concept_cov,fluency,delivery_label,
                     rubric_score,ml_score,final_score,report_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (session_id, qid, transcript, r["semantic_accuracy"], r["topic_drift"]["drift_pct"], r["concepts"]["coverage_pct"],
                   r["fluency_score"], (r.get("vocal_delivery") or {}).get("label"), r["rubric_score"], r["ml_score"],
                   r["final_score"], json.dumps(r), time.time()))

def session_report(session_id):
    with conn() as c:
        rows = c.execute("SELECT * FROM answers WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    return [dict(r) for r in rows]
