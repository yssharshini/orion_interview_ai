"""Creates a BOOTSTRAP scored dataset so model #1 can train before real human scores exist.
Replace with real data (see training/build_features_from_transcripts.py) before reporting results."""
import numpy as np, pandas as pd, argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.features import FEATURES
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--out", default="data/scored_answers_synthetic.csv")
a = ap.parse_args(); rng = np.random.default_rng(42); n = a.n
sem = np.clip(rng.normal(55, 22, n), 0, 100)
cov = np.clip(sem + rng.normal(0, 15, n), 0, 100)
drift = np.clip(100 - sem + rng.normal(0, 20, n), 0, 100) * rng.uniform(0, .6, n)
wpm = np.clip(rng.normal(135, 35, n), 60, 220)
filler = np.clip(rng.exponential(.04, n), 0, .25)
pause = np.clip(rng.exponential(.08, n), 0, .5)
ttr = np.clip(rng.normal(.7, .1, n), .3, 1)
wc = np.clip(rng.normal(70, 35, n), 5, 250)
vconf = np.clip(rng.beta(2.5, 2.5, n), 0, 1)
score = (.35*sem + .2*(100-drift) + .2*cov + .1*np.clip(100-abs(wpm-135)*.6, 0, 100) - 120*filler - 40*pause
         + 10*vconf + 4*(wc > 30) - 5 + rng.normal(0, 4, n))
df = pd.DataFrame({"semantic": sem, "drift_pct": drift, "concept_cov": cov, "wpm": wpm, "filler_rate": filler,
                   "pause_ratio": pause, "type_token_ratio": ttr, "word_count": wc, "vocal_confident_prob": vconf,
                   "human_score": np.clip(score, 0, 100)}).round(3)
df.to_csv(a.out, index=False); print("wrote", a.out, df.shape)
