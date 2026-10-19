"""MODEL 1 - Interview communication score predictor.
Features (modules/features.py) -> human score 0-100.  Gradient boosting, compared against Ridge baseline.
Usage:  python training/train_communication_score.py --data data/scored_answers.csv"""
import sys, os, argparse, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.features import FEATURES
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/scored_answers.csv"); ap.add_argument("--out", default="models/communication_score.joblib")
a = ap.parse_args()
df = pd.read_csv(a.data); X, y = df[FEATURES], df["human_score"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=0)
cands = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
         "gboost": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=0)}
cv = KFold(5, shuffle=True, random_state=0); best, best_mae = None, 1e9
for name, m in cands.items():
    mae = -cross_val_score(m, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"{name:7s} 5-fold CV MAE = {mae:.2f}")
    if mae < best_mae: best, best_mae, best_name = m, mae, name
best.fit(Xtr, ytr); p = best.predict(Xte)
rep = {"model": best_name, "test_MAE": round(mean_absolute_error(yte, p), 2), "test_R2": round(r2_score(yte, p), 3),
       "spearman": round(float(spearmanr(yte, p)[0]), 3), "n_train": len(Xtr), "n_test": len(Xte)}
print(json.dumps(rep, indent=2))
best.fit(X, y); os.makedirs(os.path.dirname(a.out), exist_ok=True); joblib.dump(best, a.out)
json.dump(rep, open(a.out.replace(".joblib", "_report.json"), "w"), indent=2); print("saved", a.out)
