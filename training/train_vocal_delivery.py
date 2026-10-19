"""Train the vocal delivery classifier (confident / nervous / flat) on RAVDESS + IESC (+ optional CREMA-D).

  python training/train_vocal_delivery.py        (uses data/raw_audio/ravdess and data/Indian Emotional Speech Corpora (IESC))
  python training/train_vocal_delivery.py --ravdess <dir> --iesc <dir> --cremad <dir>/AudioWAV
  python training/train_vocal_delivery.py --iesc ""     (leave a dataset out by passing an empty path)

IESC = Indian Emotional Speech Corpora: 8 Indian English speakers, Speaker-N/<Emotion>/*.wav.

Features come from modules/audio_features.extract - the SAME function the live API uses.
A fingerprint of that file is saved in the report; modules/vocal_delivery.py refuses to predict
if audio_features.py changes later, until you retrain. (That mismatch was the "always nervous" bug.)

Evaluation is 5-fold speaker-grouped cross-validation: every speaker is tested exactly once, by a
model that never heard them. Accuracy is also reported per dataset (e.g. Indian vs North American).
Caveat for the report: acted emotion is only a proxy for interview delivery; see DELIVERY below.
"""
import os, sys, glob, json, argparse, collections
import numpy as np
import joblib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from modules import audio_features
from modules.vocal_delivery import extractor_hash

MODELS = os.path.join(ROOT, "models")

# emotion -> delivery class (unchanged from the previous model, so results are comparable)
DELIVERY = {
    "calm": "confident", "happy": "confident", "surprised": "confident",
    "fearful": "nervous", "angry": "nervous", "disgust": "nervous",
    "neutral": "flat", "sad": "flat",
}
RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
IESC_EMO = {"anger": "angry", "fear": "fearful", "happy": "happy", "neutral": "neutral", "sad": "sad"}
CREMA_EMO = {"NEU": "neutral", "HAP": "happy", "SAD": "sad",
             "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}


def list_ravdess(d):
    # 03-01-05-01-02-01-12.wav = modality-channel-EMOTION-intensity-statement-rep-ACTOR
    items = []
    for f in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
        parts = os.path.basename(f)[:-4].split("-")
        if len(parts) == 7 and parts[0] == "03" and parts[2] in RAVDESS_EMO:
            items.append((f, RAVDESS_EMO[parts[2]], "ravdess_" + parts[6]))
    return items


def list_iesc(d):
    # Speaker-3/Anger/A-1-3-1.wav  -> emotion from the folder name, speaker from Speaker-N
    items = []
    for f in glob.glob(os.path.join(d, "Speaker-*", "*", "*.wav")):
        emo_dir = os.path.basename(os.path.dirname(f)).lower()
        speaker = os.path.basename(os.path.dirname(os.path.dirname(f))).split("-")[-1]
        if emo_dir in IESC_EMO:
            items.append((f, IESC_EMO[emo_dir], "iesc_" + speaker))
    return items


def list_cremad(d):
    # 1001_DFA_ANG_XX.wav = SPEAKER_sentence_EMOTION_level
    items = []
    for f in glob.glob(os.path.join(d, "*.wav")):
        parts = os.path.basename(f)[:-4].split("_")
        if len(parts) >= 3 and parts[2] in CREMA_EMO:
            items.append((f, CREMA_EMO[parts[2]], "cremad_" + parts[0]))
    return items


def _feat(path):
    try:
        return audio_features.extract(path)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", default=os.path.join(ROOT, "data", "raw_audio", "ravdess"))
    ap.add_argument("--iesc", default=os.path.join(ROOT, "data", "Indian Emotional Speech Corpora (IESC)"))
    ap.add_argument("--cremad", default=None, help="folder containing CREMA-D AudioWAV files")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for feature extraction")
    ap.add_argument("--folds", type=int, default=5, help="speaker-grouped cross-validation folds")
    a = ap.parse_args()

    from joblib import Parallel, delayed
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    items = []
    for name, path, lister in [("ravdess", a.ravdess, list_ravdess), ("iesc", a.iesc, list_iesc),
                               ("cremad", a.cremad, list_cremad)]:
        if not path:
            continue
        if not os.path.isdir(path):
            print(f"[{name}] folder not found, skipping: {path}")
            continue
        found = lister(path)
        print(f"[{name}] {len(found)} clips from {len({s for _, _, s in found})} speakers")
        items += found
    if not items:
        sys.exit("No audio found. Check --ravdess / --cremad paths.")
    print(f"{len(items)} clips, {len({s for _, _, s in items})} speakers. Extracting features (pyin is slow; a few minutes)...")

    feats = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_feat)(f) for f, _, _ in items)
    keep = [i for i, x in enumerate(feats) if x is not None and np.any(x)]
    X = np.array([feats[i] for i in keep])
    y = np.array([DELIVERY[items[i][1]] for i in keep])
    groups = np.array([items[i][2] for i in keep])

    # Out-of-fold predictions: each speaker is predicted by a model trained without them
    cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=0)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        m = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0))

    dataset = np.array([g.split("_")[0] for g in groups])
    per_dataset = {}
    for ds in sorted(set(dataset)):
        k = dataset == ds
        per_dataset[ds] = {
            "n": int(k.sum()),
            "class_counts": dict(sorted(collections.Counter(y[k]).items())),
            "accuracy": round(float(accuracy_score(y[k], pred[k])), 3),
            "macro_f1": round(float(f1_score(y[k], pred[k], average="macro")), 3),
        }

    counts = collections.Counter(y)
    report = {
        "n_total": int(len(y)),
        "n_speakers": int(len(set(groups))),
        "datasets": sorted(set(dataset)),
        "class_counts": dict(sorted(counts.items())),
        "evaluation": f"{a.folds}-fold speaker-grouped cross-validation (out-of-fold predictions)",
        "cv_accuracy": round(float(accuracy_score(y, pred)), 3),
        "cv_macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
        "majority_baseline_accuracy": round(max(counts.values()) / len(y), 3),
        "per_dataset": per_dataset,
        "emotion_to_delivery_mapping": DELIVERY,
        "feature_names": audio_features.FEATURE_NAMES,
        "feature_extractor_hash": extractor_hash(),
    }
    print(classification_report(y, pred))
    for ds, r in per_dataset.items():
        print(f"  {ds:8s} n={r['n']:5d}  accuracy={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")

    # Final deployed model is trained on every speaker
    model.fit(X, y)
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS, "vocal_delivery.joblib"))
    with open(os.path.join(MODELS, "vocal_delivery_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted(counts)
        cm = confusion_matrix(y, list(pred), labels=labels)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Vocal delivery (speaker-grouped CV)")
        fig.tight_layout(); fig.savefig(os.path.join(MODELS, "vocal_delivery_confusion.png"), dpi=120); plt.close(fig)

        imp = model.steps[-1][1].feature_importances_
        order = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.barh([audio_features.FEATURE_NAMES[i] for i in order], imp[order])
        ax.set_title("Top feature importances"); fig.tight_layout()
        fig.savefig(os.path.join(MODELS, "vocal_delivery_features.png"), dpi=120); plt.close(fig)
    except ImportError:
        pass

    print(json.dumps({k: v for k, v in report.items() if k not in ("feature_names", "emotion_to_delivery_mapping", "per_dataset")}, indent=2))
    print("Saved models/vocal_delivery.joblib")


if __name__ == "__main__":
    main()
