"""Quick check of the trained vocal delivery model on your own audio.
  python training/try_vocal_delivery.py my_answer.wav
  python training/try_vocal_delivery.py --folder my_recordings/      (summary over many .wav files)
Needs WAV files (Windows Voice Recorder / Audacity can export WAV; or use the web app)."""
import sys, os, glob, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import vocal_delivery
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--folder", action="store_true"); a = ap.parse_args()
if not vocal_delivery.available(): sys.exit("No model found: run training/train_vocal_delivery.py first (models/vocal_delivery.joblib)")
files = sorted(glob.glob(os.path.join(a.path, "*.wav"))) if a.folder else [a.path]
count = collections.Counter()
for f in files:
    r = vocal_delivery.predict(f)
    if r is None: print(os.path.basename(f), "-> failed (is the file a valid WAV?)"); continue
    count[r["label"]] += 1
    print(f"{os.path.basename(f):40s} {r['label']:10s}", {k: f"{v:.0%}" for k, v in r["probs"].items()})
if len(files) > 1: print("\nSummary:", dict(count))
