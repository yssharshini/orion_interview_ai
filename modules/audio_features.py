"""Audio feature extraction shared by training/train_vocal_delivery.py and the live API."""
import numpy as np

FEATURE_NAMES = (["pitch_mean", "pitch_std", "energy_mean", "energy_std", "zcr_mean", "voiced_ratio"]
                 + [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)])

def extract(path, sr=16000):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=True)
    
    # 1. Peak normalize audio amplitude to fix mic gain variations
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    # 2. Trim silence with a relaxed threshold
    y, _ = librosa.effects.trim(y, top_db=40)
    
    if len(y) < sr * 0.3:
        return np.zeros(len(FEATURE_NAMES))
        
    # 3. Extract pitch with expanded frequency bounds (50Hz to 500Hz)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 4. Pitch normalization: Use Coefficient of Variation (std / mean)
    # This prevents natural voice height or projection from inflating pitch_std
    if len(f0v) > 0:
        p_mean = float(f0v.mean())
        p_std_norm = float((f0v.std() / (p_mean + 1e-5)) * 130.0)
        pitch_mean = p_mean
        pitch_std = p_std_norm
    else:
        pitch_mean = 130.0
        pitch_std = 0.0

    voiced_ratio = float(np.mean(voiced)) if voiced is not None and len(voiced) > 0 else 0.0

    head = [
        pitch_mean,
        pitch_std,
        float(rms.mean()),
        float(rms.std()),
        float(zcr.mean()),
        voiced_ratio
    ]
    
    return np.array(head + list(mf.mean(1)) + list(mf.std(1)), dtype=float)