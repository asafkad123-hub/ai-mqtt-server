import librosa
import numpy as np
import joblib
import os

CONFIDENCE_THRESHOLD = 0.2

# Look for model in local directory
model_path = 'bark_gatekeeper_ai.pkl'
try:
    AI_GATEKEEPER = joblib.load(model_path)
except:
    AI_GATEKEEPER = None

def extract_features(y, sr):
    # n_mfcc=40 + delta + delta2 = 120 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] >= 9:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)

    mfcc_combined = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0)
    ])

    # Tone/Volume = 3 features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mask = magnitudes > np.median(magnitudes)
    pitch_val = np.mean(pitches[mask]) if np.any(mask) else 0
    amplitude_val = np.mean(librosa.feature.rms(y=y))
    brightness_val = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Total = 123 features
    return np.concatenate([mfcc_combined, [pitch_val, amplitude_val, brightness_val]]).reshape(1, -1)

def detect_all_barks(y, sr):
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=2, post_max=2, pre_avg=3, post_avg=3, delta=0.05, wait=10)

    if len(peaks) == 0: return []
    try:
        onsets = librosa.onset.onset_backtrack(peaks, onset_env)
    except: return []

    valid_segments = []
    for o in onsets:
        start = max(0, librosa.frames_to_samples(o))
        end = min(len(y), start + int(sr * 0.5))
        segment = y[start:end]

        if AI_GATEKEEPER and len(segment) > 1000:
            feats = extract_features(segment, sr)
            probs = AI_GATEKEEPER.predict_proba(feats)[0]
            bark_prob = probs[1] if len(probs) > 1 else 0.0
            if bark_prob >= CONFIDENCE_THRESHOLD:
                valid_segments.append(segment)
                print(f">>> VALID BARK: {bark_prob * 100:.1f}% confidence")
            else:
                print(f">>> REJECTED NOISE: {bark_prob * 100:.1f}% confidence")
    return valid_segments