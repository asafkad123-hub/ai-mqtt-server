import os, glob, numpy as np
import librosa
import tensorflow as tf

from model import build_mini_mamba, compute_hybrid_features, get_russell_targets

# IMPORTANT: Keep this order consistent everywhere (train + test + mapping)
CATEGORIES = ['relaxed', 'happy', 'sad', 'angry']
DATA_DIRS  = ['relaxedwavs', 'happywavs', 'sadwavs', 'angrywavs']


def load_data_minimal_aug():
    """Load data with minimal augmentation (2x per file)."""
    X, y = [], []
    print("🧪 Computing Hybrid Features (Per-Band Normalized)...")

    for idx, cat in enumerate(CATEGORIES):
        files = glob.glob(os.path.join(DATA_DIRS[idx], "*.wav"))
        print(f" Processing {cat}: {len(files)} files")

        for f in files:
            try:
                audio, sr = librosa.load(f, sr=16000)

                feats = compute_hybrid_features(audio, sr)
                X.append(feats)
                y.append(idx)

                # Minimal augmentation: slight time stretch
                audio_aug = librosa.effects.time_stretch(audio, rate=1.05)
                feats_aug = compute_hybrid_features(audio_aug, sr)
                X.append(feats_aug)
                y.append(idx)

            except Exception as e:
                print(f"Skipped {f}: {e}")
                continue

    print(f"\n✨ Total samples: {len(X)}")
    return np.array(X), np.array(y)


def train():
    X, y = load_data_minimal_aug()
    print(f"✨ Dataset: {X.shape} (N, 128, 104)")

    y_coords = get_russell_targets(y)

    print("🚀 Training Dual-Head (Class + Geometry)...")
    model = build_mini_mamba()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X,
        {
            'emotion_class': y,
            'russell_coords': y_coords
        },
        epochs=50,
        batch_size=16,
        callbacks=[callback],
        verbose=1
    )

    model.save('bark_mamba_model.keras')
    print("✅ Saved bark_mamba_model.keras")

    # Build DNA Library for DTW (one template per class)
    import joblib
    dna_lib = {}
    print("🧬 Building DNA Library...")

    for idx, cat in enumerate(CATEGORIES):
        files = glob.glob(os.path.join(DATA_DIRS[idx], "*.wav"))
        if not files:
            continue

        y_ref, _ = librosa.load(files[0], sr=16000)
        pitches, mags = librosa.piptrack(y=y_ref, sr=16000)
        pitch_curve = np.max(pitches * mags, axis=0)

        if len(pitch_curve) > 0:
            pitch_curve = (pitch_curve - np.mean(pitch_curve)) / (np.std(pitch_curve) + 1e-6)
            dna_lib[cat] = pitch_curve

    joblib.dump(dna_lib, 'bark_dna_library.pkl')
    print("✅ Saved bark_dna_library.pkl")


if __name__ == "__main__":
    train()
