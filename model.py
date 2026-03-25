import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ssqueezepy import ssq_stft
import librosa

# Configuration
INPUT_SHAPE = (128, 104)

# 4 Russell quadrants: relaxed, happy, sad, angry
NUM_CLASSES = 4


def compute_hybrid_features(y, sr=16000, max_len=16000 * 3):
    """
    SST + MFCC with per-band normalization (volume-invariant)
    Captures temporal shape patterns independent of amplitude
    """
    # 1. Fixed length (3 seconds)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        padding = max_len - len(y)
        y = np.pad(y, (0, padding))

    # --- FEATURE A: SST (Sharp Time-Frequency Resolution) ---
    try:
        Tx, _, _, _, _ = ssq_stft(y, fs=sr)
        sst_mag = np.abs(Tx)
    except Exception:
        sst_mag = np.abs(librosa.stft(y))

    # Log scale
    sst_log = np.log(sst_mag + 1e-9)

    # Per-band normalization
    sst_norm = (sst_log - np.mean(sst_log, axis=1, keepdims=True)) / (
        np.std(sst_log, axis=1, keepdims=True) + 1e-9
    )

    # Resize to (64 freq bins, 128 time steps)
    sst_tensor = tf.expand_dims(sst_norm, axis=-1)
    sst_resized = tf.image.resize(sst_tensor, (64, 128))
    sst_final = tf.transpose(sst_resized[:, :, 0], perm=[1, 0])

    # --- FEATURE B: MFCC (Timbre Shape) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Per-coefficient normalization
    mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
        np.std(mfcc, axis=1, keepdims=True) + 1e-9
    )

    mfcc_tensor = tf.expand_dims(mfcc_norm, axis=-1)
    mfcc_resized = tf.image.resize(mfcc_tensor, (40, 128))
    mfcc_final = tf.transpose(mfcc_resized[:, :, 0], perm=[1, 0])

    # --- COMBINE ---
    hybrid_features = tf.concat([sst_final, mfcc_final], axis=1)
    return hybrid_features.numpy()


def get_russell_targets(y_indices):
    """
    4-class Russell mapping:
      relaxed -> (+1, -1)  (Q4 calm/relaxed)
      happy   -> (+1, +1)  (Q1)
      sad     -> (-1, -1)  (Q3)
      angry   -> (-1, +1)  (Q2)
    """
    mapping = {
        0: [ 1.0, -1.0],  # Relaxed
        1: [ 1.0,  1.0],  # Happy
        2: [-1.0, -1.0],  # Sad
        3: [-1.0,  1.0],  # Angry
    }
    return np.array([mapping.get(int(y), [0.0, 0.0]) for y in y_indices], dtype=np.float32)


def build_mini_mamba():
    """
    Dual-Head model:
      - emotion_class (softmax over 4)
      - russell_coords (tanh -> [-1,1])
    """
    inputs = layers.Input(shape=INPUT_SHAPE)

    x = layers.Bidirectional(layers.GRU(48, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.GRU(24))(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)

    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='emotion_class')(x)
    russell_output = layers.Dense(2, activation='tanh', name='russell_coords')(x)

    model = models.Model(inputs=inputs, outputs=[class_output, russell_output])

    model.compile(
        optimizer='adam',
        loss={
            'emotion_class': 'sparse_categorical_crossentropy',
            'russell_coords': 'mse'
        },
        loss_weights={
            'emotion_class': 1.0,
            'russell_coords': 2.0
        },
        metrics={
            'emotion_class': 'accuracy',
            'russell_coords': 'mae'
        }
    )
    return model
