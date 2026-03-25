import numpy as np
import librosa
import fastdtw
from scipy.spatial.distance import euclidean


class BarkModel:
    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512

    def extract_dna(self, y):
        # Base MFCCs (40 coefficients for high detail)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=40, hop_length=self.hop_length)

        # --- SAFETY FIX FOR SHORT BARKS ---
        # librosa.feature.delta defaults to width=9. If the bark has fewer than 9 frames,
        # it crashes. We check length and pad with zeros if it's too short.
        if mfcc.shape[1] >= 9:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        else:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)

        # Pitch Tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr, hop_length=self.hop_length)

        # Safely extract pitch even if the segment is nearly silent or extremely short
        if magnitudes.size > 0 and magnitudes.shape[1] > 0:
            pitch = np.array([pitches[np.argmax(magnitudes[:, i]), i] for i in range(magnitudes.shape[1])])
        else:
            pitch = np.zeros(mfcc.shape[1])

        # RMS Amplitude and Spectral Centroid (Brightness)
        amplitude = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        brightness = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)[0]

        # Spectral Energy (Low vs High frequency balance)
        stft = np.abs(librosa.stft(y, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_e = np.mean(stft[freqs < 2000, :], axis=0)
        high_e = np.mean(stft[freqs >= 2000, :], axis=0)

        return {
            "mfcc": mfcc,
            "mfcc_delta": mfcc_delta,
            "mfcc_delta2": mfcc_delta2,
            "pitch": pitch,
            "amplitude": amplitude,
            "brightness": brightness,
            "low_energy": low_e,
            "high_energy": high_e
        }

    def compute_similarity(self, feat1, feat2):
        """
        Uses Dynamic Time Warping (DTW) to compare two bark 'DNAs'.
        Lower total_dist means the barks are more similar.
        """
        total_dist = 0

        # We focus on the most descriptive keys for similarity comparison
        keys_to_compare = ["mfcc", "pitch", "amplitude", "brightness", "low_energy", "high_energy"]

        for key in keys_to_compare:
            if key not in feat1 or key not in feat2:
                continue

            # Reshape 1D arrays to 2D for fastdtw compatibility
            f1 = feat1[key].reshape(-1, 1) if feat1[key].ndim == 1 else feat1[key].T
            f2 = feat2[key].reshape(-1, 1) if feat2[key].ndim == 1 else feat2[key].T

            # Ensure both segments actually have data frames to compare
            if len(f1) > 0 and len(f2) > 0:
                try:
                    dist, _ = fastdtw.fastdtw(f1, f2, dist=euclidean)
                    total_dist += dist
                except Exception:
                    # Skip if specific comparison fails due to extreme shortness
                    continue

        return total_dist