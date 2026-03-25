import os
import numpy as np
import librosa
import joblib
from final_model import BarkModel
from recognize_bark import detect_all_barks

# ML Pipeline Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def flatten_dna_for_ml(dna_dict):
    """Converts the dictionary into a flat 1D array for the Random Forest"""
    flat_features = []
    for key in ["mfcc", "mfcc_delta", "mfcc_delta2"]:
        flat_features.append(np.mean(dna_dict[key].T, axis=0))
    # Adding means of your other parameters
    flat_features.append([np.mean(dna_dict["pitch"]), np.mean(dna_dict["amplitude"]), np.mean(dna_dict["brightness"])])
    return np.concatenate(flat_features)


def train():
    model = BarkModel()
    # Folder mapping
    repo_map = {"happy": "happywavs", "sad": "sadwavs", "angry": "angrywavs"}

    full_dna_library = {"happy": [], "sad": [], "angry": []}
    X_ml = []
    y_ml = []

    print("--- Starting Feature Extraction ---")
    for label, folder in repo_map.items():
        if not os.path.exists(folder):
            print(f"Skipping {folder}: Folder not found.")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
        print(f"Processing {len(files)} files in {folder}...")

        for f in files:
            try:
                y, sr = librosa.load(os.path.join(folder, f), sr=model.sr)
                segments = detect_all_barks(y, sr)
                for seg in segments:
                    dna_dict = model.extract_dna(seg)

                    # Store for DTW
                    full_dna_library[label].append(dna_dict)

                    # Store for Machine Learning
                    X_ml.append(flatten_dna_for_ml(dna_dict))
                    y_ml.append(label)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if not X_ml:
        print("Error: No valid bark segments found. Check your audio files.")
        return

    X = np.array(X_ml)
    y = np.array(y_ml)

    # --- MACHINE LEARNING PIPELINE ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    clf = RandomForestClassifier(n_estimators=300, class_weight=dict(zip(classes, weights)), random_state=42)

    # 5-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        clf.fit(X_scaled[train_idx], y[train_idx])
        cv_scores.append(accuracy_score(y[test_idx], clf.predict(X_scaled[test_idx])))

    print(f"✅ 5-Fold CV Accuracy: {np.mean(cv_scores):.2%}")

    # Final training
    clf.fit(X_scaled, y)

    # --- SAVE LOCATION: CURRENT DIRECTORY ---
    # This saves the files exactly where the script is located
    joblib.dump(clf, "bark_gatekeeper_ai.pkl")
    joblib.dump(scaler, "bark_scaler.pkl")
    np.save('bark_library.npy', full_dna_library)

    print(f"Done! All models saved to current directory: {os.getcwd()}")


if __name__ == "__main__":
    train()