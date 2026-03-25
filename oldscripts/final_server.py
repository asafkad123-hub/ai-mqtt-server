import paho.mqtt.client as mqtt
import socket, wave, librosa, os, joblib, warnings
import numpy as np
import matplotlib.pyplot as plt
from final_model import BarkModel
from recognize_bark import detect_all_barks

# --- SETTINGS ---
MQTT_BROKER = "broker.hivemq.com"
TOPIC = "kurland/dog/bark"
SAVE_NAME = 'esp32_audio.wav'
DNA_LIBRARY = 'bark_library.npy'
MODEL_PATH = "../../modelmic/bark_classifier.pkl"
SCALER_PATH = "../../modelmic/bark_scaler.pkl"

# Initialize AI & Library
CLF = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
SCALER = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
EMOTION_LIB = np.load(DNA_LIBRARY, allow_pickle=True).item() if os.path.exists(DNA_LIBRARY) else None

audio_buffer = bytearray()


def extract_ml_features(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate([np.mean(mfcc.T, axis=0), np.mean(delta.T, axis=0), np.mean(delta2.T, axis=0)])

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mask = magnitudes > np.median(magnitudes)
    pitch_val = np.mean(pitches[mask]) if np.any(mask) else 0
    amplitude_val = np.mean(librosa.feature.rms(y=y))
    brightness_val = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return np.concatenate([mfcc_combined, [pitch_val, amplitude_val, brightness_val]])


def plot_research_graphs(y, sr, bark_id, emotion, distance, dna_dict, probabilities, classes):
    plt.style.use('seaborn-v0_8-whitegrid')
    stft_abs = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(stft_abs, ref=np.max).mean(axis=0) + 60

    features_map = [
        ("dB Profile", db), ("Pitch (Hz) Profile", dna_dict["pitch"]),
        ("Amplitude Profile", dna_dict["amplitude"]), ("Brightness Profile", dna_dict["brightness"]),
        ("Harmonicity Profile", librosa.feature.rms(y=librosa.effects.harmonic(y))[0]),
        ("Low Energy Profile", dna_dict["low_energy"]), ("High Energy Profile", dna_dict["high_energy"])
    ]

    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(f"Bark Analysis #{bark_id} - Predicted: {emotion.upper()} (DTW Dist: {distance:.2f})", fontsize=16)

    # Plot the 7 physical features
    for i, (title, data) in enumerate(features_map):
        plt.subplot(3, 3, i + 1)
        x = np.linspace(0, len(y) / sr, len(data))
        mask = np.isfinite(data)
        if np.any(mask):
            plt.scatter(x, data, s=10, color='royalblue', alpha=0.6)
            if len(x[mask]) > 5:
                try:
                    z = np.polyfit(x[mask], data[mask], 2)
                    plt.plot(x, np.poly1d(z)(x), "r-", linewidth=2)
                except:
                    pass
        plt.title(title, fontweight='bold')

    # --- ADDED: PROBABILITY BAR CHART ---
    plt.subplot(3, 3, 9)
    colors = ['green' if c == emotion else 'grey' for c in classes]
    plt.bar(classes, probabilities, color=colors, alpha=0.7)
    plt.ylim(0, 1.0)
    plt.title("AI Confidence Map", fontweight='bold')
    for idx, val in enumerate(probabilities):
        plt.text(idx, val + 0.02, f"{val:.1%}", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"Research_Analysis_Bark_{bark_id}.png", dpi=300)
    plt.close()
    print(f"📊 Graph saved with Probability Map: Research_Analysis_Bark_{bark_id}.png")


def classify_and_plot(filename):
    research_model = BarkModel(sr=16000)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        y, sr = librosa.load(filename, sr=16000)
        y = librosa.util.normalize(y - np.mean(y))
        barks = detect_all_barks(y, sr)
        if not barks: return

        for i, bark_audio in enumerate(barks):
            dna_vector = extract_ml_features(bark_audio, sr)

            # AI Prediction & Probability
            if CLF and SCALER:
                dna_scaled = SCALER.transform(dna_vector.reshape(1, -1))
                predicted_emotion = CLF.predict(dna_scaled)[0]
                probabilities = CLF.predict_proba(dna_scaled)[0]
                classes = CLF.classes_
                confidence = np.max(probabilities)
            else:
                predicted_emotion, confidence, probabilities, classes = "Unknown", 0.0, [0], ["Unknown"]

            # Distance Logic
            current_dna_dict = research_model.extract_dna(bark_audio)
            min_dist = 0.0
            if EMOTION_LIB:
                target_list = EMOTION_LIB.get(predicted_emotion.lower(), [])
                distances = [research_model.compute_similarity(current_dna_dict, saved_dna)
                             for saved_dna in target_list if isinstance(saved_dna, dict)]
                if distances: min_dist = np.min(distances)

            print(f"--- BARK #{i + 1}: {predicted_emotion.upper()} ({confidence:.0%} Conf) ---")
            plot_research_graphs(bark_audio, sr, i + 1, predicted_emotion, min_dist, current_dna_dict, probabilities,
                                 classes)
    except Exception as e:
        print(f"Analysis Error: {e}")


def on_message(client, userdata, msg):
    global audio_buffer
    if msg.payload == b"START":
        audio_buffer = bytearray()
    elif msg.payload == b"END":
        if audio_buffer:
            with wave.open(SAVE_NAME, "wb") as f:
                f.setnchannels(1);
                f.setsampwidth(2);
                f.setframerate(16000);
                f.writeframes(audio_buffer)
            classify_and_plot(SAVE_NAME)
    else:
        audio_buffer.extend(msg.payload)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect(MQTT_BROKER)
client.subscribe(TOPIC)
print(f"🚀 SMART SERVER LIVE: Probability Mapping Enabled")
client.loop_forever()