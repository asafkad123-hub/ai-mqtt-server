import paho.mqtt.client as mqtt
import socket, wave, librosa, os, joblib, warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from final_model import BarkModel
from recognize_bark import detect_all_barks

# --- SETTINGS ---
MQTT_BROKER = "broker.hivemq.com"
TOPIC = "kurland/dog/bark"
SAVE_NAME = 'esp32_audio.wav'
DNA_LIBRARY = 'bark_library.npy'
MODEL_PATH = "bark_gatekeeper_ai.pkl"
SCALER_PATH = "bark_scaler.pkl"

# --- INITIALIZE YAMNET ---
print("📥 Loading YAMNet Intelligence...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
DOG_VECTORS = [69, 70, 71, 72, 73, 74, 75]


# --- LOAD AI MODELS (With Error Feedback) ---
def load_ai():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"❌ WARNING: AI Files missing in {os.getcwd()}. Predictions will be UNKNOWN.")
        return None, None
    try:
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ AI Models loaded successfully.")
        return clf, scaler
    except Exception as e:
        print(f"❌ Failed to load AI: {e}")
        return None, None


CLF, SCALER = load_ai()
EMOTION_LIB = np.load(DNA_LIBRARY, allow_pickle=True).item() if os.path.exists(DNA_LIBRARY) else None

audio_buffer = bytearray()


def extract_ml_features(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
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

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mask = magnitudes > np.median(magnitudes)
    pitch_val = np.mean(pitches[mask]) if np.any(mask) else 0
    amplitude_val = np.mean(librosa.feature.rms(y=y))
    brightness_val = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return np.concatenate([mfcc_combined, [pitch_val, amplitude_val, brightness_val]])


def classify_and_plot(filename):
    research_model = BarkModel(sr=16000)
    warnings.filterwarnings("ignore")

    try:
        y, sr = librosa.load(filename, sr=16000)
        y_norm = librosa.util.normalize(y - np.mean(y))

        # 1. YAMNET CHECK
        scores, _, _ = yamnet_model(y_norm)
        yamnet_conf = np.max(np.mean(scores, axis=0)[DOG_VECTORS])

        if yamnet_conf < 0.15:
            print(f"❌ Rejected: Ambient Noise ({yamnet_conf:.1%})")
            return

        # 2. ISOLATION
        barks = detect_all_barks(y_norm, sr)
        if not barks: return

        print(f"\n--- 📋 SESSION ASSESSMENT ({len(barks)} BARKS) ---")
        session_emotions = []

        for i, bark_audio in enumerate(barks):
            dna_vector = extract_ml_features(bark_audio, sr)

            # Predict only if CLF exists
            if CLF and SCALER:
                dna_scaled = SCALER.transform(dna_vector.reshape(1, -1))
                pred = CLF.predict(dna_scaled)[0]
                prob = CLF.predict_proba(dna_scaled)[0]
                conf = np.max(prob)
                classes = CLF.classes_
            else:
                pred, conf, prob, classes = "Unknown", 0.0, [0], ["Unknown"]

            session_emotions.append(pred)
            print(f"Bark #{i + 1}: {pred.upper()} ({conf:.1%})")

            # 3. GRAPH FIX: Only plot if we have valid data
            if conf > 0:
                try:
                    dna_dict = research_model.extract_dna(bark_audio)
                    # Use a placeholder for plot_research_graphs or your actual import
                    # Assuming plot_research_graphs is imported from your other files
                    from final_model import plot_research_graphs
                    plot_research_graphs(bark_audio, sr, i + 1, pred, 0.0, dna_dict, prob, classes, yamnet_conf)
                except Exception as g_err:
                    pass

                    # FINAL SUMMARY
        most_common = max(set(session_emotions), key=session_emotions.count)
        print(f"SUMMARY: This sequence sounds mostly like: {most_common.upper()}")
        print("-------------------------------------------\n")

    except Exception as e:
        print(f"💥 Analysis Error: {e}")


def on_message(client, userdata, msg):
    global audio_buffer
    if msg.payload == b"START":
        audio_buffer = bytearray()
    elif msg.payload == b"END":
        if audio_buffer:
            with wave.open(SAVE_NAME, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_buffer)
            classify_and_plot(SAVE_NAME)
    else:
        audio_buffer.extend(msg.payload)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect(MQTT_BROKER)
client.subscribe(TOPIC)
print(f"🚀 SMART SERVER LIVE | Topic: {TOPIC}")
client.loop_forever()