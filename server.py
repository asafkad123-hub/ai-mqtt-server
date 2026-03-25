import paho.mqtt.client as mqtt
import wave, numpy as np, librosa, tensorflow as tf
import ssl, joblib, os
import tensorflow_hub as hub
from fastdtw import fastdtw
from model import compute_hybrid_features

# --- CONFIGURATION ---
BBOT_HOST = "mqtt.beebotte.com"
BBOT_TOKEN = "token_vtLYbEd3XQQuYhHi"
TOPIC = "project/dog"
SAVE_NAME = "bark_analysis.wav"

# PATH TO THE CERTIFICATE YOU DOWNLOADED
# Make sure "mqtt.beebotte.com.pem" is in the same folder as this script
CERT_FILE = "mqtt.beebotte.com.pem"

CATEGORIES = ['happy', 'sad', 'angry']

# 1. Load YAMNet
print("📥 Loading YAMNet...")
YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
DOG_CLASSES = [67, 68, 69, 70, 71]
BARK_THRESHOLD = 0.15

# 2. Load Mamba
print("📥 Loading Mini-Mamba (Hybrid)...")
MODEL = tf.keras.models.load_model('bark_mamba_model.keras')

# 3. Load DNA Library
try:
    DNA_LIB = joblib.load('bark_dna_library.pkl')
    print("🧬 DNA Library Loaded.")
except:
    DNA_LIB = None
    print("⚠️ DNA Library missing.")

audio_buffer = bytearray()

def check_dtw_shape(y, sr):
    """Finds which emotion shape is closest using FastDTW."""
    if not DNA_LIB: return "N/A"
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    target_raw = np.max(pitches * mags, axis=0)
    target_curve = [float(x) for x in target_raw.flatten()]
    if len(target_curve) == 0: return "N/A"
    mean_val = np.mean(target_curve)
    std_val = np.std(target_curve) + 1e-6
    target_curve = [(x - mean_val) / std_val for x in target_curve]
    best_emo = "None"
    min_dist = float('inf')
    simple_dist = lambda a, b: abs(a - b)
    for emo, ref_curve in DNA_LIB.items():
        try:
            ref_raw = np.array(ref_curve).flatten()
            ref_clean = [float(x) for x in ref_raw]
            dist, _ = fastdtw(target_curve, ref_clean, dist=simple_dist)
            norm_dist = dist / (len(target_curve) + len(ref_clean))
            if norm_dist < min_dist:
                min_dist = norm_dist
                best_emo = emo
        except: continue
    return best_emo

def analyze():
    print("\n" + "=" * 50)
    print("🔬 HYBRID MAMBA ANALYSIS")
    try:
        y, sr = librosa.load(SAVE_NAME, sr=16000)
    except Exception as e:
        print(f"❌ Audio Load Error: {e}")
        return
    y_yam = librosa.util.normalize(y)[:16000 * 5]
    scores, _, _ = YAMNET_MODEL(y_yam)
    dog_conf = np.max(scores.numpy().mean(axis=0)[DOG_CLASSES])
    if dog_conf < BARK_THRESHOLD:
        print(f"🚫 REJECTED: Not a dog ({dog_conf:.1%})")
        return
    features = compute_hybrid_features(y, sr)
    features = np.expand_dims(features, axis=0)
    probs = MODEL.predict(features)[0]
    mamba_pred = CATEGORIES[np.argmax(probs)]
    dtw_pred = check_dtw_shape(y, sr)
    print(f"🧠 Brain: {mamba_pred.upper()} ({np.max(probs):.1%})")
    print(f"📏 Shape: {dtw_pred.upper()}")
    print("=" * 50)

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ SUCCESS: Connected to Beebotte!")
        client.subscribe(TOPIC)
        print(f"👂 Listening for barks on: {TOPIC}")
    else:
        print(f"❌ Connection Refused. Return Code: {rc}")

def on_message(client, userdata, msg):
    global audio_buffer
    if msg.payload == b"START":
        print("\n🎤 Bark Incoming... Recording.")
        audio_buffer = bytearray()
    elif msg.payload == b"END":
        print("⏹️ END received.")
        if len(audio_buffer) > 10000:
            with wave.open(SAVE_NAME, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_buffer)
            analyze()
            audio_buffer.clear()
    else:
        audio_buffer.extend(msg.payload)

# --- CLIENT SETUP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(BBOT_TOKEN, password=None)
client.on_connect = on_connect
client.on_message = on_message

# --- CERTIFICATE SETUP ---
print(f"🔐 Setting up TLS with certificate: {CERT_FILE}...")

if not os.path.exists(CERT_FILE):
    print(f"⚠️ ERROR: Certificate file '{CERT_FILE}' not found! Download it from Beebotte.")
    exit(1)

# Create a default context that uses the specific CA file
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.load_verify_locations(CERT_FILE) # Load your file
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE # Ignore errors
# Apply context to Paho
client.tls_set_context(context)

# DEBUG LOGS (Enable if connection fails)
# client.on_log = lambda client, userdata, level, buf: print(f"LOG: {buf}")

try:
    print(f"🚀 Connecting to Beebotte ({BBOT_HOST}:8883)...")
    # Increase keepalive to 120 to reduce timeouts
    client.connect(BBOT_HOST, port=8883, keepalive=120)
    client.loop_forever()
except Exception as e:
    print(f"❌ Connection Error: {e}")
