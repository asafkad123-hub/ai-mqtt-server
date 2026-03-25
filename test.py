import paho.mqtt.client as mqtt
import wave, numpy as np, librosa, tensorflow as tf
import joblib, time
import tensorflow_hub as hub
from fastdtw import fastdtw
from collections import Counter

from model import compute_hybrid_features

# --- CONFIGURATION FOR HIVEMQ ---
MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_USE_TLS = False
TOPIC = "project/dog"
SAVE_NAME = "bark_analysis.wav"

# IMPORTANT: Must match training order exactly
CATEGORIES = ['relaxed', 'happy', 'sad', 'angry']

print("📥 Loading YAMNet...")
YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')

DOG_CLASSES = [67, 68, 69, 70, 71, 72, 73, 74, 75]

# Global gate (full recording)
BARK_THRESHOLD = 0.10

# Per-window dog gate (stronger, so non-dog windows get skipped)
WINDOW_DOG_THRESHOLD = 0.12  # try 0.10–0.20
MIN_WINDOWS_TO_DECIDE = 2    # if too many windows skipped -> fallback logic

print("📥 Loading Dual-Head model...")
MODEL = tf.keras.models.load_model('bark_mamba_model.keras')

try:
    DNA_LIB = joblib.load('bark_dna_library.pkl')
    print("🧬 DNA Library Loaded.")
except Exception:
    DNA_LIB = None
    print("⚠️ DNA Library missing.")

audio_buffer = bytearray()

# -----------------------------
# Balancing tweaks (TEST ONLY)
# -----------------------------
SOFTMAX_TEMPERATURE = 1.35
RUSSELL_MAX_VOTE = 0.55
DTW_VOTE_WEIGHT = 0.60
UNCERTAIN_MARGIN = 0.18

# -----------------------------
# Sliding window params
# -----------------------------
WIN_SEC = 3.0
HOP_SEC = 1.0

# Long crop around best YAMNet dog frame (so CROPPED yields multiple windows)
CROP_SEC = 9.0

# Optional silence trimming
TRIM_SILENCE = False
SILENCE_RMS_FRACTION = 0.25

# -----------------------------
# Anti-flip smoothing
# -----------------------------
# Only allow a change if the new label stays on top for N windows
# AND it beats the old label by at least SWITCH_MARGIN (avg prob).
HOLD_N = 2
SWITCH_MARGIN = 0.12


def save_float_wav(path, y_float, sr=16000):
    y = np.asarray(y_float, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int16.tobytes())


def apply_temperature(probs, T=1.0):
    p = np.array(probs, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0)
    if T == 1.0:
        return p.astype(np.float32)
    logits = np.log(p) / float(T)
    logits = logits - np.max(logits)
    p2 = np.exp(logits)
    p2 = p2 / np.sum(p2)
    return p2.astype(np.float32)


def yamnet_dog_conf_from_scores(scores):
    dog_scores = scores.numpy()[:, DOG_CLASSES]
    dog_conf_per_frame = np.max(dog_scores, axis=1)
    return float(np.max(dog_conf_per_frame)), float(np.mean(dog_conf_per_frame))


def crop_to_event(y, sr, window_sec=3.0, hop_length=256):
    win = int(window_sec * sr)
    if len(y) <= win:
        return np.pad(y, (0, win - len(y)))

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    peak_frame = int(np.argmax(rms))
    peak_sample = peak_frame * hop_length

    start = peak_sample - win // 2
    start = max(0, min(start, len(y) - win))
    return y[start:start + win]


def crop_to_dog_segment(y, sr, scores, window_sec=9.0, min_dog_conf=0.10):
    """
    Long crop around time where YAMNet is most confident it's a dog sound.
    """
    win = int(window_sec * sr)
    if len(y) <= win:
        return np.pad(y, (0, win - len(y)))

    dog_scores = scores.numpy()[:, DOG_CLASSES]
    dog_conf_per_frame = np.max(dog_scores, axis=1)
    best_frame = int(np.argmax(dog_conf_per_frame))
    best_conf = float(dog_conf_per_frame[best_frame])

    if best_conf < min_dog_conf:
        return None

    hop_sec = (len(y) / sr) / max(1, int(scores.shape[0]))
    center_time = best_frame * hop_sec
    center_sample = int(center_time * sr)

    start = center_sample - win // 2
    start = max(0, min(start, len(y) - win))
    return y[start:start + win]


def check_dtw_shape(y, sr):
    if not DNA_LIB:
        return "n/a", {}

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    target_raw = np.max(pitches * mags, axis=0)
    target_curve = [float(x) for x in target_raw.flatten()]
    if len(target_curve) == 0:
        return "n/a", {}

    mean_val = np.mean(target_curve)
    std_val = np.std(target_curve) + 1e-6
    target_curve = [(x - mean_val) / std_val for x in target_curve]

    simple_dist = lambda a, b: abs(a - b)

    best_emo = "n/a"
    min_dist = float('inf')
    distances = {}

    for emo, ref_curve in DNA_LIB.items():
        try:
            ref_raw = np.array(ref_curve).flatten()
            ref_clean = [float(x) for x in ref_raw]
            dist, _ = fastdtw(target_curve, ref_clean, dist=simple_dist)
            norm_dist = dist / (len(target_curve) + len(ref_clean))
            distances[str(emo).lower()] = float(norm_dist)
            if norm_dist < min_dist:
                min_dist = norm_dist
                best_emo = str(emo).lower()
        except Exception:
            continue

    return best_emo, distances


def quadrant_from_coords(valence, arousal, deadzone=0.20):
    mag = float(np.sqrt(valence**2 + arousal**2))
    if mag < deadzone:
        return "mixed/uncertain", mag

    if valence >= 0 and arousal >= 0:
        return "happy", mag
    if valence < 0 and arousal >= 0:
        return "angry", mag
    if valence < 0 and arousal < 0:
        return "sad", mag
    return "relaxed", mag


def decide_consensus(probs_cal, russell_label, russell_mag, dtw_winner):
    """
    Per-window consensus decision (classifier + Russell + DTW).
    """
    probs_cal = np.array(probs_cal, dtype=float)
    order = np.argsort(probs_cal)
    top_idx = int(order[-1])
    second_idx = int(order[-2])
    top_prob = float(probs_cal[top_idx])
    second_prob = float(probs_cal[second_idx])
    margin = top_prob - second_prob

    if margin < UNCERTAIN_MARGIN:
        return "mixed/uncertain", {"reason": f"low_margin={margin:.3f}"}

    mamba_pred = CATEGORIES[top_idx].lower()

    votes = {}
    votes[mamba_pred] = votes.get(mamba_pred, 0) + top_prob

    if russell_label in ['relaxed', 'happy', 'sad', 'angry']:
        r_weight = float(np.clip(russell_mag, 0.0, RUSSELL_MAX_VOTE))
        votes[russell_label] = votes.get(russell_label, 0) + r_weight

    if dtw_winner in ['relaxed', 'happy', 'sad', 'angry']:
        votes[dtw_winner] = votes.get(dtw_winner, 0) + DTW_VOTE_WEIGHT

    final = max(votes, key=votes.get) if votes else "mixed/uncertain"
    return final, votes


def analyze_one_window(y_win, sr):
    # Per-window YAMNet dog confidence
    y_norm = librosa.util.normalize(y_win - np.mean(y_win))
    scores, _, _ = YAMNET_MODEL(y_norm)
    dog_peak, dog_mean = yamnet_dog_conf_from_scores(scores)

    # If not dog-like, skip this window
    if dog_peak < WINDOW_DOG_THRESHOLD:
        return {
            "skipped": True,
            "reason": f"yamnet_dog_peak<{WINDOW_DOG_THRESHOLD}",
            "dog_peak": dog_peak,
            "dog_mean": dog_mean,
        }

    feats = compute_hybrid_features(y_win, sr)
    feats = np.expand_dims(feats, axis=0)

    preds = MODEL.predict(feats, verbose=0)
    probs_raw = preds[0][0]
    coords = preds[1][0]  # [valence, arousal]
    probs_cal = apply_temperature(probs_raw, T=SOFTMAX_TEMPERATURE)

    v = float(coords[0])
    a = float(coords[1])

    russell_label, russell_mag = quadrant_from_coords(v, a, deadzone=0.20)

    dtw_winner, dtw_distances = check_dtw_shape(y_win, sr)

    final, votes = decide_consensus(probs_cal, russell_label, russell_mag, dtw_winner)

    return {
        "skipped": False,
        "final": final,
        "probs_cal": probs_cal,
        "coords": coords,
        "valence": v,
        "arousal": a,
        "russell_label": russell_label,
        "russell_mag": russell_mag,
        "dtw_winner": dtw_winner,
        "dtw_distances": dtw_distances,
        "votes": votes,
        "dog_peak": dog_peak,
        "dog_mean": dog_mean,
    }


def smooth_labels(window_results, probs_list):
    """
    Prevent fast flipping:
    - base label from argmax each window (classifier head)
    - apply hold/hysteresis using average probs
    """
    labels = []
    for p in probs_list:
        labels.append(CATEGORIES[int(np.argmax(p))].lower())

    if not labels:
        return []

    smoothed = [labels[0]]
    hold_label = labels[0]

    for i in range(1, len(labels)):
        cur = labels[i]
        if cur == hold_label:
            smoothed.append(hold_label)
            continue

        # candidate change: require HOLD_N confirmations
        j_end = min(len(labels), i + HOLD_N)
        future = labels[i:j_end]
        if len(future) < HOLD_N or any(x != cur for x in future):
            smoothed.append(hold_label)
            continue

        # require probability margin over hold_label across those windows
        cur_idx = CATEGORIES.index(cur)
        hold_idx = CATEGORIES.index(hold_label)
        cur_mean = float(np.mean([probs_list[k][cur_idx] for k in range(i, j_end)]))
        hold_mean = float(np.mean([probs_list[k][hold_idx] for k in range(i, j_end)]))

        if (cur_mean - hold_mean) >= SWITCH_MARGIN:
            hold_label = cur

        smoothed.append(hold_label)

    return smoothed


def analyze_overlapping_windows(y_signal, sr, label=""):
    win = int(WIN_SEC * sr)
    hop = int(HOP_SEC * sr)

    if len(y_signal) < win:
        y_signal = np.pad(y_signal, (0, win - len(y_signal)))

    starts = list(range(0, max(1, len(y_signal) - win + 1), hop))
    windows = [y_signal[s:s + win] for s in starts]

    if TRIM_SILENCE and len(windows) >= 2:
        rms_vals = [float(np.sqrt(np.mean(w*w) + 1e-12)) for w in windows]
        max_rms = max(rms_vals) if rms_vals else 0.0
        keep = [i for i, r in enumerate(rms_vals) if r >= (SILENCE_RMS_FRACTION * max_rms)]
        if keep:
            windows = [windows[i] for i in keep]
            starts = [starts[i] for i in keep]

    results = []
    probs_list = []
    coords_list = []

    for i, y_win in enumerate(windows):
        res = analyze_one_window(y_win, sr)
        res["start_sec"] = float(starts[i] / sr)

        # keep only non-skipped windows
        if res.get("skipped"):
            continue

        results.append(res)
        probs_list.append(res["probs_cal"])
        coords_list.append([res["valence"], res["arousal"]])

    if len(results) < MIN_WINDOWS_TO_DECIDE:
        return {
            "label": label,
            "windows_total": len(windows),
            "windows_used": len(results),
            "probs_mean": None,
            "final_mean": "mixed/uncertain",
            "final_majority": "mixed/uncertain",
            "russell_mean_valence": 0.0,
            "russell_mean_arousal": 0.0,
            "russell_mean_label": "mixed/uncertain",
            "results": results,
            "smoothed_labels": [],
        }

    probs_stack = np.stack(probs_list, axis=0)
    probs_mean = np.mean(probs_stack, axis=0)

    # Mean Russell coordinates across windows (TRUE Russell aggregation)
    coords_arr = np.array(coords_list, dtype=np.float32)
    v_mean = float(np.mean(coords_arr[:, 0]))
    a_mean = float(np.mean(coords_arr[:, 1]))
    russell_mean_label, russell_mean_mag = quadrant_from_coords(v_mean, a_mean, deadzone=0.20)

    # Mean-prob decision (classifier head)
    final_mean = CATEGORIES[int(np.argmax(probs_mean))].lower()

    # Majority vote over smoothed classifier labels
    smoothed = smooth_labels(results, probs_list)
    finals_clean = [x for x in smoothed if x in ['relaxed', 'happy', 'sad', 'angry']]
    final_majority = Counter(finals_clean).most_common(1)[0][0] if finals_clean else "mixed/uncertain"

    return {
        "label": label,
        "windows_total": len(windows),
        "windows_used": len(results),
        "probs_mean": probs_mean,
        "final_mean": final_mean,
        "final_majority": final_majority,
        "russell_mean_valence": v_mean,
        "russell_mean_arousal": a_mean,
        "russell_mean_label": russell_mean_label,
        "results": results,
        "smoothed_labels": smoothed,
    }


def analyze():
    print("\n" + "=" * 60)
    print("🔬 STARTING (RUSSELL AVG + WINDOW YAMNET + SMOOTHING)")
    print("=" * 60)

    try:
        y, sr = librosa.load(SAVE_NAME, sr=16000)
        print(f"✅ Audio loaded: {len(y)} samples, {len(y) / sr:.2f} seconds")
    except Exception as e:
        print(f"❌ Audio Load Error: {e}")
        return

    print("\n--- 🧬 STEP 1: YAMNet dog gate (full recording) ---")
    y_norm = librosa.util.normalize(y - np.mean(y))
    scores, _, _ = YAMNET_MODEL(y_norm)

    dog_peak, dog_mean = yamnet_dog_conf_from_scores(scores)
    print(f" 🐕 Dog Confidence (peak): {dog_peak:.1%} | (mean): {dog_mean:.1%}")

    if dog_peak < BARK_THRESHOLD:
        print(" 🚫 REJECTED: Not a dog")
        print("=" * 60)
        return

    print(" ✅ PASSED: Valid dog sound")

    print("\n--- 🪟 STEP 2A: Sliding windows over FULL recording ---")
    out_full = analyze_overlapping_windows(y, sr, label="FULL")

    print("\n--- ✂️ STEP 2B: YAMNet crop (long) then sliding windows ---")
    y_event = crop_to_dog_segment(
        y=y,
        sr=sr,
        scores=scores,
        window_sec=CROP_SEC,
        min_dog_conf=BARK_THRESHOLD
    )
    if y_event is None:
        print("⚠️ YAMNet could not localize a strong dog segment; falling back to energy crop.")
        y_event = crop_to_event(y, sr, window_sec=CROP_SEC)

    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        detected_path = f"detected_bark_{ts}.wav"
        save_float_wav(detected_path, y_event, sr)
        print(f"🔊 Saved cropped segment to: {detected_path}")
    except Exception as e:
        print(f"⚠️ Could not save debug audio: {e}")

    out_crop = analyze_overlapping_windows(y_event, sr, label="CROPPED")

    def print_out(out):
        print("\n" + "-" * 60)
        print(f"RESULTS: {out['label']}")
        print("-" * 60)
        print(f"🪟 Windows total: {out['windows_total']} | used(after YAMNet gate): {out['windows_used']}")

        if out["probs_mean"] is not None:
            print("📊 Mean probs:")
            for i, cat in enumerate(CATEGORIES):
                print(f"  {cat.upper()}: {float(out['probs_mean'][i]):.3f}")
        else:
            print("📊 Mean probs: N/A (too few windows)")

        print(f"🎯 Final (mean probs): {out['final_mean'].upper()}")
        print(f"🧊 Final (smoothed majority): {out['final_majority'].upper()}")

        print(f"🎯 Russell mean coords: V={out['russell_mean_valence']:.3f}, A={out['russell_mean_arousal']:.3f}")
        print(f"🧭 Russell quadrant (from mean): {out['russell_mean_label'].upper()}")

    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    print_out(out_full)
    print_out(out_crop)

    print("\n" + "-" * 60)
    print("CROPPED per-window (used windows only):")
    print("-" * 60)
    for r in out_crop["results"]:
        top_idx = int(np.argmax(r["probs_cal"]))
        top_cat = CATEGORIES[top_idx]
        print(
            f"  t={r['start_sec']:.2f}s"
            f"  dog_peak={r['dog_peak']:.2f}"
            f"  pred={r['final']:>14s}"
            f"  top={top_cat}:{float(r['probs_cal'][top_idx]):.3f}"
            f"  russell=({r['valence']:.2f},{r['arousal']:.2f})->{r['russell_label']}"
        )

    print("=" * 60 + "\n")


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ SUCCESS: Connected to HiveMQ!")
        client.subscribe(TOPIC)
        print(f"👂 Listening for barks on: {TOPIC}")
    else:
        print(f"❌ Connection Failed. Return Code: {rc}")


def on_message(client, userdata, msg):
    global audio_buffer

    if msg.payload == b"START":
        print("\n🎤 Bark Incoming... Recording.")
        audio_buffer = bytearray()

    elif msg.payload == b"END":
        print(f"⏹️ END received. Total bytes: {len(audio_buffer)}")

        if len(audio_buffer) > 1000:
            with wave.open(SAVE_NAME, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_buffer)

            analyze()
            audio_buffer.clear()
        else:
            print("⚠️ Buffer too small, discarding.")

    else:
        if len(audio_buffer) % (4096 * 10) < 100:
            print(f" 📦 Recv {len(audio_buffer)} bytes...")
        audio_buffer.extend(msg.payload)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"🚀 Connecting to HiveMQ ({MQTT_HOST}:{MQTT_PORT})...")
    client.connect(MQTT_HOST, port=MQTT_PORT, keepalive=60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\n👋 Stopping...")
except Exception as e:
    print(f"❌ Connection Error: {e}")
