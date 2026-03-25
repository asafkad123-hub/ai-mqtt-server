#!/usr/bin/env python3
"""
sensor_receiver.py - Server-side script for receiving IMU + HR/HRV data
Subscribes to MQTT topic "project/dog/data" and logs all sensor readings
"""

import paho.mqtt.client as mqtt
import json
from datetime import datetime
import os
import csv

# ==========================================
# CONFIGURATION
# ==========================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_DATA = "project/dog/data"
CLIENT_ID = "computer_sensor_receiver"

# Output files
LOG_FILE = "sensor_log.txt"
CSV_FILE = "sensor_data.csv"

# ==========================================
# GLOBAL VARIABLES
# ==========================================
data_count = 0
csv_writer = None
csv_file_handle = None

# ==========================================
# CSV INITIALIZATION
# ==========================================
def init_csv():
    """Initialize CSV file with headers"""
    global csv_writer, csv_file_handle

    # Check if file exists to decide on header
    file_exists = os.path.exists(CSV_FILE)

    csv_file_handle = open(CSV_FILE, 'a', newline='')
    csv_writer = csv.writer(csv_file_handle)

    # Write header if new file
    if not file_exists:
        csv_writer.writerow([
            'timestamp',
            'avg_hr',
            'avg_rmssd',
            'avg_sdnn',
            'baseline_hr',
            'baseline_rmssd',
            'baseline_sdnn',
            'delta_hr_pct',
            'delta_rmssd_pct',
            'delta_sdnn_pct',
            'valence',
            'arousal',
            'emotion',
            'dominant_position',
            'total_beats',
            'contact_present',
            'position_counts'
        ])
        print(f"✅ Created new CSV file: {CSV_FILE}")
    else:
        print(f"📄 Appending to existing CSV file: {CSV_FILE}")

# ==========================================
# MQTT CALLBACKS
# ==========================================
def on_connect(client, userdata, flags, rc, properties=None):
    """Called when connected to MQTT broker"""
    if rc == 0:
        print("=" * 60)
        print("✅ CONNECTED TO MQTT BROKER")
        print("=" * 60)
        print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
        print(f"Topic: {TOPIC_DATA}")
        print("Waiting for sensor data from ESP32...")
        print("=" * 60)
        client.subscribe(TOPIC_DATA)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    """Called when a message is received"""
    global data_count

    try:
        # Parse JSON data
        data = json.loads(msg.payload.decode())
        data_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ==========================================
        # CONSOLE OUTPUT - DETAILED
        # ==========================================
        print("\n" + "=" * 60)
        print(f"📦 SENSOR DATA RECEIVED #{data_count}")
        print("=" * 60)
        print(f"⏰ Timestamp: {timestamp}")

        print("\n📈 HEART RATE VARIABILITY:")
        print(f"   HR:         {data.get('avg_hr', 'N/A'):>8} bpm")
        print(f"   RMSSD:      {data.get('avg_rmssd', 'N/A'):>8} ms")
        print(f"   SDNN:       {data.get('avg_sdnn', 'N/A'):>8} ms")
        print(f"   Total Beats: {data.get('total_beats', 'N/A')}")
        print(f"   Contact:     {'YES ✓' if data.get('contact_present') else 'NO ✗'}")

        print("\n📊 BASELINE COMPARISON:")
        print(f"   Baseline HR:    {data.get('baseline_hr', 'N/A')} bpm")
        print(f"   Baseline RMSSD: {data.get('baseline_rmssd', 'N/A')} ms")
        print(f"   Baseline SDNN:  {data.get('baseline_sdnn', 'N/A')} ms")
        print(f"   Delta HR:       {data.get('delta_hr_pct', 'N/A'):>+6}%")
        print(f"   Delta RMSSD:    {data.get('delta_rmssd_pct', 'N/A'):>+6}%")
        print(f"   Delta SDNN:     {data.get('delta_sdnn_pct', 'N/A'):>+6}%")

        print("\n🧭 RUSSELL'S CIRCUMPLEX MODEL:")
        valence = data.get('valence', 0)
        arousal = data.get('arousal', 0)
        print(f"   Valence: {valence:>+7.3f}  ({'Positive' if valence >= 0 else 'Negative'})")
        print(f"   Arousal: {arousal:>+7.3f}  ({'High' if arousal >= 0.5 else 'Low'})")

        print("\n🏃 POSITION ANALYSIS:")
        pos_counts = data.get('position_counts', {})
        total_samples = sum(pos_counts.values()) if pos_counts else 0
        print(f"   Dominant Position: {data.get('dominant_position', 'UNKNOWN')}")
        if pos_counts:
            print("   Distribution:")
            for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1])[:5]:
                pct = (count / total_samples) * 100 if total_samples > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"      {pos:30s}: {count:4d} ({pct:5.1f}%) {bar}")

        print("\n🎭 FINAL EMOTION:")
        emotion = data.get('emotion', 'UNKNOWN')
        print(f"   >>> {emotion.upper()} <<<")

        print("=" * 60)

        # ==========================================
        # LOG TO TEXT FILE
        # ==========================================
        with open(LOG_FILE, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Emotion: {emotion}\n")
            f.write(f"HR: {data.get('avg_hr', 'N/A')} bpm, RMSSD: {data.get('avg_rmssd', 'N/A')} ms, SDNN: {data.get('avg_sdnn', 'N/A')} ms\n")
            f.write(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}\n")
            f.write(f"Dominant Position: {data.get('dominant_position', 'UNKNOWN')}\n")
            f.write(f"Position Counts: {json.dumps(pos_counts)}\n")
            f.write(f"{'=' * 60}\n")

        # ==========================================
        # LOG TO CSV FILE
        # ==========================================
        csv_writer.writerow([
            timestamp,
            data.get('avg_hr', ''),
            data.get('avg_rmssd', ''),
            data.get('avg_sdnn', ''),
            data.get('baseline_hr', ''),
            data.get('baseline_rmssd', ''),
            data.get('baseline_sdnn', ''),
            data.get('delta_hr_pct', ''),
            data.get('delta_rmssd_pct', ''),
            data.get('delta_sdnn_pct', ''),
            data.get('valence', ''),
            data.get('arousal', ''),
            data.get('emotion', ''),
            data.get('dominant_position', ''),
            data.get('total_beats', ''),
            data.get('contact_present', ''),
            json.dumps(pos_counts)
        ])
        csv_file_handle.flush()  # Ensure data is written immediately

        print(f"\n💾 Data saved to {LOG_FILE} and {CSV_FILE}")

    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON: {e}")
        print(f"Raw payload: {msg.payload}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")

def on_disconnect(client, userdata, rc, properties=None):
    """Called when disconnected from MQTT broker"""
    if rc != 0:
        print(f"\n⚠️ Unexpected disconnection. Code: {rc}")
        print("Attempting to reconnect...")

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    print("\n" + "=" * 60)
    print("🐕 DOG EMOTION SENSOR DATA RECEIVER")
    print("=" * 60)
    print("This script receives IMU + HR/HRV data from ESP32")
    print("(Audio is handled by a separate script)")
    print("=" * 60 + "\n")

    # Initialize CSV
    init_csv()

    # Setup MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        print(f"🔗 Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)

        # Start loop
        print("✅ Connected! Listening for sensor data...")
        print("Press Ctrl+C to stop\n")
        client.loop_forever()

    except KeyboardInterrupt:
        print("\n\n👋 Stopping sensor receiver...")
        client.disconnect()
        if csv_file_handle:
            csv_file_handle.close()
        print(f"✅ Data saved to:")
        print(f"   • {LOG_FILE}")
        print(f"   • {CSV_FILE}")
        print(f"\nTotal readings received: {data_count}")

    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        print("Make sure you have internet connection!")

if __name__ == "__main__":
    main()
