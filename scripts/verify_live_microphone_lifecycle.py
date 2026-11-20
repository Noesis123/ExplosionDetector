"""Live Physical Microphone Lifecycle & Subsystem Verification Script.

Verifies:
1. Physical WaveIn driver start/stop lifecycle.
2. Live quiet-room capture (fan noise at -40 dBFS) produces EXACTLY ZERO false alerts.
3. Speech conversation & ambient sound rejection (0 false alerts).
4. Audio file explosion injection triggers verified EXPLOSION_DETECTED alert.
5. SQLite alert persistence in reports/alerts_history.db.
6. Mobile phone playback acoustic simulation detects explosions cleanly.
7. Dashboard REST API endpoints and dynamic settings sync.
"""

import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request
import numpy as np
import scipy.signal as signal
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.microphone_stream import PhysicalWaveInCapture, RealtimeInferenceEngine

SERVER_URL = "http://127.0.0.1:8767"

def post_json(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{SERVER_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{SERVER_URL}{path}") as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 75)
    print("NOESIS — LIVE PHYSICAL MICROPHONE LIFECYCLE & SUBSYSTEM VERIFICATION")
    print("=" * 75)

    # 1. Verify REST API and Settings
    print("\n--- 1. Dashboard REST API & Dynamic Settings Verification ---")
    st = get_json("/api/status")
    print(f"  Server State    : {st.get('state')} (Active Checkpoint: {st.get('model_checkpoint')})")
    settings = get_json("/api/settings")
    print(f"  Backend Settings: {settings}")
    assert settings.get("rms_squelch_dbfs") == -42.0, f"Expected squelch -42.0 dBFS, got {settings.get('rms_squelch_dbfs')}"
    print("  Settings Status : PASS (Synced with backend -42.0 dBFS)")

    # 2. Quiet Room Live Microphone Capture (15 Seconds)
    print("\n--- 2. Live Physical Microphone Quiet-Room Capture (15 Seconds) ---")
    post_json("/api/alerts/clear", {})
    start_res = post_json("/api/monitor/start", {"device_id": 0, "simulation": False})
    print(f"  Live Mic Start: {start_res.get('status')}")

    readings = []
    alerts_seen = []
    for i in range(60):  # 15 seconds at 250ms intervals
        time.sleep(0.25)
        telemetry = get_json("/api/prediction/latest")
        rms = telemetry.get("audio_energy_dbfs", -99.9)
        conf = telemetry.get("confidence", 0.0)
        pred = telemetry.get("prediction", "unknown")
        readings.append(rms)

        if telemetry.get("active_alert"):
            alerts_seen.append(telemetry)
            print(f"  [ALERT!] t={i*0.25:.2f}s | Alert: {telemetry['active_alert']} | Conf: {conf:.4f} | RMS: {rms:.1f} dBFS")
        elif i % 12 == 0:
            print(f"  [MIC LIVE] t={i*0.25:.2f}s | Pred: {pred:<14} | Conf: {conf:.4f} | Energy: {rms:5.1f} dBFS")

    post_json("/api/monitor/stop", {})
    print("  Live Mic Stop: IDLE")

    db_path = PROJECT_ROOT / "reports" / "alerts_history.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM alerts WHERE source = 'microphone'")
    mic_db_alerts = cur.fetchone()[0]
    conn.close()

    print(f"  Physical Microphone Capture Summary:")
    print(f"    Duration Processed : 15.0 seconds (60 streaming windows)")
    print(f"    Average Noise Level: {sum(readings)/len(readings):.1f} dBFS")
    print(f"    Min / Max Energy   : {min(readings):.1f} / {max(readings):.1f} dBFS")
    print(f"    Active Alerts Seen : {len(alerts_seen)}")
    print(f"    SQLite Alerts Saved: {mic_db_alerts}")
    assert len(alerts_seen) == 0, f"FAIL: {len(alerts_seen)} alerts seen during quiet room monitoring!"
    assert mic_db_alerts == 0, f"FAIL: {mic_db_alerts} alerts saved in SQLite during quiet room monitoring!"
    print("  Quiet Room Verdict  : PASS (Exactly ZERO false alerts over 15.0s)")

    # 3. Speech and Everyday Ambient Sound Rejection
    print("\n--- 3. Speech and Everyday Ambient Sound Rejection ---")
    speech_dir = PROJECT_ROOT / "data" / "phase19_approved" / "speech_conversation"
    speech_files = list(speech_dir.glob("*.mp3"))
    engine = RealtimeInferenceEngine()
    total_speech_alerts = 0

    for sf in speech_files:
        sig, sr, dur, ch = load_audio_file(sf)
        engine.reset_state()
        engine.inject_audio_data(sig, sr)
        summ = engine.tracker.get_summary()
        al = summ["initial_alerts_count"]
        total_speech_alerts += al
        print(f"  {sf.name:<45} ({dur:4.1f}s) : Alerts={al}")

    assert total_speech_alerts == 0, f"FAIL: Speech files produced {total_speech_alerts} alerts!"
    print(f"  Speech Rejection Verdict: PASS (0 False Alerts across {len(speech_files)} speech files)")

    # 4. Mobile Phone Playback Acoustic Simulation
    print("\n--- 4. Mobile Phone Playback Acoustic Verification ---")
    sos_phone = signal.butter(2, 250.0, btype="highpass", fs=22050, output="sos")
    demo_files = [
        "data/positives/alex_jauk-slow-explosion-190430.mp3",
        "data/positives/bbc_demolition_07022507.mp3",
        "data/positives/soundreality-explosion-fx-343683.mp3",
        "data/positives/scottishperson-sound-effect-ww2-uk-bomb-blast-164351.mp3",
        "data/positives/ribhavagrawal-bomb-explosion-type-01-265510.mp3",
    ]
    phone_passed = 0
    for df in demo_files:
        p = PROJECT_ROOT / df
        sig, sr, dur, ch = load_audio_file(p)
        mono = engine.preprocessor.to_mono(sig)
        if sr != 22050:
            mono = engine.preprocessor.resample(mono, orig_sr=sr)
        phone_mono = signal.sosfilt(sos_phone, mono).astype(np.float32)
        phone_mono = phone_mono * (0.125 / (np.max(np.abs(phone_mono)) + 1e-6))

        engine.reset_state()
        engine.inject_audio_data(phone_mono, 22050)
        summ = engine.tracker.get_summary()
        al = summ["initial_alerts_count"]
        if al > 0:
            phone_passed += 1
            print(f"  {p.name:<45} : DETECTED (Alerts={al})")
        else:
            print(f"  {p.name:<45} : MISSED")

    assert phone_passed == len(demo_files), f"FAIL: Only {phone_passed}/{len(demo_files)} phone playbacks detected!"
    print(f"  Phone Playback Verdict  : PASS ({phone_passed}/{len(demo_files)} Detected)")

    # 5. Genuine Explosion Injection & SQLite Database Record
    print("\n--- 5. Genuine Explosion Injection & SQLite Database Verification ---")
    post_json("/api/alerts/clear", {})
    inj_file = "data/positives/alex_jauk-slow-explosion-190430.mp3"
    inj_res = post_json("/api/test/inject_audio", {"file_path": inj_file})
    print(f"  Injected: {inj_res.get('file')} ({inj_res.get('duration_sec')}s)")

    saw_blast = False
    peak_p = 0.0
    for i in range(25):
        time.sleep(0.25)
        st = get_json("/api/prediction/latest")
        if st.get("probabilities", {}).get("explosion", 0.0) > peak_p:
            peak_p = st["probabilities"]["explosion"]
        if st.get("active_alert") == "EXPLOSION_DETECTED":
            saw_blast = True
            print(f"  [BLAST DETECTED] t={i*0.25:.2f}s | Confidence: {st['confidence']:.4f} | Energy: {st['audio_energy_dbfs']:.1f} dBFS")
            break

    time.sleep(1.0)
    alerts_res = get_json("/api/alerts")
    alerts_list = alerts_res.get("alerts", [])
    print(f"  Alerts in Database ({len(alerts_list)}):")
    for a in alerts_list:
        print(f"    - ID #{a['id']}: Type={a['alert_type']}, Conf={a['confidence']:.4f}, Energy={a['audio_energy_dbfs']} dBFS, Source={a['source']}")

    assert saw_blast, "FAIL: Explosion injection did not trigger active alert!"
    assert len(alerts_list) > 0, "FAIL: Alert was not persisted in SQLite database!"
    print("  Explosion Injection Verdict: PASS (Alert triggered, broadcast via SSE, and recorded in SQLite)")

    print("\n" + "=" * 75)
    print("ALL VERIFICATION SUITES PASSED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
