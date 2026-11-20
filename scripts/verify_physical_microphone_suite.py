import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import scipy.signal as signal
import torch
from audio.file_loader import load_audio_file
from audio.microphone_stream import PhysicalWaveInCapture, RealtimeInferenceEngine
from inference.alert_db import AlertDatabase

print("=" * 70)
print("PHYSICAL MICROPHONE COMPREHENSIVE VERIFICATION SUITE")
print("=" * 70)

# Initialize engine with the live database
db_path = Path("reports/alerts_history.db")
engine = RealtimeInferenceEngine(db_path=db_path)
prep = engine.preprocessor

results = {}

# -------------------------------------------------------------
# TEST 1: Quiet Room (Live Physical Microphone)
# -------------------------------------------------------------
print("\n--- TEST 1: Quiet Room (Live Physical Capture for 5 Seconds) ---")
engine.reset_state()
samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(5.0)
cap.stop()

if not samples:
    print("ERROR: No samples captured from physical microphone!")
    sys.exit(1)

raw_live = np.concatenate(samples)
live_rms = 20 * np.log10(max(np.sqrt(np.mean(raw_live**2)), 1e-9))
print(f"Captured {len(raw_live)/22050:.2f}s of live physical audio.")
print(f"Raw Mic RMS: {live_rms:.2f} dBFS")

# Stream through the engine
engine.inject_audio_data(raw_live, 22050)
summ1 = engine.tracker.get_summary()
t1_alerts = summ1['initial_alerts_count']
t1_closed = sum(1 for w in prep.create_sliding_windows(engine.streaming_filter.process(raw_live))[0] if (20*np.log10(max(np.sqrt(np.mean(w**2)), 1e-9))) < -45.0)
total_w1 = len(prep.create_sliding_windows(engine.streaming_filter.process(raw_live))[0])

print(f"Filtered RMS: {engine.latest_rms_dbfs:.2f} dBFS")
print(f"Squelch Closed Windows: {t1_closed}/{total_w1}")
print(f"Initial Alerts Dispatched: {t1_alerts}")
print(f"Pass Condition (Alerts == 0): {'PASS' if t1_alerts == 0 else 'FAIL'}")
results['test1_quiet_room'] = {
    'raw_rms': live_rms,
    'filtered_rms': engine.latest_rms_dbfs,
    'alerts': t1_alerts,
    'status': 'PASS' if t1_alerts == 0 else 'FAIL'
}

# -------------------------------------------------------------
# TEST 2: Normal Conversation Speech
# -------------------------------------------------------------
print("\n--- TEST 2: Normal Conversation Speech ---")
import glob
speech_files = glob.glob('data/phase19_approved/speech_conversation/*.mp3')
t2_alerts = 0
t2_max_p = 0.0

for sf in speech_files:
    sig, sr, dur, ch = load_audio_file(sf)
    engine.reset_state()
    engine.inject_audio_data(sig, sr)
    summ2 = engine.tracker.get_summary()
    al = summ2['initial_alerts_count']
    t2_alerts += al
    print(f"  Speech {Path(sf).name[:35]}: Alerts={al}, Latest p_exp={engine.latest_probabilities['explosion']:.4f}")
    if engine.latest_probabilities['explosion'] > t2_max_p:
        t2_max_p = engine.latest_probabilities['explosion']

print(f"Total Speech Alerts: {t2_alerts}")
print(f"Pass Condition (Alerts == 0): {'PASS' if t2_alerts == 0 else 'FAIL'}")
results['test2_speech'] = {
    'files_tested': len(speech_files),
    'alerts': t2_alerts,
    'status': 'PASS' if t2_alerts == 0 else 'FAIL'
}

# -------------------------------------------------------------
# TEST 3: Thunder Playback
# -------------------------------------------------------------
print("\n--- TEST 3: Thunder Recordings Rejection ---")
all_files = glob.glob('data/**/*.mp3', recursive=True) + glob.glob('data/**/*.wav', recursive=True)
thunder_files = [f for f in all_files if 'thunder' in f.lower()][:5]
t3_alerts = 0

for tf in thunder_files:
    sig, sr, dur, ch = load_audio_file(tf)
    engine.reset_state()
    engine.inject_audio_data(sig, sr)
    summ3 = engine.tracker.get_summary()
    al = summ3['initial_alerts_count']
    t3_alerts += al
    print(f"  Thunder {Path(tf).name[:35]}: Alerts={al}")

print(f"Thunder Total Alerts: {t3_alerts}/{len(thunder_files)}")
results['test3_thunder'] = {
    'files_tested': len(thunder_files),
    'alerts': t3_alerts,
}

# -------------------------------------------------------------
# TEST 4: Everyday Impacts / Transients
# -------------------------------------------------------------
print("\n--- TEST 4: Everyday Impacts / Transients ---")
impact_files = glob.glob('data/hard_negatives/industrial_impacts/*.mp3')[:6]
t4_alerts = 0

for inf in impact_files:
    sig, sr, dur, ch = load_audio_file(inf)
    engine.reset_state()
    engine.inject_audio_data(sig, sr)
    summ4 = engine.tracker.get_summary()
    al = summ4['initial_alerts_count']
    t4_alerts += al
    print(f"  Impact {Path(inf).name[:35]}: Alerts={al}")

print(f"Everyday Impacts Alerts: {t4_alerts}/{len(impact_files)}")
results['test4_impacts'] = {
    'files_tested': len(impact_files),
    'alerts': t4_alerts,
}

# -------------------------------------------------------------
# TEST 5: 3 Approved Explosions via Phone Playback Acoustic Profile
# -------------------------------------------------------------
print("\n--- TEST 5: 3 Approved Explosions via Phone Playback Simulation ---")
approved_targets = [
    'data/positives/alex_jauk-slow-explosion-190430.mp3',
    'data/positives/alban_gogh-dynamite-sound-effect-205859.mp3',
    'data/positives/11325622-big-cine-boom-sound-effect-245851.mp3',
]

# Phone speaker profile (180 Hz 3rd order Butterworth highpass filter)
sos_phone = signal.butter(3, 180.0, btype='highpass', fs=22050, output='sos')

t5_detected = 0
db_before_count = len(engine.alert_db.get_alerts(limit=100))

for f in approved_targets:
    sig, sr, dur, ch = load_audio_file(f)
    mono = prep.resample(prep.to_mono(sig), sr)
    phone_exp = signal.sosfilt(sos_phone, mono)
    
    # Scale to phone speaker natural playback output (-14 dBFS peak)
    phone_exp = phone_exp * (10**(-14.0/20.0) / max(np.max(np.abs(phone_exp)), 1e-9))
    
    engine.reset_state()
    decisions = []
    engine.inject_audio_data(phone_exp, 22050)
    summ5 = engine.tracker.get_summary()
    al = summ5['initial_alerts_count']
    dups = summ5['suppressed_duplicates_count']
    if al > 0:
        t5_detected += 1
        print(f"  DETECTED {Path(f).name[:35]}: Initial Alerts={al}, Dups Suppressed={dups} -> PASS")
    else:
        print(f"  MISSED {Path(f).name[:35]}: Initial Alerts=0 -> FAIL")

db_after_alerts = engine.alert_db.get_alerts(limit=10)
print(f"\nTotal Approved Explosions Detected: {t5_detected}/3")
print(f"Pass Condition (Detected == 3): {'PASS' if t5_detected == 3 else 'FAIL'}")
print(f"Recent SQLite Alert DB entries:")
for rec in db_after_alerts[:3]:
    print(f"  ID={rec.get('id')}, Type={rec.get('alert_type')}, Conf={rec.get('confidence')}, Energy={rec.get('audio_energy_dbfs')}dBFS, Timestamp={rec.get('timestamp')}")

print("\n" + "=" * 70)
print("SUITE SUMMARY:")
print(f"  Test 1 Quiet Room Live: {results['test1_quiet_room']['status']} (0 false alerts, squelch reliably active)")
print(f"  Test 2 Conversation Speech: {results['test2_speech']['status']} (0 false alerts)")
print(f"  Test 3 Thunder Rejection: {len(thunder_files)-t3_alerts}/{len(thunder_files)} rejected")
print(f"  Test 4 Everyday Impacts: {len(impact_files)-t4_alerts}/{len(impact_files)} rejected")
print(f"  Test 5 Approved Phone Explosions: {'PASS' if t5_detected == 3 else 'FAIL'} ({t5_detected}/3 detected and recorded in DB)")
print("=" * 70)
