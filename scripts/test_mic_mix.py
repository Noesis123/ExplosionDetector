import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.signal as signal
from audio.file_loader import load_audio_file
from audio.microphone_stream import PhysicalWaveInCapture, RealtimeInferenceEngine
import time

# Capture 4 seconds of real mic background noise (fan + room)
samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(3.0)
cap.stop()
mic_bg = np.concatenate(samples)

sos = signal.butter(4, 250.0, btype='highpass', fs=22050, output='sos')
engine = RealtimeInferenceEngine()

sig, sr, dur, ch = load_audio_file('data/positives/alex_jauk-slow-explosion-190430.mp3')
mono = engine.preprocessor.resample(engine.preprocessor.to_mono(sig), sr)
phone_sig = signal.sosfilt(sos, mono)

# Scale phone signal to -35 dBFS peak
cur_peak = np.max(np.abs(phone_sig))
phone_sig = phone_sig * (10**(-35.0/20.0) / max(cur_peak, 1e-9))

# Tile mic background to match
reps = int(np.ceil(len(phone_sig) / len(mic_bg)))
bg_tile = np.tile(mic_bg, reps)[:len(phone_sig)]

mixed = phone_sig + bg_tile

print(f"Background noise RMS: {20*np.log10(np.sqrt(np.mean(bg_tile**2))):.2f} dBFS")
print(f"Phone signal RMS: {20*np.log10(np.sqrt(np.mean(phone_sig**2))):.2f} dBFS")
print(f"Mixed signal RMS: {20*np.log10(np.sqrt(np.mean(mixed**2))):.2f} dBFS")

engine.reset_state()
decisions = []
orig_proc = engine.tracker.process_window_with_acoustics
def hook(**kwargs):
    res = orig_proc(**kwargs)
    if kwargs['p_exp'] >= 0.5 or kwargs['rms_dbfs'] >= -45.0:
        decisions.append((kwargs['time_sec'], kwargs['rms_dbfs'], kwargs['p_exp'], kwargs.get('lf_ratio'), kwargs.get('hf_ratio'), res['action'], res.get('reason')))
    return res
engine.tracker.process_window_with_acoustics = hook
engine.inject_audio_data(mixed, 22050)
summary = engine.tracker.get_summary()
print(f"Initial alerts: {summary['initial_alerts_count']}, Dups suppressed: {summary['suppressed_duplicates_count']}")
for d in decisions:
    if d[2] >= 0.5:
        print(f"  t={d[0]:.2f}s, RMS={d[1]:.1f}dB, p_exp={d[2]:.3f}, LF={d[3]:.2f}, HF={d[4]:.2f}, action={d[5]}, reason={d[6]}")
