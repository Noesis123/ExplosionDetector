import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import time
from audio.microphone_stream import PhysicalWaveInCapture, RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(3.0)
cap.stop()

raw = np.concatenate(samples)
print(f"Captured {len(raw)} samples ({len(raw)/22050:.2f}s)")
rms = np.sqrt(np.mean(raw**2))
print(f"RMS: {20*np.log10(max(rms, 1e-9)):.2f} dBFS, Peak: {np.max(np.abs(raw)):.4f}, Mean: {raw.mean():.6f}")

engine.inject_audio_data(raw, 22050)
summ = engine.tracker.get_summary()
print("Tracker summary on current ambient room:")
print(f"  Alerts: {summ['initial_alerts_count']}, Dups suppressed: {summ['suppressed_duplicates_count']}")
print(f"  Latest prediction: {engine.latest_prediction}, Conf: {engine.latest_confidence:.4f}")
print(f"  Latest p_exp: {engine.latest_probabilities['explosion']:.4f}")
