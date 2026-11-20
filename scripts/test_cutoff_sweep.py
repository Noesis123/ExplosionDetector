import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.signal as signal
from audio.microphone_stream import PhysicalWaveInCapture
import time

samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(3.0)
cap.stop()

raw = np.concatenate(samples)
raw_rms = 20 * np.log10(max(np.sqrt(np.mean(raw**2)), 1e-9))
print(f"Raw Mic RMS: {raw_rms:.2f} dBFS")

for cut in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]:
    sos = signal.butter(2, cut, btype="highpass", fs=22050, output="sos")
    filt = signal.sosfilt(sos, raw)
    f_rms = 20 * np.log10(max(np.sqrt(np.mean(filt**2)), 1e-9))
    status = "CLOSED" if f_rms < -45.0 else "OPEN"
    print(f"Cutoff {cut:4.1f} Hz: Filtered RMS = {f_rms:.2f} dBFS | Attenuation = {raw_rms - f_rms:.2f} dB | Squelch (-45): {status}")
