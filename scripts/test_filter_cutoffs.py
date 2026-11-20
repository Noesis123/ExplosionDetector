import numpy as np
import scipy.signal as signal
import math

live_audio = np.load("scripts/live_mic_diagnostic_sample.npy")
# Steady-state portion: 1 to 4 seconds
sample = live_audio[22050 : 22050*4]

raw_rms = float(np.sqrt(np.mean(sample**2)))
raw_dbfs = float(20.0 * math.log10(max(raw_rms, 1e-9)))
print(f"Raw live audio RMS: {raw_dbfs:.2f} dBFS")

for cutoff in [20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0]:
    sos = signal.butter(2, cutoff, btype="highpass", fs=22050, output="sos")
    filtered = signal.sosfilt(sos, sample)
    filt_rms = float(np.sqrt(np.mean(filtered**2)))
    filt_dbfs = float(20.0 * math.log10(max(filt_rms, 1e-9)))
    drop_db = raw_dbfs - filt_dbfs
    print(f"Cutoff {cutoff:4.1f} Hz -> Filtered RMS: {filt_dbfs:6.2f} dBFS (Drop: {drop_db:5.2f} dB)")

