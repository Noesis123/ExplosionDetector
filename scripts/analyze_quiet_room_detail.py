import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.signal as signal
from audio.microphone_stream import PhysicalWaveInCapture, StreamingHighpassFilter
import time

print("=" * 70)
print("QUIET ROOM DETAILED SPECTRUM & RMS ANALYSIS")
print("=" * 70)

samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(5.0)
cap.stop()

raw = np.concatenate(samples)
hpf90 = StreamingHighpassFilter(cutoff_hz=90.0, sample_rate=22050)
filt = hpf90.process(raw)

fft_raw = np.abs(np.fft.rfft(raw))
fft_filt = np.abs(np.fft.rfft(filt))
freqs = np.fft.rfftfreq(len(raw), d=1.0 / 22050)

print(f"Captured {len(raw)/22050:.2f} seconds ({len(raw)} samples).")
print(f"Raw Mic RMS: {20 * np.log10(max(np.sqrt(np.mean(raw**2)), 1e-9)):.2f} dBFS")
print(f"90 Hz Filtered RMS: {20 * np.log10(max(np.sqrt(np.mean(filt**2)), 1e-9)):.2f} dBFS")

print("\nTop 10 Frequency Peaks in Filtered Audio:")
top_idx = np.argsort(fft_filt)[::-1][:10]
for idx in top_idx:
    print(f"  Freq: {freqs[idx]:.1f} Hz | Mag: {fft_filt[idx]:.2f}")

tot_e = np.sum(fft_filt ** 2) + 1e-12
e_sub100 = np.sum(fft_filt[freqs < 100] ** 2) / tot_e
e_100_200 = np.sum(fft_filt[(freqs >= 100) & (freqs < 200)] ** 2) / tot_e
e_200_500 = np.sum(fft_filt[(freqs >= 200) & (freqs < 500)] ** 2) / tot_e
e_500_1000 = np.sum(fft_filt[(freqs >= 500) & (freqs < 1000)] ** 2) / tot_e
e_above1000 = np.sum(fft_filt[freqs >= 1000] ** 2) / tot_e

print(f"\nFiltered Signal Energy Distribution:")
print(f"  < 100 Hz   : {e_sub100*100:.2f}%")
print(f"  100-200 Hz : {e_100_200*100:.2f}%")
print(f"  200-500 Hz : {e_200_500*100:.2f}%")
print(f"  500-1000 Hz: {e_500_1000*100:.2f}%")
print(f"  > 1000 Hz  : {e_above1000*100:.2f}%")

# Sliding 1.0s windows with 0.25s hop
n_wins = max(1, (len(filt) - 22050) // 5512 + 1)
win_rms = []
for i in range(n_wins):
    w = filt[i * 5512 : i * 5512 + 22050]
    rms = 20 * np.log10(max(np.sqrt(np.mean(w ** 2)), 1e-9))
    win_rms.append(rms)

open_count = sum(1 for r in win_rms if r >= -45.0)
print(f"\nSliding Window Squelch (-45.0 dBFS) Stats ({n_wins} windows):")
print(f"  Min RMS : {min(win_rms):.2f} dBFS")
print(f"  Mean RMS: {np.mean(win_rms):.2f} dBFS")
print(f"  Max RMS : {max(win_rms):.2f} dBFS")
print(f"  Squelch Open Windows: {open_count}/{n_wins} ({open_count/n_wins*100:.1f}%)")
