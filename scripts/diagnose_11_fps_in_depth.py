import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

prep = AudioPreprocessor(sample_rate=22050)

manifest_path = Path("data/manifests/dataset_manifest_phase21_expanded.csv")
manifest_df = pd.read_csv(manifest_path)
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

fps_to_check = [
    ("artillery_firing", "fs_161341_qubodup.mp3"),
    ("artillery_firing", "fs_171475_2887679652.mp3"),
    ("artillery_firing", "fs_184275_qubodup.mp3"),
    ("fireworks", "fs_410347_inspectorj.mp3"),
    ("gunfire", "bbc_gunfire---_07034194.mp3"),
    ("gunfire", "bbc_gunfire---_07034201.mp3"),
    ("gunfire", "fs_213925_Diboz.mp3"),
    ("thunder", "fs_36044_inchadney.mp3"),
    ("thunder", "fs_195529_klankbeeld.mp3"),
    ("vehicle_backfires", "fs_105351_CeebFrack.mp3"),
    ("vehicle_backfires", "fs_350672_Dominik_W.mp3"),
]

print("=" * 80)
print("IN-DEPTH ACOUSTIC & PHYSICAL ANALYSIS OF THE 11 FALSE POSITIVES")
print("=" * 80)

for cat, fname in fps_to_check:
    matches = eval_df[eval_df["file_path"].str.contains(fname)]
    if matches.empty:
        print(f"Could not find {fname} in manifest")
        continue
    fpath = Path(matches.iloc[0]["file_path"])
    if not fpath.exists():
        print(f"File does not exist: {fpath}")
        continue
    sig, sr, dur, ch = load_audio_file(fpath)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)

    # Compute full-recording and windowed metrics
    peak_amp = float(np.max(np.abs(mono)))
    rms_full = float(np.sqrt(np.mean(mono**2) + 1e-12))
    dbfs_full = 20.0 * np.log10(rms_full + 1e-12)

    # Attack rise time: time from 10% of peak to 90% of peak in ms
    env = np.abs(signal.hilbert(mono))
    peak_idx = int(np.argmax(env))
    p_val = env[peak_idx]
    t10_idx = np.where(env[:peak_idx] <= 0.10 * p_val)[0]
    t90_idx = np.where(env[:peak_idx] <= 0.90 * p_val)[0]
    rise_ms = (t90_idx[-1] - t10_idx[-1]) / 22050.0 * 1000.0 if (len(t10_idx) > 0 and len(t90_idx) > 0 and t90_idx[-1] > t10_idx[-1]) else 0.0

    # Spectral centroid and spread
    fft_mag = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / 22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    centroid = np.sum(freqs * (fft_mag ** 2)) / tot_e
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * (fft_mag ** 2)) / tot_e)

    # Low frequency decay: ratio of energy in 0.5s after peak to 0.5s at peak
    win_len = int(0.5 * 22050)
    e_peak = np.sum(mono[max(0, peak_idx - win_len//2) : min(len(mono), peak_idx + win_len//2)] ** 2) + 1e-9
    e_post = np.sum(mono[min(len(mono), peak_idx + win_len//2) : min(len(mono), peak_idx + 3*win_len//2)] ** 2) + 1e-9
    decay_ratio = e_post / e_peak

    # Band energies:
    # Infrasound / sub-bass (20 - 80 Hz)
    e_sub = np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e
    # Low-mid bass (80 - 250 Hz)
    e_lowmid = np.sum(fft_mag[(freqs >= 80) & (freqs < 250)] ** 2) / tot_e
    # High-mid (1000 - 4000 Hz)
    e_mid = np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e
    # Treble (4000+ Hz)
    e_high = np.sum(fft_mag[freqs >= 4000] ** 2) / tot_e

    # Temporal kurtosis of envelope (peakedness of sound)
    env_norm = env / (np.mean(env) + 1e-9)
    kurt = float(np.mean(env_norm ** 4))

    print(f"\n[{cat.upper()}] {fname}:")
    print(f"  Dur: {dur:.2f}s | Full RMS: {dbfs_full:.1f} dBFS | Peak Amp: {peak_amp:.3f} | Rise Time: {rise_ms:.1f} ms")
    print(f"  Centroid: {centroid:.1f} Hz | Spread: {spread:.1f} Hz | Kurtosis: {kurt:.1f} | Decay Ratio: {decay_ratio:.3f}")
    print(f"  Bands -> Sub(20-80Hz): {e_sub:.3f} | LowMid(80-250Hz): {e_lowmid:.3f} | Mid(1-4kHz): {e_mid:.3f} | High(>4kHz): {e_high:.3f}")
