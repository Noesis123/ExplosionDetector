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

exp_df = eval_df[eval_df["label"] == "explosion"]

print("=" * 80)
print("IN-DEPTH ACOUSTIC & PHYSICAL ANALYSIS OF ALL 24 GENUINE EXPLOSIONS")
print("=" * 80)

metrics = []
for idx, r in exp_df.iterrows():
    p = Path(r["file_path"])
    if not p.exists():
        continue
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)

    peak_amp = float(np.max(np.abs(mono)))
    rms_full = float(np.sqrt(np.mean(mono**2) + 1e-12))
    dbfs_full = 20.0 * np.log10(rms_full + 1e-12)

    env = np.abs(signal.hilbert(mono))
    peak_idx = int(np.argmax(env))
    p_val = env[peak_idx]
    t10_idx = np.where(env[:peak_idx] <= 0.10 * p_val)[0]
    t90_idx = np.where(env[:peak_idx] <= 0.90 * p_val)[0]
    rise_ms = (t90_idx[-1] - t10_idx[-1]) / 22050.0 * 1000.0 if (len(t10_idx) > 0 and len(t90_idx) > 0 and t90_idx[-1] > t10_idx[-1]) else 0.0

    fft_mag = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / 22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    centroid = np.sum(freqs * (fft_mag ** 2)) / tot_e
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * (fft_mag ** 2)) / tot_e)

    win_len = int(0.5 * 22050)
    e_peak = np.sum(mono[max(0, peak_idx - win_len//2) : min(len(mono), peak_idx + win_len//2)] ** 2) + 1e-9
    e_post = np.sum(mono[min(len(mono), peak_idx + win_len//2) : min(len(mono), peak_idx + 3*win_len//2)] ** 2) + 1e-9
    decay_ratio = e_post / e_peak

    e_sub = np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e
    e_lowmid = np.sum(fft_mag[(freqs >= 80) & (freqs < 250)] ** 2) / tot_e
    e_mid = np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e
    e_high = np.sum(fft_mag[freqs >= 4000] ** 2) / tot_e

    env_norm = env / (np.mean(env) + 1e-9)
    kurt = float(np.mean(env_norm ** 4))

    metrics.append({
        "fname": p.name, "dur": dur, "dbfs": dbfs_full, "peak": peak_amp,
        "rise_ms": rise_ms, "centroid": centroid, "decay": decay_ratio,
        "sub": e_sub, "lowmid": e_lowmid, "mid": e_mid, "high": e_high
    })
    print(f"{p.name[:35]:<35}: Centroid={centroid:5.1f}Hz | Sub={e_sub:.3f} | LowMid={e_lowmid:.3f} | Mid={e_mid:.3f} | High={e_high:.3f} | Rise={rise_ms:4.1f}ms | Decay={decay_ratio:.3f}")

df = pd.DataFrame(metrics)
print("\n" + "=" * 80)
print(f"EXPLOSIONS SUMMARY (N={len(df)}):")
print(f"  Centroid: mean={df['centroid'].mean():.1f} Hz, min={df['centroid'].min():.1f} Hz, max={df['centroid'].max():.1f} Hz")
print(f"  Sub(20-80Hz): mean={df['sub'].mean():.3f}, min={df['sub'].min():.3f}, max={df['sub'].max():.3f}")
print(f"  Mid(1-4kHz): mean={df['mid'].mean():.3f}, min={df['mid'].min():.3f}, max={df['mid'].max():.3f}")
print(f"  High(>4kHz): mean={df['high'].mean():.3f}, min={df['high'].min():.3f}, max={df['high'].max():.3f}")
