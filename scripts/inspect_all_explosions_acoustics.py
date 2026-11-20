import numpy as np
import pandas as pd
import torch
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

manifest_path = ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()
exp_df = eval_df[eval_df["label"] == "explosion"]

ckpt = torch.load("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
pre = AudioPreprocessor()

print("Acoustic parameters of triggering windows for all 22 detected explosions:")
print(f"{'Filename':<32} | {'p':<6} | {'RMS':<6} | {'SubBass':<8} | {'LF':<6} | {'HF':<6} | {'Centroid':<8} | {'RollOff':<8}")
print("-" * 95)

for idx, r in exp_df.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.resample(pre.to_mono(sig), sr) if sr != 22050 else pre.to_mono(sig)
    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    first_trig = None
    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            if dbfs < -38.0:
                continue
            spec = pre.compute_log_mel_spectrogram(w)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
            p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())
            if p_exp >= 0.70 and first_trig is None:
                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
                tot_e = np.sum(fft_mag**2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
                sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)]**2) / tot_e)
                centroid = float(np.sum(freqs * fft_mag**2) / tot_e)
                cum_e = np.cumsum(fft_mag**2)
                rolloff_idx = np.searchsorted(cum_e, 0.85 * tot_e)
                rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
                first_trig = {
                    "p": p_exp, "dbfs": dbfs, "sub_bass": sub_bass,
                    "lf": lf, "hf": hf, "centroid": centroid, "rolloff": rolloff
                }

    if first_trig:
        print(f"{p.name[:32]:<32} | {first_trig['p']:<6.3f} | {first_trig['dbfs']:<6.1f} | {first_trig['sub_bass']:<8.3f} | {first_trig['lf']:<6.3f} | {first_trig['hf']:<6.3f} | {first_trig['centroid']:<8.1f} | {first_trig['rolloff']:<8.1f}")
    else:
        print(f"{p.name[:32]:<32} | MISSED")

