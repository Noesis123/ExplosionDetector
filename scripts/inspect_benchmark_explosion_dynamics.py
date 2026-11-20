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

print(f"Analyzing all {len(exp_df)} benchmark explosion files:")
print(f"{'Filename':<35} | {'Max p':<7} | {'WinRMS':<8} | {'SubRange':<8} | {'OnsetSurge':<10} | {'Crest':<6}")
print("-" * 85)

for idx, r in exp_df.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050:
        mono = pre.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    max_p = 0.0
    best_subrange = 0.0
    best_rms = -99.9
    best_surge = 0.0
    best_crest = 0.0

    recent_rms = []
    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            crest = peak / max(rms, 1e-9)

            if len(recent_rms) >= 2:
                baseline_rms = min(recent_rms[-8:])
                onset_surge = dbfs - baseline_rms
            else:
                onset_surge = 0.0
            recent_rms.append(dbfs)

            sub4 = [w[k*5512 : (k+1)*5512] for k in range(4)]
            sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf**2))), 1e-9)) for sf in sub4]
            sub_range = max(sub4_rms) - min(sub4_rms)

            spec = pre.compute_log_mel_spectrogram(w)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
            p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())

            if p_exp > max_p:
                max_p = p_exp
                best_rms = dbfs
                best_subrange = sub_range
                best_surge = onset_surge
                best_crest = crest

    print(f"{p.name[:35]:<35} | {max_p:<7.4f} | {best_rms:<8.1f} | {best_subrange:<8.2f} | {best_surge:<10.2f} | {best_crest:<6.2f}")

