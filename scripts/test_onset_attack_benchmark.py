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

print("Testing onset attack definition across all 24 benchmark explosions:")
for idx, r in exp_df.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050:
        mono = pre.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    detected = False
    detected_win = None
    info = {}

    prev_rms = None
    recent_rms = []

    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))

            if len(recent_rms) >= 2:
                baseline = min(recent_rms[-8:])
                surge = dbfs - baseline
            else:
                surge = 0.0
            recent_rms.append(dbfs)

            sub4 = [w[k*5512 : (k+1)*5512] for k in range(4)]
            sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf**2))), 1e-9)) for sf in sub4]

            if dbfs < -38.0:
                p_exp = 0.0
            else:
                spec = pre.compute_log_mel_spectrogram(w)
                inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
                p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())

            # Attack check:
            # 1. Rising within window: subframe 1, 2, or 3 is louder than subframe 0 by at least 1.5 dB, OR
            # 2. Window surged above baseline by >= 3.0 dB and is not lower than previous window by > 1.5 dB, OR
            # 3. Loud blast >= -28 dBFS
            is_rising_sub = (max(sub4_rms[1:]) >= sub4_rms[0] + 1.5)
            is_surging_win = (surge >= 3.0 and (prev_rms is None or dbfs >= prev_rms - 1.5))
            is_loud = (dbfs >= -28.0)
            has_attack = (is_rising_sub or is_surging_win or is_loud)

            if p_exp >= 0.70 and has_attack and not detected:
                detected = True
                detected_win = i
                info = {"dbfs": round(dbfs, 1), "surge": round(surge, 1), "is_loud": is_loud, "is_rising": is_rising_sub}

            prev_rms = dbfs

    status = "DETECTED" if detected else "MISSED"
    print(f"{p.name[:35]:<35} | {status:<8} | Win {str(detected_win):<4} | {info}")

