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

print("Checking preceding window profile of all 24 benchmark explosions:")
for idx, r in exp_df.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050:
        mono = pre.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    first_trigger_win = None
    preceding_above_squelch_non_exp_count = 0

    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))

            if dbfs < -38.0:
                p_exp = 0.0
            else:
                spec = pre.compute_log_mel_spectrogram(w)
                inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
                p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())

            if p_exp >= 0.70 and first_trigger_win is None:
                first_trigger_win = i
                break
            elif dbfs >= -38.0 and p_exp < 0.65 and first_trigger_win is None:
                preceding_above_squelch_non_exp_count += 1

    print(f"{p.name[:35]:<35} | FirstTrigWin: {str(first_trigger_win):<4} | PrecedingActiveNonExp: {preceding_above_squelch_non_exp_count}")

