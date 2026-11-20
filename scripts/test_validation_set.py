import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

prep = AudioPreprocessor(sample_rate=22050)
val_data = torch.load(PROJECT_ROOT / "training" / "data" / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)
val_records = val_data["file_records"] # 190 source-isolated files

ckpt_f = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(sd_f)
model_f.eval()

print("=" * 80)
print(f"EVALUATING ON SOURCE-ISOLATED VALIDATION SET ({len(val_records)} files)")
print("=" * 80)

# Evaluate Baseline ExpF alone on val records
tp_raw, fp_raw, tn_raw, fn_raw = 0, 0, 0, 0
for r in val_records:
    lbl = r["binary_label"]
    max_p = 0.0
    for w in r["windows"]:
        spec = w["spec"].unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            p = float(torch.softmax(model_f(spec), dim=-1)[0, 1])
        if p > max_p: max_p = p
    if lbl == 1:
        if max_p >= 0.70: tp_raw += 1
        else: fn_raw += 1
    else:
        if max_p >= 0.70: fp_raw += 1
        else: tn_raw += 1

prec_raw = tp_raw / (tp_raw + fp_raw) if (tp_raw + fp_raw) > 0 else 0
rec_raw = tp_raw / (tp_raw + fn_raw) if (tp_raw + fn_raw) > 0 else 0
f1_raw = 2 * prec_raw * rec_raw / (prec_raw + rec_raw) if (prec_raw + rec_raw) > 0 else 0

print(f"[RAW ExpF CNN alone at th=0.70 on Val (190)]")
print(f"  Recall: {rec_raw*100:.2f}% ({tp_raw}/{tp_raw+fn_raw}), Precision: {prec_raw*100:.2f}% ({tp_raw}/{tp_raw+fp_raw}), F1: {f1_raw*100:.2f}%, FP: {fp_raw}/{fp_raw+tn_raw}")

# Now at th=0.80
tp_80, fp_80, tn_80, fn_80 = 0, 0, 0, 0
for r in val_records:
    lbl = r["binary_label"]
    max_p = 0.0
    for w in r["windows"]:
        spec = w["spec"].unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            p = float(torch.softmax(model_f(spec), dim=-1)[0, 1])
        if p > max_p: max_p = p
    if lbl == 1:
        if max_p >= 0.80: tp_80 += 1
        else: fn_80 += 1
    else:
        if max_p >= 0.80: fp_80 += 1
        else: tn_80 += 1

prec_80 = tp_80 / (tp_80 + fp_80) if (tp_80 + fp_80) > 0 else 0
rec_80 = tp_80 / (tp_80 + fn_80) if (tp_80 + fn_80) > 0 else 0
f1_80 = 2 * prec_80 * rec_80 / (prec_80 + rec_80) if (prec_80 + rec_80) > 0 else 0

print(f"\n[RAW ExpF CNN alone at th=0.80 on Val (190)]")
print(f"  Recall: {rec_80*100:.2f}% ({tp_80}/{tp_80+fn_80}), Precision: {prec_80*100:.2f}% ({tp_80}/{tp_80+fp_80}), F1: {f1_80*100:.2f}%, FP: {fp_80}/{fp_80+tn_80}")
