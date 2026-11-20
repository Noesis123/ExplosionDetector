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

ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)

def evaluate_raw_cnn_on_files(file_list, name="Dataset", threshold=0.70, squelch=-34.0):
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_cats = {}
    
    for r in file_list:
        p = Path(r["file_path"])
        if not p.exists():
            continue
        is_pos = (r["label"] == "explosion")
        cat = r.get("category", "unknown")
        
        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.to_mono(sig)
        if sr != 22050:
            mono = prep.resample(mono, orig_sr=sr)
        if len(mono) < 22050:
            mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
            
        nw = max(1, (len(mono) - 22050) // 5512 + 1)
        file_alerted = False
        
        with torch.no_grad():
            for i in range(nw):
                w = mono[i * 5512 : i * 5512 + 22050]
                rms = float(np.sqrt(np.mean(w**2) + 1e-12))
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                if dbfs >= squelch:
                    spec = prep.compute_log_mel_spectrogram(w)
                    tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
                    p_exp = float(probs[1])
                    if p_exp >= threshold:
                        file_alerted = True
                        break
                        
        if is_pos:
            if file_alerted: tp += 1
            else: fn += 1
        else:
            if file_alerted:
                fp += 1
                fp_cats[cat] = fp_cats.get(cat, 0) + 1
            else: tn += 1
            
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    print(f"\n--- {name} (Raw CNN @ threshold={threshold:.2f}, squelch={squelch:.1f}) ---")
    print(f"  Recall:    {rec*100:.2f}% ({tp}/{tp+fn})")
    print(f"  Precision: {prec*100:.2f}% ({tp}/{tp+fp})")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  Accuracy:  {acc*100:.2f}% ({tp+tn}/{tp+tn+fp+fn})")
    print(f"  FP Files:  {fp}/{tn+fp}")
    print(f"  FP breakdown: {dict(sorted(fp_cats.items(), key=lambda x: -x[1]))}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "rec": rec, "prec": prec, "f1": f1}

# 1. Historical Benchmark (92 files)
m_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
m_df = pd.read_csv(m_path)
eval_df = m_df[m_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()
bench_files = eval_df.to_dict("records")
evaluate_raw_cnn_on_files(bench_files, name="Historical Benchmark (92 files)", threshold=0.70, squelch=-34.0)
