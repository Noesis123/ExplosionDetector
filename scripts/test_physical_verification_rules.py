import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

prep = AudioPreprocessor(sample_rate=22050)
ckpt_path = Path("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

manifest_df = pd.read_csv("data/manifests/dataset_manifest_phase21_expanded.csv")
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

# Precompute all window data for the 92 files
print("Precomputing windows for all 92 files...")
file_windows = {}

for idx, r in eval_df.iterrows():
    p = Path(r["file_path"])
    fname = p.name
    is_pos = (r["label"] == "explosion")
    cat = r.get("category", "unknown")
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    
    win_list = []
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = float(np.sqrt(np.mean(w**2) + 1e-12))
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        peak_val = float(np.max(np.abs(w)))
        
        if dbfs < -45.0:
            p_exp = 0.0
            p_nt = 1.0
            lf = 1.0
            hf = 0.0
            sub_bass = 0.5
            crest = 1.0
            rise_ms = 100.0
        else:
            spec = prep.compute_log_mel_spectrogram(w)
            tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
            p_exp = float(probs[1])
            p_nt = float(probs[0])
            
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
            sub_bass = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e)
            crest = peak_val / (rms + 1e-9)
            
            peak_idx = int(np.argmax(np.abs(w)))
            thresh_val = 0.10 * peak_val
            pre_peak = np.abs(w[:peak_idx+1])
            onset_candidates = np.where(pre_peak < thresh_val)[0]
            onset_idx = onset_candidates[-1] if len(onset_candidates) > 0 else 0
            rise_ms = ((peak_idx - onset_idx) / 22050.0) * 1000.0
            
        win_list.append({
            "time_sec": i * 0.25,
            "rms_dbfs": dbfs,
            "peak": peak_val,
            "p_exp": p_exp,
            "p_nt": p_nt,
            "lf": lf,
            "hf": hf,
            "sub_bass": sub_bass,
            "crest": crest,
            "rise_ms": rise_ms,
        })
        
    file_windows[fname] = {
        "is_pos": is_pos,
        "category": cat,
        "windows": win_list,
    }

print(f"Loaded {len(file_windows)} files.")

# Test baseline configuration
def eval_config(name, filter_fn):
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_files = []
    lost_pos = []
    
    for fname, data in file_windows.items():
        is_pos = data["is_pos"]
        alerted = False
        
        for w in data["windows"]:
            if filter_fn(w):
                alerted = True
                break
                
        if is_pos:
            if alerted:
                tp += 1
            else:
                fn += 1
                lost_pos.append(fname)
        else:
            if alerted:
                fp += 1
                fp_files.append((fname, data["category"]))
            else:
                tn += 1
                
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n--- {name} ---")
    print(f"Recall: {rec*100:.2f}% ({tp}/{tp+fn}) | Precision: {prec*100:.2f}% ({tp}/{tp+fp}) | F1: {f1*100:.2f}% | Acc: {acc*100:.2f}% | FP: {fp}/{tn+fp}")
    if lost_pos:
        print(f"  Lost positives ({len(lost_pos)}): {lost_pos}")
    return rec, prec, f1, fp

# 1. Active Baseline ExpF + AEV Config C
eval_config(
    "Baseline Config C (p>=0.70, LF>=0.35, HF<=0.15, RMS>=-45)",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.35 and w["hf"] <= 0.15
)

# 2. Test near-squelch margin (+3 dB, i.e. RMS >= -42.0 dBFS)
eval_config(
    "Config with Squelch -42.0 dBFS",
    lambda w: w["rms_dbfs"] >= -42.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.35 and w["hf"] <= 0.15
)

# 3. Test near-squelch margin only for lower confidence: if p in [0.70, 0.75), require RMS >= -40
eval_config(
    "Config with Near-Squelch Margin (+3dB for p < 0.75)",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.35 and w["hf"] <= 0.15 and not (w["rms_dbfs"] < -42.0 and w["p_exp"] < 0.75)
)

# 4. Test HF ratio tightening (HF <= 0.10 or 0.08)
eval_config(
    "Config with HF <= 0.08",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.35 and w["hf"] <= 0.08
)

# 5. Test LF ratio tightening (LF >= 0.45 or 0.50)
eval_config(
    "Config with LF >= 0.50",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.50 and w["hf"] <= 0.15
)

# 6. Test Crest factor (Explosions have peak/RMS >= 3.0)
eval_config(
    "Config with Crest >= 3.0",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.35 and w["hf"] <= 0.15 and w["crest"] >= 3.0
)

# 7. Test Combined Physical Rules
eval_config(
    "Combined Physical Verification: LF>=0.40, HF<=0.08, Near-Squelch Guard, Crest>=3.0",
    lambda w: w["rms_dbfs"] >= -45.0 and w["p_exp"] >= 0.70 and w["lf"] >= 0.40 and w["hf"] <= 0.08 and w["crest"] >= 3.0 and not (w["rms_dbfs"] < -42.0 and w["p_exp"] < 0.75)
)
