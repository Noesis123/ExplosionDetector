"""Comprehensive Benchmark and Evaluation: Candidate ExpF vs ExpE, ExpD, and Baseline Config C.

Evaluates on the 92 historical benchmark recordings:
1. Baseline Config C (0.75, EnhancedTracker)
2. Candidate ExpD (0.70, EnhancedTracker)
3. Candidate ExpE (0.70, EnhancedTracker)
4. Candidate ExpE + AEV (0.70, AEVTracker)
5. Candidate ExpF (0.70, EnhancedTracker)
6. Candidate ExpF + AEV (0.70, AEVTracker)
7. Candidate ExpF + AEV (0.75, AEVTracker - High Precision Mode)

Computes:
- Binary Confusion Matrices, Recall, Precision, F1, Accuracy, Specificity.
- Total Dispatched Alert Events and False Alert Multiplicity.
- Category-by-Category False Positive Breakdown.
- Recovered and Missed Explosions.
- Gain Sweep (0 dB, -6 dB, -12 dB).
- Real-Time CPU Latency Profiling.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
PROD_DIR = Path("D:/NOESIS/Explosion/Explosion")
EXP_DIR = LAB_DIR / "experiments/binary_explosion"
CKPT_DIR = EXP_DIR / "checkpoints"

if str(LAB_DIR) not in sys.path: sys.path.insert(0, str(LAB_DIR))
if str(PROD_DIR) not in sys.path: sys.path.insert(0, str(PROD_DIR))
if str(LAB_DIR / "phase24") not in sys.path: sys.path.insert(0, str(LAB_DIR / "phase24"))
if str(EXP_DIR) not in sys.path: sys.path.insert(0, str(EXP_DIR))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from models.multiclass_cnn import AudioMulticlassCNN
from model_binary import ExplosionBinaryCNN
from enhanced_tracker import EnhancedTemporalEventTracker
from tracker_verified import AcousticVerifiedEventTracker

TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

# 1. Load Baseline Config C
base_ckpt = LAB_DIR / "checkpoints/phase21_improved_best.pt"
base_sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
base_sd = base_sd["model_state_dict"] if "model_state_dict" in base_sd else base_sd
base_model = AudioMulticlassCNN(num_classes=3)
base_model.load_state_dict(base_sd)
base_model.eval()

# 2. Load ExpD
exp_d_ckpt = CKPT_DIR / "candidate_expD_recall_boosted_best.pt"
exp_d_raw = torch.load(exp_d_ckpt, map_location="cpu", weights_only=False)
exp_d_sd = exp_d_raw["model_state_dict"] if "model_state_dict" in exp_d_raw else exp_d_raw
model_d = ExplosionBinaryCNN(freeze_bn=True)
model_d.load_state_dict(exp_d_sd)
model_d.eval()

# 3. Load ExpE
exp_e_ckpt = CKPT_DIR / "candidate_expE_hard_neg_refined_best.pt"
exp_e_raw = torch.load(exp_e_ckpt, map_location="cpu", weights_only=False)
exp_e_sd = exp_e_raw["model_state_dict"] if "model_state_dict" in exp_e_raw else exp_e_raw
model_e = ExplosionBinaryCNN(freeze_bn=True)
model_e.load_state_dict(exp_e_sd)
model_e.eval()

# 4. Load ExpF
exp_f_ckpt = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
exp_f_raw = torch.load(exp_f_ckpt, map_location="cpu", weights_only=False)
exp_f_sd = exp_f_raw["model_state_dict"] if "model_state_dict" in exp_f_raw else exp_f_raw
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(exp_f_sd)
model_f.eval()

print("All models loaded successfully:")
print(f"  Baseline: {base_ckpt.name}")
print(f"  ExpD    : {exp_d_ckpt.name}")
print(f"  ExpE    : {exp_e_ckpt.name}")
print(f"  ExpF    : {exp_f_ckpt.name}")

# Manifest & 92-eval recordings
manifest_df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

configs = [
    {"id": "baseline_075", "name": "Baseline Config C (Thresh 0.75)", "model": base_model, "is_binary": False, "thresh": 0.75, "use_aev": False},
    {"id": "expD_070", "name": "Candidate ExpD (Thresh 0.70)", "model": model_d, "is_binary": True, "thresh": 0.70, "use_aev": False},
    {"id": "expE_070", "name": "Candidate ExpE (Thresh 0.70)", "model": model_e, "is_binary": True, "thresh": 0.70, "use_aev": False},
    {"id": "expE_aev_070", "name": "Candidate ExpE + AEV (Thresh 0.70)", "model": model_e, "is_binary": True, "thresh": 0.70, "use_aev": True},
    {"id": "expF_070", "name": "Candidate ExpF (Thresh 0.70)", "model": model_f, "is_binary": True, "thresh": 0.70, "use_aev": False},
    {"id": "expF_aev_070", "name": "Candidate ExpF + AEV (Thresh 0.70)", "model": model_f, "is_binary": True, "thresh": 0.70, "use_aev": True},
    {"id": "expF_aev_075", "name": "Candidate ExpF + AEV (Thresh 0.75)", "model": model_f, "is_binary": True, "thresh": 0.75, "use_aev": True},
]

def compute_binary_metrics(cm):
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "specificity": float(spec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_positives": tp + fn, "total_negatives": tn + fp,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }

cm_dict = {c["id"]: np.zeros((2, 2), dtype=int) for c in configs}
dispatched_events_count = {c["id"]: 0 for c in configs}
fp_events_count = {c["id"]: 0 for c in configs}
fp_files_by_cat = {c["id"]: {} for c in configs}
per_file_records = []

print(f"\nEvaluating {len(eval_df)} benchmark recordings across 7 pipeline configurations...")
t0 = time.time()

for idx, r in eval_df.iterrows():
    p = LAB_DIR / r["file_path"]
    if not p.exists(): p = PROD_DIR / r["file_path"]
    
    orig_lbl = r["label"]
    binary_true = 1 if orig_lbl == "explosion" else 0
    cat = r.get("category", "")
    fname = Path(r["file_path"]).name
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    res = preprocessor.resample(mono, sr)
    n_w = max(1, (len(res) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    
    trackers = {}
    for c in configs:
        if c["use_aev"]:
            trackers[c["id"]] = AcousticVerifiedEventTracker(squelch_dbfs=SQUELCH_DBFS, conf_threshold=c["thresh"])
        else:
            trackers[c["id"]] = EnhancedTemporalEventTracker(squelch_dbfs=SQUELCH_DBFS, conf_threshold=c["thresh"])
            
    max_probs = {c["id"]: 0.0 for c in configs}
    dispatched = {c["id"]: 0 for c in configs}
    
    for i in range(n_w):
        s_i = i * HOP_SAMPLES
        t_sec = i * 0.25
        w = res[s_i : s_i + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
        rms = float(np.sqrt(np.mean(w ** 2)))
        dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
        peak = float(np.max(np.abs(w)))
        
        if dbfs < SQUELCH_DBFS:
            probs_base = [1.0, 0.0, 0.0]
            p_d = 0.0
            p_e = 0.0
            p_f = 0.0
            lf_ratio, hf_ratio = 1.0, 0.0
        else:
            mel = preprocessor.compute_log_mel_spectrogram(w)
            t_mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                probs_base = torch.softmax(base_model(t_mel), dim=-1).squeeze().numpy().tolist()
                p_d = float(torch.softmax(model_d(t_mel), dim=-1)[0, 1])
                p_e = float(torch.softmax(model_e(t_mel), dim=-1)[0, 1])
                p_f = float(torch.softmax(model_f(t_mel), dim=-1)[0, 1])
                
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            lf_ratio = float(np.sum(fft_mag[freqs < 1000] ** 2) / tot_e)
            hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
            
        p_dict = {
            "baseline_075": probs_base[1],
            "expD_070": p_d,
            "expE_070": p_e,
            "expE_aev_070": p_e,
            "expF_070": p_f,
            "expF_aev_070": p_f,
            "expF_aev_075": p_f,
        }
        
        for c in configs:
            cid = c["id"]
            p_exp = p_dict[cid]
            if p_exp > max_probs[cid]: max_probs[cid] = p_exp
            
            if c["use_aev"]:
                act = trackers[cid].process_window_with_acoustics(
                    time_sec=t_sec, rms_dbfs=dbfs, p_nt=1.0-p_exp, p_exp=p_exp, p_fw=0.0,
                    peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio
                )
            elif c["is_binary"]:
                act = trackers[cid].process_window(t_sec, dbfs, 1.0 - p_exp, p_exp, 0.0, peak)
            else:
                act = trackers[cid].process_window(t_sec, dbfs, probs_base[0], probs_base[1], probs_base[2], peak)
                
            if act.get("action") == "initial_alert":
                a_cls = act.get("alert", {}).get("alert_class") if act.get("alert") else act.get("alert_class")
                if a_cls == "explosion":
                    dispatched[cid] += 1
                    
    file_record = {
        "file_name": fname,
        "label": orig_lbl,
        "is_explosion": binary_true,
        "category": cat,
        "results": {}
    }
    
    for c in configs:
        cid = c["id"]
        pred_pos = (dispatched[cid] > 0)
        binary_pred = 1 if pred_pos else 0
        cm_dict[cid][binary_true, binary_pred] += 1
        dispatched_events_count[cid] += dispatched[cid]
        
        if binary_true == 0 and binary_pred == 1:
            fp_events_count[cid] += dispatched[cid]
            fp_files_by_cat[cid][cat] = fp_files_by_cat[cid].get(cat, 0) + 1
            
        file_record["results"][cid] = {
            "max_p": float(max_probs[cid]),
            "dispatched_alerts": dispatched[cid],
            "detected": pred_pos
        }
        
    per_file_records.append(file_record)

elapsed = time.time() - t0
print(f"Completed 92-recording evaluation in {elapsed:.1f}s.")

# Print Summary Table
print("\n" + "="*95)
print(f"{'Configuration':<35} | {'Thresh':<6} | {'Recall':<14} | {'Precision':<10} | {'F1':<7} | {'FP Files':<8} | {'FP Alerts':<9}")
print("="*95)

final_results = {}
for c in configs:
    cid = c["id"]
    metrics = compute_binary_metrics(cm_dict[cid])
    metrics["total_dispatched_alerts"] = dispatched_events_count[cid]
    metrics["false_positive_alerts"] = fp_events_count[cid]
    metrics["false_positive_files_by_category"] = fp_files_by_cat[cid]
    final_results[cid] = metrics
    
    print(f"{c['name']:<35} | {c['thresh']:<6.2f} | {metrics['recall']*100:.2f}% ({metrics['tp']}/{metrics['total_positives']}) | {metrics['precision']*100:.2f}% | {metrics['f1']:.4f} | {metrics['fp']:<8} | {fp_events_count[cid]:<9}")

print("="*95)

# Gain Sweeps for ExpF + AEV
print("\nRunning Gain Robustness Sweep for ExpF + AEV @ 0.70...")
gain_levels = [0.0, -6.0, -12.0]
gain_results = {}

for g in gain_levels:
    gain_linear = 10.0 ** (g / 20.0)
    cm_g = np.zeros((2, 2), dtype=int)
    fp_events_g = 0
    
    for idx, r in eval_df.iterrows():
        p = LAB_DIR / r["file_path"]
        if not p.exists(): p = PROD_DIR / r["file_path"]
        orig_lbl = r["label"]
        binary_true = 1 if orig_lbl == "explosion" else 0
        
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        res = preprocessor.resample(mono, sr) * gain_linear
        n_w = max(1, (len(res) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
        
        tracker = AcousticVerifiedEventTracker(squelch_dbfs=SQUELCH_DBFS, conf_threshold=0.70)
        disp = 0
        
        for i in range(n_w):
            s_i = i * HOP_SAMPLES
            t_sec = i * 0.25
            w = res[s_i : s_i + WINDOW_SAMPLES]
            if len(w) < WINDOW_SAMPLES:
                w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
            rms = float(np.sqrt(np.mean(w ** 2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            
            if dbfs < SQUELCH_DBFS:
                continue
                
            mel = preprocessor.compute_log_mel_spectrogram(w)
            t_mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                p_exp = float(torch.softmax(model_f(t_mel), dim=-1)[0, 1])
                
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            lf_ratio = float(np.sum(fft_mag[freqs < 1000] ** 2) / tot_e)
            hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
            
            act = tracker.process_window_with_acoustics(
                time_sec=t_sec, rms_dbfs=dbfs, p_nt=1.0-p_exp, p_exp=p_exp, p_fw=0.0,
                peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio
            )
            if act.get("action") == "initial_alert":
                a_cls = act.get("alert", {}).get("alert_class") if act.get("alert") else act.get("alert_class")
                if a_cls == "explosion":
                    disp += 1
                    
        pred_pos = (disp > 0)
        cm_g[binary_true, 1 if pred_pos else 0] += 1
        if binary_true == 0 and pred_pos:
            fp_events_g += disp
            
    m_g = compute_binary_metrics(cm_g)
    m_g["false_positive_alerts"] = fp_events_g
    gain_results[f"{g:.1f}dB"] = m_g
    print(f"Gain {g:+5.1f} dB: Recall={m_g['recall']*100:.1f}% ({m_g['tp']}/{m_g['total_positives']}), Prec={m_g['precision']*100:.1f}%, FP Files={m_g['fp']}, FP Events={fp_events_g}")

# Latency Profiling
print("\nProfiling CPU Hop Latency for ExpF + AEV (1,000 runs)...")
sample_w = np.random.randn(22050).astype(np.float32) * 0.1
times_cnn = []
times_mel = []
times_aev = []

for _ in range(1000):
    t0 = time.perf_counter()
    mel = preprocessor.compute_log_mel_spectrogram(sample_w)
    t1 = time.perf_counter()
    t_mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model_f(t_mel)
        p_exp = float(torch.softmax(logits, dim=-1)[0, 1])
    t2 = time.perf_counter()
    fft_mag = np.abs(np.fft.rfft(sample_w))
    freqs = np.fft.rfftfreq(len(sample_w), d=1.0/22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    lf_ratio = float(np.sum(fft_mag[freqs < 1000] ** 2) / tot_e)
    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
    act = tracker.process_window_with_acoustics(
        time_sec=1.0, rms_dbfs=-30.0, p_nt=1.0-p_exp, p_exp=p_exp, p_fw=0.0,
        peak=0.1, lf_ratio=lf_ratio, hf_ratio=hf_ratio
    )
    t3 = time.perf_counter()
    
    times_mel.append((t1 - t0) * 1000.0)
    times_cnn.append((t2 - t1) * 1000.0)
    times_aev.append((t3 - t2) * 1000.0)

times_total = [m + c + a for m, c, a in zip(times_mel, times_cnn, times_aev)]

latency_summary = {
    "mel_preprocessor_p95_ms": float(np.percentile(times_mel, 95)),
    "cnn_forward_p95_ms": float(np.percentile(times_cnn, 95)),
    "aev_verification_p95_ms": float(np.percentile(times_aev, 95)),
    "total_hop_p95_ms": float(np.percentile(times_total, 95)),
    "budget_ms": 250.0,
    "budget_utilization_percent": float(np.percentile(times_total, 95) / 250.0 * 100.0)
}
print(f"Latency P95: Total={latency_summary['total_hop_p95_ms']:.2f} ms ({latency_summary['budget_utilization_percent']:.1f}% of 250ms budget)")

output_package = {
    "benchmark_summary": final_results,
    "gain_robustness": gain_results,
    "latency_profiling": latency_summary,
    "file_records": per_file_records
}

with open(EXP_DIR / "final_expF_benchmark_results.json", "w") as f:
    json.dump(output_package, f, indent=2)

print("\nBenchmark complete. Saved to final_expF_benchmark_results.json.")
