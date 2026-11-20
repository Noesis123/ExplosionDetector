import math
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
PROD_DIR = Path("D:/NOESIS/Explosion/Explosion")
sys.path.insert(0, str(LAB_DIR))
sys.path.insert(0, str(PROD_DIR))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from models.multiclass_cnn import AudioMulticlassCNN

# 1. Load Model & Preprocessor
preprocessor = AudioPreprocessor(sample_rate=22050)
manifest_df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

ckpt_p = LAB_DIR / "checkpoints/phase21_improved_best.pt"
raw = torch.load(ckpt_p, map_location="cpu", weights_only=True)
sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
model = AudioMulticlassCNN(num_classes=3)
model.load_state_dict(sd)
model.eval()

# Verify architecture & checkpoint parameters
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_state_elements = sum(v.numel() for v in sd.values())
print(f"Verified Model Architecture: AudioMulticlassCNN(num_classes=3)")
print(f"Checkpoint Path: {ckpt_p.name}")
print(f"Trainable Parameters: {num_trainable} | Total state_dict elements: {num_state_elements}")

# CPU Latency Benchmark (1,000 iterations)
print("\nBenchmarking CPU inference latency (1,000 iterations)...", flush=True)
dummy = torch.randn(1, 1, 64, 44)
for _ in range(50): model(dummy)
t0 = time.perf_counter()
for _ in range(1000): model(dummy)
lat_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0
print(f"CPU Latency per 1.0s window: {lat_ms:.2f} ms\n", flush=True)

# 2. Extract & Cache all 92 Evaluation Recordings
eval_splits = ['val', 'holdout', 'test', 'challenge']
eval_df = manifest_df[manifest_df['split'].isin(eval_splits)]
print(f"Total Evaluation Recordings to Cache: {len(eval_df)}")
print(eval_df.groupby(['split', 'label']).size())

cache = []
print("\nExtracting sliding windows and model probabilities...", flush=True)
t_start_cache = time.perf_counter()

for idx, r in eval_df.iterrows():
    p = LAB_DIR / r['file_path']
    if not p.exists(): p = PROD_DIR / r['file_path']
    lbl = r['label']
    split = r['split']
    category = r.get('category', 'unknown')
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    res = preprocessor.resample(mono, sr)
    windows, _ = preprocessor.create_sliding_windows(res, preserve_tail=True)
    
    file_windows = []
    with torch.no_grad():
        for i, w in enumerate(windows):
            rms = float(np.sqrt(np.mean(w ** 2)))
            raw_dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            
            # Spectrogram and probabilities
            spec = preprocessor.compute_log_mel_spectrogram(w)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :])
            probs = torch.softmax(model(inp), dim=-1).squeeze().numpy()
            
            file_windows.append({
                "idx": i,
                "raw_dbfs": raw_dbfs,
                "peak": peak,
                "p_nt": float(probs[0]),
                "p_exp": float(probs[1]),
                "p_fw": float(probs[2])
            })
            
    cache.append({
        'stem': p.stem,
        'file_path': r['file_path'],
        'label': lbl,
        'split': split,
        'category': category,
        'duration': dur,
        'windows': file_windows
    })

print(f"Cached all {len(cache)} files in {time.perf_counter() - t_start_cache:.1f}s.\n", flush=True)

# 3. Evaluation Function across Splits & Gain Attenuations
def evaluate_dataset(split_filter=None, squelch_dbfs=-40.0, gain_att_db=0.0, conf_thresh=0.75):
    """
    Evaluates dataset under specified squelch threshold and gain attenuation.
    Preserves raw amplitude: window_rms_attenuated = raw_dbfs + gain_att_db.
    Alert decision: window qualifies if window_rms_attenuated >= squelch_dbfs AND p >= conf_thresh.
    """
    records = cache if split_filter is None else [c for c in cache if c['split'] in split_filter]
    
    cm = np.zeros((3, 3), dtype=int)
    file_decisions = []
    
    for item in records:
        lbl = item['label']
        t_id = 0 if lbl == 'non_target' else (1 if lbl == 'explosion' else 2)
        
        exp_trig = False
        fw_trig = False
        max_exp = 0.0
        max_fw = 0.0
        
        for w in item['windows']:
            eff_dbfs = w['raw_dbfs'] + gain_att_db
            if eff_dbfs >= squelch_dbfs:
                if w['p_exp'] > max_exp: max_exp = w['p_exp']
                if w['p_fw'] > max_fw: max_fw = w['p_fw']
                if w['p_exp'] >= conf_thresh: exp_trig = True
                if w['p_fw'] >= conf_thresh: fw_trig = True
                
        if exp_trig and fw_trig:
            p_id = 1 if max_exp >= max_fw else 2
        elif exp_trig:
            p_id = 1
        elif fw_trig:
            p_id = 2
        else:
            p_id = 0
            
        cm[t_id, p_id] += 1
        file_decisions.append({
            'stem': item['stem'],
            'split': item['split'],
            'category': item['category'],
            'true_label': lbl,
            'pred_label': 'non_target' if p_id == 0 else ('explosion' if p_id == 1 else 'fireworks'),
            'max_exp': max_exp,
            'max_fw': max_fw,
            'correct': bool(t_id == p_id)
        })
        
    num_nt = sum(1 for c in records if c['label'] == 'non_target')
    num_exp = sum(1 for c in records if c['label'] == 'explosion')
    num_fw = sum(1 for c in records if c['label'] == 'fireworks')
    total = len(records)
    
    clean_nt = cm[0, 0]
    fa = cm[0, 1] + cm[0, 2]
    exp_rec = cm[1, 1] / num_exp if num_exp > 0 else 0.0
    fw_rec = cm[2, 2] / num_fw if num_fw > 0 else 0.0
    rec_acc = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / total if total > 0 else 0.0
    
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    
    return {
        'total': total,
        'counts': {'nt': num_nt, 'exp': num_exp, 'fw': num_fw},
        'cm': cm.tolist(),
        'clean_nt': int(clean_nt),
        'fa': int(fa),
        'exp_rec': float(exp_rec),
        'exp_tp': int(cm[1, 1]),
        'fw_rec': float(fw_rec),
        'fw_tp': int(cm[2, 2]),
        'rec_acc': float(rec_acc),
        'macro_f1': float(macro_f1),
        'file_decisions': file_decisions
    }

# ----------------------------------------------------------------------------------
# Test A: Squelch Threshold Comparison on Validation Set (0 dB Attenuation)
# ----------------------------------------------------------------------------------
squelch_list = [-58.0, -45.0, -42.0, -40.0, -38.0]
gain_list = [0.0, -6.0, -12.0, -18.0]

print("=" * 105)
print("PART 1: SQUELCH THRESHOLD COMPARISON ON VALIDATION SET (37 files, 0 dB Attenuation)")
print("=" * 105)
print(f"{'Squelch':<12} | {'EXP Recall':<14} | {'FW Recall':<14} | {'Clean NT':<14} | {'FA':<6} | {'Accuracy':<10} | {'Macro F1':<10}")
print("-" * 105)

val_squelch_results = {}
for sq in squelch_list:
    res = evaluate_dataset(['val'], squelch_dbfs=sq, gain_att_db=0.0)
    val_squelch_results[sq] = res
    print(f"{sq:5.1f} dBFS   | {res['exp_tp']:2d}/12 ({res['exp_rec']*100:5.1f}%) | {res['fw_tp']:2d}/3 ({res['fw_rec']*100:5.1f}%)  | {res['clean_nt']:2d}/22 ({res['clean_nt']/22*100:5.1f}%) | {res['fa']:2d}/22 | {res['rec_acc']*100:5.2f}%    | {res['macro_f1']:.4f}")

# ----------------------------------------------------------------------------------
# Test B: Gain Attenuation Matrix on Validation Set (0, -6, -12, -18 dB)
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 2: GAIN SENSITIVITY MATRIX ON VALIDATION SET (37 files)")
print("=" * 105)
print(f"{'Squelch':<12} | {'Gain Atten':<12} | {'EXP Recall':<14} | {'FW Recall':<14} | {'Clean NT':<14} | {'FA':<6} | {'Accuracy':<10} | {'Macro F1':<10}")
print("-" * 105)

val_matrix_results = {}
for sq in squelch_list:
    val_matrix_results[sq] = {}
    for att in gain_list:
        res = evaluate_dataset(['val'], squelch_dbfs=sq, gain_att_db=att)
        val_matrix_results[sq][att] = res
        print(f"{sq:5.1f} dBFS   | {att:5.1f} dB     | {res['exp_tp']:2d}/12 ({res['exp_rec']*100:5.1f}%) | {res['fw_tp']:2d}/3 ({res['fw_rec']*100:5.1f}%)  | {res['clean_nt']:2d}/22 ({res['clean_nt']/22*100:5.1f}%) | {res['fa']:2d}/22 | {res['rec_acc']*100:5.2f}%    | {res['macro_f1']:.4f}")

# ----------------------------------------------------------------------------------
# Test C: Independent Test Set (20 files: 10 EXP, 1 FW, 9 NT)
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 3: INDEPENDENT TEST SET (20 files: 10 Explosion, 1 Fireworks, 9 Non-target)")
print("=" * 105)
print(f"{'Squelch':<12} | {'Gain Atten':<12} | {'EXP Recall':<14} | {'FW Recall':<14} | {'Clean NT':<14} | {'FA':<6} | {'Accuracy':<10} | {'Macro F1':<10}")
print("-" * 105)

test_matrix_results = {}
for sq in squelch_list:
    test_matrix_results[sq] = {}
    for att in [0.0, -6.0, -12.0, -18.0]:
        res = evaluate_dataset(['test'], squelch_dbfs=sq, gain_att_db=att)
        test_matrix_results[sq][att] = res
        print(f"{sq:5.1f} dBFS   | {att:5.1f} dB     | {res['exp_tp']:2d}/10 ({res['exp_rec']*100:5.1f}%) | {res['fw_tp']:2d}/1 ({res['fw_rec']*100:5.1f}%)  | {res['clean_nt']:2d}/9  ({res['clean_nt']/9*100:5.1f}%) | {res['fa']:2d}/9  | {res['rec_acc']*100:5.2f}%    | {res['macro_f1']:.4f}")

# ----------------------------------------------------------------------------------
# Test D: Independent Holdout Set (25 files: 2 EXP, 3 FW, 20 NT)
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 4: INDEPENDENT HOLDOUT SET (25 files: 2 Explosion, 3 Fireworks, 20 Non-target)")
print("=" * 105)
print(f"{'Squelch':<12} | {'Gain Atten':<12} | {'EXP Recall':<14} | {'FW Recall':<14} | {'Clean NT':<14} | {'FA':<6} | {'Accuracy':<10} | {'Macro F1':<10}")
print("-" * 105)

holdout_matrix_results = {}
for sq in squelch_list:
    holdout_matrix_results[sq] = {}
    for att in [0.0, -6.0, -12.0, -18.0]:
        res = evaluate_dataset(['holdout'], squelch_dbfs=sq, gain_att_db=att)
        holdout_matrix_results[sq][att] = res
        print(f"{sq:5.1f} dBFS   | {att:5.1f} dB     | {res['exp_tp']:2d}/2  ({res['exp_rec']*100:5.1f}%) | {res['fw_tp']:2d}/3 ({res['fw_rec']*100:5.1f}%)  | {res['clean_nt']:2d}/20 ({res['clean_nt']/20*100:5.1f}%) | {res['fa']:2d}/20 | {res['rec_acc']*100:5.2f}%    | {res['macro_f1']:.4f}")

# ----------------------------------------------------------------------------------
# Test E: ALL 55 GENUINELY INDEPENDENT RECORDINGS (Holdout + Test + Challenge)
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 5: ALL 55 INDEPENDENT EVALUATION RECORDINGS (Holdout + Test + Challenge)")
print("Counts: 12 Explosion, 4 Fireworks, 39 Non-target = 55 recordings")
print("=" * 105)
print(f"{'Squelch':<12} | {'Gain Atten':<12} | {'EXP Recall':<14} | {'FW Recall':<14} | {'Clean NT':<14} | {'FA':<6} | {'Accuracy':<10} | {'Macro F1':<10}")
print("-" * 105)

indep_matrix_results = {}
for sq in squelch_list:
    indep_matrix_results[sq] = {}
    for att in [0.0, -6.0, -12.0, -18.0]:
        res = evaluate_dataset(['holdout', 'test', 'challenge'], squelch_dbfs=sq, gain_att_db=att)
        indep_matrix_results[sq][att] = res
        print(f"{sq:5.1f} dBFS   | {att:5.1f} dB     | {res['exp_tp']:2d}/12 ({res['exp_rec']*100:5.1f}%) | {res['fw_tp']:2d}/4 ({res['fw_rec']*100:5.1f}%)  | {res['clean_nt']:2d}/39 ({res['clean_nt']/39*100:5.1f}%) | {res['fa']:2d}/39 | {res['rec_acc']*100:5.2f}%    | {res['macro_f1']:.4f}")

# ----------------------------------------------------------------------------------
# Test F: Audit of the Four Eliminated False Alerts
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 6: AUDIT OF THE FOUR ELIMINATED FALSE ALERTS")
print("=" * 105)

eliminated_stems = [
    "p19_silence_ambience_3_ahriik_cb1a0293",
    "p19_shouting_screaming_3_halgrimm_44e0bfe0",
    "p19_doors_slamming_falling_3_actioncircus_ecb5d04a",
    "p19_construction_impacts_3_xserra_6ad8f42d"
]

for item in cache:
    if any(s in item['stem'] for s in eliminated_stems):
        # find qualifying windows at -58 dBFS
        qual_wins = [w for w in item['windows'] if (w['p_exp'] >= 0.75 or w['p_fw'] >= 0.75)]
        max_qual_rms = max((w['raw_dbfs'] for w in qual_wins), default=-999.0)
        overall_rms = float(20.0 * math.log10(max(np.sqrt(np.mean([10**(w['raw_dbfs']/10) for w in item['windows']])), 1e-9)))
        print(f"\nFile: {item['stem']}")
        print(f"  Category: {item['category']} | Total Windows: {len(item['windows'])}")
        print(f"  Overall RMS: {overall_rms:.1f} dBFS | Max Window Raw RMS: {max(w['raw_dbfs'] for w in item['windows']):.1f} dBFS")
        print(f"  Number of windows qualifying at -58 dBFS: {len(qual_wins)}")
        print(f"  Maximum Raw RMS among qualifying windows: {max_qual_rms:.1f} dBFS")
        print(f"  Why eliminated at -40 dBFS: Peak qualifying window RMS ({max_qual_rms:.1f} dBFS) < -40.0 dBFS gate.")

# ----------------------------------------------------------------------------------
# Test G: Audit of the Remaining Misses & False Alerts under Attenuation
# ----------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("PART 7: AUDIT OF PERSISTENT MISSES & FALSE ALERTS UNDER ATTENUATION")
print("=" * 105)

focus_stems = [
    "bbc_demolition_07022510",
    "fs_323316_alfrodou",
    "bbc_gunfire---_07034194",
    "bbc_gunfire---_07034197",
    "fs_195529_klankbeeld",
    "bbc_demolition_07022508",
    "bbc_demolition_07022507"
]

for item in cache:
    if any(s in item['stem'] for s in focus_stems):
        qual_wins = [w for w in item['windows'] if (w['p_exp'] >= 0.75 or w['p_fw'] >= 0.75)]
        max_qual_rms = max((w['raw_dbfs'] for w in qual_wins), default=-999.0)
        max_all_rms = max(w['raw_dbfs'] for w in item['windows'])
        print(f"\nFile: {item['stem']} (true={item['label']}, category={item['category']})")
        print(f"  Max Raw RMS (all windows): {max_all_rms:.1f} dBFS | Max Raw RMS (qualifying windows): {max_qual_rms:.1f} dBFS")
        for att in [0.0, -6.0, -12.0, -18.0]:
            eff = max_qual_rms + att
            status_40 = "PASS" if eff >= -40.0 else "BLOCKED"
            status_42 = "PASS" if eff >= -42.0 else "BLOCKED"
            status_45 = "PASS" if eff >= -45.0 else "BLOCKED"
            print(f"    Atten {att:5.1f} dB: Eff RMS = {eff:6.1f} dBFS -> -40dBFS: {status_40:<7} | -42dBFS: {status_42:<7} | -45dBFS: {status_45}")

# Save full results
report_data = {
    'val_squelch_results': {str(k): v for k, v in val_squelch_results.items()},
    'val_matrix_results': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in val_matrix_results.items()},
    'test_matrix_results': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in test_matrix_results.items()},
    'holdout_matrix_results': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in holdout_matrix_results.items()},
    'indep_matrix_results': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in indep_matrix_results.items()},
    'latency_ms': lat_ms,
}

report_out = LAB_DIR / "reports" / "squelch_robustness_verification_report.json"
with open(report_out, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print(f"\nSaved full report to {report_out.name}")
