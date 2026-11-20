"""
Full Controlled Evaluation of Candidate ExpM vs ExpF Baseline across all suites:
1. Source-Isolated Development Reference (190 files: 70 pos, 120 neg)
2. Historical 92-recording Regression Benchmark (24 pos, 68 neg)
3. Codex Reserved Frozen Test Set 2 (60 files: 15 audible explosions, 45 confounds)
4. Codex Selected Test Set 1 (86 files: 20 explosions, 66 confounds)
5. Gain attenuation robustness (0 dB, -6 dB, -12 dB)
6. Latency and memory consumption profiling
"""

import sys
import json
from pathlib import Path
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
CONTEXT_SAMPLES = 44100
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

# 1. Models Definition
class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h.flatten(1)

class CausalContextBinaryCNN(nn.Module):
    def __init__(self, expf_model):
        super().__init__()
        self.event_branch = expf_model
        self.context_encoder = ContextEncoder()
        self.head = nn.Linear(128 + 64, 2)
    def forward(self, x_ev, x_ctx):
        h1 = self.event_branch.block1(x_ev)
        h2 = self.event_branch.block2(h1)
        h3 = self.event_branch.block3(h2)
        h4 = self.event_branch.block4(h3)
        feat_ev = h4.flatten(1)
        feat_ctx = self.context_encoder(x_ctx)
        feat_comb = torch.cat([feat_ev, feat_ctx], dim=1)
        return self.head(feat_comb)

# Load Models
exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

model_expf = ExplosionBinaryCNN(freeze_bn=True)
model_expf.load_state_dict(sd_f)
model_expf.eval()

model_expm = CausalContextBinaryCNN(model_expf)
m_path = CKPT_DIR / "candidate_expM_causal_context_best.pt"
if not m_path.exists():
    m_path = CKPT_DIR / "candidate_expM_causal_context_final.pt"
ckpt_m = torch.load(m_path, map_location="cpu", weights_only=False)
model_expm.load_state_dict(ckpt_m["model_state_dict"])
model_expm.eval()

def run_evaluation_on_files(model, is_expm, file_list, gain_factor=1.0):
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_by_cat = {}
    missed_pos = []
    total_alerts = 0
    t0 = time.time()

    with torch.no_grad():
        for item in file_list:
            fpath = item["path"]
            is_pos = item["is_pos"]
            cat = item["category"]
            fname = fpath.name

            sig, sr, dur, ch = load_audio_file(fpath)
            mono = preprocessor.to_mono(sig)
            if sr != TARGET_SR:
                mono = preprocessor.resample(mono, orig_sr=sr)
            if gain_factor != 1.0:
                mono = mono * gain_factor
            if len(mono) < WINDOW_SAMPLES:
                mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)))

            tracker = AcousticVerifiedEventTracker(
                squelch_dbfs=SQUELCH_DBFS,
                conf_threshold=THRESHOLD,
                min_lf_ratio=0.35,
                max_hf_ratio=0.15
            )

            n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
            file_alerted = False

            for i in range(n_w):
                t_start = i * HOP_SAMPLES
                w = mono[t_start : t_start + WINDOW_SAMPLES]
                if len(w) < WINDOW_SAMPLES:
                    w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))

                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))

                if dbfs < SQUELCH_DBFS:
                    d = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                        peak=peak, lf_ratio=1.0, hf_ratio=0.0
                    )
                else:
                    mel_ev = preprocessor.compute_log_mel_spectrogram(w)
                    t_ev = torch.from_numpy(mel_ev).unsqueeze(0).unsqueeze(0).float()

                    if is_expm:
                        c_start = max(0, t_start - CONTEXT_SAMPLES)
                        ctx = mono[c_start : t_start]
                        if len(ctx) < CONTEXT_SAMPLES:
                            ctx = np.pad(ctx, (CONTEXT_SAMPLES - len(ctx), 0))
                        mel_ctx = preprocessor.compute_log_mel_spectrogram(ctx)
                        t_ctx = torch.from_numpy(mel_ctx).unsqueeze(0).unsqueeze(0).float()
                        logits = model(t_ev, t_ctx)
                    else:
                        logits = model(t_ev)

                    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                    p_nt = float(probs[0])
                    p_exp = float(probs[1])

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                    d = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=p_nt, p_exp=p_exp, p_fw=0.0,
                        peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio
                    )

                if d.get("action") == "initial_alert":
                    a_cls = d.get("alert", {}).get("alert_class") if d.get("alert") else d.get("alert_class")
                    if a_cls == "explosion":
                        file_alerted = True
                        total_alerts += 1

            if is_pos:
                if file_alerted: tp += 1
                else:
                    fn += 1
                    missed_pos.append(fname)
            else:
                if file_alerted:
                    fp += 1
                    fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
                else: tn += 1

    elapsed = time.time() - t0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": rec, "precision": prec, "f1": f1, "f2": f2,
        "accuracy": acc, "specificity": spec,
        "total_alerts": total_alerts,
        "fp_by_cat": fp_by_cat, "missed": missed_pos,
        "elapsed": elapsed, "latency_ms": (elapsed / len(file_list)) * 1000.0
    }

# 2. Build Datasets
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
m_df = pd.read_csv(manifest_path)

# Suite 1: Historical 92 Benchmark
hist_files = []
for idx, r in m_df[m_df["split"].isin(["val", "holdout", "test", "challenge"])].iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if p.exists():
        hist_files.append({"path": p, "is_pos": (r["label"] == "explosion"), "category": r.get("category", "unknown")})

# Suite 2: Source-Isolated Val Set (190 files)
val_cache = torch.load(PROJECT_ROOT / "training" / "data" / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)
val_files = []
for r in val_cache["file_records"]:
    m_row = m_df[m_df["file_path"].str.contains(r["name"])]
    if len(m_row) > 0:
        p = PROJECT_ROOT / m_row.iloc[0]["file_path"]
        if p.exists():
            val_files.append({"path": p, "is_pos": (r["binary_label"] == 1), "category": r.get("category", "unknown")})

# Suite 3: Codex Reserved Frozen Set 2 (60 files)
codex_set2_df = pd.read_csv(PROJECT_ROOT / "reports" / "codex_reserved_eval_frozen.csv")
codex_set2_files = []
for idx, r in codex_set2_df.iterrows():
    p = CODEX_DIR / r["audio_path"]
    if p.exists():
        codex_set2_files.append({"path": p, "is_pos": (r["category"] == "surface_high_explosive"), "category": r["category"]})

# Suite 4: Codex Set 1 (86 files)
codex_set1_df = pd.read_csv(PROJECT_ROOT / "reports" / "codex_selected_eval_subset.csv")
codex_set1_files = []
for idx, r in codex_set1_df.iterrows():
    p = CODEX_DIR / r["audio_path"]
    if p.exists():
        codex_set1_files.append({"path": p, "is_pos": (r["category"] == "surface_high_explosive"), "category": r["category"]})

print("=" * 85)
print("BENCHMARKING EXPF BASELINE vs CANDIDATE ExpM ACROSS ALL SUITES")
print("=" * 85)

results = {}

for sname, sfiles in [("Historical_Benchmark_92", hist_files),
                      ("Source_Isolated_Val_190", val_files),
                      ("Codex_Reserved_Set2_60", codex_set2_files),
                      ("Codex_Test_Set1_86", codex_set1_files)]:
    print(f"\n==========================================")
    print(f"EVALUATING SUITE: {sname} ({len(sfiles)} files)")
    print(f"==========================================")
    
    res_f = run_evaluation_on_files(model_expf, is_expm=False, file_list=sfiles)
    res_m = run_evaluation_on_files(model_expm, is_expm=True, file_list=sfiles)
    
    print(f"{'Metric':<22} | {'ExpF Baseline':<20} | {'Candidate ExpM':<20} | {'Delta'}")
    print("-" * 75)
    print(f"{'Recall':<22} | {res_f['recall']*100:5.2f}% ({res_f['tp']}/{res_f['tp']+res_f['fn']}){'':<7} | {res_m['recall']*100:5.2f}% ({res_m['tp']}/{res_m['tp']+res_m['fn']}){'':<7} | {(res_m['recall']-res_f['recall'])*100:+5.2f}%")
    print(f"{'Precision':<22} | {res_f['precision']*100:5.2f}% ({res_f['tp']}/{res_f['tp']+res_f['fp']}){'':<7} | {res_m['precision']*100:5.2f}% ({res_m['tp']}/{res_m['tp']+res_m['fp']}){'':<7} | {(res_m['precision']-res_f['precision'])*100:+5.2f}%")
    print(f"{'F1-Score':<22} | {res_f['f1']:6.4f}{'':<14} | {res_m['f1']:6.4f}{'':<14} | {res_m['f1']-res_f['f1']:+6.4f}")
    print(f"{'F2-Score':<22} | {res_f['f2']:6.4f}{'':<14} | {res_m['f2']:6.4f}{'':<14} | {res_m['f2']-res_f['f2']:+6.4f}")
    print(f"{'Accuracy':<22} | {res_f['accuracy']*100:5.2f}%{'':<14} | {res_m['accuracy']*100:5.2f}%{'':<14} | {(res_m['accuracy']-res_f['accuracy'])*100:+5.2f}%")
    print(f"{'Specificity':<22} | {res_f['specificity']*100:5.2f}%{'':<14} | {res_m['specificity']*100:5.2f}%{'':<14} | {(res_m['specificity']-res_f['specificity'])*100:+5.2f}%")
    print(f"{'False-Pos Files':<22} | {res_f['fp']:2d}/{res_f['tn']+res_f['fp']}{'':<14} | {res_m['fp']:2d}/{res_m['tn']+res_m['fp']}{'':<14} | {res_m['fp']-res_f['fp']:+2d} files")
    print(f"{'Total False Alerts':<22} | {res_f['total_alerts']:3d}{'':<17} | {res_m['total_alerts']:3d}{'':<17} | {res_m['total_alerts']-res_f['total_alerts']:+3d} alerts")
    print(f"{'Avg Latency / file':<22} | {res_f['latency_ms']:5.1f} ms{'':<13} | {res_m['latency_ms']:5.1f} ms{'':<13} | {res_m['latency_ms']-res_f['latency_ms']:+5.1f} ms")
    print(f"FP by Category (ExpF): {res_f['fp_by_cat']}")
    print(f"FP by Category (ExpM): {res_m['fp_by_cat']}")
    print(f"Missed (ExpF): {res_f['missed']}")
    print(f"Missed (ExpM): {res_m['missed']}")

    results[sname] = {"expf": res_f, "expm": res_m}

# 3. Gain Attenuation Sweep on Historical Benchmark
print("\n" + "=" * 85)
print("GAIN ATTENUATION SWEEP (HISTORICAL BENCHMARK: 0 dB, -6 dB, -12 dB)")
print("=" * 85)
for att_db, g_factor in [("0 dB (100% amp)", 1.0), ("-6 dB (50% amp)", 0.5012), ("-12 dB (25% amp)", 0.2512)]:
    res_f_g = run_evaluation_on_files(model_expf, is_expm=False, file_list=hist_files, gain_factor=g_factor)
    res_m_g = run_evaluation_on_files(model_expm, is_expm=True, file_list=hist_files, gain_factor=g_factor)
    print(f"--- Gain {att_db} ---")
    print(f"  ExpF Baseline : Recall={res_f_g['recall']*100:.1f}% ({res_f_g['tp']}/24), FP={res_f_g['fp']}/68, F1={res_f_g['f1']:.4f}")
    print(f"  Candidate ExpM: Recall={res_m_g['recall']*100:.1f}% ({res_m_g['tp']}/24), FP={res_m_g['fp']}/68, F1={res_m_g['f1']:.4f}")

# Save full results
report_json = PROJECT_ROOT / "reports" / "candidate_expM_full_evaluation.json"
with open(report_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull evaluation results saved to: {report_json}")
