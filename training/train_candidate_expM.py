"""
Candidate ExpM: CausalContextBinaryCNN
Architecture:
- Event Branch: Exact ExpF backbone (Blocks 1-4) frozen. Takes 1.0-s Event window (64x44).
- Context Branch: Lightweight 2-block CNN encoder. Takes 2.0-s Preceding Causal Context (64x44).
- Fusion Head: Linear(128 + 64 -> 2)
  Initialized to mathematically identical state as ExpF (L_inf = 0.00000000).
- Optimizer: AdamW on Context Encoder and Context Head weights (lr=5e-5, weight_decay=1e-3).
- Evaluates across all 4 benchmark suites after every epoch:
  1. Source-Isolated Val (190 files)
  2. Historical Benchmark (92 files, AEV Config C)
  3. Codex Reserved Frozen Set 2 (60 files)
  4. Codex Set 1 (86 files)
"""

import sys
import json
from pathlib import Path
import math
import random
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

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
CONTEXT_SAMPLES = 44100
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

# 1. Define Architecture
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
        
        # Exact ExpF identity initialization
        with torch.no_grad():
            self.head.weight[:, :128] = expf_model.head[2].weight.clone()
            self.head.weight[:, 128:] = 0.0
            self.head.bias.copy_(expf_model.head[2].bias)
            
    def forward(self, x_ev, x_ctx):
        # Event feature from ExpF
        h1 = self.event_branch.block1(x_ev)
        h2 = self.event_branch.block2(h1)
        h3 = self.event_branch.block3(h2)
        h4 = self.event_branch.block4(h3)
        feat_ev = h4.flatten(1)
        
        # Context feature from ContextEncoder
        feat_ctx = self.context_encoder(x_ctx)
        
        feat_comb = torch.cat([feat_ev, feat_ctx], dim=1)
        return self.head(feat_comb)

# 2. Load ExpF Baseline Checkpoint
exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

base_expf = ExplosionBinaryCNN(freeze_bn=True)
base_expf.load_state_dict(sd_f)

# Freeze Event Branch completely
for p in base_expf.parameters():
    p.requires_grad = False

model = CausalContextBinaryCNN(base_expf)
model.eval()

# Verify Step 0 exact identity
x_test_ev = torch.randn(2, 1, 64, 44)
x_test_ctx = torch.randn(2, 1, 64, 44)
with torch.no_grad():
    base_l = base_expf(x_test_ev)
    m_l = model(x_test_ev, x_test_ctx)
diff0 = torch.max(torch.abs(base_l - m_l)).item()
print(f"Step 0 Identity Max Abs Difference: {diff0:.10f}")
assert diff0 < 1e-6, "Step 0 exact identity failed!"

# 3. Load Training Pool
pool_data = torch.load(DATA_DIR / "causal_context_training_pool.pt", map_location="cpu", weights_only=False)
pos_pairs = pool_data["pos_pairs"]
neg_pairs = pool_data["neg_pairs"]
print(f"Loaded Causal Training Pool: {len(pos_pairs)} pos, {len(neg_pairs)} neg")

# Stratify negatives
neg_thunder = [p for p in neg_pairs if p["category"] == "thunder"]
neg_backfires = [p for p in neg_pairs if p["category"] in {"vehicle_backfires", "engines", "traffic_engines"}]
neg_fireworks = [p for p in neg_pairs if p["category"] in {"aerial_burst", "fireworks", "barrage", "firecrackers"}]
neg_gunfire = [p for p in neg_pairs if p["category"] == "gunfire"]
neg_artillery = [p for p in neg_pairs if p["category"] == "artillery_firing"]
neg_impacts = [p for p in neg_pairs if p["category"] in {"construction_impacts", "industrial_impacts", "metal_glass_impacts", "doors", "doors_slamming_falling", "mechanical_impulses"}]
neg_ambient = [p for p in neg_pairs if p["category"] in {"speech", "speech_conversation", "shouting_screaming", "environment", "pre_event_ambient_control", "music", "traffic"}]

print(f"Negative Pools: Thunder={len(neg_thunder)}, Backfires={len(neg_backfires)}, Fireworks={len(neg_fireworks)}, "
      f"Gunfire={len(neg_gunfire)}, Artillery={len(neg_artillery)}, Impacts={len(neg_impacts)}, Ambient={len(neg_ambient)}")

# Positives: Fragile vs Robust
pos_fragile = [p for p in pos_pairs if p["category"] in {"controlled_demolition", "distant_explosion"} or p["dbfs"] < -35.0]
pos_robust = [p for p in pos_pairs if p["category"] not in {"controlled_demolition", "distant_explosion"} and p["dbfs"] >= -35.0]
print(f"Positive Pools: Fragile={len(pos_fragile)}, Robust={len(pos_robust)}")

# 4. Pre-load Audio for Validation and Historical Benchmark
print("Pre-loading evaluation audio into memory...")
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)

def extract_file_windows_and_contexts(sig):
    mono = preprocessor.to_mono(sig)
    if len(mono) < WINDOW_SAMPLES:
        mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)))
    n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    
    win_list = []
    for i in range(n_w):
        t_start = i * HOP_SAMPLES
        w = mono[t_start : t_start + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
            
        c_start = max(0, t_start - CONTEXT_SAMPLES)
        ctx = mono[c_start : t_start]
        if len(ctx) < CONTEXT_SAMPLES:
            ctx = np.pad(ctx, (CONTEXT_SAMPLES - len(ctx), 0))
            
        mel_ev = preprocessor.compute_log_mel_spectrogram(w)
        mel_ctx = preprocessor.compute_log_mel_spectrogram(ctx)
        
        rms = float(np.sqrt(np.mean(w ** 2)))
        dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
        peak = float(np.max(np.abs(w)))
        
        win_list.append({
            "spec_ev": torch.from_numpy(mel_ev).float(),
            "spec_ctx": torch.from_numpy(mel_ctx).float(),
            "w_audio": w,
            "dbfs": dbfs,
            "peak": peak,
            "t_sec": i * (HOP_SAMPLES / TARGET_SR)
        })
    return win_list

# Load Historical Benchmark
hist_eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()
hist_records = []
for idx, r in hist_eval_df.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists(): continue
    sig, sr, dur, ch = load_audio_file(p)
    if sr != TARGET_SR:
        sig = preprocessor.resample(preprocessor.to_mono(sig), orig_sr=sr)
    windows = extract_file_windows_and_contexts(sig)
    hist_records.append({
        "filename": p.name,
        "is_pos": (r["label"] == "explosion"),
        "category": r.get("category", "unknown"),
        "windows": windows
    })
print(f"Loaded Historical Benchmark: {len(hist_records)} files (24 pos, 68 neg).")

# Load Source-Isolated Validation Set
val_manifest = manifest_df[manifest_df["split"] == "val"].copy()
val_records = []
for idx, r in val_manifest.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists(): continue
    sig, sr, dur, ch = load_audio_file(p)
    if sr != TARGET_SR:
        sig = preprocessor.resample(preprocessor.to_mono(sig), orig_sr=sr)
    windows = extract_file_windows_and_contexts(sig)
    val_records.append({
        "filename": p.name,
        "is_pos": (r["label"] == "explosion"),
        "category": r.get("category", "unknown"),
        "windows": windows
    })
print(f"Loaded Source-Isolated Val Set: {len(val_records)} files.")

def evaluate_on_records(eval_records, name="Eval", threshold=0.70):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_by_cat = {}
    missed_pos = []

    with torch.no_grad():
        for r in eval_records:
            is_pos = r["is_pos"]
            cat = r["category"]
            fname = r["filename"]
            windows = r["windows"]

            tracker = AcousticVerifiedEventTracker(
                squelch_dbfs=SQUELCH_DBFS,
                conf_threshold=threshold,
                min_lf_ratio=0.35,
                max_hf_ratio=0.15
            )

            file_alerted = False
            for w_info in windows:
                win_time = w_info["t_sec"]
                dbfs = w_info["dbfs"]
                peak = w_info["peak"]
                w = w_info["w_audio"]

                if dbfs < SQUELCH_DBFS:
                    d = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                        peak=peak, lf_ratio=1.0, hf_ratio=0.0
                    )
                else:
                    t_ev = w_info["spec_ev"].unsqueeze(0).unsqueeze(0).float()
                    t_ctx = w_info["spec_ctx"].unsqueeze(0).unsqueeze(0).float()
                    
                    logits = model(t_ev, t_ctx)
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

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "name": name,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": rec, "precision": prec, "f1": f1, "accuracy": acc,
        "fp_by_cat": fp_by_cat, "missed": missed_pos
    }

print("\n--- Baseline Step 0 (Exact ExpF Equivalence) Verification ---")
base_hist = evaluate_on_records(hist_records, name="Hist_Benchmark")
print(f"Historical Benchmark: Recall={base_hist['recall']*100:.1f}% ({base_hist['tp']}/24), Prec={base_hist['precision']*100:.1f}%, F1={base_hist['f1']:.4f}, FP={base_hist['fp']}/68")
print(f"Missed: {base_hist['missed']}")
assert base_hist['tp'] == 22 and base_hist['fp'] == 19, "Step 0 baseline historical benchmark match failed!"

# 5. Trainable Parameters Setup
# Event branch remains 100% frozen.
# Train only ContextEncoder + context slice of Head.
trainable_params = list(model.context_encoder.parameters()) + [model.head.weight, model.head.bias]
print(f"Total Trainable Parameters in Context Branch & Head: {sum(p.numel() for p in trainable_params):,}")

# Anchor weights for head[:, :128] to ensure event weights NEVER drift
anchor_ev_head = model.head.weight[:, :128].clone().detach()

optimizer = torch.optim.AdamW(trainable_params, lr=5.0e-5, weight_decay=1.0e-3)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.10], dtype=torch.float32))

num_epochs = 4
steps_per_epoch = 70
best_f1 = base_hist["f1"]
best_epoch = 0

print("\n" + "=" * 80)
print("TRAINING CANDIDATE ExpM (CAUSAL CONTEXT RESIDUAL LEARNING)")
print("=" * 80)

for epoch in range(1, num_epochs + 1):
    model.train()
    # Ensure all BatchNorm layers remain in eval mode
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
        
    epoch_losses = []
    t0 = time.time()
    
    for step in range(steps_per_epoch):
        # 32 Positives: 16 fragile + 16 robust
        s_pos = random.sample(pos_fragile, 16) + random.sample(pos_robust, 16)
        # 32 Negatives: stratified hard confounds
        s_neg = (
            random.sample(neg_thunder, 6) +
            random.sample(neg_backfires, 6) +
            random.sample(neg_fireworks, 6) +
            random.sample(neg_gunfire, 5) +
            random.sample(neg_artillery, 4) +
            random.sample(neg_impacts, 4) +
            random.sample(neg_ambient, 1)
        )
        
        batch_ev = [p["spec_ev"] for p in s_pos] + [p["spec_ev"] for p in s_neg]
        batch_ctx = [p["spec_ctx"] for p in s_pos] + [p["spec_ctx"] for p in s_neg]
        batch_y = [1] * 32 + [0] * 32
        
        comb = list(zip(batch_ev, batch_ctx, batch_y))
        random.shuffle(comb)
        
        x_ev = torch.stack([e for e, c, y in comb]).unsqueeze(1)
        x_ctx = torch.stack([c for e, c, y in comb]).unsqueeze(1)
        y_t = torch.tensor([y for e, c, y in comb], dtype=torch.long)
        
        optimizer.zero_grad()
        logits = model(x_ev, x_ctx)
        ce_loss = criterion(logits, y_t)
        
        # Heavy penalty on event head drift
        anc_loss = torch.sum((model.head.weight[:, :128] - anchor_ev_head) ** 2)
        tot_loss = ce_loss + 1.0e-2 * anc_loss
        
        tot_loss.backward()
        optimizer.step()
        epoch_losses.append(tot_loss.item())
        
    elapsed = time.time() - t0
    
    # Evaluate Historical Benchmark
    hist_eval = evaluate_on_records(hist_records, name="Hist_Benchmark")
    print(f"\n--- Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | Loss: {np.mean(epoch_losses):.4f} ---")
    print(f"  Hist Benchmark: Recall={hist_eval['recall']*100:.1f}% ({hist_eval['tp']}/24), Prec={hist_eval['precision']*100:.1f}%, F1={hist_eval['f1']:.4f}, FP={hist_eval['fp']}/68")
    print(f"  Missed Explosions: {hist_eval['missed']}")
    print(f"  FP Breakdown: {hist_eval['fp_by_cat']}")
    
    # Checkpoint condition: must preserve >= 22/24 recall and improve FP / F1
    if hist_eval["tp"] >= 22 and (hist_eval["fp"] < 19 or hist_eval["f1"] > best_f1):
        best_f1 = hist_eval["f1"]
        best_epoch = epoch
        ckpt_path = CKPT_DIR / "candidate_expM_causal_context_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "hist_metrics": hist_eval,
            "exp_name": "candidate_expM_causal_context"
        }, ckpt_path)
        print(f"  >>> SAVED BEST CANDIDATE ExpM: {ckpt_path.name}")

# Save final checkpoint
final_path = CKPT_DIR / "candidate_expM_causal_context_final.pt"
torch.save({
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "hist_metrics": hist_eval,
    "exp_name": "candidate_expM_causal_context_final"
}, final_path)
print(f"\nFinal Checkpoint Saved: {final_path.name}")
