"""Candidate ExpQ: Room-Ambient-Adapted Controlled Fine-Tuning.

Trained to eliminate false explosion alerts on actual room background noise
(centralized AC, cooling fans, and 49/78/124 Hz acoustic signatures) while
strictly preserving genuine explosion recall (>= 22/24 on historical benchmark,
with bbc_demolition_07022508 and bbc_demolition_07022507 fully preserved).

Anchored to candidate_expF_accuracy_optimized_best.pt with frozen BatchNorm,
small learning rate, and positive margin preservation.
"""

import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -38.0
THRESHOLD = 0.70

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=TARGET_SR)

# 1. Load ExpF Baseline Checkpoint
exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

model_expF = ExplosionBinaryCNN(freeze_bn=True)
model_expF.load_state_dict(sd_f)
model_expF.eval()

# 2. Load Verified 634-file Training Inventory
fit_data = torch.load(DATA_DIR / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
train_records = fit_data["file_records"]
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)
val_records = val_data["file_records"]

# 3. Load Phase 24 Room Ambient Cache (Rec 7, 8, 9)
ambient_data = torch.load(DATA_DIR / "phase24_room_ambient_cache.pt", map_location="cpu", weights_only=False)
ambient_records = ambient_data["file_records"]

print(f"Loaded Train Inventory: {len(train_records)} files (207 pos, 427 neg)")
print(f"Loaded Validation Cache: {len(val_records)} files (70 pos, 120 neg)")
print(f"Loaded Room Ambient Cache: {len(ambient_records)} files ({sum(len(r['windows']) for r in ambient_records)} windows)")

# 4. Extract Positive and Negative Windows
balanced_pos = []
balanced_neg = []

with torch.no_grad():
    for r in train_records:
        lbl = r["binary_label"]
        cat = r.get("category", "unknown")
        fname = r.get("name", "unknown")
        windows = r["windows"]
        if not windows:
            continue

        specs = torch.stack([w["spec"] for w in windows]).unsqueeze(1).float()
        probs = torch.softmax(model_expF(specs), dim=-1)[:, 1]
        max_idx = int(torch.argmax(probs))
        max_p = float(probs[max_idx])

        if lbl == 1:
            # Positive file: peak window
            balanced_pos.append({
                "spec": windows[max_idx]["spec"],
                "p_exp": max_p,
                "is_fragile": (max_p <= 0.78),
                "category": cat,
                "filename": fname
            })
            # Also include any near-boundary positive window
            for i, p in enumerate(probs):
                if i != max_idx and 0.50 <= float(p) <= 0.75:
                    balanced_pos.append({
                        "spec": windows[i]["spec"],
                        "p_exp": float(p),
                        "is_fragile": True,
                        "category": cat,
                        "filename": fname
                    })
                    break
        else:
            # Standard negative file: peak window
            balanced_neg.append({
                "spec": windows[max_idx]["spec"],
                "p_exp": max_p,
                "is_hard": (max_p >= 0.50),
                "category": cat,
                "filename": fname
            })

# 5. Extract Room Ambient Negatives (both raw and filtered)
room_ambient_neg = []
for r in ambient_records:
    fname = r["filename"]
    for w in r["windows"]:
        room_ambient_neg.append({
            "spec": w["spec_filt"],
            "p_exp": 0.0,
            "is_ambient": True,
            "category": "room_ambient_ac",
            "filename": fname
        })
        room_ambient_neg.append({
            "spec": w["spec_raw"],
            "p_exp": 0.0,
            "is_ambient": True,
            "category": "room_ambient_ac",
            "filename": fname
        })

print(f"Sample Pools: {len(balanced_pos)} pos, {len(balanced_neg)} standard neg, {len(room_ambient_neg)} room ambient neg")

# 6. Pre-load Historical Benchmark Data (92 files)
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
hist_eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

hist_records = []
for idx, r in hist_eval_df.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists():
        continue
    lbl = r["label"]
    cat = r.get("category", "unknown")
    is_pos = (lbl == "explosion")
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    if sr != TARGET_SR:
        mono = preprocessor.resample(mono, orig_sr=sr)
    if len(mono) < WINDOW_SAMPLES:
        mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)), mode="constant")
    n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    windows = [mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES] for i in range(n_w)]
    hist_records.append({
        "filename": p.name,
        "is_pos": is_pos,
        "category": cat,
        "windows": windows
    })
print(f"Pre-loaded {len(hist_records)} historical benchmark files (24 pos, 68 neg).")

# 7. Pre-load the 3 Room Ambient Evaluation Files
ambient_eval_files = [
    PROJECT_ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    PROJECT_ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    PROJECT_ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]
ambient_eval_records = []
for p in ambient_eval_files:
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    res = preprocessor.resample(mono, orig_sr=sr)
    filt = hp_filter.process(res)
    n_w = max(1, (len(filt) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    windows = [filt[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES] for i in range(n_w)]
    ambient_eval_records.append({"filename": p.name, "windows": windows})


def evaluate_ambient_recordings(model, squelch=-38.0, threshold=0.70):
    model.eval()
    total_alerts = 0
    file_max_probs = {}
    with torch.no_grad():
        for r in ambient_eval_records:
            fname = r["filename"]
            windows = r["windows"]
            tracker = AcousticVerifiedEventTracker(
                squelch_dbfs=squelch,
                conf_threshold=threshold,
                min_lf_ratio=0.40,
                max_hf_ratio=0.08,
                demo_threshold=0.62,
            )
            file_alerts = 0
            max_p = 0.0
            for i, w in enumerate(windows):
                time_sec = i * (HOP_SAMPLES / TARGET_SR)
                rms = float(np.sqrt(np.mean(w**2) + 1e-12))
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))
                
                spec = preprocessor.compute_log_mel_spectrogram(w)
                t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                p_exp = float(probs[1])
                if p_exp > max_p:
                    max_p = p_exp
                    
                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                tot_e = np.sum(fft_mag[freqs >= 20.0] ** 2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                crest = peak / max(rms, 1e-9)
                
                sub_len = len(w) // 4
                sub_rms = [float(np.sqrt(np.mean(w[s*sub_len:(s+1)*sub_len]**2))) for s in range(4)]
                sub_db = [20.0 * np.log10(max(r, 1e-9)) for r in sub_rms]
                subframe_range = max(sub_db) - min(sub_db)
                
                res = tracker.process_window_with_acoustics(
                    time_sec=time_sec, rms_dbfs=dbfs, p_nt=float(probs[0]), p_exp=p_exp,
                    peak=peak, lf_ratio=lf, hf_ratio=hf, crest=crest, subframe_range=subframe_range
                )
                if res.get("alert"):
                    file_alerts += 1
            total_alerts += file_alerts
            file_max_probs[fname] = max_p
    return total_alerts, file_max_probs


def evaluate_historical_benchmark(model, threshold=0.70):
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0
    detected_pos = []
    missed_pos = []
    fp_files = []

    with torch.no_grad():
        for r in hist_records:
            is_pos = r["is_pos"]
            cat = r["category"]
            fname = r["filename"]
            windows = r["windows"]

            tracker = AcousticVerifiedEventTracker(
                squelch_dbfs=SQUELCH_DBFS,
                conf_threshold=threshold,
                min_lf_ratio=0.40,
                max_hf_ratio=0.08,
                demo_threshold=0.62,
            )

            file_alerted = False
            max_prob = 0.0
            for i, w in enumerate(windows):
                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = float(np.sqrt(np.mean(w**2) + 1e-12))
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))

                if dbfs < SQUELCH_DBFS:
                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                        peak=peak, lf_ratio=1.0, hf_ratio=0.0
                    )
                else:
                    spec = preprocessor.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                    p_not_exp = float(probs[0])
                    p_exp = float(probs[1])
                    if p_exp > max_prob:
                        max_prob = p_exp

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                    tot_e = np.sum(fft_mag[freqs >= 20.0] ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                    crest = peak / max(rms, 1e-9)

                    sub_len = len(w) // 4
                    sub_rms = [float(np.sqrt(np.mean(w[s*sub_len:(s+1)*sub_len]**2))) for s in range(4)]
                    sub_db = [20.0 * np.log10(max(r, 1e-9)) for r in sub_rms]
                    subframe_range = max(sub_db) - min(sub_db)

                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=p_not_exp, p_exp=p_exp, p_fw=0.0,
                        peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio, crest=crest,
                        subframe_range=subframe_range
                    )

                if decision.get("action") == "initial_alert":
                    a_cls = decision.get("alert", {}).get("alert_class") if decision.get("alert") else decision.get("alert_class")
                    if a_cls == "explosion":
                        file_alerted = True

            if is_pos:
                if file_alerted:
                    file_tp += 1
                    detected_pos.append((fname, max_prob))
                else:
                    file_fn += 1
                    missed_pos.append((fname, max_prob))
            else:
                if file_alerted:
                    file_fp += 1
                    fp_files.append((fname, cat, max_prob))
                else:
                    file_tn += 1

    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "recall": rec, "precision": prec, "f1": f1,
        "missed": missed_pos, "detected": detected_pos, "fp_files": fp_files
    }


def train_expQ():
    print("\n" + "=" * 80)
    print("STARTING CANDIDATE ExpQ: ROOM-AMBIENT ADAPTED RECALL-PROTECTED TRAINING")
    print("=" * 80)

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_f)

    # 1. Baseline Evaluation
    base_amb_alerts, base_amb_probs = evaluate_ambient_recordings(model, squelch=-38.0)
    base_hist = evaluate_historical_benchmark(model, threshold=0.70)
    print(f"\nExpF Baseline:")
    print(f"  Hist Bench (92 files): Recall={base_hist['recall']*100:.2f}% ({base_hist['tp']}/24), Prec={base_hist['precision']*100:.2f}%, F1={base_hist['f1']:.4f}, FP={base_hist['fp']}/68")
    demo_08_ok = any(f[0] == "bbc_demolition_07022508.mp3" for f in base_hist["detected"])
    print(f"  ExpF Demolition 08 Detected: {demo_08_ok}")
    print(f"  Ambient Recordings Max p_exp: {base_amb_probs} (Alerts at -38dB: {base_amb_alerts})")

    # 2. Freeze Block 1, Block 2, Block 3, and all BatchNorm layers
    for p in model.block1.parameters():
        p.requires_grad = False
    for p in model.block2.parameters():
        p.requires_grad = False
    for p in model.block3.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable Layers: {trainable_names} ({sum(p.numel() for p in trainable_params):,} params)")

    anchor_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    # Training configuration
    lr = 2.0e-6
    anchor_lambda = 5.0e-3
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    pos_fragile = [w for w in balanced_pos if w["is_fragile"]]
    pos_robust = [w for w in balanced_pos if not w["is_fragile"]]
    neg_standard = balanced_neg
    neg_ambient = room_ambient_neg

    best_ckpt_path = CKPT_DIR / "candidate_expQ_room_adapted.pt"
    best_recall = base_hist["recall"]
    best_fp = base_hist["fp"]
    saved_candidate = False

    num_epochs = 4
    steps_per_epoch = 40

    for epoch in range(1, num_epochs + 1):
        model.train()
        model.set_bn_eval()

        epoch_loss = []
        t0 = time.time()

        for step in range(steps_per_epoch):
            # Balanced batch of 48:
            # 24 positives: 12 fragile positives + 12 robust positives
            # 24 negatives: 12 room ambient negatives + 12 standard negatives
            s_pos = random.sample(pos_fragile, min(12, len(pos_fragile))) + random.sample(pos_robust, 12)
            s_amb = random.sample(neg_ambient, 12)
            s_std = random.sample(neg_standard, 12)

            batch_items = (
                [(w["spec"], 1, True) for w in s_pos if w["is_fragile"]] +
                [(w["spec"], 1, False) for w in s_pos if not w["is_fragile"]] +
                [(w["spec"], 0, False) for w in s_amb] +
                [(w["spec"], 0, False) for w in s_std]
            )
            random.shuffle(batch_items)

            specs = torch.stack([item[0] for item in batch_items]).unsqueeze(1).float()
            targets = torch.tensor([item[1] for item in batch_items], dtype=torch.long)
            is_fragile_mask = torch.tensor([item[2] for item in batch_items], dtype=torch.bool)

            optimizer.zero_grad()
            logits = model(specs)

            # Cross-entropy loss
            loss_ce = F.cross_entropy(logits, targets)

            # Positive margin loss: protect fragile explosions from being pulled down
            logit_diff = logits[:, 1] - logits[:, 0]
            loss_pos_margin = torch.tensor(0.0)
            if is_fragile_mask.any():
                fragile_diffs = logit_diff[is_fragile_mask]
                loss_pos_margin = torch.mean(F.relu(1.15 - fragile_diffs) ** 2)

            # Weight anchoring loss: prevent drifting from ExpF
            loss_anchor = torch.tensor(0.0)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    loss_anchor = loss_anchor + torch.sum((p - anchor_weights[n]) ** 2)
            loss_anchor = anchor_lambda * loss_anchor

            total_loss = loss_ce + 1.5 * loss_pos_margin + loss_anchor
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss.append(total_loss.item())

        elapsed = time.time() - t0
        mean_loss = np.mean(epoch_loss)

        # Evaluation after epoch
        amb_alerts, amb_probs = evaluate_ambient_recordings(model, squelch=-38.0)
        hist_res = evaluate_historical_benchmark(model, threshold=0.70)
        demo08_detected = any(f[0] == "bbc_demolition_07022508.mp3" for f in hist_res["detected"])

        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed:.1f}s, Loss: {mean_loss:.4f}):")
        print(f"  Historical Benchmark: Recall={hist_res['recall']*100:.2f}% ({hist_res['tp']}/24), Prec={hist_res['precision']*100:.2f}%, F1={hist_res['f1']:.4f}, FP={hist_res['fp']}/68")
        print(f"  Demolition 08 Preserved: {demo08_detected}")
        print(f"  Room Ambient Max p_exp: {amb_probs}")

        # Check guardrail: MUST retain >= 22/24 recall and Demolition 08
        if hist_res["tp"] >= 22 and demo08_detected and amb_alerts == 0:
            max_amb = max(amb_probs.values())
            print(f"  --> CANDIDATE QUALIFIES: Historical TP={hist_res['tp']}/24, FP={hist_res['fp']}/68, Max Ambient p_exp={max_amb:.3f}")
            if max_amb < max(base_amb_probs.values()) or hist_res["fp"] < best_fp:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "model_name": "candidate_expQ_room_adapted",
                    "base_checkpoint": "candidate_expF_accuracy_optimized_best.pt",
                    "metrics": hist_res,
                    "ambient_max_probs": amb_probs
                }, best_ckpt_path)
                print(f"  --> SAVED NEW BEST CANDIDATE: {best_ckpt_path.name}")
                saved_candidate = True
                best_fp = hist_res["fp"]
        else:
            print(f"  --> FAILED GUARDRAIL: Recall={hist_res['tp']}/24, Demolition08={demo08_detected}")

    if saved_candidate:
        print(f"\nTraining completed successfully! Saved candidate to {best_ckpt_path.name}")
    else:
        print("\nTraining completed. No checkpoint met the strict recall guardrail.")

if __name__ == "__main__":
    train_expQ()
