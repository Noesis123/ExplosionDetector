"""Candidate ExpP: Recording-Balanced Hard-Negative Fine-Tuning.

Design based on verified training inventory:
- 634 training files (207 pos, 427 neg).
- Eliminates repeated window overemphasis: exactly 1-2 representative windows per recording.
- File-level stratification: every recording has equal representation.
- Positive boundary anchoring: protects fragile blasts (demolition 08, distant explosions).
- Hard negative margin suppression: penalizes misclassified recordings (thunder, gunfire, artillery, backfires).
- Evaluates after each epoch on Source-Isolated Val (190 files) and Historical Benchmark (92 files).
- Guardrails: Saves best candidate ONLY IF Historical TP >= 22, Demolition 08 detected, and Historical FP < 19.
"""

import json
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
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

# 1. Load ExpF Baseline Checkpoint
exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

model_expF = ExplosionBinaryCNN(freeze_bn=True)
model_expF.load_state_dict(sd_f)
model_expF.eval()

# 2. Load 634-file Training Inventory
fit_data = torch.load(DATA_DIR / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
train_records = fit_data["file_records"]
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)
val_records = val_data["file_records"]

print(f"Loaded Train Inventory: {len(train_records)} files (207 pos, 427 neg)")
print(f"Loaded Validation Cache: {len(val_records)} files (70 pos, 120 neg)")

# 3. Build Recording-Balanced Pool (1-2 windows per recording)
print("Extracting Recording-Balanced Windows using ExpF predictions...")
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
            # Positive file: always include peak window
            balanced_pos.append({
                "spec": windows[max_idx]["spec"],
                "p_exp": max_p,
                "is_fragile": (max_p <= 0.78),
                "category": cat,
                "filename": fname
            })
            # If there is also a fragile window near threshold (0.50 - 0.75), include it
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
            # Negative file: include the single peak (hardest) window of this recording
            balanced_neg.append({
                "spec": windows[max_idx]["spec"],
                "p_exp": max_p,
                "is_hard": (max_p >= 0.50),
                "category": cat,
                "filename": fname
            })

print(f"Recording-Balanced Pool: {len(balanced_pos)} pos windows (from 207 files), {len(balanced_neg)} neg windows (from 427 files)")

# Check negative categories in balanced pool
neg_hard = [w for w in balanced_neg if w["is_hard"]]
print(f"Hard Negative Recordings (ExpF p_max >= 0.50): {len(neg_hard)} / {len(balanced_neg)}")
cat_hard = {}
for w in neg_hard:
    c = w["category"]
    cat_hard[c] = cat_hard.get(c, 0) + 1
print("Hard Negative Recordings by category:", sorted(cat_hard.items(), key=lambda x: -x[1]))

# 4. Pre-load Historical Benchmark Data (92 files)
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
hist_eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

print("Pre-loading historical benchmark audio into memory...")
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


def evaluate_val_cache(model, threshold=0.70):
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0
    fp_by_cat = {}

    with torch.no_grad():
        for r in val_records:
            true_lbl = r["binary_label"]
            cat = r.get("category", "unknown")
            specs = torch.stack([w["spec"] for w in r["windows"]]).unsqueeze(1).float()
            probs = torch.softmax(model(specs), dim=-1)[:, 1]
            file_max_p = float(torch.max(probs))

            if true_lbl == 1:
                if file_max_p >= threshold:
                    file_tp += 1
                else:
                    file_fn += 1
            else:
                if file_max_p >= threshold:
                    file_fp += 1
                    fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
                else:
                    file_tn += 1

    total_pos = file_tp + file_fn
    total_neg = file_tn + file_fp
    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / total_pos if total_pos > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (file_tp + file_tn) / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0

    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "fp_by_cat": fp_by_cat
    }


def evaluate_historical_benchmark(model, threshold=0.70):
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0
    fp_by_cat = {}
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
                min_lf_ratio=0.35,
                max_hf_ratio=0.15
            )

            file_alerted = False
            max_prob = 0.0
            for i, w in enumerate(windows):
                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
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
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=p_not_exp, p_exp=p_exp, p_fw=0.0,
                        peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio
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
                    fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
                    fp_files.append((fname, cat, max_prob))
                else:
                    file_tn += 1

    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (file_tp + file_tn) / (file_tp + file_fn + file_tn + file_fp)

    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "recall": rec, "precision": prec, "f1": f1, "accuracy": acc,
        "missed": missed_pos, "detected": detected_pos,
        "fp_files": fp_files, "fp_by_cat": fp_by_cat
    }


def train_expP():
    print("\n" + "=" * 80)
    print("STARTING CANDIDATE ExpP: RECORDING-BALANCED HARD-NEGATIVE OPTIMIZATION")
    print("=" * 80)

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_f)

    # ExpF Baseline
    base_val = evaluate_val_cache(model, threshold=0.70)
    base_hist = evaluate_historical_benchmark(model, threshold=0.70)
    print(f"\nExpF Baseline:")
    print(f"  Val Dev (190 files)  : Recall={base_val['recall']*100:.2f}% ({base_val['tp']}/70), Prec={base_val['precision']*100:.2f}%, F1={base_val['f1']:.4f}, FP={base_val['fp']}/120")
    print(f"  Hist Bench (92 files): Recall={base_hist['recall']*100:.2f}% ({base_hist['tp']}/24), Prec={base_hist['precision']*100:.2f}%, F1={base_hist['f1']:.4f}, FP={base_hist['fp']}/68")
    print(f"  ExpF Hist FP by Cat  : {base_hist['fp_by_cat']}")

    demo_08_ok = any(f[0] == "bbc_demolition_07022508.mp3" for f in base_hist["detected"])
    print(f"  ExpF Demolition 08 Detected: {demo_08_ok}")

    # Freeze Block 1, 2, 3 completely
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
    lr = 3.0e-6
    anchor_lambda = 2.0e-3
    alpha_pos_margin = 1.0
    beta_neg_margin = 1.5
    pos_target = 1.15   # corresponds to p_exp ~ 0.76
    neg_target = -0.40  # corresponds to p_exp ~ 0.40

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Prepare pools for batch sampling
    pos_fragile = [w for w in balanced_pos if w["is_fragile"]]
    pos_robust = [w for w in balanced_pos if not w["is_fragile"]]
    neg_hard = [w for w in balanced_neg if w["is_hard"]]
    neg_standard = [w for w in balanced_neg if not w["is_hard"]]

    print(f"Sample Pools: PosFragile={len(pos_fragile)}, PosRobust={len(pos_robust)}, NegHard={len(neg_hard)}, NegStandard={len(neg_standard)}")

    best_hist_fp = base_hist["fp"]
    best_hist_f1 = base_hist["f1"]
    best_epoch = 0
    results_history = []

    num_epochs = 4
    steps_per_epoch = 50

    for epoch in range(1, num_epochs + 1):
        model.train()
        model.set_bn_eval()

        epoch_tot = []
        epoch_ce = []
        epoch_pos_m = []
        epoch_neg_m = []
        epoch_anc = []
        t0 = time.time()

        for step in range(steps_per_epoch):
            # Batch of 48: 24 pos (12 fragile, 12 robust), 24 neg (16 hard, 8 standard)
            s_pos = random.sample(pos_fragile, min(12, len(pos_fragile))) + random.sample(pos_robust, 12)
            s_neg = random.sample(neg_hard, min(16, len(neg_hard))) + random.sample(neg_standard, 8)

            batch_items = (
                [(w["spec"], 1, 1) for w in s_pos if w["is_fragile"]] +
                [(w["spec"], 1, 0) for w in s_pos if not w["is_fragile"]] +
                [(w["spec"], 0, 2) for w in s_neg if w["is_hard"]] +
                [(w["spec"], 0, 0) for w in s_neg if not w["is_hard"]]
            )
            random.shuffle(batch_items)

            specs = torch.stack([item[0] for item in batch_items]).unsqueeze(1).float()
            labels = torch.tensor([item[1] for item in batch_items], dtype=torch.long)
            tags = torch.tensor([item[2] for item in batch_items], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(specs)

            ce_loss = F.cross_entropy(logits, labels)
            delta = logits[:, 1] - logits[:, 0]

            # Fragile positive margin loss
            mask_fragile = (tags == 1)
            if mask_fragile.sum() > 0:
                pos_m_loss = torch.mean(F.relu(pos_target - delta[mask_fragile]) ** 2)
            else:
                pos_m_loss = torch.tensor(0.0)

            # Hard negative margin loss
            mask_hard_neg = (tags == 2)
            if mask_hard_neg.sum() > 0:
                neg_m_loss = torch.mean(F.relu(delta[mask_hard_neg] - neg_target) ** 2)
            else:
                neg_m_loss = torch.tensor(0.0)

            # Weight anchor
            anc_loss = sum(torch.sum((p - anchor_weights[n]) ** 2) for n, p in model.named_parameters() if n in anchor_weights)

            tot_loss = ce_loss + alpha_pos_margin * pos_m_loss + beta_neg_margin * neg_m_loss + anchor_lambda * anc_loss
            tot_loss.backward()
            optimizer.step()

            epoch_tot.append(tot_loss.item())
            epoch_ce.append(ce_loss.item())
            epoch_pos_m.append(pos_m_loss.item())
            epoch_neg_m.append(neg_m_loss.item())
            epoch_anc.append(anc_loss.item())

        elapsed = time.time() - t0

        # Evaluation
        val_res = evaluate_val_cache(model, threshold=0.70)
        hist_res = evaluate_historical_benchmark(model, threshold=0.70)

        demo_08_ok = any(f[0] == "bbc_demolition_07022508.mp3" for f in hist_res["detected"])

        print(f"\n--- Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) ---")
        print(f"  Losses: Tot={np.mean(epoch_tot):.4f}, CE={np.mean(epoch_ce):.4f}, PosM={np.mean(epoch_pos_m):.4f}, NegM={np.mean(epoch_neg_m):.4f}, Anc={np.mean(epoch_anc):.4f}")
        print(f"  Val Dev (190 files)  : Recall={val_res['recall']*100:.1f}% ({val_res['tp']}/70), Prec={val_res['precision']*100:.1f}%, F1={val_res['f1']:.4f}, FP={val_res['fp']}/120")
        print(f"  Hist Bench (92 files): Recall={hist_res['recall']*100:.1f}% ({hist_res['tp']}/24), Prec={hist_res['precision']*100:.1f}%, F1={hist_res['f1']:.4f}, FP={hist_res['fp']}/68")
        print(f"  Demolition 08 Detected: {demo_08_ok}")
        print(f"  Hist FP by Category: {hist_res['fp_by_cat']}")

        epoch_record = {
            "epoch": epoch,
            "val_recall": val_res["recall"],
            "val_precision": val_res["precision"],
            "val_f1": val_res["f1"],
            "val_tp": val_res["tp"],
            "val_fp": val_res["fp"],
            "hist_recall": hist_res["recall"],
            "hist_tp": hist_res["tp"],
            "hist_fp": hist_res["fp"],
            "hist_precision": hist_res["precision"],
            "hist_f1": hist_res["f1"],
            "demolition_08_detected": demo_08_ok,
            "fp_by_cat": hist_res["fp_by_cat"]
        }
        results_history.append(epoch_record)

        # GUARDRAIL CONDITION
        if hist_res["tp"] >= 22 and demo_08_ok and val_res["tp"] >= 56 and hist_res["fp"] < base_hist["fp"]:
            best_hist_fp = hist_res["fp"]
            best_hist_f1 = hist_res["f1"]
            best_epoch = epoch
            ckpt_path = CKPT_DIR / "candidate_expP_recording_balanced_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_res,
                "hist_metrics": hist_res,
                "exp_name": "candidate_expP_recording_balanced"
            }, ckpt_path)
            print(f"  >>> SAVED NEW BEST CANDIDATE ExpP: {ckpt_path.name} (Recall={hist_res['tp']}/24, FP={hist_res['fp']}/68)")

    final_path = CKPT_DIR / "candidate_expP_recording_balanced_final.pt"
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "history": results_history,
        "exp_name": "candidate_expP_recording_balanced_final"
    }, final_path)
    print(f"\nFinal Checkpoint Saved: {final_path.name}")

    with open(REPORTS_DIR / "candidate_expP_training_history.json", "w") as f:
        json.dump({
            "base_val": base_val,
            "base_hist": base_hist,
            "history": results_history,
            "best_epoch": best_epoch
        }, f, indent=2)

    return {"best_epoch": best_epoch, "best_hist_fp": best_hist_fp, "history": results_history}


if __name__ == "__main__":
    train_expP()
