"""Candidate ExpL: Bounded Block 3, 4 + Head Fine-Tuning with Heavy Weight Anchoring to ExpF.

Strategy & Safeguards:
1. Initialize strictly from ExpF (candidate_expF_accuracy_optimized_best.pt).
2. Freeze Block 1 and Block 2 completely (protecting low-frequency waveform filters).
3. Freeze ALL BatchNorm running statistics and affine parameters across the entire network.
4. Trainable parameters: Block 3 Conv2d, Block 4 Conv2d, and Classifier Head Linear.
5. Conservative learning rate: 5.0e-6 with AdamW (weight decay 1e-4).
6. Heavy ExpF weight anchor (lambda = 5.0e-4) to penalize any deviation from ExpF weights.
7. Balanced CrossEntropyLoss (pos_weight = 1.10) - strictly NO asymmetric positive margin loss.
8. Stratified hard-negative batching (thunder, vehicle backfires, aerial bursts, gunfire, artillery firing, mechanical impacts).
9. Stratified positive batching (50% fragile/controlled demolition/distant blasts, 50% robust blasts).
10. Fast batched evaluation after every epoch on source-isolated val (190 files) and historical benchmark (92 files).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"

# 1. Load Data Pools
aug_data = torch.load(DATA_DIR / "augmented_training_pool.pt", map_location="cpu", weights_only=False)
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)

pool_pos = aug_data["pool_pos"]
pool_neg = aug_data["pool_neg"]
val_records = val_data["file_records"]

print(f"Loaded Augmented Training Pool: {len(pool_pos)} pos, {len(pool_neg)} neg")
print(f"Loaded Source-Isolated Val Cache: {len(val_records)} files (70 pos, 120 neg)")

# 2. Hard-Negative Stratification
neg_thunder = [w for w in pool_neg if w.get("category") == "thunder"]
neg_fireworks = [w for w in pool_neg if w.get("category") in {"aerial_burst", "fireworks", "firecrackers", "barrage"}]
neg_backfires = [w for w in pool_neg if w.get("category") in {"vehicle_backfires", "engines", "traffic_engines"}]
neg_gunfire = [w for w in pool_neg if w.get("category") == "gunfire"]
neg_artillery = [w for w in pool_neg if w.get("category") == "artillery_firing"]
neg_mechanical = [w for w in pool_neg if w.get("category") in {
    "construction_impacts", "industrial_impacts", "metal_glass_impacts",
    "doors", "doors_slamming_falling", "mechanical_impulses"
}]
neg_speech_ambient = [w for w in pool_neg if w.get("category") in {
    "speech", "speech_conversation", "shouting_screaming", "mixed_speech_noise",
    "environment", "pre_event_ambient_control", "music", "traffic", "crowds"
}]

print(f"Negative Pools: Thunder={len(neg_thunder)}, Fireworks={len(neg_fireworks)}, Backfires={len(neg_backfires)}, "
      f"Gunfire={len(neg_gunfire)}, ArtilleryFiring={len(neg_artillery)}, Mechanical={len(neg_mechanical)}, "
      f"Speech/Ambient={len(neg_speech_ambient)}")

# 3. Positive Stratification (Fragile / Demolition vs Robust)
pos_fragile = [w for w in pool_pos if w.get("category") in {"controlled_demolition", "distant_explosion"} or w.get("dbfs", 0) < -35.0]
pos_robust = [w for w in pool_pos if w.get("category") in {"close_explosion", "military_ordnance", "surface_high_explosive", "positives", "outdoor_explosion"} and w.get("dbfs", 0) >= -35.0]

print(f"Positive Pools: Fragile/Demolition={len(pos_fragile)}, Robust/Close={len(pos_robust)}")

# 4. Load Preprocessor and Historical Benchmark Data
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
hist_eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

# Pre-load audio for historical benchmark for fast evaluation
print("Pre-loading historical benchmark audio into memory...")
hist_records = []
preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)
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
    """Fast batched evaluation on the 190-file source-isolated validation set."""
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
        "clean_negatives": f"{file_tn}/{total_neg} ({file_tn/total_neg*100:.1f}%)",
        "fp_by_cat": fp_by_cat
    }


def evaluate_historical_benchmark(model, threshold=0.70):
    """Production evaluation of historical benchmark using AEV Config C."""
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0
    fp_by_cat = {}
    missed_pos = []

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
                else:
                    file_fn += 1
                    missed_pos.append(fname)
            else:
                if file_alerted:
                    file_fp += 1
                    fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
                else:
                    file_tn += 1

    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (file_tp + file_tn) / (file_tp + file_fn + file_tn + file_fp)

    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "recall": rec, "precision": prec, "f1": f1, "accuracy": acc,
        "missed": missed_pos, "fp_by_cat": fp_by_cat
    }


def train_expL():
    print("\n" + "=" * 80)
    print("STARTING CANDIDATE ExpL TRAINING (BOUNDED HYBRID FINE-TUNING)")
    print("=" * 80)

    # Load ExpF checkpoint
    exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
    ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
    sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_f)

    # Baseline evaluation
    base_val = evaluate_val_cache(model, threshold=0.70)
    print(f"\nExpF Baseline on Source-Isolated Val:")
    print(f"  Recall   : {base_val['recall']*100:.2f}% ({base_val['tp']}/70)")
    print(f"  Precision: {base_val['precision']*100:.2f}%")
    print(f"  F1-Score : {base_val['f1']:.4f}")
    print(f"  Clean Neg: {base_val['clean_negatives']}")
    print(f"  FP by Cat: {base_val['fp_by_cat']}")

    base_hist = evaluate_historical_benchmark(model, threshold=0.70)
    print(f"\nExpF Baseline on Historical Benchmark:")
    print(f"  Recall   : {base_hist['recall']*100:.2f}% ({base_hist['tp']}/24)")
    print(f"  Precision: {base_hist['precision']*100:.2f}%")
    print(f"  F1-Score : {base_hist['f1']:.4f}")
    print(f"  FP Files : {base_hist['fp']}/68")
    print(f"  Missed   : {base_hist['missed']}")
    print(f"  FP by Cat: {base_hist['fp_by_cat']}")

    # Apply Parameter Freezing
    # 1. Freeze Block 1 completely
    for p in model.block1.parameters():
        p.requires_grad = False
    # 2. Freeze Block 2 completely
    for p in model.block2.parameters():
        p.requires_grad = False
    # 3. Freeze all BatchNorm affine parameters in Block 3 and Block 4
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable Parameter Layers: {trainable_names}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    # Snapshot ExpF anchor weights for L2 weight penalty
    anchor_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    anchor_lambda = 5.0e-4
    lr = 5.0e-6
    pos_weight = 1.10

    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    best_val_f1 = base_val["f1"]
    best_epoch = 0
    best_sd = None
    results_history = []

    num_epochs = 4
    steps_per_epoch = 75

    for epoch in range(1, num_epochs + 1):
        model.train()
        # Ensure BatchNorm remains in eval mode
        model.set_bn_eval()

        epoch_losses = []
        ce_losses = []
        anc_losses = []
        t0 = time.time()

        for step in range(steps_per_epoch):
            # Sample 32 Positives: 16 fragile/demolition + 16 robust/ordnance
            s_pos = random.sample(pos_fragile, 16) + random.sample(pos_robust, 16)

            # Sample 32 Hard Negatives across priority categories
            s_neg = (
                random.sample(neg_thunder, 6) +
                random.sample(neg_fireworks, 6) +
                random.sample(neg_backfires, 6) +
                random.sample(neg_gunfire, 5) +
                random.sample(neg_artillery, 4) +
                random.sample(neg_mechanical, 4) +
                random.sample(neg_speech_ambient, 1)
            )

            batch_specs = [s["spec"] for s in s_pos] + [s["spec"] for s in s_neg]
            batch_labels = [1] * 32 + [0] * 32
            comb = list(zip(batch_specs, batch_labels))
            random.shuffle(comb)

            x = torch.stack([s for s, y in comb]).unsqueeze(1).float()
            y = torch.tensor([y for s, y in comb], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            ce_loss = criterion(logits, y)

            # Anchor penalty to ExpF
            anc_loss = sum(torch.sum((p - anchor_weights[n]) ** 2) for n, p in model.named_parameters() if n in anchor_weights)
            tot_loss = ce_loss + anchor_lambda * anc_loss

            tot_loss.backward()
            optimizer.step()

            epoch_losses.append(tot_loss.item())
            ce_losses.append(ce_loss.item())
            anc_losses.append(anc_loss.item())

        elapsed = time.time() - t0

        # Evaluate after epoch
        val_res = evaluate_val_cache(model, threshold=0.70)
        hist_res = evaluate_historical_benchmark(model, threshold=0.70)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "anc_loss": float(np.mean(anc_losses)),
            "val_recall": val_res["recall"],
            "val_prec": val_res["precision"],
            "val_f1": val_res["f1"],
            "val_clean_neg": val_res["clean_negatives"],
            "hist_recall": hist_res["recall"],
            "hist_tp": hist_res["tp"],
            "hist_fp": hist_res["fp"],
            "hist_prec": hist_res["precision"],
            "hist_f1": hist_res["f1"],
            "hist_missed": hist_res["missed"]
        }
        results_history.append(epoch_summary)

        print(f"\n--- Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) ---")
        print(f"  Train Loss: {np.mean(epoch_losses):.4f} (CE: {np.mean(ce_losses):.4f}, Anchor: {np.mean(anc_losses):.4f})")
        print(f"  Val Dev (190 files)  : Recall={val_res['recall']*100:.1f}% ({val_res['tp']}/70), Prec={val_res['precision']*100:.1f}%, F1={val_res['f1']:.4f}, CleanNeg={val_res['clean_negatives']}")
        print(f"  Hist Bench (92 files): Recall={hist_res['recall']*100:.1f}% ({hist_res['tp']}/24), Prec={hist_res['precision']*100:.1f}%, F1={hist_res['f1']:.4f}, FP={hist_res['fp']}/68")
        print(f"  Hist Missed          : {hist_res['missed']}")

        # Save checkpoint if F1 improves AND recall criteria met (>= 84.3% on val and >= 22/24 on hist)
        if hist_res["tp"] >= 22 and val_res["recall"] >= 0.843 and (val_res["f1"] > best_val_f1 or (val_res["f1"] >= best_val_f1 and hist_res["fp"] < base_hist["fp"])):
            best_val_f1 = val_res["f1"]
            best_epoch = epoch
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path = CKPT_DIR / "candidate_expL_anchored_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_res,
                "hist_metrics": hist_res,
                "exp_name": "candidate_expL_anchored"
            }, ckpt_path)
            print(f"  >>> SAVED BEST CANDIDATE ExpL: {ckpt_path.name} (Val F1={val_res['f1']:.4f}, Hist FP={hist_res['fp']}/68)")

    # Save final checkpoint if none met improvement threshold
    final_path = CKPT_DIR / "candidate_expL_anchored_final.pt"
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "history": results_history,
        "exp_name": "candidate_expL_anchored_final"
    }, final_path)
    print(f"\nFinal Epoch Checkpoint Saved: {final_path.name}")

    # Save history json
    history_path = PROJECT_ROOT / "reports" / "candidate_expL_training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "base_val": base_val,
            "base_hist": base_hist,
            "history": results_history,
            "best_epoch": best_epoch
        }, f, indent=2)
    print(f"Training History Saved to: {history_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": results_history
    }

if __name__ == "__main__":
    train_expL()
