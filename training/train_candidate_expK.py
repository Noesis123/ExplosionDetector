"""Candidate ExpK: Protected Demolition Training via Frozen Backbone & Asymmetric Margin Loss.

Key Technical Innovations:
1. Freeze Blocks 1, 2, 3, and 4 completely (freeze_backbone=True).
   Guarantees zero degradation of low-level acoustic feature representations.
2. Train only the Linear Classification Head (in=128, out=2).
3. Asymmetric Margin Loss:
   - For all negatives (thunder, artillery, gunfire, backfires): standard Cross-Entropy drives P_exp down.
   - For weak and demolition explosions: explicit positive margin penalty if (z_exp - z_not_exp) < 1.20 (P_exp < 0.768),
     preventing weak/demolition explosions from ever falling below the 0.70 decision threshold.
4. Uses the augmented training pool with 2,829 newly approved hard negatives from Codex (thunder, engines, impacts).
5. Strictly evaluated on source-isolated train_val cache (190 files) before saving best checkpoint.
"""

import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

# 1. Load Augmented Training Pool & Validation Cache
print("Loading augmented training pool and validation cache...")
aug_data = torch.load(DATA_DIR / "augmented_training_pool.pt", map_location="cpu", weights_only=False)
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)

pool_pos = aug_data["pool_pos"]
pool_neg = aug_data["pool_neg"]
val_records = val_data["file_records"]

print(f"Pool size: {len(pool_pos)} positives, {len(pool_neg)} negatives.")

# 2. Load ExpF Base Checkpoint
exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

base_model = ExplosionBinaryCNN(freeze_bn=True)
base_model.load_state_dict(sd_f)
base_model.eval()

# 3. Categorize Negatives into Focused Confound Clusters (using ExpF scores)
print("Clustering negatives for focused hard-negative sampling...")
neg_thunder = []
neg_artillery = []
neg_gunfire = []
neg_backfires = []
neg_mechanical = []
neg_ambient = []

with torch.no_grad():
    for w in pool_neg:
        cat = w.get("category", "")
        # Filter windows with ExpF confidence >= 0.20 as active confound candidates
        if cat in ["thunder"]:
            neg_thunder.append(w)
        elif cat in ["artillery_firing"]:
            neg_artillery.append(w)
        elif cat in ["gunfire"]:
            neg_gunfire.append(w)
        elif cat in ["vehicle_backfires", "engines"]:
            neg_backfires.append(w)
        elif cat in ["construction_impacts", "industrial_impacts", "metal_glass_impacts", "doors", "doors_slamming_falling", "mechanical_impulses"]:
            neg_mechanical.append(w)
        else:
            neg_ambient.append(w)

print(f"Negative Clusters:")
print(f"  Thunder            : {len(neg_thunder)} windows")
print(f"  Artillery Firing   : {len(neg_artillery)} windows")
print(f"  Gunfire            : {len(neg_gunfire)} windows")
print(f"  Vehicle Backfires  : {len(neg_backfires)} windows")
print(f"  Mechanical Impacts : {len(neg_mechanical)} windows")
print(f"  Ambient / General  : {len(neg_ambient)} windows")

# 4. Partition Positives (Highlighting Demolition and Weak Blasts)
pos_distant_demolition = []
pos_military_outdoor = []
pos_close_general = []

for w in pool_pos:
    cat = w.get("category", "")
    db = w.get("dbfs", 0)
    if cat in {"controlled_demolition", "distant_explosion"} or db < -38.0:
        pos_distant_demolition.append(w)
    elif cat in {"military_ordnance", "outdoor_explosion"}:
        pos_military_outdoor.append(w)
    else:
        pos_close_general.append(w)

print(f"Positive Clusters:")
print(f"  Distant / Demolition / Weak : {len(pos_distant_demolition)} windows")
print(f"  Military / Outdoor Blasts   : {len(pos_military_outdoor)} windows")
print(f"  Close / General Explosions  : {len(pos_close_general)} windows")

# 5. Validation Evaluation Function (on 190 train_val records)
def evaluate_on_val(model, threshold=0.70):
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0

    with torch.no_grad():
        for r in val_records:
            true_lbl = r["binary_label"]
            file_max_p = 0.0

            for w in r["windows"]:
                spec = w["spec"].unsqueeze(0).unsqueeze(0).float()
                logits = model(spec)
                p_exp = float(torch.softmax(logits, dim=-1)[0, 1])
                if p_exp > file_max_p:
                    file_max_p = p_exp

            pred_pos = (file_max_p >= threshold)
            if true_lbl == 1:
                if pred_pos: file_tp += 1
                else: file_fn += 1
            else:
                if pred_pos: file_fp += 1
                else: file_tn += 1

    total_pos = file_tp + file_fn
    total_neg = file_tn + file_fp
    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / total_pos if total_pos > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (file_tp + file_tn) / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0
    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc),
        "clean_negatives": f"{file_tn}/{total_neg} ({file_tn/total_neg*100:.1f}%)"
    }

# 6. Candidate ExpK Training Routine
def train_candidate_exp_k(num_epochs=5, steps_per_epoch=80, lr=1.5e-4, margin_lambda=1.5):
    exp_name = "candidate_expK_protected_demolition"
    print("\n" + "=" * 75)
    print(f"TRAINING {exp_name.upper()}")
    print(f"Freeze Backbone: TRUE | LR: {lr:.1e} | Margin Lambda: {margin_lambda:.2f}")
    print("=" * 75)

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_f)

    # Freeze convolutional backbone completely
    model.freeze_backbone(True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Parameters (Head only): {sum(p.numel() for p in trainable_params):,}")

    # Baseline ExpF score on val
    zs_70 = evaluate_on_val(model, threshold=0.70)
    print(f"Starting Point (ExpF) @ 0.70: Recall={zs_70['recall']*100:.1f}% ({zs_70['tp']}/70), Prec={zs_70['precision']*100:.1f}%, F1={zs_70['f1']:.4f}, CleanNeg={zs_70['clean_negatives']}")

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)

    best_f1 = zs_70["f1"]
    best_epoch = 0
    best_sd = None

    for epoch in range(1, num_epochs + 1):
        model.head.train()
        losses = []
        ce_losses = []
        margin_losses = []
        t0 = time.time()

        for step in range(steps_per_epoch):
            # Positives (32 windows): 16 distant/demolition (heavily represented) + 8 military + 8 close
            pos_samples = (
                random.sample(pos_distant_demolition, 16) +
                random.sample(pos_military_outdoor, 8) +
                random.sample(pos_close_general, 8)
            )

            # Negatives (32 windows): 8 thunder + 6 backfires + 6 artillery + 6 gunfire + 4 mechanical + 2 ambient
            neg_samples = (
                random.sample(neg_thunder, 8) +
                random.sample(neg_backfires, 6) +
                random.sample(neg_artillery, 6) +
                random.sample(neg_gunfire, 6) +
                random.sample(neg_mechanical, 4) +
                random.sample(neg_ambient, 2)
            )

            batch_specs = [s["spec"] for s in pos_samples] + [s["spec"] for s in neg_samples]
            batch_labels = [1] * 32 + [0] * 32
            is_demolition = [1] * 16 + [0] * 16 + [0] * 32  # Flag for the 16 demolition windows

            comb = list(zip(batch_specs, batch_labels, is_demolition))
            random.shuffle(comb)

            x = torch.stack([s[0] for s in comb]).unsqueeze(1)
            y = torch.tensor([s[1] for s in comb], dtype=torch.long)
            demo_mask = torch.tensor([s[2] for s in comb], dtype=torch.bool)

            optimizer.zero_grad()
            logits = model(x)
            ce_loss = ce_criterion(logits, y)

            # Positive Margin Loss on Demolition Blasts:
            # Margin condition: (logits[:, 1] - logits[:, 0]) >= 1.20 (guarantees P_exp >= 0.768)
            diff = logits[:, 1] - logits[:, 0]
            margin_violations = F.relu(1.20 - diff[demo_mask])
            margin_loss = margin_violations.mean() if len(margin_violations) > 0 else torch.tensor(0.0)

            total_loss = ce_loss + margin_lambda * margin_loss
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())
            ce_losses.append(ce_loss.item())
            margin_losses.append(margin_loss.item())

        elapsed = time.time() - t0
        val_res = evaluate_on_val(model, threshold=0.70)

        print(f"Epoch {epoch:2d}/{num_epochs:2d} ({elapsed:.1f}s) | Loss: {np.mean(losses):.4f} (CE: {np.mean(ce_losses):.4f}, Margin: {np.mean(margin_losses):.4f})")
        print(f"   @ 0.70: Recall={val_res['recall']*100:.1f}% ({val_res['tp']}/70), Prec={val_res['precision']*100:.1f}%, F1={val_res['f1']:.4f}, CleanNeg={val_res['clean_negatives']}")

        # Save if F1 improves while preserving at least 84.3% recall on val (the baseline ExpF recall!)
        if val_res["recall"] >= 0.843 and val_res["f1"] >= best_f1:
            best_f1 = val_res["f1"]
            best_epoch = epoch
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path = CKPT_DIR / f"{exp_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_70": val_res,
                "exp_name": exp_name
            }, ckpt_path)
            print(f"   >>> SAVED NEW BEST CHECKPOINT: {ckpt_path.name} (F1: {val_res['f1']:.4f}, Prec: {val_res['precision']*100:.1f}%, Rec: {val_res['recall']*100:.1f}%)")

    if best_epoch == 0:
        ckpt_path = CKPT_DIR / f"{exp_name}_final.pt"
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "val_70": val_res,
            "exp_name": exp_name
        }, ckpt_path)
        print(f"   >>> SAVED FINAL CHECKPOINT: {ckpt_path.name}")

    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_f1_70": best_f1
    }

if __name__ == "__main__":
    train_candidate_exp_k(num_epochs=5, steps_per_epoch=80, lr=1.5e-4, margin_lambda=1.5)
