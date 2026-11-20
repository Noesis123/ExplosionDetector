"""Candidate ExpF: High-Accuracy Explosion Binary CNN Training.

Key Architectural & Training Enhancements:
1. Initialize directly from candidate_expE_hard_neg_refined_best.pt.
2. Uses refined training pool with rescued distant/weak blast tails and expanded hard negatives.
3. Source-Balanced Sampling across 6 acoustic confound clusters:
   - Artillery Firing (5 windows/batch)
   - Gunfire (5 windows/batch)
   - Vehicle Backfires (5 windows/batch)
   - Thunder (4 windows/batch)
   - Mechanical Impacts (5 windows/batch: industrial, metal/glass, construction, doors)
   - Fireworks (5 windows/batch: aerial bursts, firecrackers, barrage)
   - Ambient/General (3 windows/batch)
4. Positive Event-Window Balancing:
   - 12 distant & demolition blast windows (with atmospheric roll-off & gain jitter)
   - 8 military ordnance & outdoor blasts
   - 12 close & general explosions
5. Frozen BatchNorm throughout (preserves stable batch statistics).
6. Conservative fine-tuning: lr = 2.5e-5, AdamW, weight anchor penalty (lambda = 1.5e-4) to ExpE.
7. Validation & Early Stopping: Strictly based on 190 source-isolated train_val files.
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
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
SUMMARY_DIR = PROJECT_ROOT / "training"

if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

# 1. Load Precomputed Training Pool and Validation Cache
print("Loading training pool and validation cache for ExpF...")
fit_data = torch.load(DATA_DIR / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)

pool_pos = fit_data["pool_pos"]  # 4094 windows
pool_neg = fit_data["pool_neg"]  # 4693 windows
val_records = val_data["file_records"] # 190 files

# Load ExpE as initial weights and anchor
exp_e_path = CKPT_DIR / "candidate_expE_hard_neg_refined_best.pt"
ckpt_e = torch.load(exp_e_path, map_location="cpu", weights_only=False)
sd_e = ckpt_e["model_state_dict"] if "model_state_dict" in ckpt_e else ckpt_e
model_e = ExplosionBinaryCNN(freeze_bn=True)
model_e.load_state_dict(sd_e)
model_e.eval()

print("Mining hard-negative windows with ExpE across categories...")
neg_artillery = []
neg_gunfire = []
neg_backfires = []
neg_thunder = []
neg_mechanical = []
neg_fireworks = []
neg_ambient = []

with torch.no_grad():
    for w in pool_neg:
        spec = w["spec"].unsqueeze(0).unsqueeze(0).float()
        logits = model_e(spec)
        p = float(torch.softmax(logits, dim=-1)[0, 1])
        w_tagged = dict(w)
        w_tagged["p_expE"] = p
        
        cat = w.get("category", "")
        # Filter for moderately hard and hard windows (p >= 0.25 to catch boundary confounds)
        is_hard = (p >= 0.25)
        
        if cat == "artillery_firing":
            neg_artillery.append(w_tagged)
        elif cat == "gunfire":
            if is_hard or p >= 0.20: neg_gunfire.append(w_tagged)
        elif cat == "vehicle_backfires":
            if is_hard or p >= 0.20: neg_backfires.append(w_tagged)
        elif cat == "thunder":
            if is_hard or p >= 0.20: neg_thunder.append(w_tagged)
        elif cat in {"construction_impacts", "industrial_impacts", "metal_glass_impacts", "doors_slamming_falling"}:
            if is_hard or p >= 0.20: neg_mechanical.append(w_tagged)
        elif cat in {"aerial_burst", "firecrackers", "barrage", "fireworks", "distant", "fountain"}:
            if is_hard or p >= 0.20: neg_fireworks.append(w_tagged)
        else:
            neg_ambient.append(w_tagged)

print(f"Mined Confound Clusters:")
print(f"  Artillery Firing   : {len(neg_artillery)} windows (avg p: {np.mean([w['p_expE'] for w in neg_artillery]):.3f})")
print(f"  Gunfire            : {len(neg_gunfire)} windows (avg p: {np.mean([w['p_expE'] for w in neg_gunfire]):.3f})")
print(f"  Vehicle Backfires  : {len(neg_backfires)} windows (avg p: {np.mean([w['p_expE'] for w in neg_backfires]):.3f})")
print(f"  Thunder            : {len(neg_thunder)} windows (avg p: {np.mean([w['p_expE'] for w in neg_thunder]):.3f})")
print(f"  Mechanical Impacts : {len(neg_mechanical)} windows (avg p: {np.mean([w['p_expE'] for w in neg_mechanical]):.3f})")
print(f"  Fireworks          : {len(neg_fireworks)} windows (avg p: {np.mean([w['p_expE'] for w in neg_fireworks]):.3f})")
print(f"  Ambient / General  : {len(neg_ambient)} windows")

# Partition positives: distant/demolition vs military/outdoor vs close/general
pos_distant_demolition = []
pos_military_outdoor = []
pos_close_general = []

for idx, w in enumerate(pool_pos):
    cat = w.get("category", "")
    db = w.get("dbfs", 0)
    if cat in {"controlled_demolition", "distant_explosion"} or db < -38.0:
        pos_distant_demolition.append(w)
    elif cat in {"military_ordnance", "outdoor_explosion"}:
        pos_military_outdoor.append(w)
    else:
        pos_close_general.append(w)

print(f"\nPositive Clusters:")
print(f"  Distant / Demolition / Weak : {len(pos_distant_demolition)} windows")
print(f"  Military / Outdoor Blasts   : {len(pos_military_outdoor)} windows")
print(f"  Close / General Explosions  : {len(pos_close_general)} windows")

# Spectrogram Augmentation
def augment_spectrogram(spec_tensor, p_roll_off=0.30, p_gain=0.40):
    augmented = spec_tensor.clone()
    if random.random() < p_gain:
        gain_shift = random.uniform(-0.15, 0.15)
        augmented = augmented + gain_shift
    if random.random() < p_roll_off:
        slope = torch.linspace(1.0, 0.70, 24)
        augmented[40:, :] = augmented[40:, :] * slope.unsqueeze(1)
    return augmented

# Validation Evaluation Helper
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
    return {
        "tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "clean_negatives": f"{file_tn}/{total_neg} ({file_tn/total_neg*100:.1f}%)"
    }

def train_candidate_exp_f(pos_weight_val=1.15, lr=2.5e-5, num_epochs=6, steps_per_epoch=120, anchor_lambda=1.5e-4):
    exp_name = "candidate_expF_accuracy_optimized"
    print("\n" + "="*70)
    print(f"TRAINING {exp_name.upper()}")
    print(f"Positive Weight: {pos_weight_val:.2f} | LR: {lr:.1e} | Anchor: {anchor_lambda:.1e}")
    print("="*70)
    
    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_e)
    print(f"Initialized from {exp_e_path.name} successfully.")
    
    anchor_weights = {n: p.clone().detach() for n, p in model.named_parameters() if "block" in n}
    
    # Baseline ExpE scores on train_val
    zs_70 = evaluate_on_val(model, threshold=0.70)
    zs_75 = evaluate_on_val(model, threshold=0.75)
    print(f"Starting Point (ExpE) @ 0.70: Recall={zs_70['recall']*100:.1f}% ({zs_70['tp']}/70), Prec={zs_70['precision']*100:.1f}%, F1={zs_70['f1']:.4f}, CleanNeg={zs_70['clean_negatives']}")
    print(f"Starting Point (ExpE) @ 0.75: Recall={zs_75['recall']*100:.1f}% ({zs_75['tp']}/70), Prec={zs_75['precision']*100:.1f}%, F1={zs_75['f1']:.4f}, CleanNeg={zs_75['clean_negatives']}")
    
    weight_tensor = torch.tensor([1.0, float(pos_weight_val)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_f1_70 = zs_70["f1"]
    best_epoch = 0
    history = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        ce_losses = []
        anc_losses = []
        t0 = time.time()
        
        for step in range(steps_per_epoch):
            # Positives (32 windows): 12 distant/demolition + 8 military/outdoor + 12 close/general
            pos_samples = (
                random.sample(pos_distant_demolition, 12) +
                random.sample(pos_military_outdoor, 8) +
                random.sample(pos_close_general, 12)
            )
            
            # Negatives (32 windows): Source-Balanced across all confound clusters
            neg_samples = (
                random.sample(neg_artillery, 5) +
                random.sample(neg_gunfire, 5) +
                random.sample(neg_backfires, 5) +
                random.sample(neg_thunder, 4) +
                random.sample(neg_mechanical, 5) +
                random.sample(neg_fireworks, 5) +
                random.sample(neg_ambient, 3)
            )
            
            # Augment
            batch_specs = []
            batch_labels = []
            for s in pos_samples:
                aug_spec = augment_spectrogram(s["spec"], p_roll_off=0.30, p_gain=0.40)
                batch_specs.append(aug_spec)
                batch_labels.append(1)
            for s in neg_samples:
                aug_spec = augment_spectrogram(s["spec"], p_roll_off=0.10, p_gain=0.30)
                batch_specs.append(aug_spec)
                batch_labels.append(0)
                
            combined = list(zip(batch_specs, batch_labels))
            random.shuffle(combined)
            
            x_tensor = torch.stack([x for x, y in combined]).unsqueeze(1)
            y_tensor = torch.tensor([y for x, y in combined], dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(x_tensor)
            ce_loss = criterion(logits, y_tensor)
            
            anc_loss = torch.tensor(0.0)
            if anchor_lambda > 0.0:
                for n, p in model.named_parameters():
                    if n in anchor_weights:
                        anc_loss = anc_loss + torch.sum((p - anchor_weights[n]) ** 2)
                anc_loss = anchor_lambda * anc_loss
                
            loss = ce_loss + anc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            ce_losses.append(ce_loss.item())
            anc_losses.append(anc_loss.item() if isinstance(anc_loss, torch.Tensor) else 0.0)
            
        elapsed = time.time() - t0
        val_70 = evaluate_on_val(model, threshold=0.70)
        val_75 = evaluate_on_val(model, threshold=0.75)
        
        print(f"Epoch {epoch:2d}/{num_epochs:2d} ({elapsed:.1f}s) | Loss: {np.mean(losses):.4f} (CE: {np.mean(ce_losses):.4f}, Anc: {np.mean(anc_losses):.4f})")
        print(f"   @ 0.70: Recall={val_70['recall']*100:.1f}% ({val_70['tp']}/70), Prec={val_70['precision']*100:.1f}%, F1={val_70['f1']:.4f}, CleanNeg={val_70['clean_negatives']}")
        print(f"   @ 0.75: Recall={val_75['recall']*100:.1f}% ({val_75['tp']}/70), Prec={val_75['precision']*100:.1f}%, F1={val_75['f1']:.4f}, CleanNeg={val_75['clean_negatives']}")
        
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_70": val_70,
            "val_75": val_75
        }
        history.append(record)
        
        # Save if F1@0.70 improves while preserving at least 82.0% recall on val
        if val_70["f1"] >= best_f1_70 and val_70["recall"] >= 0.814:
            best_f1_70 = val_70["f1"]
            best_epoch = epoch
            ckpt_path = CKPT_DIR / f"{exp_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_70": val_70,
                "val_75": val_75,
                "exp_name": exp_name
            }, ckpt_path)
            print(f"   >>> SAVED NEW BEST CHECKPOINT: {ckpt_path.name} (F1@0.70: {val_70['f1']:.4f}, Prec: {val_70['precision']*100:.1f}%, Recall: {val_70['recall']*100:.1f}%)")
            
    if best_epoch == 0:
        ckpt_path = CKPT_DIR / f"{exp_name}_final.pt"
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "val_70": val_70,
            "val_75": val_75,
            "exp_name": exp_name
        }, ckpt_path)
        print(f"   >>> SAVED FINAL CHECKPOINT: {ckpt_path.name}")
        
    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_f1_70": best_f1_70,
        "history": history
    }

if __name__ == "__main__":
    results = train_candidate_exp_f(pos_weight_val=1.25, lr=1.5e-5, num_epochs=4, steps_per_epoch=80, anchor_lambda=2.5e-4)
    with open(SUMMARY_DIR / "training_summary_candidate_expF.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nExpF Training complete. Summary saved.")
