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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

fit_data = torch.load(DATA_DIR / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
val_data = torch.load(DATA_DIR / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)

pool_pos = fit_data["pool_pos"]
pool_neg = fit_data["pool_neg"]
val_records = val_data["file_records"]

exp_f_path = CKPT_DIR / "candidate_expF_accuracy_optimized_best.pt"
ckpt_f = torch.load(exp_f_path, map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f

base_model = ExplosionBinaryCNN(freeze_bn=True)
base_model.load_state_dict(sd_f)
base_model.eval()

# Mine hard negatives
neg_artillery = []
neg_gunfire = []
neg_backfires = []
neg_thunder = []
neg_other = []

with torch.no_grad():
    for w in pool_neg:
        spec = w["spec"].unsqueeze(0).unsqueeze(0).float()
        p = float(torch.softmax(base_model(spec), dim=-1)[0, 1])
        w_tagged = dict(w)
        w_tagged["p_expF"] = p
        cat = w.get("category", "")
        if cat == "artillery_firing" and p >= 0.25:
            neg_artillery.append(w_tagged)
        elif cat == "gunfire" and p >= 0.25:
            neg_gunfire.append(w_tagged)
        elif cat == "vehicle_backfires" and p >= 0.25:
            neg_backfires.append(w_tagged)
        elif cat == "thunder" and p >= 0.25:
            neg_thunder.append(w_tagged)
        else:
            neg_other.append(w_tagged)

pos_distant = []
pos_other = []
for w in pool_pos:
    cat = w.get("category", "")
    db = w.get("dbfs", 0)
    if cat in {"controlled_demolition", "distant_explosion"} or db < -38.0:
        pos_distant.append(w)
    else:
        pos_other.append(w)

def evaluate_on_val(model, threshold=0.70):
    model.eval()
    file_tp = 0
    file_fp = 0
    file_tn = 0
    file_fn = 0
    with torch.no_grad():
        for r in val_records:
            true_lbl = r["binary_label"]
            file_max_p = max(float(torch.softmax(model(w["spec"].unsqueeze(0).unsqueeze(0).float()), dim=-1)[0, 1]) for w in r["windows"])
            if true_lbl == 1:
                if file_max_p >= threshold: file_tp += 1
                else: file_fn += 1
            else:
                if file_max_p >= threshold: file_fp += 1
                else: file_tn += 1
    prec = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0.0
    rec = file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": file_tp, "fp": file_fp, "tn": file_tn, "fn": file_fn, "precision": prec, "recall": rec, "f1": f1}

def train_exp_h():
    print("Training Candidate ExpH (Strong Anchor, High Recall Protection)...")
    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd_f)

    anchor_weights = {n: p.clone().detach() for n, p in model.named_parameters() if "block" in n}
    anchor_lambda = 3.5e-4
    pos_weight = 1.20
    lr = 8.0e-6

    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = 0.0
    best_sd = None

    for epoch in range(1, 4):
        model.train()
        losses = []
        for step in range(60):
            pos_samples = random.sample(pos_distant, 16) + random.sample(pos_other, 16)
            neg_samples = (
                random.sample(neg_artillery, 6) +
                random.sample(neg_gunfire, 6) +
                random.sample(neg_backfires, 6) +
                random.sample(neg_thunder, 6) +
                random.sample(neg_other, 8)
            )

            batch_specs = [s["spec"] for s in pos_samples] + [s["spec"] for s in neg_samples]
            batch_labels = [1] * 32 + [0] * 32
            comb = list(zip(batch_specs, batch_labels))
            random.shuffle(comb)

            x = torch.stack([s for s, y in comb]).unsqueeze(1)
            y = torch.tensor([y for s, y in comb], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            ce_loss = criterion(logits, y)
            anc_loss = sum(torch.sum((p - anchor_weights[n]) ** 2) for n, p in model.named_parameters() if n in anchor_weights)
            loss = ce_loss + anchor_lambda * anc_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_res = evaluate_on_val(model, threshold=0.70)
        print(f"Epoch {epoch}: Loss={np.mean(losses):.4f} | Val Recall={val_res['recall']*100:.1f}% ({val_res['tp']}/70), Prec={val_res['precision']*100:.1f}%, F1={val_res['f1']:.4f}")
        if val_res["recall"] >= 0.843 and val_res["f1"] >= best_f1:
            best_f1 = val_res["f1"]
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    if best_sd:
        torch.save(best_sd, CKPT_DIR / "candidate_expH_recall_anchored_best.pt")
        print("Saved candidate_expH_recall_anchored_best.pt")
    else:
        torch.save(model.state_dict(), CKPT_DIR / "candidate_expH_recall_anchored_best.pt")
        print("Saved candidate_expH_recall_anchored_best.pt (final)")

train_exp_h()
