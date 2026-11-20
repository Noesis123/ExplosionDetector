"""Prepare Audited Binary Datasets for Explosion Binary CNN.

Enforces:
1. Approved training sources only (training_approved == True, 824 files).
2. Quarantined / conflict files excluded.
3. Source-group isolated 80/20 partition (0% leakage between train_fit and train_val).
4. Zero leakage into frozen 92-recording evaluation suite.
5. Strict Explosion blast-window energy filtering (purging background-only windows).
6. Preserving hard-negative diversity (gunfire, artillery, backfires, machinery, thunder, impacts, fireworks).
7. Full per-window provenance recording.
"""

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
DATA_DIR = EXP_DIR / "data"

if str(LAB_DIR) not in sys.path: sys.path.insert(0, str(LAB_DIR))
if str(PROD_DIR) not in sys.path: sys.path.insert(0, str(PROD_DIR))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
BLAST_FILTER_OFFSET_DB = 18.0
BLAST_MIN_DBFS = -40.0
MAX_WINDOWS_PER_NEGATIVE_FILE = 12

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

manifest_df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

# 1. Enforce approved training recordings
raw_train = manifest_df[manifest_df["split"] == "train"]
approved_train = manifest_df[(manifest_df["split"] == "train") & (manifest_df["training_approved"] == True)].copy()
quarantined_train = manifest_df[(manifest_df["split"] == "train") & (manifest_df["training_approved"] == False)].copy()

print(f"Total raw train recordings in manifest: {len(raw_train)}")
print(f"Quarantined / Unapproved excluded: {len(quarantined_train)}")
for idx, r in quarantined_train.iterrows():
    print(f"  EXCLUDED: {r['label']} | {r['file_path']} | {r['review_status']}")

print(f"\nApproved train recordings: {len(approved_train)}")
print(approved_train["label"].value_counts())

assert len(approved_train) == 824, f"Expected 824 approved files, got {len(approved_train)}"
assert len(quarantined_train) == 4, f"Expected 4 quarantined files, got {len(quarantined_train)}"

# 2. Group-aware 80/20 partition
np.random.seed(42)
group_labels = approved_train.groupby("group_id")["label"].first()

fit_groups = []
val_groups = []

for lbl in ["explosion", "fireworks", "non_target"]:
    lbl_groups = group_labels[group_labels == lbl].index.tolist()
    np.random.shuffle(lbl_groups)
    n_val = max(1, int(len(lbl_groups) * 0.20))
    val_groups.extend(lbl_groups[:n_val])
    fit_groups.extend(lbl_groups[n_val:])

fit_df = approved_train[approved_train["group_id"].isin(fit_groups)].copy()
val_df = approved_train[approved_train["group_id"].isin(val_groups)].copy()

# Integrity checks
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])]
eval_files = set(eval_df["file_path"])
eval_groups = set(eval_df["group_id"].dropna())

assert len(set(fit_df["group_id"]).intersection(set(val_df["group_id"]))) == 0, "Leakage between fit and val groups!"
assert len(set(fit_df["file_path"]).intersection(eval_files)) == 0, "Leakage into eval files from fit!"
assert len(set(val_df["file_path"]).intersection(eval_files)) == 0, "Leakage into eval files from val!"
assert len(set(fit_df["group_id"]).intersection(eval_groups)) == 0, "Leakage into eval groups from fit!"
assert len(set(val_df["group_id"]).intersection(eval_groups)) == 0, "Leakage into eval groups from val!"

print(f"\nPartitioning Verified:")
print(f"  train_fit: {len(fit_df)} recordings across {len(fit_groups)} groups")
print(f"    {fit_df['label'].value_counts().to_dict()}")
print(f"  train_val: {len(val_df)} recordings across {len(val_groups)} groups")
print(f"    {val_df['label'].value_counts().to_dict()}")

# Helper to process a split and extract windows
def extract_split_windows(df_subset, split_name):
    print(f"\nProcessing {split_name} ({len(df_subset)} files)...")
    t0 = time.time()
    
    records = []
    pool_pos = []
    pool_neg = []
    
    file_stats = {
        "total_files": len(df_subset),
        "explosion_files": 0,
        "fireworks_files": 0,
        "nontarget_files": 0,
        "total_active_windows": 0,
        "pos_windows": 0,
        "neg_windows": 0,
        "neg_fireworks_windows": 0,
        "neg_nontarget_windows": 0,
        "quarantined_background_windows_purged": 0
    }
    
    for idx, r in df_subset.iterrows():
        p = LAB_DIR / r["file_path"]
        if not p.exists(): p = PROD_DIR / r["file_path"]
        
        lbl = r["label"]
        cat = r.get("category", "")
        gid = r["group_id"]
        fname = Path(r["file_path"]).name
        
        if lbl == "explosion": file_stats["explosion_files"] += 1
        elif lbl == "fireworks": file_stats["fireworks_files"] += 1
        else: file_stats["nontarget_files"] += 1
        
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        res = preprocessor.resample(mono, sr)
        n_w = max(1, (len(res) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
        
        # Pass 1: compute RMS dbfs for all windows
        raw_windows = []
        dbfs_list = []
        peaks = []
        
        for i in range(n_w):
            s_i = i * HOP_SAMPLES
            w = res[s_i : s_i + WINDOW_SAMPLES]
            if len(w) < WINDOW_SAMPLES:
                w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
            rms = float(np.sqrt(np.mean(w ** 2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            raw_windows.append(w)
            dbfs_list.append(dbfs)
            peaks.append(peak)
            
        max_dbfs = max(dbfs_list) if dbfs_list else -100.0
        
        # Pass 2: Filter and assign valid windows
        file_specs = []
        file_window_records = []
        
        if lbl == "explosion":
            # Explosion Positive Window Selection
            blast_threshold = max(max_dbfs - BLAST_FILTER_OFFSET_DB, BLAST_MIN_DBFS)
            for i, (w, db, pk) in enumerate(zip(raw_windows, dbfs_list, peaks)):
                if db < SQUELCH_DBFS:
                    continue
                if db < blast_threshold:
                    # Low-energy background before/after blast in explosion file -> purged from positive pool
                    file_stats["quarantined_background_windows_purged"] += 1
                    continue
                    
                spec = preprocessor.compute_log_mel_spectrogram(w)
                win_data = {
                    "spec": torch.from_numpy(spec).float(),
                    "dbfs": db,
                    "peak": pk,
                    "t_sec": i * 0.25,
                    "binary_label": 1,
                    "original_label": "explosion",
                    "category": cat,
                    "group_id": gid,
                    "filename": fname
                }
                pool_pos.append(win_data)
                file_window_records.append(win_data)
                file_stats["pos_windows"] += 1
        else:
            # Negative Window Selection (Fireworks or Non-target)
            active_indices = [i for i, db in enumerate(dbfs_list) if db >= SQUELCH_DBFS]
            if not active_indices:
                # If everything below squelch, take top window
                active_indices = [int(np.argmax(dbfs_list))]
                
            # If long file, select evenly spaced energetic windows up to max limit
            if len(active_indices) > MAX_WINDOWS_PER_NEGATIVE_FILE:
                step = len(active_indices) / float(MAX_WINDOWS_PER_NEGATIVE_FILE)
                selected_indices = [active_indices[int(i * step)] for i in range(MAX_WINDOWS_PER_NEGATIVE_FILE)]
            else:
                selected_indices = active_indices
                
            for i in selected_indices:
                w = raw_windows[i]
                db = dbfs_list[i]
                pk = peaks[i]
                spec = preprocessor.compute_log_mel_spectrogram(w)
                win_data = {
                    "spec": torch.from_numpy(spec).float(),
                    "dbfs": db,
                    "peak": pk,
                    "t_sec": i * 0.25,
                    "binary_label": 0,
                    "original_label": lbl,
                    "category": cat,
                    "group_id": gid,
                    "filename": fname
                }
                pool_neg.append(win_data)
                file_window_records.append(win_data)
                file_stats["neg_windows"] += 1
                if lbl == "fireworks":
                    file_stats["neg_fireworks_windows"] += 1
                else:
                    file_stats["neg_nontarget_windows"] += 1
                    
        records.append({
            "name": fname,
            "label": lbl,
            "binary_label": 1 if lbl == "explosion" else 0,
            "category": cat,
            "group_id": gid,
            "max_dbfs": max_dbfs,
            "windows_count": len(file_window_records),
            "windows": file_window_records
        })
        
    file_stats["total_active_windows"] = file_stats["pos_windows"] + file_stats["neg_windows"]
    elapsed = time.time() - t0
    print(f"Completed {split_name} in {elapsed:.1f}s:")
    print(f"  Positive windows (Explosion blast/concussive): {file_stats['pos_windows']}")
    print(f"  Negative windows (Total Not-Explosion): {file_stats['neg_windows']}")
    print(f"    Fireworks negative windows : {file_stats['neg_fireworks_windows']}")
    print(f"    Non-target negative windows: {file_stats['neg_nontarget_windows']}")
    print(f"  Quarantined background windows purged from explosion files: {file_stats['quarantined_background_windows_purged']}")
    
    return {
        "file_records": records,
        "pool_pos": pool_pos,
        "pool_neg": pool_neg,
        "stats": file_stats
    }

# Execute window extraction for both splits
fit_data = extract_split_windows(fit_df, "train_fit")
val_data = extract_split_windows(val_df, "train_val")

# Save pools and caches
print("\nSaving dataset pools and caches...")
torch.save(fit_data, DATA_DIR / "train_fit_binary_pool.pt")
torch.save(val_data, DATA_DIR / "train_val_binary_cache.pt")

provenance_summary = {
    "manifest_source": "dataset_manifest_phase21_expanded.csv",
    "total_approved_recordings": 824,
    "quarantined_recordings_excluded": 4,
    "quarantined_files": [r["file_path"] for idx, r in quarantined_train.iterrows()],
    "partition_seed": 42,
    "fit_split": fit_data["stats"],
    "val_split": val_data["stats"],
    "extraction_parameters": {
        "sample_rate": TARGET_SR,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "squelch_dbfs": SQUELCH_DBFS,
        "blast_filter_offset_db": BLAST_FILTER_OFFSET_DB,
        "blast_min_dbfs": BLAST_MIN_DBFS,
        "max_negative_windows_per_file": MAX_WINDOWS_PER_NEGATIVE_FILE
    }
}

with open(DATA_DIR / "dataset_provenance_summary.json", "w") as f:
    json.dump(provenance_summary, f, indent=2)

print(f"Saved dataset caches and {DATA_DIR / 'dataset_provenance_summary.json'} successfully.")
