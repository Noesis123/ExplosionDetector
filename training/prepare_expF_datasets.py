"""Prepare Refined Audited Datasets for Candidate ExpF.

Fixes genuine event-window extraction defects:
1. Rescues weak/distant explosion blast windows by lowering BLAST_MIN_DBFS from -40.0 to -44.0 dBFS
   (with BLAST_FILTER_OFFSET_DB = 20.0), recovering genuine blast rumbling tails and 2 completely omitted files.
2. Expands rare hard-negative transient sampling (artillery, vehicle backfires, industrial impacts,
   metal/glass impacts) from 12 up to 36 windows per recording to capture varied blast shock profiles.
3. Preserves exact 80/20 group isolation (partition seed = 42, 0% leakage between train_fit and train_val).
4. Strictly preserves zero leakage into the 92-recording evaluation benchmark and Codex evaluation set.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"

if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
BLAST_FILTER_OFFSET_DB = 20.0
BLAST_MIN_DBFS = -44.0
DEFAULT_MAX_NEG_WINS = 12
EXPANDED_MAX_NEG_WINS = 36
RARE_HARD_NEG_CATEGORIES = {
    "artillery_firing",
    "industrial_impacts",
    "metal_glass_impacts",
    "vehicle_backfires"
}

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")

approved_train = manifest_df[(manifest_df["split"] == "train") & (manifest_df["training_approved"] == True)].copy()
quarantined_train = manifest_df[(manifest_df["split"] == "train") & (manifest_df["training_approved"] == False)].copy()

assert len(approved_train) == 824, f"Expected 824 approved files, got {len(approved_train)}"
assert len(quarantined_train) == 4, f"Expected 4 quarantined files, got {len(quarantined_train)}"

# Group-aware 80/20 partition (identical deterministic seed 42)
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

# Leakage checks
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])]
eval_files = set(eval_df["file_path"])
eval_groups = set(eval_df["group_id"].dropna())

assert len(set(fit_df["group_id"]).intersection(set(val_df["group_id"]))) == 0, "Leakage between fit and val groups!"
assert len(set(fit_df["file_path"]).intersection(eval_files)) == 0, "Leakage into eval files from fit!"
assert len(set(val_df["file_path"]).intersection(eval_files)) == 0, "Leakage into eval files from val!"
assert len(set(fit_df["group_id"]).intersection(eval_groups)) == 0, "Leakage into eval groups from fit!"
assert len(set(val_df["group_id"]).intersection(eval_groups)) == 0, "Leakage into eval groups from val!"

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
        p = PROJECT_ROOT / r["file_path"]
        
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
        file_window_records = []
        
        if lbl == "explosion":
            # Explosion Positive Window Selection
            blast_threshold = max(max_dbfs - BLAST_FILTER_OFFSET_DB, BLAST_MIN_DBFS)
            for i, (w, db, pk) in enumerate(zip(raw_windows, dbfs_list, peaks)):
                if db < SQUELCH_DBFS:
                    continue
                if db < blast_threshold:
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
            max_wins = EXPANDED_MAX_NEG_WINS if cat in RARE_HARD_NEG_CATEGORIES else DEFAULT_MAX_NEG_WINS
            
            # For validation split, preserve full evaluation sequence; for fit pool, take top spaced windows
            active_indices = [i for i, db in enumerate(dbfs_list) if db >= SQUELCH_DBFS]
            if not active_indices:
                active_indices = [int(np.argmax(dbfs_list))]
                
            if split_name == "train_fit" and len(active_indices) > max_wins:
                step = len(active_indices) / float(max_wins)
                selected_indices = [active_indices[int(i * step)] for i in range(max_wins)]
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

if __name__ == "__main__":
    fit_data = extract_split_windows(fit_df, "train_fit")
    val_data = extract_split_windows(val_df, "train_val")
    
    print("\nSaving dataset pools and caches for ExpF...")
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
            "default_max_neg_wins": DEFAULT_MAX_NEG_WINS,
            "expanded_max_neg_wins": EXPANDED_MAX_NEG_WINS,
            "rare_hard_neg_categories": list(RARE_HARD_NEG_CATEGORIES)
        }
    }
    
    with open(DATA_DIR / "dataset_provenance_summary.json", "w") as f:
        json.dump(provenance_summary, f, indent=2)
        
    print(f"Saved dataset caches and {DATA_DIR / 'dataset_provenance_summary.json'} successfully.")
