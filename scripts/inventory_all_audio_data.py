r"""NOESIS Audio Data Inventory: Comprehensive Read-Only Audit of All Audio Recordings across D:\NOESIS.

Determines:
1. Total audio files on disk
2. Exact deduplication by SHA-256
3. Categorization into:
   - Already in ExpF training (634 files)
   - Protected Validation / Benchmark / Evaluation:
     * 190-file validation set
     * 92-file historical regression benchmark
     * Codex reserved test sets (Set 1: 86 files, Set 2: 60 files)
   - Codex evaluation collection remainder (906 files)
   - Quarantined / rejected files
   - Any unassigned / new unique audio files available on disk
"""

import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import pandas as pd
import torch

PROJECT_ROOT = Path("D:/NOESIS/Explosion/Explosion")
CODEX_ROOT = Path("D:/NOESIS/Codex_Evaluation_Dataset")

def compute_sha256(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

print("=" * 80)
print("1. LOADING BASELINE INVENTORY")
print("=" * 80)

# Load dataset provenance summary for ExpF
prov_path = PROJECT_ROOT / "training" / "data" / "dataset_provenance_summary.json"
with open(prov_path, "r") as f:
    prov = json.load(f)

print(f"ExpF fit files: {prov['fit_split']['total_files']} ({prov['fit_split']['explosion_files']} pos, {prov['fit_split']['nontarget_files'] + prov['fit_split']['fireworks_files']} neg)")
print(f"ExpF val files: {prov['val_split']['total_files']} ({prov['val_split']['explosion_files']} pos, {prov['val_split']['nontarget_files'] + prov['val_split']['fireworks_files']} neg)")
print(f"Quarantined files in Phase 21: {len(prov['quarantined_files'])}")

# Load Phase 21 expanded manifest
p21_csv = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
p21_df = pd.read_csv(p21_csv)
print(f"Phase 21 Expanded Manifest Total Rows: {len(p21_df)}")

# Load train_fit_binary_pool to get exact filenames of ExpF training files
fit_data = torch.load(PROJECT_ROOT / "training" / "data" / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
fit_fnames = set(w.get("filename") for w in fit_data["pool_pos"] + fit_data["pool_neg"])
print(f"Unique filenames in train_fit_binary_pool: {len(fit_fnames)}")

# Load train_val_binary_cache to get exact filenames of 190 validation files
val_data = torch.load(PROJECT_ROOT / "training" / "data" / "train_val_binary_cache.pt", map_location="cpu", weights_only=False)
val_fnames = set(r.get("name") or r.get("filename") for r in val_data["file_records"])
print(f"Unique filenames in train_val_binary_cache: {len(val_fnames)}")

# Historical Benchmark files (92 files)
hist_bench_df = p21_df[p21_df["split"].isin(["val", "holdout", "test", "challenge"])]
hist_fnames = set(Path(p).name for p in hist_bench_df["file_path"])
print(f"Unique filenames in Historical Benchmark: {len(hist_fnames)}")

# Check for overlap between fit, val, and hist
print(f"Overlap fit vs val : {len(fit_fnames.intersection(val_fnames))}")
print(f"Overlap fit vs hist: {len(fit_fnames.intersection(hist_fnames))}")
print(f"Overlap val vs hist: {len(val_fnames.intersection(hist_fnames))}")

# Check all audio files physically present in D:\NOESIS\Explosion\Explosion\data
print("\n" + "=" * 80)
print("2. SCANNING AUDIO FILES IN D:\\NOESIS\\Explosion\\Explosion\\data")
print("=" * 80)

data_dir = PROJECT_ROOT / "data"
exts = {".wav", ".mp3", ".ogg", ".flac"}
disk_audio_files = []
for root, dirs, files in os.walk(data_dir):
    for f in files:
        p = Path(root) / f
        if p.suffix.lower() in exts:
            disk_audio_files.append(p)

print(f"Total audio files found on disk under data/: {len(disk_audio_files)}")

# Check how many disk audio files are in p21_df
p21_paths_rel = set(Path(p).as_posix() for p in p21_df["file_path"])
p21_fnames = set(Path(p).name for p in p21_df["file_path"])

disk_not_in_p21 = []
for p in disk_audio_files:
    rel = p.relative_to(PROJECT_ROOT).as_posix()
    if rel not in p21_paths_rel and p.name not in p21_fnames:
        disk_not_in_p21.append(p)

print(f"Audio files on disk in data/ not in Phase 21 manifest: {len(disk_not_in_p21)}")
if disk_not_in_p21:
    print(f"Sample unlisted files: {[p.name for p in disk_not_in_p21[:10]]}")

# Scan Codex Evaluation Dataset
print("\n" + "=" * 80)
print("3. SCANNING CODEX EVALUATION DATASET")
print("=" * 80)

codex_manifest_csv = CODEX_ROOT / "manifest.csv"
if codex_manifest_csv.exists():
    codex_df = pd.read_csv(codex_manifest_csv)
    print(f"Codex Manifest Total Rows: {len(codex_df)}")
    print(f"Codex Ready Status:\n{codex_df['status'].value_counts()}")
    print(f"Codex Ready for AGY:\n{codex_df['ready_for_antigravity_validation'].value_counts()}")
    print(f"Codex Categories:\n{codex_df[codex_df['status'] == 'ready_for_validation']['category'].value_counts()}")
    
    # Check frozen test sets
    c_set1 = PROJECT_ROOT / "data" / "manifests" / "codex_selected_test_set1.csv"
    c_set2 = PROJECT_ROOT / "data" / "manifests" / "codex_frozen_test_set2.csv"
    c1_fnames = set(pd.read_csv(c_set1)["filename"]) if c_set1.exists() else set()
    c2_fnames = set(pd.read_csv(c_set2)["filename"]) if c_set2.exists() else set()
    print(f"Codex Frozen Set 1 Files: {len(c1_fnames)}")
    print(f"Codex Frozen Set 2 Files: {len(c2_fnames)}")
    print(f"Codex Overlap Set 1 vs Set 2: {len(c1_fnames.intersection(c2_fnames))}")
