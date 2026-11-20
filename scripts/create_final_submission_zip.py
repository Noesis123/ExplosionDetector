import sys
import os
import time
import zipfile
import hashlib
from pathlib import Path
import json

PROJECT_DIR = Path("D:/NOESIS/Explosion/Explosion")
CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
OUTPUT_ZIP = Path("D:/NOESIS/NOESIS_Explosion_Final_Submission.zip")

print("=" * 80)
print("NOESIS EXPLOSION DETECTION SYSTEM — FINAL SUBMISSION PACKAGING")
print(f"Target Archive: {OUTPUT_ZIP}")
print("=" * 80)

# 1. Inventory source files
print("\n[1/4] Inventorying source files...")
files_to_pack = []

# Root START_HERE.md
start_here_file = PROJECT_DIR / "START_HERE.md"
if start_here_file.exists():
    files_to_pack.append((start_here_file, "START_HERE.md"))

# All files in PROJECT_DIR
for p in PROJECT_DIR.rglob("*"):
    if p.is_file():
        rel = p.relative_to(PROJECT_DIR).as_posix()
        arcname = f"Explosion/{rel}"
        files_to_pack.append((p, arcname))

# All files in CODEX_DIR
if CODEX_DIR.exists():
    for p in CODEX_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(CODEX_DIR).as_posix()
            arcname = f"external_dependencies/Codex_Evaluation_Dataset/{rel}"
            files_to_pack.append((p, arcname))

total_files = len(files_to_pack)
total_bytes = sum(src.stat().st_size for src, _ in files_to_pack)
print(f"  Total files to archive: {total_files:,}")
print(f"  Total uncompressed size: {total_bytes / (1024**3):.2f} GB ({total_bytes:,} bytes)")

# 2. Check ExpF Checkpoint in source
expf_path = PROJECT_DIR / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
assert expf_path.exists(), f"ERROR: ExpF checkpoint not found at {expf_path}"
expf_sha256 = hashlib.sha256(expf_path.read_bytes()).hexdigest()
REQUIRED_SHA256 = "3395acbe99251e7445e89342011176a744e8e515ba7241dd255059928dde0287"
print(f"\n[2/4] Verifying active ExpF checkpoint:")
print(f"  Path: {expf_path}")
print(f"  SHA-256: {expf_sha256}")
assert expf_sha256 == REQUIRED_SHA256, f"ERROR: SHA256 mismatch! Got {expf_sha256}, expected {REQUIRED_SHA256}"
print("  Checkpoint verification: PASS")

# 3. Create ZIP64 Archive
print(f"\n[3/4] Creating ZIP64 Archive at {OUTPUT_ZIP}...")
t0 = time.time()
processed_bytes = 0
last_print_time = t0

# If an old temp zip exists, preserve or remove
temp_zip = OUTPUT_ZIP.with_suffix(".zip.tmp")
if temp_zip.exists():
    temp_zip.unlink()

with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
    for idx, (src, arcname) in enumerate(files_to_pack, 1):
        try:
            zf.write(src, arcname=arcname)
            processed_bytes += src.stat().st_size
        except Exception as e:
            print(f"Error packing {src}: {e}")
            raise e

        now = time.time()
        if now - last_print_time >= 5.0 or idx == total_files:
            pct = (processed_bytes / total_bytes) * 100.0 if total_bytes > 0 else 100.0
            mb_sec = (processed_bytes / (1024**2)) / max(now - t0, 0.001)
            print(f"  Progress: {idx:,}/{total_files:,} files ({pct:.1f}%) | {processed_bytes/(1024**3):.2f}/{total_bytes/(1024**3):.2f} GB | Speed: {mb_sec:.1f} MB/s")
            last_print_time = now

# Rename temp to final
if OUTPUT_ZIP.exists():
    backup_zip = OUTPUT_ZIP.with_name("NOESIS_Explosion_Final_Submission_backup.zip")
    print(f"  Preserving existing ZIP as {backup_zip.name}")
    OUTPUT_ZIP.rename(backup_zip)

temp_zip.rename(OUTPUT_ZIP)
t_elapsed = time.time() - t0
zip_size = OUTPUT_ZIP.stat().st_size
print(f"\n[4/4] Archive successfully created!")
print(f"  Location: {OUTPUT_ZIP}")
print(f"  Compressed Size: {zip_size / (1024**3):.2f} GB ({zip_size:,} bytes)")
print(f"  Compression Ratio: {(1.0 - zip_size / total_bytes) * 100.0:.1f}% space saved")
print(f"  Packaging Time: {t_elapsed:.1f} seconds")
print("=" * 80)
