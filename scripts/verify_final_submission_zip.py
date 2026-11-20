import sys
import os
import time
import zipfile
import hashlib
from pathlib import Path
import shutil
import torch
import numpy as np

ZIP_PATH = Path("D:/NOESIS/NOESIS_Explosion_Final_Submission.zip")
TEMP_EXTRACT_DIR = Path("D:/NOESIS/temp_restore_verify")
REQUIRED_EXPF_SHA256 = "3395acbe99251e7445e89342011176a744e8e515ba7241dd255059928dde0287"

print("=" * 80)
print("NOESIS SUBMISSION PACKAGE — COMPREHENSIVE INTEGRITY VERIFICATION")
print(f"Archive: {ZIP_PATH}")
print("=" * 80)

# 1. Check ZIP existence and size
assert ZIP_PATH.exists(), f"ERROR: ZIP not found at {ZIP_PATH}"
zip_size = ZIP_PATH.stat().st_size
print(f"[CHECK 1] ZIP File Found:")
print(f"  Path: {ZIP_PATH}")
print(f"  Size: {zip_size / (1024**3):.2f} GB ({zip_size:,} bytes)")

# 2. Archive CRC / Integrity Check
print(f"\n[CHECK 2] Testing ZIP Archive Integrity (CRC32 Check)...")
t0 = time.time()
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    namelist = zf.namelist()
    total_entries = len(namelist)
    print(f"  Total entries in ZIP: {total_entries:,}")
    
    # Fast testzip check
    corrupt_file = zf.testzip()
    if corrupt_file is None:
        print("  Archive integrity (CRC32): PASS (0 corrupted files)")
    else:
        print(f"  ERROR: Corrupted file detected: {corrupt_file}")
        sys.exit(1)

    # 3. Check ExpF Checkpoint Integrity in ZIP
    print(f"\n[CHECK 3] Verifying Active ExpF Checkpoint inside ZIP...")
    ckpt_arcname = "Explosion/explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt"
    assert ckpt_arcname in namelist, f"ERROR: {ckpt_arcname} missing from ZIP!"
    
    ckpt_bytes = zf.read(ckpt_arcname)
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
    print(f"  Extracted Checkpoint Size: {len(ckpt_bytes):,} bytes")
    print(f"  Computed SHA-256: {ckpt_sha256}")
    print(f"  Required SHA-256: {REQUIRED_EXPF_SHA256}")
    assert ckpt_sha256 == REQUIRED_EXPF_SHA256, "ERROR: Checkpoint SHA-256 mismatch!"
    print("  ExpF Checkpoint Integrity: PASS")

    # 4. Check Config E and Config C Preservation
    print(f"\n[CHECK 4] Verifying Config E & Config C Preservation...")
    tracker_arcname = "Explosion/inference/tracker_verified.py"
    assert tracker_arcname in namelist, f"ERROR: {tracker_arcname} missing from ZIP!"
    tracker_code = zf.read(tracker_arcname).decode("utf-8")
    assert "Config E" in tracker_code, "ERROR: Config E missing from tracker!"
    assert "Config C" in tracker_code, "ERROR: Config C preset missing from tracker!"
    assert "config_preset: str = \"Config E\"" in tracker_code or "set_config_preset" in tracker_code, "ERROR: Config preset methods missing!"
    print("  Config E Default & Config C Reversibility: PASS")

    # 5. Check Essential Components
    print(f"\n[CHECK 5] Checking Critical Component Categories...")
    critical_files = [
        "START_HERE.md",
        "Explosion/START_HERE.md",
        "Explosion/ARCHITECTURE.md",
        "Explosion/README.md",
        "Explosion/requirements.txt",
        "Explosion/server/app.py",
        "Explosion/server/static/index.html",
        "Explosion/audio/microphone_stream.py",
        "Explosion/audio/file_loader.py",
        "Explosion/audio/preprocessing.py",
        "Explosion/explosion_detection/models/binary_cnn.py",
        "Explosion/data/manifests/dataset_manifest_phase21_expanded.csv",
        "Explosion/training/data/train_val_binary_cache.pt",
        "Explosion/reports/alerts_history_submission_backup.db",
        "external_dependencies/Codex_Evaluation_Dataset/README.md",
    ]
    missing_critical = []
    for cf in critical_files:
        if cf in namelist:
            print(f"  Found: {cf}")
        else:
            print(f"  MISSING: {cf}")
            missing_critical.append(cf)
    
    assert len(missing_critical) == 0, f"ERROR: {len(missing_critical)} critical files missing!"
    print("  Critical File Presence: PASS")

# 6. Restoration Verification (Extraction & Smoke Test)
print(f"\n[CHECK 6] Verifying Restoration & Running Smoke Test...")
if TEMP_EXTRACT_DIR.exists():
    shutil.rmtree(TEMP_EXTRACT_DIR)
TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Selectively extract test subset to verify unzipping and execution without filling disk
test_extract_prefixes = [
    "START_HERE.md",
    "Explosion/START_HERE.md",
    "Explosion/explosion_detection/",
    "Explosion/inference/",
    "Explosion/audio/",
    "Explosion/server/",
    "Explosion/configs/",
]

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for m in zf.namelist():
        if any(m.startswith(prefix) for prefix in test_extract_prefixes):
            zf.extract(m, TEMP_EXTRACT_DIR)

extracted_proj = TEMP_EXTRACT_DIR / "Explosion"
assert (extracted_proj / "START_HERE.md").exists(), "Extracted START_HERE.md missing"
assert (extracted_proj / "server" / "app.py").exists(), "Extracted server/app.py missing"

# Test loading extracted checkpoint in PyTorch
sys.path.insert(0, str(extracted_proj))
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

ext_ckpt_path = extracted_proj / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
sd = torch.load(ext_ckpt_path, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]

model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

# Dummy forward pass smoke test
dummy_spec = torch.randn(1, 1, 64, 44)
with torch.no_grad():
    out = model(dummy_spec)
    probs = torch.softmax(out, dim=-1)

print(f"  Extracted Model Forward Pass: Output shape {out.shape}, Probs: {probs.tolist()}")
assert out.shape == (1, 2), "Forward pass failed shape check"

# Test Tracker Smoke Test
tracker = AcousticVerifiedEventTracker(config_preset="Config E")
assert tracker.config_preset == "Config E"
assert tracker.squelch_dbfs == -34.0
assert tracker.conf_threshold == 0.80

tracker.set_config_preset("Config C")
assert tracker.config_preset == "Config C"
assert tracker.squelch_dbfs == -45.0
assert tracker.conf_threshold == 0.70

tracker.set_config_preset("Config E")
assert tracker.config_preset == "Config E"
print("  Extracted Tracker Preset Switching: PASS")

# Clean up temporary test directory
shutil.rmtree(TEMP_EXTRACT_DIR)
print(f"  Cleaned up temporary extraction directory: {TEMP_EXTRACT_DIR}")

print("\n" + "=" * 80)
print("ALL MANDATORY CHECKS PASSED: SUBMISSION PACKAGE IS 100% READY!")
print("=" * 80)
