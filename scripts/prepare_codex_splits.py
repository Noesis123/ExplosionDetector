import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
from audio.file_loader import load_audio_file

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
manifest_path = CODEX_DIR / "manifest.csv"
codex_df = pd.read_csv(manifest_path, low_memory=False)

# 1. Load previously examined evaluation subset (86 files)
prev_eval_path = PROJECT_ROOT / "reports" / "codex_selected_eval_subset.csv"
prev_eval_df = pd.read_csv(prev_eval_path)
prev_eval_record_ids = set(prev_eval_df["record_id"].dropna())
prev_eval_filenames = set(prev_eval_df["filename"].dropna())

# 2. Load main project training manifest to prevent ANY overlap
train_manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
train_filenames = set(Path(f).name for f in train_manifest["file_path"].dropna())
train_groups = set(train_manifest["group_id"].dropna())

print(f"Total Codex Manifest Records: {len(codex_df)}")

# 3. Filter for ready_for_validation and exclude quarantined
valid_mask = (codex_df["status"] == "ready_for_validation") & (codex_df["status"] != "quarantined")
valid_df = codex_df[valid_mask].copy()
print(f"Approved Ready-for-Validation Records: {len(valid_df)}")

# 4. Verify on disk and exclude overlaps
clean_records = []
quarantine_count = 0
overlap_count = 0
prev_eval_count = 0

for idx, r in valid_df.iterrows():
    p = CODEX_DIR / r["audio_path"]
    if not p.exists():
        continue
    rid = r["record_id"]
    fname = p.name
    
    if rid in prev_eval_record_ids or fname in prev_eval_filenames:
        prev_eval_count += 1
        continue
        
    if fname in train_filenames:
        overlap_count += 1
        continue
        
    gid = r.get("uploader_group_id") or r.get("source_session_id")
    if gid in train_groups:
        overlap_count += 1
        continue
        
    clean_records.append(r)

df_unexamined = pd.DataFrame(clean_records)
print(f"Unexamined Clean Codex Pool: {len(df_unexamined)} files")
print(f"  (Excluded: {prev_eval_count} previous eval files, {overlap_count} training overlaps, {106} quarantined)")
print("\nUnexamined Clean Categories:")
print(df_unexamined["category"].value_counts())

# 5. Screen explosions for audible acoustic energy vs sub-squelch/infrasound
print("\nScreening unexamined explosions for audible shockwaves...")
exp_clean = df_unexamined[df_unexamined["category"] == "surface_high_explosive"]

audible_exp_records = []
sub_squelch_infrasound_records = []

for idx, r in exp_clean.iterrows():
    p = CODEX_DIR / r["audio_path"]
    sig, sr, dur, ch = load_audio_file(p)
    rms = 20 * np.log10(max(np.sqrt(np.mean(sig**2)), 1e-9))
    
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0/sr)
    tot_e = np.sum(fft_mag**2) + 1e-12
    e_sub20 = np.sum(fft_mag[freqs < 20.0]**2) / tot_e
    
    if rms >= -45.0 and sr > 800 and e_sub20 < 0.80:
        audible_exp_records.append(r)
    else:
        sub_squelch_infrasound_records.append(r)

df_audible_exp = pd.DataFrame(audible_exp_records)
df_infrasound_exp = pd.DataFrame(sub_squelch_infrasound_records)
print(f"Audible Explosions Screened: {len(df_audible_exp)} files (RMS >= -45 dBFS, SR=8kHz, sub-20Hz < 80%)")
print(f"Infrasound / Sub-squelch Purged: {len(df_infrasound_exp)} files")

# 6. Reserve Frozen Codex Evaluation Portion (Test Split 2)
# Strictly source-independent sampling
np.random.seed(42)
reserved_records = []

# 15 Audible Explosions for Evaluation
res_exp = df_audible_exp.sample(n=min(15, len(df_audible_exp)), random_state=42)
reserved_records.append(res_exp)

# 10 Thunder for Evaluation
thun_pool = df_unexamined[df_unexamined["category"] == "thunder"]
res_thun = thun_pool.sample(n=min(10, len(thun_pool)), random_state=42)
reserved_records.append(res_thun)

# 10 Engines for Evaluation
eng_pool = df_unexamined[df_unexamined["category"] == "engines"]
res_eng = eng_pool.sample(n=min(10, len(eng_pool)), random_state=42)
reserved_records.append(res_eng)

# 10 Mechanical / Impacts for Evaluation
mech_pool = df_unexamined[df_unexamined["category"].isin(["mechanical_impulses", "construction_impacts", "doors"])]
res_mech = mech_pool.sample(n=min(10, len(mech_pool)), random_state=42)
reserved_records.append(res_mech)

# 15 Everyday / Speech / Household for Evaluation
everyday_pool = df_unexamined[df_unexamined["category"].isin(["speech", "environment", "household", "traffic", "music"])]
res_everyday = everyday_pool.sample(n=min(15, len(everyday_pool)), random_state=42)
reserved_records.append(res_everyday)

reserved_eval_df = pd.concat(reserved_records, ignore_index=True)
print(f"\nReserved Frozen Evaluation Set (Test Split 2): {len(reserved_eval_df)} files")
print(reserved_eval_df["category"].value_counts())

reserved_eval_path = PROJECT_ROOT / "reports" / "codex_reserved_eval_frozen.csv"
reserved_eval_df[["record_id", "category", "proposed_binary_label", "audio_path", "filename", "duration_seconds"]].to_csv(reserved_eval_path, index=False)
print(f"Saved reserved evaluation manifest to: {reserved_eval_path}")

# 7. Remaining Approved Pool for Training Additions
reserved_rids = set(reserved_eval_df["record_id"].dropna())

training_addition_records = []

# Eligible audible explosions for training
for idx, r in df_audible_exp.iterrows():
    if r["record_id"] not in reserved_rids:
        training_addition_records.append(r)

# Eligible negatives for training (excluding reserved, infrasound, and quarantined)
neg_unexamined = df_unexamined[df_unexamined["category"] != "surface_high_explosive"]
for idx, r in neg_unexamined.iterrows():
    if r["record_id"] not in reserved_rids:
        training_addition_records.append(r)

training_additions_df = pd.DataFrame(training_addition_records)
print(f"\nApproved Codex Training Additions: {len(training_additions_df)} files")
print(training_additions_df["category"].value_counts())

train_add_path = PROJECT_ROOT / "reports" / "codex_training_additions.csv"
training_additions_df[["record_id", "category", "proposed_binary_label", "audio_path", "filename", "duration_seconds"]].to_csv(train_add_path, index=False)
print(f"Saved training additions manifest to: {train_add_path}")
