import pandas as pd
import numpy as np
from pathlib import Path

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
manifest_path = CODEX_DIR / "manifest.csv"
output_path = Path("data/manifests/codex_expanded_eval_manifest.csv")

# 1. Load Codex Manifest
df_codex = pd.read_csv(manifest_path, low_memory=False)

# 2. Filter out quarantined and non-ready
valid = df_codex[(df_codex["status"] == "ready_for_validation") & (df_codex["status"] != "quarantined")].copy()

# 3. Verify disk existence
on_disk = []
for idx, r in valid.iterrows():
    p = CODEX_DIR / str(r["audio_path"])
    if p.exists():
        on_disk.append(r)
df_disk = pd.DataFrame(on_disk)
print(f"Total verified files on disk in Codex: {len(df_disk)}")

# 4. Check zero overlap with training / benchmark data
train_manifest = pd.read_csv("data/manifests/dataset_manifest_phase21_expanded.csv")
train_filenames = set(Path(f).name for f in train_manifest["file_path"].dropna())
train_shas = set(train_manifest["sha256"].dropna()) if "sha256" in train_manifest.columns else set()

df_clean = df_disk[~df_disk["audio_path"].apply(lambda p: Path(p).name in train_filenames)].copy()
if "sha256" in df_clean.columns:
    df_clean = df_clean[~df_clean["sha256"].isin(train_shas)].copy()
print(f"Files after strict training/benchmark exclusion: {len(df_clean)}")

# 5. Deterministic Stratified Sampling (Seed 42)
np.random.seed(42)

# A: 100 Surface High Explosives
exp_pool = df_clean[df_clean["category"] == "surface_high_explosive"]
selected_exp = exp_pool.sample(n=min(100, len(exp_pool)), random_state=42).copy()
selected_exp["eval_binary_label"] = "explosion"

# B: 200 Diverse Negative Recordings
# Categories: thunder, gunfire, fireworks, engines, mechanical_impulses, construction_impacts, speech, crowds, environment, household, traffic, pre_event_ambient_control
neg_dfs = []

# Thunder: 40
thun_pool = df_clean[df_clean["category"] == "thunder"]
neg_dfs.append(thun_pool.sample(n=min(40, len(thun_pool)), random_state=42))

# Gunfire: all available (16)
gun_pool = df_clean[df_clean["category"] == "gunfire"]
neg_dfs.append(gun_pool.sample(n=min(16, len(gun_pool)), random_state=42))

# Fireworks: all available (16)
fw_pool = df_clean[df_clean["category"] == "fireworks"]
neg_dfs.append(fw_pool.sample(n=min(16, len(fw_pool)), random_state=42))

# Engines / vehicle backfires: 35
eng_pool = df_clean[df_clean["category"] == "engines"]
neg_dfs.append(eng_pool.sample(n=min(35, len(eng_pool)), random_state=42))

# Mechanical & Construction impacts: 35
mech_pool = df_clean[df_clean["category"].isin(["mechanical_impulses", "construction_impacts"])]
neg_dfs.append(mech_pool.sample(n=min(35, len(mech_pool)), random_state=42))

# Speech & Crowds: 25
speech_pool = df_clean[df_clean["category"].isin(["speech", "crowds"])]
neg_dfs.append(speech_pool.sample(n=min(25, len(speech_pool)), random_state=42))

# Ambient / Environment / Traffic / Household: 33
ambient_pool = df_clean[df_clean["category"].isin(["pre_event_ambient_control", "environment", "traffic", "household", "doors", "music"])]
neg_dfs.append(ambient_pool.sample(n=min(33, len(ambient_pool)), random_state=42))

selected_neg = pd.concat(neg_dfs).copy()
selected_neg["eval_binary_label"] = "non_explosion"

# Combine into frozen manifest
expanded_eval = pd.concat([selected_exp, selected_neg]).reset_index(drop=True)
expanded_eval["absolute_audio_path"] = expanded_eval["audio_path"].apply(lambda p: str((CODEX_DIR / p).resolve()))

print(f"\nExpanded Evaluation Dataset composition (Total: {len(expanded_eval)}):")
print(expanded_eval["eval_binary_label"].value_counts())
print("\nCategory breakdown:")
print(expanded_eval["category"].value_counts())

output_path.parent.mkdir(parents=True, exist_ok=True)
expanded_eval.to_csv(output_path, index=False)
print(f"\nSaved frozen expanded manifest to {output_path} ({len(expanded_eval)} records).")
