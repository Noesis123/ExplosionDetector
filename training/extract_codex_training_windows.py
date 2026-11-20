import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import math
import numpy as np
import pandas as pd
import torch
from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

print("=" * 75)
print("EXTRACTING APPROVED CODEX TRAINING WINDOWS")
print("=" * 75)

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
manifest_path = PROJECT_ROOT / "reports" / "codex_training_additions.csv"
train_df = pd.read_csv(manifest_path)

prep = AudioPreprocessor(sample_rate=22050)
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0

codex_pos_windows = []
codex_neg_windows = []

t0 = time.time()

for idx, r in train_df.iterrows():
    p = CODEX_DIR / r["audio_path"]
    cat = r["category"]
    is_pos = (cat == "surface_high_explosive")
    gid = r.get("source_session_id") or r.get("uploader_group_id") or "codex"
    fname = p.name
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != TARGET_SR:
        mono = prep.resample(mono, orig_sr=sr)
        
    if len(mono) < WINDOW_SAMPLES:
        mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)), mode="constant")
        
    n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    
    # Cap max negative windows per file to prevent single file dominance
    max_wins = 8 if cat in ["pre_event_ambient_control", "music", "traffic"] else 12
    step = max(1, n_w // max_wins) if n_w > max_wins else 1
    selected_indices = range(0, n_w, step)[:max_wins]
    
    for i in selected_indices:
        w = mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)), mode="constant")
            
        rms = float(np.sqrt(np.mean(w ** 2)))
        dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
        peak = float(np.max(np.abs(w)))
        
        # In negatives, if below squelch, only keep a few ambient controls
        if not is_pos and dbfs < SQUELCH_DBFS and cat != "pre_event_ambient_control":
            continue
            
        spec = prep.compute_log_mel_spectrogram(w)
        win_data = {
            "spec": torch.from_numpy(spec).float(),
            "dbfs": dbfs,
            "peak": peak,
            "t_sec": i * 0.25,
            "binary_label": 1 if is_pos else 0,
            "original_label": "explosion" if is_pos else "non_target",
            "category": cat,
            "group_id": gid,
            "filename": fname,
            "source": "codex"
        }
        
        if is_pos:
            codex_pos_windows.append(win_data)
        else:
            codex_neg_windows.append(win_data)

elapsed = time.time() - t0
print(f"Extracted Codex Windows in {elapsed:.1f}s:")
print(f"  Positive Explosion Windows: {len(codex_pos_windows)}")
print(f"  Negative Confound Windows : {len(codex_neg_windows)}")

# Load existing training fit pool
fit_data = torch.load(PROJECT_ROOT / "training" / "data" / "train_fit_binary_pool.pt", map_location="cpu", weights_only=False)
base_pos = fit_data["pool_pos"]
base_neg = fit_data["pool_neg"]

print(f"\nExisting Training Pool: {len(base_pos)} pos, {len(base_neg)} neg")

# Merge pools
combined_pos = list(base_pos) + codex_pos_windows
combined_neg = list(base_neg) + codex_neg_windows

print(f"Augmented Combined Training Pool: {len(combined_pos)} pos, {len(combined_neg)} neg")

# Save augmented pool
out_path = PROJECT_ROOT / "training" / "data" / "augmented_training_pool.pt"
torch.save({
    "pool_pos": combined_pos,
    "pool_neg": combined_neg,
    "num_pos": len(combined_pos),
    "num_neg": len(combined_neg)
}, out_path)
print(f"Saved augmented pool to: {out_path}")
