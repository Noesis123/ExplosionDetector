"""
Extract and cache paired (event_mel, context_mel) samples for CausalContextBinaryCNN (Candidate ExpM).
Ensures 100% causal 2.0-s context extraction with zero future audio leakage.
"""

import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

TARGET_SR = 22050
WINDOW_SAMPLES = 22050   # 1.0 s
CONTEXT_SAMPLES = 44100  # 2.0 s preceding
HOP_SAMPLES = 5512       # 0.25 s
SQUELCH_DBFS = -45.0

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

print("=" * 75)
print("PREPARING CAUSAL CONTEXT TRAINING & EVALUATION CACHES")
print("=" * 75)

# 1. Load manifest and approved Codex additions
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
codex_additions_path = PROJECT_ROOT / "reports" / "codex_training_additions.csv"
codex_df = pd.read_csv(codex_additions_path) if codex_additions_path.exists() else pd.DataFrame()

# Exclude val, test, challenge from training
train_manifest = manifest_df[manifest_df["split"].isin(["train", "fit", "synth"])].copy()

def extract_causal_mels(sig, t_start):
    # Event window
    w = sig[t_start : t_start + WINDOW_SAMPLES]
    if len(w) < WINDOW_SAMPLES:
        w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
    mel_ev = preprocessor.compute_log_mel_spectrogram(w)
    
    # Preceding 2.0s context
    ctx_start = max(0, t_start - CONTEXT_SAMPLES)
    ctx = sig[ctx_start : t_start]
    if len(ctx) < CONTEXT_SAMPLES:
        ctx = np.pad(ctx, (CONTEXT_SAMPLES - len(ctx), 0))
    mel_ctx = preprocessor.compute_log_mel_spectrogram(ctx)
    
    rms = float(np.sqrt(np.mean(w ** 2)))
    dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
    
    return torch.from_numpy(mel_ev).float(), torch.from_numpy(mel_ctx).float(), dbfs

# Extract training pool
pos_pairs = []
neg_pairs = []

print(f"Extracting causal pairs from {len(train_manifest)} base training files...")
for idx, r in train_manifest.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists(): continue
    cat = r.get("category", "unknown")
    is_pos = (r["label"] == "explosion")
    
    try:
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        if sr != TARGET_SR:
            mono = preprocessor.resample(mono, orig_sr=sr)
        if len(mono) < WINDOW_SAMPLES:
            mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)))
            
        n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
        max_wins = 8 if not is_pos else 12
        step = max(1, n_w // max_wins) if n_w > max_wins else 1
        
        for i in range(0, n_w, step)[:max_wins]:
            t_start = i * HOP_SAMPLES
            ev_spec, ctx_spec, dbfs = extract_causal_mels(mono, t_start)
            
            if not is_pos and dbfs < SQUELCH_DBFS and cat != "pre_event_ambient_control":
                continue
                
            pair = {
                "spec_ev": ev_spec,
                "spec_ctx": ctx_spec,
                "dbfs": dbfs,
                "category": cat,
                "is_pos": 1 if is_pos else 0,
                "filename": p.name
            }
            if is_pos:
                pos_pairs.append(pair)
            else:
                neg_pairs.append(pair)
    except Exception:
        continue

print(f"Extracted Base Pairs: {len(pos_pairs)} pos, {len(neg_pairs)} neg")

# Add Codex approved additions
if len(codex_df) > 0:
    print(f"Extracting causal pairs from {len(codex_df)} approved Codex training additions...")
    codex_pos = 0
    codex_neg = 0
    for idx, r in codex_df.iterrows():
        p = CODEX_DIR / r["audio_path"]
        if not p.exists(): continue
        cat = r["category"]
        is_pos = (cat == "surface_high_explosive")
        try:
            sig, sr, dur, ch = load_audio_file(p)
            mono = preprocessor.to_mono(sig)
            if sr != TARGET_SR:
                mono = preprocessor.resample(mono, orig_sr=sr)
            if len(mono) < WINDOW_SAMPLES:
                mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)))
                
            n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
            max_wins = 6 if not is_pos else 10
            step = max(1, n_w // max_wins) if n_w > max_wins else 1
            
            for i in range(0, n_w, step)[:max_wins]:
                t_start = i * HOP_SAMPLES
                ev_spec, ctx_spec, dbfs = extract_causal_mels(mono, t_start)
                
                if not is_pos and dbfs < SQUELCH_DBFS and cat != "pre_event_ambient_control":
                    continue
                    
                pair = {
                    "spec_ev": ev_spec,
                    "spec_ctx": ctx_spec,
                    "dbfs": dbfs,
                    "category": cat,
                    "is_pos": 1 if is_pos else 0,
                    "filename": p.name
                }
                if is_pos:
                    pos_pairs.append(pair)
                    codex_pos += 1
                else:
                    neg_pairs.append(pair)
                    codex_neg += 1
        except Exception:
            continue
    print(f"Added Codex Pairs: {codex_pos} pos, {codex_neg} neg")

print(f"\nTotal Causal Training Pool: {len(pos_pairs)} pos, {len(neg_pairs)} neg")

# Save pool
out_path = PROJECT_ROOT / "training" / "data" / "causal_context_training_pool.pt"
torch.save({
    "pos_pairs": pos_pairs,
    "neg_pairs": neg_pairs
}, out_path)
print(f"Saved Causal Context Pool to: {out_path}")
