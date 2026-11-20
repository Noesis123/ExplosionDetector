"""Comprehensive Acoustic Feature and Representation Investigation.

Investigates physical and spectral characteristics of genuine explosions vs:
1. Thunder (rolling thunder and close thunderclaps)
2. Vehicle exhaust backfires and engines
3. Gunfire (pistol, rifle, burst)
4. Artillery firing (propellant boom) vs Artillery impact detonations
5. Industrial and mechanical impacts (doors, construction)
6. Fireworks and pyrotechnic aerial bursts
7. Speech and impulsive ambient sounds

Measures:
- Attack time (Trise: time from onset to peak)
- Peak dBFS, RMS dBFS, Crest factor (dB)
- Spectral centroid (Hz) and spectral bandwidth (Hz)
- Low-frequency (< 250 Hz), Mid-frequency (250-2000 Hz), High-frequency (> 4000 Hz) energy ratios
- Spectral flux (frame-to-frame log-Mel transition)
- Pre-event context energy ratio: Energy(t-2s to t) vs Energy(t to t+1s)
- Pre-event harmonicity / crest stability (distinguishing engine revs vs sudden transients)
- ExpF Block 4 128-d embedding distance
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
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

TARGET_SR = 22050
WINDOW_SAMPLES = 22050  # 1.0 s
CONTEXT_SAMPLES = 44100 # 2.0 s preceding
HOP_SAMPLES = 5512      # 0.25 s
SQUELCH_DBFS = -45.0

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

# Load ExpF model
ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

# Hook into Block 4 to extract 128-dimensional latent embeddings
block4_features = {}
def hook_fn(module, input, output):
    block4_features["feat"] = output.squeeze().detach().cpu()
model.block4.register_forward_hook(hook_fn)

def compute_detailed_acoustics(sig, trigger_start_idx):
    """Computes comprehensive acoustic properties for event window and preceding 2-s context."""
    # Event window (1.0 s)
    w = sig[trigger_start_idx : trigger_start_idx + WINDOW_SAMPLES]
    if len(w) < WINDOW_SAMPLES:
        w = np.pad(w, (0, WINDOW_SAMPLES - len(w)), mode="constant")
        
    # Preceding context (2.0 s immediately before trigger_start_idx)
    ctx_start = max(0, trigger_start_idx - CONTEXT_SAMPLES)
    ctx = sig[ctx_start : trigger_start_idx]
    if len(ctx) < CONTEXT_SAMPLES:
        # Prepend zeros if cold start
        ctx = np.pad(ctx, (CONTEXT_SAMPLES - len(ctx), 0), mode="constant")
        
    # First 1.0 s of context [t-2s to t-1s] and second 1.0 s of context [t-1s to t0]
    ctx_sec1 = ctx[:WINDOW_SAMPLES]
    ctx_sec2 = ctx[WINDOW_SAMPLES:]
    
    # RMS & Peak
    rms_event = float(np.sqrt(np.mean(w ** 2)))
    rms_event_db = 20.0 * math.log10(max(rms_event, 1e-9))
    peak_event = float(np.max(np.abs(w)))
    peak_event_db = 20.0 * math.log10(max(peak_event, 1e-9))
    crest_event_db = peak_event_db - rms_event_db
    
    # Pre-event RMS
    rms_ctx1 = float(np.sqrt(np.mean(ctx_sec1 ** 2)))
    rms_ctx1_db = 20.0 * math.log10(max(rms_ctx1, 1e-9))
    rms_ctx2 = float(np.sqrt(np.mean(ctx_sec2 ** 2)))
    rms_ctx2_db = 20.0 * math.log10(max(rms_ctx2, 1e-9))
    
    # Pre-event to Event Energy Ratio (Delta dB)
    pre_to_event_delta_db = rms_event_db - max(rms_ctx1_db, rms_ctx2_db)
    
    # Attack time Trise (time in ms from onset to peak within event window)
    peak_idx = np.argmax(np.abs(w))
    # Threshold at 10% of peak amplitude
    onset_thresh = 0.10 * peak_event
    onset_indices = np.where(np.abs(w[:peak_idx+1]) >= onset_thresh)[0]
    if len(onset_indices) > 0:
        onset_idx = onset_indices[0]
        trise_ms = float((peak_idx - onset_idx) / TARGET_SR * 1000.0)
    else:
        trise_ms = float(peak_idx / TARGET_SR * 1000.0)
        
    # Spectral properties of event window
    fft_mag = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
    tot_e = np.sum(fft_mag ** 2) + 1e-12
    
    # Frequency energy bands
    lf_sub250 = float(np.sum(fft_mag[freqs < 250] ** 2) / tot_e)
    mid_250_2k = float(np.sum(fft_mag[(freqs >= 250) & (freqs < 2000)] ** 2) / tot_e)
    hf_above4k = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
    
    # Spectral Centroid & Bandwidth
    centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-12)))
    
    # Log-Mel Spectrogram & Spectral Flux
    mel = preprocessor.compute_log_mel_spectrogram(w)
    # Spectral flux: average L2 norm of difference between adjacent time frames
    flux = float(np.mean(np.linalg.norm(np.diff(mel, axis=1), axis=0)))
    
    # ExpF Inference and Embedding
    t_in = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(t_in)
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        p_exp = float(probs[1])
        emb = block4_features["feat"].numpy().copy()
        
    return {
        "rms_event_db": rms_event_db,
        "crest_event_db": crest_event_db,
        "trise_ms": trise_ms,
        "lf_sub250": lf_sub250,
        "mid_250_2k": mid_250_2k,
        "hf_above4k": hf_above4k,
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "flux": flux,
        "rms_ctx1_db": rms_ctx1_db,
        "rms_ctx2_db": rms_ctx2_db,
        "pre_to_event_delta_db": pre_to_event_delta_db,
        "p_exp": p_exp,
        "emb": emb
    }

# Gather representative files across all 7 target & non-target categories
manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
m_df = pd.read_csv(manifest_path)

# Add Codex additions
codex_additions_path = PROJECT_ROOT / "reports" / "codex_training_additions.csv"
codex_df = pd.read_csv(codex_additions_path) if codex_additions_path.exists() else pd.DataFrame()

categories_to_analyze = [
    ("Explosion: Controlled Demolition", m_df[(m_df["category"] == "controlled_demolition") & (m_df["label"] == "explosion")]),
    ("Explosion: Military Ordnance", m_df[(m_df["category"] == "military_ordnance") & (m_df["label"] == "explosion")]),
    ("Explosion: Close Blast", m_df[(m_df["category"] == "close_explosion") & (m_df["label"] == "explosion")]),
    ("Explosion: Distant Blast", m_df[(m_df["category"] == "distant_explosion") & (m_df["label"] == "explosion")]),
    ("Non-Target: Thunder (Rolling & Claps)", m_df[m_df["category"] == "thunder"]),
    ("Non-Target: Vehicle Backfires & Engines", m_df[m_df["category"].isin(["vehicle_backfires", "traffic_engines"])]),
    ("Non-Target: Gunfire", m_df[m_df["category"] == "gunfire"]),
    ("Non-Target: Artillery Firing (Muzzle)", m_df[m_df["category"] == "artillery_firing"]),
    ("Non-Target: Mechanical / Doors", m_df[m_df["category"].isin(["doors_slamming_falling", "metal_glass_impacts", "construction_impacts"])]),
    ("Non-Target: Fireworks & Aerial Bursts", m_df[m_df["category"].isin(["aerial_burst", "fireworks", "barrage"])]),
    ("Non-Target: Speech & Shouting", m_df[m_df["category"].isin(["speech_conversation", "shouting_screaming"])])
]

records_summary = []
all_embs = []
all_labels = []

print("Analyzing acoustic features across categories...")
for cat_name, df_subset in categories_to_analyze:
    print(f"\nProcessing {cat_name} ({len(df_subset)} candidate files)...")
    cat_stats = []
    
    count = 0
    for idx, r in df_subset.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        if not p.exists():
            continue
        try:
            sig, sr, dur, ch = load_audio_file(p)
            mono = preprocessor.to_mono(sig)
            if sr != TARGET_SR:
                mono = preprocessor.resample(mono, orig_sr=sr)
            if len(mono) < WINDOW_SAMPLES:
                continue
                
            # Find the peak energy window
            n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
            best_idx = 0
            best_rms = -100.0
            for i in range(n_w):
                w = mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES]
                rms = float(20.0 * math.log10(max(np.sqrt(np.mean(w ** 2)), 1e-9)))
                if rms > best_rms:
                    best_rms = rms
                    best_idx = i * HOP_SAMPLES
                    
            if best_rms < -45.0:
                continue
                
            feats = compute_detailed_acoustics(mono, best_idx)
            cat_stats.append(feats)
            all_embs.append(feats["emb"])
            all_labels.append(cat_name)
            count += 1
            if count >= 15: # 15 representative files per category
                break
        except Exception as e:
            continue
            
    if cat_stats:
        avg_rms = np.mean([s["rms_event_db"] for s in cat_stats])
        avg_crest = np.mean([s["crest_event_db"] for s in cat_stats])
        avg_trise = np.mean([s["trise_ms"] for s in cat_stats])
        avg_lf = np.mean([s["lf_sub250"] for s in cat_stats])
        avg_mid = np.mean([s["mid_250_2k"] for s in cat_stats])
        avg_hf = np.mean([s["hf_above4k"] for s in cat_stats])
        avg_cent = np.mean([s["centroid_hz"] for s in cat_stats])
        avg_bw = np.mean([s["bandwidth_hz"] for s in cat_stats])
        avg_flux = np.mean([s["flux"] for s in cat_stats])
        avg_pre_delta = np.mean([s["pre_to_event_delta_db"] for s in cat_stats])
        avg_p_exp = np.mean([s["p_exp"] for s in cat_stats])
        
        records_summary.append({
            "category": cat_name,
            "count": len(cat_stats),
            "rms_db": avg_rms,
            "crest_db": avg_crest,
            "trise_ms": avg_trise,
            "lf_sub250": avg_lf,
            "mid_250_2k": avg_mid,
            "hf_above4k": avg_hf,
            "centroid_hz": avg_cent,
            "bandwidth_hz": avg_bw,
            "flux": avg_flux,
            "pre_delta_db": avg_pre_delta,
            "p_exp": avg_p_exp
        })

df_res = pd.DataFrame(records_summary)
print("\n" + "=" * 120)
print(f"{'Category':<35} | {'N':<3} | {'Trise(ms)':<9} | {'Crest(dB)':<9} | {'LF(<250)':<8} | {'Mid(2k)':<8} | {'Centroid':<8} | {'PreDelta(dB)':<12} | {'ExpF P(exp)'}")
print("-" * 120)
for idx, r in df_res.iterrows():
    print(f"{r['category']:<35} | {r['count']:<3} | {r['trise_ms']:9.1f} | {r['crest_db']:9.1f} | {r['lf_sub250']:8.3f} | {r['mid_250_2k']:8.3f} | {r['centroid_hz']:8.0f} | {r['pre_delta_db']:12.1f} | {r['p_exp']:7.3f}")
print("=" * 120)

# Save to reports
report_csv = PROJECT_ROOT / "reports" / "acoustic_feature_investigation_summary.csv"
df_res.to_csv(report_csv, index=False)
print(f"\nAcoustic feature summary saved to: {report_csv}")
