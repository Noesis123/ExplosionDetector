import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
PROD_DIR = Path("D:/NOESIS/Explosion/Explosion")
EXP_DIR = LAB_DIR / "experiments/binary_explosion"

if str(LAB_DIR) not in sys.path: sys.path.insert(0, str(LAB_DIR))
if str(PROD_DIR) not in sys.path: sys.path.insert(0, str(PROD_DIR))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(sample_rate=22050)

# Load benchmark results
with open(EXP_DIR / "all_candidates_benchmark_results.json", "r") as f:
    bench_data = json.load(f)

per_file = bench_data["per_file_records"]

print(f"{'Type':<10} | {'Filename':<38} | {'Category':<16} | {'Max_P':<6} | {'Peak_dBFS':<9} | {'RMS_dBFS':<9} | {'Crest_dB':<8} | {'LF_Ratio':<8} | {'HF_Ratio':<8} | {'Dur':<5}")
print("-" * 125)

def analyze_audio_features(p):
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    res = preprocessor.resample(mono, sr)
    
    # Analyze window with max energy / max peak
    win_len = 22050
    hop_len = 5512
    n_w = max(1, (len(res) - win_len) // hop_len + 1)
    
    max_peak = float(np.max(np.abs(res)))
    rms = float(np.sqrt(np.mean(res ** 2)))
    peak_db = 20.0 * math.log10(max(max_peak, 1e-9))
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    crest_db = peak_db - rms_db
    
    # FFT spectral energy distribution
    fft_mag = np.abs(np.fft.rfft(res[:min(len(res), 22050 * 3)]))
    freqs = np.fft.rfftfreq(min(len(res), 22050 * 3), d=1.0/22050)
    total_energy = np.sum(fft_mag ** 2) + 1e-9
    lf_energy = np.sum(fft_mag[freqs < 1000] ** 2) / total_energy
    hf_energy = np.sum(fft_mag[freqs > 4000] ** 2) / total_energy
    
    return peak_db, rms_db, crest_db, lf_energy, hf_energy, dur

manifest_df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

for r in per_file:
    fname = r["file_name"]
    is_exp = (r["binary_ground_truth"] == 1)
    pred_pos = r["results"]["binary_d_070"]["predicted_positive"]
    max_p = r["results"]["binary_d_070"]["max_prob"]
    
    # Only genuine explosions or false positives
    if is_exp or pred_pos:
        ftype = "EXP-TP" if (is_exp and pred_pos) else ("EXP-MISS" if is_exp else "FP")
        
        # locate file
        m_row = manifest_df[manifest_df["file_path"].str.contains(fname)]
        if len(m_row) > 0:
            p = LAB_DIR / m_row.iloc[0]["file_path"]
            if not p.exists(): p = PROD_DIR / m_row.iloc[0]["file_path"]
        else:
            continue
            
        peak_db, rms_db, crest_db, lf_e, hf_e, dur = analyze_audio_features(p)
        print(f"{ftype:<10} | {fname:<38} | {r['category']:<16} | {max_p:6.3f} | {peak_db:9.1f} | {rms_db:9.1f} | {crest_db:8.1f} | {lf_e:8.3f} | {hf_e:8.3f} | {dur:5.1f}")
