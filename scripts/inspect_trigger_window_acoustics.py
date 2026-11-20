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
from model_binary import ExplosionBinaryCNN

preprocessor = AudioPreprocessor(sample_rate=22050)
ckpt = torch.load(EXP_DIR / "checkpoints/candidate_expD_recall_boosted_best.pt", map_location="cpu", weights_only=False)
model = ExplosionBinaryCNN(freeze_bn=True)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(sd)
model.eval()

manifest_df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

with open(EXP_DIR / "all_candidates_benchmark_results.json", "r") as f:
    bench_data = json.load(f)

per_file = bench_data["per_file_records"]

print(f"{'Type':<8} | {'Filename':<38} | {'Cat':<16} | {'t_sec':<6} | {'p_exp':<6} | {'RMS_dB':<7} | {'Crest_dB':<9} | {'HF_Ratio_Win'}")
print("-" * 115)

for r in per_file:
    fname = r["file_name"]
    is_exp = (r["binary_ground_truth"] == 1)
    pred_pos = r["results"]["binary_d_070"]["predicted_positive"]
    
    if is_exp or pred_pos:
        ftype = "EXP-TP" if (is_exp and pred_pos) else ("EXP-MISS" if is_exp else "FP")
        m_row = manifest_df[manifest_df["file_path"].str.contains(fname)]
        if len(m_row) == 0: continue
        p = LAB_DIR / m_row.iloc[0]["file_path"]
        if not p.exists(): p = PROD_DIR / m_row.iloc[0]["file_path"]
        
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        res = preprocessor.resample(mono, sr)
        
        n_w = max(1, (len(res) - 22050) // 5512 + 1)
        for i in range(n_w):
            w = res[i*5512 : i*5512 + 22050]
            if len(w) < 22050: w = np.pad(w, (0, 22050 - len(w)))
            rms = float(np.sqrt(np.mean(w ** 2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            if dbfs < -45.0: continue
            
            mel = preprocessor.compute_log_mel_spectrogram(w)
            t_mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                p_e = float(torch.softmax(model(t_mel), dim=-1)[0, 1])
                
            if p_e >= 0.70:
                t_sec = i * 0.25
                peak = float(np.max(np.abs(w)))
                crest_db = 20.0 * math.log10(max(peak, 1e-9)) - dbfs
                
                # Window FFT
                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
                tot_e = np.sum(fft_mag ** 2) + 1e-9
                hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
                
                print(f"{ftype:<8} | {fname:<38} | {r['category']:<16} | {t_sec:6.2f} | {p_e:6.3f} | {dbfs:7.1f} | {crest_db:9.1f} | {hf_ratio:8.4f}")
                break # only first trigger window
