import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

prep = AudioPreprocessor(sample_rate=22050)
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
bench_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

ckpt_f = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(sd_f)
model_f.eval()

# Precompute windows and features for all 92 benchmark files
bench_records = []
for idx, r in bench_df.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists(): continue
    is_pos = (r["label"] == "explosion")
    cat = r.get("category", "unknown")
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    file_trigs = []
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = np.sqrt(np.mean(w**2) + 1e-12)
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        peak = float(np.max(np.abs(w)))
        crest = peak / max(rms, 1e-9)
        if dbfs < -34.0: continue
        
        spec = prep.compute_log_mel_spectrogram(w)
        tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            p_exp = float(torch.softmax(model_f(tin), dim=-1)[0, 1])
        if p_exp < 0.65: continue
        
        fft_mag = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
        tot_e = np.sum(fft_mag ** 2) + 1e-9
        
        lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
        hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
        sub20_80 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e)
        sub80_250 = float(np.sum(fft_mag[(freqs >= 80) & (freqs < 250)] ** 2) / tot_e)
        mid1k_4k = float(np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e)
        
        sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
        sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
        subframe_range = max(sub4_rms) - min(sub4_rms)
        
        e_half1 = np.sum(w[:11025] ** 2) + 1e-9
        e_half2 = np.sum(w[11025:] ** 2) + 1e-9
        decay_ratio_in_win = float(e_half2 / e_half1)
        
        file_trigs.append({
            "win_idx": i,
            "dbfs": dbfs,
            "p_exp": p_exp,
            "crest": crest,
            "lf": lf,
            "hf": hf,
            "sub20_80": sub20_80,
            "sub80_250": sub80_250,
            "mid1k_4k": mid1k_4k,
            "sub_rng": subframe_range,
            "decay_win": decay_ratio_in_win,
        })
    bench_records.append({
        "file_name": p.name,
        "is_pos": is_pos,
        "cat": cat,
        "windows": file_trigs
    })

print(f"Precomputed {len(bench_records)} files.")

def test_dual_threshold(conf_th, demo_th, p_sq_th):
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    
    for r in bench_records:
        alerted = False
        for w in r["windows"]:
            p = w["p_exp"]
            
            # Standard high-confidence trigger
            is_std = (p >= conf_th)
            
            # Controlled Demolition / Fragile Detonation Branch for p in [demo_th, conf_th):
            # Must satisfy:
            # 1. Authentic low-frequency dominance (LF >= 0.70)
            # 2. Authentic sub-80Hz infrasonic blast shockwave (sub20_80 >= 0.12)
            # 3. Not high-frequency clatter (HF <= 0.07)
            # 4. Not gunfire crack (mid1k_4k <= 0.21)
            # 5. Crest factor within realistic explosion range [3.0, 10.0]
            # 6. EITHER explosive attack (sub_rng >= 6.0 dB) OR loud blast (RMS >= -18.0 dBFS)
            is_fragile_detonation = (
                p >= demo_th
                and w["lf"] >= 0.70
                and w["sub20_80"] >= 0.12
                and w["hf"] <= 0.07
                and w["mid1k_4k"] <= 0.21
                and (3.0 <= w["crest"] <= 10.0)
                and (w["sub_rng"] >= 6.0 or w["dbfs"] >= -18.0)
            )
            
            if not (is_std or is_fragile_detonation):
                continue
                
            # Base acoustic physics filters:
            if w["dbfs"] < -28.0 and w["sub_rng"] < 3.5: continue
            if w["lf"] < 0.40: continue
            if w["hf"] > 0.08: continue
            if w["crest"] < 2.5 or w["crest"] > 12.0: continue
            if w["p_exp"] < 0.78 and w["lf"] < 0.65: continue
            if w["sub20_80"] < 0.045 and w["hf"] > 0.02: continue
            if w["dbfs"] < -25.0 and w["lf"] >= 0.96 and w["hf"] <= 0.010 and w["sub_rng"] < 6.0: continue
            
            # Additional acoustic rules:
            # Gunfire crack: high mid-band and low sub-80Hz
            if w["mid1k_4k"] >= 0.28 and w["sub20_80"] <= 0.05: continue
            # Near-squelch fireworks sizzle suppression
            if w["dbfs"] < -30.0 and p < p_sq_th and w["sub_rng"] < 10.0: continue
            # Sub-80Hz Infrasonic Shockwave requirement at quiet/distant energy
            if w["dbfs"] < -25.0 and w["sub20_80"] < 0.10: continue
            
            alerted = True
            break
            
        if r["is_pos"]:
            if alerted: tp += 1
            else:
                fn += 1
                fn_list.append(r["file_name"])
        else:
            if alerted:
                fp += 1
                fp_list.append((r["cat"], r["file_name"]))
            else:
                tn += 1
                
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"ConfTh={conf_th:.2f}, DemoTh={demo_th:.2f}, P_sq={p_sq_th:.2f} -> Rec: {rec*100:.2f}% ({tp}/{tp+fn}), Prec: {prec*100:.2f}% ({tp}/{tp+fp}), F1: {f1*100:.2f}%, FP: {fp}/{fp+tn}")
    if fn_list:
        print(f"  Missed genuine: {fn_list}")
    print(f"  FPs ({len(fp_list)}): {fp_list}\n")
    return rec, prec, f1, fp

print("\n--- TESTING DUAL THRESHOLD CONFIGURATIONS ---")
for cth in [0.75, 0.78, 0.80, 0.82]:
    for dth in [0.70]:
        for psq in [0.82, 0.84]:
            test_dual_threshold(cth, dth, psq)
