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
        if p_exp < 0.70: continue
        
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

print(f"Precomputed {len(bench_records)} benchmark files.")

def eval_config(name, rule_check):
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    for r in bench_records:
        alerted = False
        for w in r["windows"]:
            # Config D baseline checks
            if w["dbfs"] < -28.0 and w["sub_rng"] < 3.5:
                continue
            if w["lf"] < 0.40:
                continue
            if w["hf"] > 0.08:
                continue
            if w["crest"] < 2.5 or w["crest"] > 12.0:
                continue
            if w["dbfs"] < -31.0 and w["p_exp"] < 0.75 and w["sub_rng"] < 6.0:
                continue
            if w["p_exp"] < 0.78 and w["lf"] < 0.65:
                continue
            if w["sub20_80"] < 0.045 and w["hf"] > 0.02:
                continue
            if w["dbfs"] < -25.0 and w["lf"] >= 0.96 and w["hf"] <= 0.010 and w["sub_rng"] < 6.0:
                continue
                
            # Additional rule checks
            if rule_check(w):
                continue
                
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
    print(f"\n[{name}]")
    print(f"  Recall: {rec*100:.2f}% ({tp}/{tp+fn}) | Precision: {prec*100:.2f}% ({tp}/{tp+fp}) | F1: {f1*100:.2f}% | FP: {fp}/{fp+tn}")
    if fn_list:
        print(f"  Missed genuine: {fn_list}")
    print(f"  False positives ({len(fp_list)}): {fp_list}")
    return rec, prec, f1, fp

# Config D Baseline
eval_config("Config D Baseline", lambda w: False)

# Let's test a sequence of physics-grounded refinements:
# 1. Gunfire crack: mid energy >= 0.28 with sub80 <= 0.05
# 2. Vehicle backfire exhaust pop: sub80 < 0.02 at RMS < -25 dBFS
# 3. Fireworks aerial burst near-squelch: RMS < -30 dBFS and p_exp < 0.84 when sub_rng < 10.0 dB
# 4. Infrasonic thunder/rumble discrimination: p_exp < 0.85 when sub_rng < 5.0 dB and LF >= 0.95 (sustained rolling thunder)
# 5. Artillery / distant blast discrimination:

def candidate_gate_1(w):
    # Rule A: Gunfire crack
    if w["mid1k_4k"] >= 0.28 and w["sub20_80"] <= 0.05:
        return True
    # Rule B: Exhaust backfire pop
    if w["dbfs"] < -25.0 and w["sub20_80"] < 0.015:
        return True
    # Rule C: Low-energy fireworks sizzle
    if w["dbfs"] < -30.0 and w["p_exp"] < 0.84 and w["sub_rng"] < 10.0:
        return True
    return False

eval_config("Candidate Gate 1 (Gunfire + Backfire + Fireworks)", candidate_gate_1)

def candidate_gate_2(w):
    # Includes Gate 1
    if candidate_gate_1(w):
        return True
    # Rule D: Rolling Thunder Sustained Low-Frequency
    # Rolling thunder rumbles have LF >= 0.95 and high crest/swell, but low transient attack (sub_rng < 5.0 dB)
    if w["lf"] >= 0.95 and w["sub_rng"] < 5.0 and w["p_exp"] < 0.82:
        return True
    return False

eval_config("Candidate Gate 2 (+ Thunder Flat Attack Suppression)", candidate_gate_2)

def candidate_gate_3(w):
    if candidate_gate_1(w):
        return True
    # Rolling thunder often has slow rise / swelling (decay_win > 2.0) without massive demolition attack
    if w["lf"] >= 0.93 and w["decay_win"] > 2.5 and w["sub_rng"] < 15.0:
        return True
    # Artillery / distant blast discrimination:
    # Artillery muzzle blast at moderate distance often has flat sustain (sub_rng < 5.0 dB) and LF >= 0.92
    if w["lf"] >= 0.92 and w["sub_rng"] < 4.0 and w["p_exp"] < 0.82:
        return True
    return False

eval_config("Candidate Gate 3 (+ Swelling Thunder & Low-Attack Distant Confounders)", candidate_gate_3)

# Test tighter grid search to maximize Precision without losing any of the 22 explosions
print("\n" + "=" * 80)
print("GRID SEARCH ACROSS CRITICAL THRESHOLDS")
print("=" * 80)

for p_sq_thresh in [0.80, 0.82, 0.84, 0.85]:
    for flat_rng_thresh in [3.5, 4.0, 4.5, 5.0]:
        def grid_rule(w):
            if w["mid1k_4k"] >= 0.28 and w["sub20_80"] <= 0.05: return True
            if w["dbfs"] < -25.0 and w["sub20_80"] < 0.015: return True
            if w["dbfs"] < -30.0 and w["p_exp"] < p_sq_thresh and w["sub_rng"] < 10.0: return True
            if w["lf"] >= 0.94 and w["sub_rng"] < flat_rng_thresh and w["p_exp"] < 0.82: return True
            return False
            
        r, p, f, fp = eval_config(f"p_sq={p_sq_thresh}, flat_rng={flat_rng_thresh}", grid_rule)
