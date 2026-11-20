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

# Let's inspect the 3 fragile positives vs the 8 remaining FPs
fragile_pos = [
    "bbc_demolition_07022508.mp3",
    "fs_367622_fxkid2.mp3",
    "fs_323316_alfrodou.mp3",
]

remaining_fps = [
    "fs_161341_qubodup.mp3",        # Artillery
    "fs_171475_2887679652.mp3",     # Artillery
    "fs_184275_qubodup.mp3",        # Artillery
    "bbc_gunfire---_07034194.mp3",  # Gunfire
    "fs_213925_Diboz.mp3",          # Gunfire
    "fs_36044_inchadney.mp3",       # Thunder
    "fs_195529_klankbeeld.mp3",     # Thunder
    "fs_105351_CeebFrack.mp3",      # Backfire
]

def get_peak_trigger_window(fn):
    m = bench_df[bench_df["file_path"].str.contains(fn)]
    if m.empty: return None
    p = PROJECT_ROOT / m.iloc[0]["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    best_w = None
    max_p = 0.0
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
            
        if p_exp > max_p:
            max_p = p_exp
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            
            lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
            sub20_80 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e)
            mid1k_4k = float(np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e)
            
            sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
            sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
            subframe_range = max(sub4_rms) - min(sub4_rms)
            
            e_half1 = np.sum(w[:11025] ** 2) + 1e-9
            e_half2 = np.sum(w[11025:] ** 2) + 1e-9
            decay_win = float(e_half2 / e_half1)
            
            best_w = {
                "fn": fn,
                "dur": dur,
                "win_idx": i,
                "p_exp": p_exp,
                "rms": dbfs,
                "crest": crest,
                "lf": lf,
                "hf": hf,
                "mid": mid1k_4k,
                "sub80": sub20_80,
                "sub_rng": subframe_range,
                "decay_win": decay_win,
            }
    return best_w

print("=" * 105)
print("SIDE-BY-SIDE PHYSICAL COMPARISON: 3 FRAGILE POSITIVES vs 8 HARD NEGATIVES")
print("=" * 105)
print(f"{'FILE':<28} | {'TYPE':<4} | {'p_exp':<5} | {'RMS':<5} | {'Crest':<5} | {'LF':<5} | {'HF':<5} | {'Mid':<5} | {'Sub80':<5} | {'SubRng':<6} | {'DecWin':<6}")
print("-" * 105)

for fn in fragile_pos:
    w = get_peak_trigger_window(fn)
    if w:
        print(f"{w['fn'][:28]:<28} | {'POS':<4} | {w['p_exp']:.3f} | {w['rms']:5.1f} | {w['crest']:5.2f} | {w['lf']:.3f} | {w['hf']:.3f} | {w['mid']:.3f} | {w['sub80']:.3f} | {w['sub_rng']:6.1f} | {w['decay_win']:6.2f}")

print("-" * 105)
for fn in remaining_fps:
    w = get_peak_trigger_window(fn)
    if w:
        print(f"{w['fn'][:28]:<28} | {'NEG':<4} | {w['p_exp']:.3f} | {w['rms']:5.1f} | {w['crest']:5.2f} | {w['lf']:.3f} | {w['hf']:.3f} | {w['mid']:.3f} | {w['sub80']:.3f} | {w['sub_rng']:6.1f} | {w['decay_win']:6.2f}")
print("=" * 105)
