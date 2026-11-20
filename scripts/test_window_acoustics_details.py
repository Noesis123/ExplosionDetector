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
from inference.tracker_verified import AcousticVerifiedEventTracker

prep = AudioPreprocessor(sample_rate=22050)
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
bench_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

ckpt_f = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(sd_f)
model_f.eval()

# Let's inspect each of the 11 false positives in detail during streaming window processing:
fps_to_check = [
    "fs_161341_qubodup.mp3",
    "fs_171475_2887679652.mp3",
    "fs_184275_qubodup.mp3",
    "fs_410347_inspectorj.mp3",
    "bbc_gunfire---_07034194.mp3",
    "bbc_gunfire---_07034201.mp3",
    "fs_213925_Diboz.mp3",
    "fs_36044_inchadney.mp3",
    "fs_195529_klankbeeld.mp3",
    "fs_105351_CeebFrack.mp3",
    "fs_350672_Dominik_W.mp3",
]

fragile_positives = [
    "bbc_demolition_07022507.mp3",
    "bbc_demolition_07022508.mp3",
    "fs_323316_alfrodou.mp3",
    "fs_367622_fxkid2.mp3",
]

print("=" * 80)
print("ANALYZING TRIGGER WINDOWS FOR 11 FPS AND FRAGILE POSITIVES")
print("=" * 80)

def analyze_file(fname):
    matches = bench_df[bench_df["file_path"].str.contains(fname)]
    if matches.empty:
        return
    row = matches.iloc[0]
    p = PROJECT_ROOT / row["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    
    triggers = []
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        t_sec = i * (5512 / 22050)
        rms = np.sqrt(np.mean(w**2) + 1e-12)
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        peak = float(np.max(np.abs(w)))
        crest = peak / max(rms, 1e-9)
        
        if dbfs < -34.0:
            continue
            
        spec = prep.compute_log_mel_spectrogram(w)
        tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            p_exp = float(torch.softmax(model_f(tin), dim=-1)[0, 1])
            
        if p_exp < 0.70:
            continue
            
        # Frequency band decomposition of the triggering window
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
        
        # Intra-window decay ratio: 2nd half energy / 1st half energy
        e_half1 = np.sum(w[:11025] ** 2) + 1e-9
        e_half2 = np.sum(w[11025:] ** 2) + 1e-9
        decay_ratio_in_win = float(e_half2 / e_half1)
        
        triggers.append({
            "win_idx": i,
            "t_sec": t_sec,
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
    return triggers

print("\n--- 11 FALSE POSITIVES TRIGGER WINDOWS ---")
for fn in fps_to_check:
    trigs = analyze_file(fn)
    if trigs:
        t0 = trigs[0]
        print(f"FP {fn[:28]:<28} | p={t0['p_exp']:.3f} | RMS={t0['dbfs']:.1f} | LF={t0['lf']:.3f} | HF={t0['hf']:.3f} | Mid={t0['mid1k_4k']:.3f} | Sub80={t0['sub20_80']:.3f} | SubRng={t0['sub_rng']:.1f}dB | DecWin={t0['decay_win']:.2f}")

print("\n--- 4 FRAGILE GENUINE POSITIVES TRIGGER WINDOWS ---")
for fn in fragile_positives:
    trigs = analyze_file(fn)
    if trigs:
        t0 = trigs[0]
        print(f"TP {fn[:28]:<28} | p={t0['p_exp']:.3f} | RMS={t0['dbfs']:.1f} | LF={t0['lf']:.3f} | HF={t0['hf']:.3f} | Mid={t0['mid1k_4k']:.3f} | Sub80={t0['sub20_80']:.3f} | SubRng={t0['sub_rng']:.1f}dB | DecWin={t0['decay_win']:.2f}")
