import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

prep = AudioPreprocessor(sample_rate=22050)
ckpt_path = Path("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

manifest_df = pd.read_csv("data/manifests/dataset_manifest_phase21_expanded.csv")
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

# Target files: 22 TPs and 12 FPs
target_files = [
    # 12 FPs
    'bbc_gunfire---_07034194.mp3', 'bbc_gunfire---_07034201.mp3', 'fs_410347_inspectorj.mp3',
    'fs_36044_inchadney.mp3', 'fs_195529_klankbeeld.mp3', 'fs_161341_qubodup.mp3',
    'fs_171475_2887679652.mp3', 'fs_184275_qubodup.mp3', 'fs_213925_Diboz.mp3',
    'fs_105351_CeebFrack.mp3', 'fs_350672_Dominik_W.mp3', 'p19_shouting_screaming_3_halgrimm_44e0bfe0.mp3',
    # Demolition and weak explosions
    'bbc_demolition_07022508.mp3', 'bbc_demolition_07022507.mp3', 'fs_367622_fxkid2.mp3',
    'scottishperson-sound-effect-ww2-uk-bomb-blast-164351.mp3'
]

results = []

for idx, r in eval_df.iterrows():
    p = Path(r["file_path"])
    fname = p.name
    if fname not in target_files:
        continue
    is_pos = (r["label"] == "explosion")
    cat = r.get("category", "unknown")
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = float(np.sqrt(np.mean(w**2) + 1e-12))
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        if dbfs < -42.0:
            continue
        spec = prep.compute_log_mel_spectrogram(w)
        tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
        p_exp = float(probs[1])
        if p_exp < 0.70:
            continue
            
        peak_val = float(np.max(np.abs(w)))
        crest = peak_val / (rms + 1e-9)
        
        fft_mag = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
        tot_e = np.sum(fft_mag ** 2) + 1e-9
        
        # Spectral bands
        sub_bass_e = np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e
        mid_bass_e = np.sum(fft_mag[(freqs >= 200) & (freqs < 1000)] ** 2) / tot_e
        mid_freq_e = np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e
        hf_e = np.sum(fft_mag[freqs >= 4000] ** 2) / tot_e
        lf_e = sub_bass_e + mid_bass_e
        
        # Spectral Centroid
        centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))
        
        # Spectral Rolloff (85% of energy below this freq)
        cum_e = np.cumsum(fft_mag ** 2)
        rolloff_idx = np.where(cum_e >= 0.85 * tot_e)[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 11025.0
        
        results.append({
            "fname": fname,
            "is_pos": is_pos,
            "category": cat,
            "t": i * 0.25,
            "p_exp": p_exp,
            "dbfs": dbfs,
            "crest": crest,
            "lf": lf_e,
            "sub_bass": sub_bass_e,
            "mid_bass": mid_bass_e,
            "mid_freq": mid_freq_e,
            "hf": hf_e,
            "centroid": centroid,
            "rolloff": rolloff
        })

df_res = pd.DataFrame(results)
print(f"Total qualifying windows analyzed: {len(df_res)}")

print("\n--- Summary by file (first qualifying window) ---")
for fname, grp in df_res.groupby("fname"):
    row = grp.iloc[0]
    kind = "TP" if row["is_pos"] else "FP"
    print(f"[{kind}] {fname:<35} | p={row['p_exp']:.3f} | LF={row['lf']:.2f} | SubB={row['sub_bass']:.2f} | MidB={row['mid_bass']:.2f} | MidF={row['mid_freq']:.2f} | HF={row['hf']:.3f} | Cent={row['centroid']:.0f}Hz | Roll={row['rolloff']:.0f}Hz | Crest={row['crest']:.1f}")
