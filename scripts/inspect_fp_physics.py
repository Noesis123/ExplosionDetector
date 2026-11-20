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

def analyze_file(file_path):
    sig, sr, dur, ch = load_audio_file(file_path)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    
    peak_p = 0.0
    best_w = None
    
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = np.sqrt(np.mean(w**2) + 1e-12)
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        if dbfs < -45.0:
            continue
        spec = prep.compute_log_mel_spectrogram(w)
        tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
        p_exp = float(probs[1])
        if p_exp > peak_p:
            peak_p = p_exp
            best_w = w
            
    if best_w is None:
        return None
        
    rms = float(np.sqrt(np.mean(best_w**2) + 1e-12))
    dbfs = float(20.0 * np.log10(rms + 1e-12))
    peak_val = float(np.max(np.abs(best_w)))
    crest = peak_val / (rms + 1e-9)
    
    # Rise time to peak
    peak_idx = int(np.argmax(np.abs(best_w)))
    thresh_val = 0.10 * peak_val
    pre_peak = np.abs(best_w[:peak_idx+1])
    onset_candidates = np.where(pre_peak < thresh_val)[0]
    onset_idx = onset_candidates[-1] if len(onset_candidates) > 0 else 0
    rise_time_ms = ((peak_idx - onset_idx) / 22050.0) * 1000.0
    
    fft_mag = np.abs(np.fft.rfft(best_w))
    freqs = np.fft.rfftfreq(len(best_w), d=1.0/22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
    hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
    sub_bass = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e)
    
    return {
        "peak_p": peak_p, "dbfs": dbfs, "crest": crest, "rise_ms": rise_time_ms,
        "lf": lf, "hf": hf, "sub_bass": sub_bass
    }

print(f"{'Filename':<32} | {'Category':<16} | {'p_exp':<6} | {'dBFS':<6} | {'Crest':<6} | {'Rise_ms':<8} | {'LF':<5} | {'HF':<5} | {'SubBass':<7}")
print("-" * 105)

fp_names = [
    "bbc_gunfire---_07034194.mp3", "bbc_gunfire---_07034201.mp3", "bbc_right-whal_nhu0501615.mp3",
    "bbc_transport-_07012036.mp3", "fs_410347_inspectorj.mp3", "fs_36044_inchadney.mp3",
    "fs_195529_klankbeeld.mp3", "fs_161341_qubodup.mp3", "fs_171475_2887679652.mp3",
    "fs_184275_qubodup.mp3", "fs_184650_Isaac200000.mp3", "fs_187542_qubodup.mp3",
    "fs_213925_Diboz.mp3", "fs_427597_michorvath.mp3", "fs_105351_CeebFrack.mp3",
    "fs_350672_Dominik_W.mp3", "p19_shouting_screaming_3_halgrimm_44e0bfe0.mp3",
    "p19_silence_ambience_4_frederik_sunne_0e4ca5cf.mp3", "p19_household_ambience_3_bdunis4_0be312d3.mp3"
]

print("=== 19 FALSE POSITIVE FILES ===")
for name in fp_names:
    matches = eval_df[eval_df["file_path"].str.contains(name)]
    if len(matches) == 0:
        continue
    r = matches.iloc[0]
    res = analyze_file(Path(r["file_path"]))
    if res:
        print(f"{name[:32]:<32} | {r['category'][:16]:<16} | {res['peak_p']:.4f} | {res['dbfs']:6.1f} | {res['crest']:6.2f} | {res['rise_ms']:8.2f} | {res['lf']:.3f} | {res['hf']:.3f} | {res['sub_bass']:.3f}")

print("\n=== 24 GENUINE EXPLOSIONS ===")
pos_df = eval_df[eval_df["label"] == "explosion"]
for idx, r in pos_df.iterrows():
    name = Path(r["file_path"]).name
    res = analyze_file(Path(r["file_path"]))
    if res:
        print(f"{name[:32]:<32} | {'explosion':<16} | {res['peak_p']:.4f} | {res['dbfs']:6.1f} | {res['crest']:6.2f} | {res['rise_ms']:8.2f} | {res['lf']:.3f} | {res['hf']:.3f} | {res['sub_bass']:.3f}")
