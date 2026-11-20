import sys
from pathlib import Path
import numpy as np
import torch
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

ckpt_path = Path("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)

fragile_exps = [
    ("EXP", "data/positives/bbc_demolition_07022508.mp3"),
    ("EXP", "data/phase13_approved/positives/fs_367622_fxkid2.mp3"),
    ("EXP", "data/phase13_approved/positives/fs_323316_alfrodou.mp3"),
    ("EXP", "data/positives/bbc_explosions_07023354.mp3"),
]

confused_negs = [
    ("BACKFIRE", "data/phase15_approved/vehicle_backfires/fs_350672_Dominik_W.mp3"),
    ("ARTILLERY", "data/phase15_approved/artillery_firing/fs_184275_qubodup.mp3"),
    ("THUNDER", "data/phase13_approved/thunder/fs_36044_inchadney.mp3"),
    ("ARTILLERY", "data/phase15_approved/artillery_firing/fs_161341_qubodup.mp3"),
    ("GUNFIRE", "data/phase15_approved/gunfire/fs_213925_Diboz.mp3"),
    ("GUNFIRE", "data/hard_negatives/gunfire/bbc_gunfire---_07034194.mp3"),
    ("BACKFIRE", "data/phase15_approved/vehicle_backfires/fs_105351_CeebFrack.mp3"),
    ("THUNDER", "data/phase13_approved/thunder/fs_195529_klankbeeld.mp3"),
    ("FIREWORKS", "data/phase13_approved/fireworks/fs_410347_inspectorj.mp3"),
    ("GUNFIRE", "data/hard_negatives/gunfire/bbc_gunfire---_07034201.mp3"),
    ("ARTILLERY", "data/phase15_approved/artillery_firing/fs_171475_2887679652.mp3"),
]

def analyze_file(cat, fpath):
    sig, sr, dur, ch = load_audio_file(fpath)
    mono = prep.resample(prep.to_mono(sig), orig_sr=sr)
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    
    best_w = None
    best_p = 0.0
    best_idx = 0
    
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = float(20 * np.log10(np.sqrt(np.mean(w**2)) + 1e-12))
        if rms >= -34.0:
            spec = prep.compute_log_mel_spectrogram(w)
            tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
            p = float(probs[1])
            if p > best_p:
                best_p = p
                best_w = w
                best_idx = i
                
    if best_w is None:
        return
        
    w = best_w
    rms = float(20 * np.log10(np.sqrt(np.mean(w**2)) + 1e-12))
    peak = float(np.max(np.abs(w)))
    crest = peak / max(10**(rms/20), 1e-9)
    
    fft_mag = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    
    sub_20_80 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e)
    subbass_20_200 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e)
    lf_20_1000 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
    mid_1_4k = float(np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e)
    hf_4k_plus = float(np.sum(fft_mag[freqs >= 4000] ** 2) / tot_e)
    
    # Subframe dynamic range across four 250ms subframes
    sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
    sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
    sub_rng = max(sub4_rms) - min(sub4_rms)
    
    # Subframe attack: subframe 0 or 1 vs tail subframes
    sf_attack_diff = max(sub4_rms[0], sub4_rms[1]) - min(sub4_rms[2], sub4_rms[3])
    
    # Temporal decay inside the window: energy in first half vs second half
    e_first_half = np.sum(w[:11025] ** 2) + 1e-9
    e_second_half = np.sum(w[11025:] ** 2) + 1e-9
    decay_ratio_in_win = e_second_half / e_first_half

    print(f"[{cat:<9}] {Path(fpath).name[:25]:<25} t={best_idx*0.25:4.2f}s | p={best_p:.3f} RMS={rms:5.1f}dB | Sub20-80={sub_20_80:.3f} SB20-200={subbass_20_200:.3f} Mid1-4k={mid_1_4k:.3f} | SubRng={sub_rng:4.1f}dB Decay={decay_ratio_in_win:.3f} Crest={crest:4.2f}")

print("=" * 115)
print("FRAGILE GENUINE EXPLOSIONS (p in [0.70, 0.78]):")
print("=" * 115)
for c, p in fragile_exps:
    analyze_file(c, p)

print("\n" + "=" * 115)
print("HARD NEGATIVES (CONFUSED NON-EXPLOSIONS):")
print("=" * 115)
for c, p in confused_negs:
    analyze_file(c, p)
