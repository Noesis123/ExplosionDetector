import numpy as np
import pandas as pd
import torch
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

ckpt = torch.load("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
pre = AudioPreprocessor()

remaining_11 = [
    "data/hard_negatives/gunfire/bbc_gunfire---_07034194.mp3",
    "data/phase15_approved/gunfire/fs_213925_Diboz.mp3",
    "data/phase15_approved/gunfire/fs_427597_michorvath.mp3",
    "data/phase15_approved/artillery_firing/fs_161341_qubodup.mp3",
    "data/phase15_approved/artillery_firing/fs_171475_2887679652.mp3",
    "data/phase15_approved/artillery_firing/fs_184275_qubodup.mp3",
    "data/phase15_approved/vehicle_backfires/fs_105351_CeebFrack.mp3",
    "data/phase15_approved/vehicle_backfires/fs_350672_Dominik_W.mp3",
    "data/phase13_approved/thunder/fs_36044_inchadney.mp3",
    "data/phase13_approved/thunder/fs_195529_klankbeeld.mp3",
    "data/phase13_approved/fireworks/fs_410347_inspectorj.mp3",
]

print("Analyzing Triggering Windows of Remaining 11 False Positives:")
print(f"{'Filename':<30} | {'Cat':<12} | {'p':<6} | {'RMS':<7} | {'Crest':<6} | {'LF':<6} | {'HF':<6} | {'SubBass':<8} | {'Centroid':<8} | {'RollOff':<8}")
print("-" * 115)

for fpath in remaining_11:
    p = ROOT / fpath
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.resample(pre.to_mono(sig), sr) if sr != 22050 else pre.to_mono(sig)
    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    cat = Path(fpath).parent.name
    best_p = 0.0
    best_info = {}

    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            if dbfs < -38.0:
                continue
            spec = pre.compute_log_mel_spectrogram(w)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
            p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())
            if p_exp >= 0.70 and p_exp > best_p:
                peak = float(np.max(np.abs(w)))
                crest = peak / max(rms, 1e-9)
                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
                tot_e = np.sum(fft_mag**2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
                sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)]**2) / tot_e)
                centroid = float(np.sum(freqs * fft_mag**2) / tot_e)
                cum_e = np.cumsum(fft_mag**2)
                rolloff_idx = np.searchsorted(cum_e, 0.85 * tot_e)
                rolloff = freqs[min(rolloff_idx, len(freqs)-1)]

                best_p = p_exp
                best_info = {
                    "p": p_exp, "dbfs": dbfs, "crest": crest, "lf": lf, "hf": hf,
                    "sub_bass": sub_bass, "centroid": centroid, "rolloff": rolloff
                }

    if best_info:
        print(f"{p.name[:30]:<30} | {cat:<12} | {best_info['p']:<6.3f} | {best_info['dbfs']:<7.1f} | {best_info['crest']:<6.2f} | {best_info['lf']:<6.3f} | {best_info['hf']:<6.3f} | {best_info['sub_bass']:<8.3f} | {best_info['centroid']:<8.1f} | {best_info['rolloff']:<8.1f}")

