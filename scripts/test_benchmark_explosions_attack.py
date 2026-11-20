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
from inference.tracker_verified import AcousticVerifiedEventTracker

manifest_path = ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

ckpt = torch.load("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
pre = AudioPreprocessor()

print(f"Testing acoustic attack verification across {len(eval_df)} benchmark files...")

# Let's inspect each explosion file and check what its attack characteristics are on the triggering window
exp_files = eval_df[eval_df["label"] == "explosion"]

for idx, r in exp_files.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050:
        mono = pre.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    tracker = AcousticVerifiedEventTracker(squelch_dbfs=-38.0, conf_threshold=0.70)
    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    
    alerted = False
    first_alert_win = None
    first_alert_info = {}

    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            crest = peak / max(rms, 1e-9)

            if dbfs < -38.0:
                d = tracker.process_window_with_acoustics(i*0.25, dbfs, 1.0, 0.0, crest=1.0)
            else:
                spec = pre.compute_log_mel_spectrogram(w)
                inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
                p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())

                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
                tot_e = np.sum(fft_mag[freqs >= 20.0]**2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
                sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)]**2) / tot_e)

                sub4 = [w[k*5512 : (k+1)*5512] for k in range(4)]
                sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf**2))), 1e-9)) for sf in sub4]
                sub_range = max(sub4_rms) - min(sub4_rms)

                d = tracker.process_window_with_acoustics(
                    i*0.25, dbfs, 1.0 - p_exp, p_exp,
                    peak=peak, lf_ratio=lf, hf_ratio=hf, crest=crest,
                    sub_bass_ratio=sub_bass, subframe_range=sub_range
                )

                if d.get("action") == "initial_alert" and not alerted:
                    alerted = True
                    first_alert_win = i
                    first_alert_info = {
                        "p": round(p_exp, 3), "dbfs": round(dbfs, 1),
                        "sub_range": round(sub_range, 2), "crest": round(crest, 2)
                    }

    status = "DETECTED" if alerted else "MISSED"
    print(f"{p.name[:32]:<32} | {status:<8} | Win {str(first_alert_win):<4} | {first_alert_info}")

