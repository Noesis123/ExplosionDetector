import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
exp_files = manifest_df[(manifest_df["label"] == "explosion") & (manifest_df["split"].isin(["val", "holdout", "test", "challenge"]))]

print("=" * 70)
print("GAIN ATTENUATION SWEEP ON ACTIVE EXPF MODEL (24 BENCHMARK EXPLOSIONS)")
print("=" * 70)

gains_db = [+6.0, 0.0, -6.0, -12.0, -18.0]
results = {}

for g in gains_db:
    factor = 10.0 ** (g / 20.0)
    detected = 0
    total = 0

    for idx, r in exp_files.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        if not p.exists():
            continue
        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.to_mono(sig)
        if sr != 22050:
            mono = prep.resample(mono, orig_sr=sr)
        if len(mono) < 22050:
            mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

        # Apply gain
        scaled = np.clip(mono * factor, -1.0, 1.0).astype(np.float32)

        tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
        n_windows = max(1, (len(scaled) - 22050) // 5512 + 1)
        alerted = False

        with torch.no_grad():
            for i in range(n_windows):
                w = scaled[i * 5512 : i * 5512 + 22050]
                win_time = i * (5512 / 22050)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))

                if dbfs < -45.0:
                    dec = tracker.process_window_with_acoustics(win_time, dbfs, 1.0, 0.0, 0.0, peak, 1.0, 0.0)
                else:
                    spec = prep.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                    p_exp = float(probs[1])

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                    dec = tracker.process_window_with_acoustics(win_time, dbfs, float(probs[0]), p_exp, 0.0, peak, lf_ratio, hf_ratio)

                if dec.get("action") == "initial_alert":
                    alerted = True

        total += 1
        if alerted:
            detected += 1

    results[g] = (detected, total)
    print(f"Gain {g:+5.1f} dB: Detected = {detected:2d}/{total} ({detected/total*100:.1f}%)")
