import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)

# Inspect two files: one with P=0.0000 and one with P=0.9239
files_to_inspect = [
    "explosion/shared_075_INL_20220811_03_1637622024_explosion.wav",
    "explosion/shared_281_NNSS_1637610019_explosion.wav",
    "explosion/shared_103_NNSS_1637610022_explosion.wav"
]

print("=" * 75)
print("INSPECTING CODEX EXPLOSION FILES WINDOW-BY-WINDOW")
print("=" * 75)

for fpath in files_to_inspect:
    p = CODEX_DIR / fpath
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    print(f"\nFile: {p.name} | Original SR: {sr} | Dur: {dur:.2f}s | Channels: {ch}")
    tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
    n_windows = max(1, (len(mono) - 22050) // 5512 + 1)

    for i in range(n_windows):
        w = mono[i * 5512 : i * 5512 + 22050]
        win_time = i * (5512 / 22050)
        rms = np.sqrt(np.mean(w**2) + 1e-12)
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        peak = float(np.max(np.abs(w)))

        if dbfs < -45.0:
            dec = tracker.process_window_with_acoustics(win_time, dbfs, 1.0, 0.0, 0.0, peak, 1.0, 0.0)
            if i < 5 or i % 10 == 0:
                print(f"  t={win_time:.2f}s: RMS={dbfs:.1f} dBFS < Squelch (-45) -> SQUELCHED")
        else:
            spec = prep.compute_log_mel_spectrogram(w)
            t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(t_in)
                probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
            p_exp = float(probs[1])

            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

            dec = tracker.process_window_with_acoustics(win_time, dbfs, float(probs[0]), p_exp, 0.0, peak, lf, hf)
            print(f"  t={win_time:.2f}s: RMS={dbfs:.1f} dBFS | p={p_exp:.4f} | LF={lf:.3f} | HF={hf:.3f} | Action={dec.get('action')}, Reason={dec.get('reason')}")
