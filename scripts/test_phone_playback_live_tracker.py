import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

prep = AudioPreprocessor(sample_rate=22050)
ckpt_path = Path("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

# Phone speaker acoustic simulation: 2nd order Butterworth high-pass at 250 Hz (sub-bass cut of phone transducers)
sos_phone = signal.butter(2, 250.0, btype="highpass", fs=22050, output="sos")

demo_files = [
    "data/positives/alex_jauk-slow-explosion-190430.mp3",
    "data/positives/bbc_demolition_07022507.mp3",
    "data/positives/soundreality-explosion-fx-343683.mp3",
    "data/positives/scottishperson-sound-effect-ww2-uk-bomb-blast-164351.mp3",
    "data/positives/ribhavagrawal-bomb-explosion-type-01-265510.mp3",
]

print("=" * 70)
print("PHONE PLAYBACK RECEPTION TEST VIA PRODUCTION AEV TRACKER")
print("=" * 70)

all_detected = True
for fpath_str in demo_files:
    p = Path(fpath_str)
    if not p.exists():
        continue
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)

    # Filter with phone speaker roll-off (<250Hz cut)
    phone_mono = signal.sosfilt(sos_phone, mono).astype(np.float32)
    # Scale to typical phone playback level into laptop mic (~ -18 dBFS peak)
    phone_mono = phone_mono * (0.125 / (np.max(np.abs(phone_mono)) + 1e-6))

    tracker = AcousticVerifiedEventTracker(
        squelch_dbfs=-34.0,
        conf_threshold=0.70,
        demo_threshold=0.70,
    )

    nw = max(1, (len(phone_mono) - 22050) // 5512 + 1)
    file_alerted = False
    best_p = 0.0

    for i in range(nw):
        w = phone_mono[i * 5512 : i * 5512 + 22050]
        rms = float(np.sqrt(np.mean(w**2) + 1e-12))
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        peak_val = float(np.max(np.abs(w)))
        crest = peak_val / (rms + 1e-9)

        if dbfs < -34.0:
            dec = tracker.process_window_with_acoustics(
                time_sec=i * 0.25, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0, peak=peak_val
            )
        else:
            spec = prep.compute_log_mel_spectrogram(w)
            tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
            p_exp = float(probs[1])
            if p_exp > best_p:
                best_p = p_exp

            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
            tot_e = np.sum(fft_mag ** 2) + 1e-9
            lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
            sub_bass = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e)

            sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
            sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
            subframe_range = max(sub4_rms) - min(sub4_rms)

            dec = tracker.process_window_with_acoustics(
                time_sec=i * 0.25,
                rms_dbfs=dbfs,
                p_nt=float(probs[0]),
                p_exp=p_exp,
                p_fw=0.0,
                peak=peak_val,
                lf_ratio=lf,
                hf_ratio=hf,
                crest=crest,
                sub_bass_ratio=sub_bass,
                subframe_range=subframe_range,
            )

        if dec.get("action") == "initial_alert":
            file_alerted = True

    status = "DETECTED" if file_alerted else "MISSED"
    if not file_alerted:
        all_detected = False
    print(f"  {p.name[:38]:<38} : {status} (Peak p_exp={best_p:.4f})")

print("=" * 70)
print(f"Result: {'ALL 5 DETECTED' if all_detected else 'SOME MISSED'}")
