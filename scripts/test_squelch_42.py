import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
FILES = [
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]

device = torch.device("cpu")
model = ExplosionBinaryCNN()
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)

for sq in [-42.0, -40.0, -38.0]:
    print(f"\n==================== SQUELCH = {sq} dBFS ====================")
    for fpath in FILES:
        raw_audio, sr, dur, ch = load_audio_file(fpath)
        mono_audio = preprocessor.to_mono(raw_audio)
        resampled_audio = preprocessor.resample(mono_audio, orig_sr=sr)
        hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)
        filtered_audio = hp_filter.process(resampled_audio)
        
        win_len = 22050
        hop_len = int(0.25 * 22050)
        num_windows = (len(filtered_audio) - win_len) // hop_len + 1
        
        tracker = AcousticVerifiedEventTracker(
            squelch_dbfs=sq,
            conf_threshold=0.70,
            min_lf_ratio=0.40,
            max_hf_ratio=0.08,
            demo_threshold=0.62,
        )
        
        alerts = []
        for i in range(num_windows):
            window = filtered_audio[i*hop_len : i*hop_len + win_len]
            time_sec = i * hop_len / 22050.0
            rms_lin = float(np.sqrt(np.mean(window ** 2)))
            filt_rms = 20.0 * np.log10(max(rms_lin, 1e-9))
            peak = float(np.max(np.abs(window)))
            
            spec = preprocessor.compute_log_mel_spectrogram(window)
            spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            
            with torch.no_grad():
                probs = torch.softmax(model(spec_tensor), dim=1).squeeze().numpy()
                p_nt, p_exp = probs[0], probs[1]
                
            fft_mag = np.abs(np.fft.rfft(window))
            freqs = np.fft.rfftfreq(len(window), d=1.0 / 22050.0)
            tot_e = np.sum(fft_mag[freqs >= 20.0] ** 2) + 1e-9
            lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
            hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
            sub_bass_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
            crest = peak / max(rms_lin, 1e-9)
            
            sub_len = len(window) // 4
            sub_rms = [float(np.sqrt(np.mean(window[s*sub_len:(s+1)*sub_len]**2))) for s in range(4)]
            sub_db = [20.0 * np.log10(max(r, 1e-9)) for r in sub_rms]
            subframe_range = max(sub_db) - min(sub_db)
            
            res = tracker.process_window_with_acoustics(
                time_sec=time_sec,
                rms_dbfs=filt_rms,
                p_nt=float(p_nt),
                p_exp=float(p_exp),
                peak=peak,
                lf_ratio=lf_ratio,
                hf_ratio=hf_ratio,
                crest=crest,
                sub_bass_ratio=sub_bass_ratio,
                subframe_range=subframe_range,
            )
            
            if res.get("alert"):
                alerts.append((i, time_sec, p_exp, filt_rms, subframe_range, res["action"], res.get("reason", "")))
                
        print(f"{fpath.name}: {len(alerts)} alerts (rejections: {tracker.total_acoustically_rejected})")
        for a in alerts:
            print(f"  Alert: Win {a[0]} t={a[1]:.2f}s p={a[2]:.3f} RMS={a[3]:.1f} SubRng={a[4]:.2f} act={a[5]} reason={a[6]}")
