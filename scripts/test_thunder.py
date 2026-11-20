import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import numpy as np
import scipy.signal as signal
import torch
from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

prep = AudioPreprocessor(sample_rate=22050)
model = ExplosionBinaryCNN(freeze_bn=True)
sd = torch.load('explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt', map_location='cpu', weights_only=False)
if 'model_state_dict' in sd:
    sd = sd['model_state_dict']
model.load_state_dict(sd)
model.eval()

# Find thunder files in data/
all_files = glob.glob('data/**/*.mp3', recursive=True) + glob.glob('data/**/*.wav', recursive=True)
thunder_files = [f for f in all_files if 'thunder' in f.lower()][:10]
print(f"Found {len(thunder_files)} thunder files:")

sos = signal.butter(2, 75.0, btype='highpass', fs=22050, output='sos')

alerts_count = 0
for tf in thunder_files:
    sig, sr, dur, ch = load_audio_file(tf)
    mono = prep.resample(prep.to_mono(sig), sr)
    mono_filt = signal.sosfilt(sos, mono - np.mean(mono))
    wins, ts = prep.create_sliding_windows(mono_filt)
    tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
    for w in wins:
        rms_dbfs = 20 * np.log10(max(np.sqrt(np.mean(w**2)), 1e-9))
        if rms_dbfs < -45.0:
            p_exp, p_nt, lf_ratio, hf_ratio = 0.0, 1.0, 1.0, 0.0
        else:
            spec = prep.compute_log_mel_spectrogram(w)
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(spec[None, None, :, :]).float()), dim=-1)[0]
            p_nt, p_exp = probs[0].item(), probs[1].item()
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
            tot_e = np.sum(fft_mag**2) + 1e-9
            lf_ratio = float(np.sum(fft_mag[freqs < 1000]**2) / tot_e)
            hf_ratio = float(np.sum(fft_mag[freqs > 4000]**2) / tot_e)
        tracker.process_window_with_acoustics(time_sec=0.0, rms_dbfs=rms_dbfs, p_nt=p_nt, p_exp=p_exp, lf_ratio=lf_ratio, hf_ratio=hf_ratio)
    tracker.finalize_stream()
    summ = tracker.get_summary()
    if summ['initial_alerts_count'] > 0:
        alerts_count += 1
        print(f"  ALERT on {Path(tf).name}: {summ['initial_alerts_count']} alerts")
    else:
        print(f"  REJECTED {Path(tf).name}: 0 alerts")

print(f"Thunder alerts: {alerts_count}/{len(thunder_files)}")
