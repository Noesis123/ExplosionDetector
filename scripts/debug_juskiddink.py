import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

ckpt_path = Path('explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

fpath = 'data/phase13_approved/positives/fs_108641_juskiddink.mp3'
sig, sr, dur, ch = load_audio_file(fpath)
preprocessor = AudioPreprocessor(sample_rate=22050)
mono = preprocessor.to_mono(sig)
if sr != 22050:
    mono = preprocessor.resample(mono, orig_sr=sr)

tracker = AcousticVerifiedEventTracker(squelch_dbfs=-38.0, conf_threshold=0.70)

n_windows = max(1, (len(mono) - 22050) // 5512 + 1)
windows = [mono[i * 5512 : i * 5512 + 22050] for i in range(n_windows)]

for i, w in enumerate(windows):
    win_time = i * (5512 / 22050)
    rms = np.sqrt(np.mean(w**2) + 1e-12)
    dbfs = float(20.0 * np.log10(rms + 1e-12))
    peak = float(np.max(np.abs(w)))
    crest = peak / max(rms, 1e-9)
    if dbfs >= -38.0:
        spec = preprocessor.compute_log_mel_spectrogram(w)
        t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(t_in)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        p_exp = float(probs[1])
        fft_mag = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
        tot_e = np.sum(fft_mag ** 2) + 1e-9
        lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
        hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
        sub_bass_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
        sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
        sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
        subframe_range = max(sub4_rms) - min(sub4_rms)

        dec = tracker.process_window_with_acoustics(
            time_sec=win_time, rms_dbfs=dbfs, p_nt=float(probs[0]), p_exp=p_exp, p_fw=0.0,
            peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio, crest=crest,
            sub_bass_ratio=sub_bass_ratio, subframe_range=subframe_range
        )
        if p_exp >= 0.5 or dec.get('action') != 'none':
            act = dec.get('action')
            rsn = dec.get('reason')
            print(f"t={win_time:.2f}s RMS={dbfs:.1f} p_exp={p_exp:.3f} LF={lf_ratio:.3f} HF={hf_ratio:.3f} SubBass={sub_bass_ratio:.3f} SubRng={subframe_range:.1f} act={act} rsn={rsn}")
