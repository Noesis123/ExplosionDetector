import sys
from pathlib import Path
import numpy as np
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.microphone_stream import PhysicalWaveInCapture, StreamingHighpassFilter
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
import torch

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
model = ExplosionBinaryCNN()
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

preprocessor = AudioPreprocessor(sample_rate=22050)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

print("Capturing 5 seconds of physical audio from default device...")
chunks = []
def callback(chunk_np, sr):
    chunks.append(chunk_np.copy())

mic = PhysicalWaveInCapture(device_id=0, sample_rate=48000, chunk_callback=callback)
mic.start()
time.sleep(5.0)
mic.stop()

if chunks:
    audio_48k = np.concatenate(chunks)
    audio_22k = preprocessor.resample(audio_48k, orig_sr=48000)
    filtered = hp_filter.process(audio_22k)
    
    rms_raw = 20.0 * np.log10(max(np.sqrt(np.mean(audio_22k**2)), 1e-9))
    rms_filt = 20.0 * np.log10(max(np.sqrt(np.mean(filtered**2)), 1e-9))
    
    print(f"\nCaptured {len(audio_48k)} samples ({len(audio_48k)/48000:.2f}s):")
    print(f"  Raw RMS: {rms_raw:.1f} dBFS | 35Hz Filtered RMS: {rms_filt:.1f} dBFS")
    
    win_len = 22050
    hop_len = int(0.25 * 22050)
    n_w = (len(filtered) - win_len) // hop_len + 1
    print(f"  Evaluating {n_w} windows:")
    for i in range(n_w):
        w = filtered[i*hop_len : i*hop_len + win_len]
        w_raw = audio_22k[i*hop_len : i*hop_len + win_len]
        r_w = 20.0 * np.log10(max(np.sqrt(np.mean(w**2)), 1e-9))
        spec = preprocessor.compute_log_mel_spectrogram(w)
        probs = torch.softmax(model(torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()), dim=1).squeeze().detach().numpy()
        
        fft_mag = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), d=1.0/22050.0)
        tot_e = np.sum(fft_mag[freqs >= 20.0]**2) + 1e-9
        lf = np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e
        hf = np.sum(fft_mag[freqs > 4000.0]**2) / tot_e
        peak = np.max(np.abs(w))
        crest = peak / max(np.sqrt(np.mean(w**2)), 1e-9)
        
        sub_len = len(w) // 4
        sub_rms = [20.0 * np.log10(max(np.sqrt(np.mean(w[s*sub_len:(s+1)*sub_len]**2)), 1e-9)) for s in range(4)]
        sub_rng = max(sub_rms) - min(sub_rms)
        
        # Check dominant freq
        band_mask = (freqs >= 35.0) & (freqs <= 1000.0)
        top_idx = np.argmax(fft_mag[band_mask])
        top_hz = freqs[band_mask][top_idx]
        
        print(f"    Win {i:2d} ({i*0.25:4.2f}s): p_exp={probs[1]:.4f}, RMS={r_w:5.1f} dBFS, SubRng={sub_rng:4.2f} dB, LF={lf:.3f}, HF={hf:.4f}, Crest={crest:4.2f}, Top={top_hz:.1f}Hz")
else:
    print("No audio chunks captured.")
