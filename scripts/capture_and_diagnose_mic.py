import numpy as np
import pandas as pd
import torch
import math
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from audio.microphone_stream import PhysicalWaveInCapture

# 1. Capture 5 seconds of actual live physical microphone
print("Capturing 5 seconds from physical microphone...")
raw_audio_chunks = []

def chunk_cb(chunk, sr):
    raw_audio_chunks.append(chunk.copy())

cap = PhysicalWaveInCapture(sample_rate=22050, chunk_callback=chunk_cb)
cap.start()
time.sleep(5.0)
cap.stop()

if not raw_audio_chunks:
    print("ERROR: No audio chunks captured from physical microphone!")
else:
    live_audio = np.concatenate(raw_audio_chunks)
    print(f"Captured {len(live_audio)} samples ({len(live_audio)/22050:.2f} seconds)")
    
    # Save raw capture for reproducible diagnostic inspection
    np.save("scripts/live_mic_diagnostic_sample.npy", live_audio)
    
    # Analyze in 1-second sliding windows (hop 0.25s)
    ckpt = torch.load("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    pre = AudioPreprocessor()

    n_win = max(1, (len(live_audio) - 22050) // 5512 + 1)
    print(f"\nAnalyzing {n_win} windows of live microphone audio:")
    print(f"{'Win':<4} | {'RMS(dB)':<8} | {'Peak':<7} | {'Crest':<6} | {'p_exp':<7} | {'LF':<6} | {'HF':<6} | {'SubBass':<8} | {'SubRange':<8}")
    print("-" * 75)

    with torch.no_grad():
        for i in range(n_win):
            w = live_audio[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            crest = peak / max(rms, 1e-9)

            spec = pre.compute_log_mel_spectrogram(w)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
            probs = torch.softmax(model(inp), dim=-1).squeeze().numpy()
            p_exp = float(probs[1])

            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
            tot_e = np.sum(fft_mag[freqs >= 20.0]**2) + 1e-9
            lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
            sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)]**2) / tot_e)

            sub4 = [w[k*5512 : (k+1)*5512] for k in range(4)]
            sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf**2))), 1e-9)) for sf in sub4]
            sub_range = max(sub4_rms) - min(sub4_rms)

            print(f"{i:<4} | {dbfs:<8.2f} | {peak:<7.4f} | {crest:<6.2f} | {p_exp:<7.4f} | {lf:<6.3f} | {hf:<6.3f} | {sub_bass:<8.3f} | {sub_range:<8.2f}")
