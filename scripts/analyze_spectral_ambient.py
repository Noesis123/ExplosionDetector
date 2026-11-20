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

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
FILES = [
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]

preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

for fpath in FILES:
    raw_audio, sr, dur, ch = load_audio_file(fpath)
    mono_audio = preprocessor.to_mono(raw_audio)
    resampled_audio = preprocessor.resample(mono_audio, orig_sr=sr)
    filtered = hp_filter.process(resampled_audio)
    
    # Analyze window with peak p_exp
    # Let's find top 3 FFT peak frequencies
    fft_mag = np.abs(np.fft.rfft(filtered[:22050]))
    freqs = np.fft.rfftfreq(22050, d=1.0 / 22050.0)
    
    # Top 5 peaks between 35 Hz and 500 Hz
    band_mask = (freqs >= 35.0) & (freqs <= 500.0)
    band_freqs = freqs[band_mask]
    band_mags = fft_mag[band_mask]
    top_indices = np.argsort(band_mags)[-5:][::-1]
    
    print(f"\n{fpath.name}:")
    print(f"  Dominant frequencies in 35-500 Hz band:")
    for idx in top_indices:
        print(f"    {band_freqs[idx]:.1f} Hz (magnitude: {band_mags[idx]:.2f})")
    
    # Check spectral flatness: exp(mean(ln(x))) / mean(x)
    pos_mag = np.maximum(fft_mag[freqs >= 35.0], 1e-9)
    flatness = np.exp(np.mean(np.log(pos_mag))) / np.mean(pos_mag)
    print(f"  Spectral Flatness (35-11025 Hz): {flatness:.6f}")
