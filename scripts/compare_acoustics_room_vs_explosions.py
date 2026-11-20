import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"

preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

model = ExplosionBinaryCNN()
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

# Select test files
AMBIENT_FILES = [
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]

# Find benchmark explosions
manifest_path = ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
pos_files = manifest_df[(manifest_df["label"] == "explosion") & (manifest_df["split"].isin(["val", "holdout", "test", "challenge"]))]

TARGET_POS = [
    "bbc_demolition_07022508.mp3",
    "bbc_demolition_07022507.mp3",
    "fs_367622_fxkid2.mp3",
    "fs_323316_alfrodou.mp3",
    "fs_14615_dave_welch.mp3"
]

all_paths = [("Room Ambient", p) for p in AMBIENT_FILES]
for name in TARGET_POS:
    match = pos_files[pos_files["file_path"].str.contains(name)]
    if not match.empty:
        all_paths.append(("Genuine Explosion", ROOT / match.iloc[0]["file_path"]))

print(f"{'Group':<18} {'File':<28} {'Win':<4} {'p_exp':<7} {'FiltRMS':<8} {'LF':<6} {'HF':<6} {'SubRng':<7} {'Crest':<6} {'Top1_Hz':<8} {'Top1_Share':<10} {'Flatness':<9}")
print("-" * 125)

for grp, fpath in all_paths:
    if not fpath.exists():
        print(f"Missing: {fpath}")
        continue
    raw_audio, sr, dur, ch = load_audio_file(fpath)
    mono_audio = preprocessor.to_mono(raw_audio)
    resampled_audio = preprocessor.resample(mono_audio, orig_sr=sr)
    filtered = hp_filter.process(resampled_audio)
    
    win_len = 22050
    hop_len = int(0.25 * 22050)
    num_windows = max(1, (len(filtered) - win_len) // hop_len + 1)
    
    # Find peak window by p_exp
    best_p = -1.0
    best_stats = {}
    
    for i in range(num_windows):
        window = filtered[i*hop_len : i*hop_len + win_len]
        if len(window) < win_len:
            window = np.pad(window, (0, win_len - len(window)))
        rms_lin = float(np.sqrt(np.mean(window ** 2)))
        filt_rms = 20.0 * np.log10(max(rms_lin, 1e-9))
        peak = float(np.max(np.abs(window)))
        
        spec = preprocessor.compute_log_mel_spectrogram(window)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(model(spec_tensor), dim=1).squeeze().numpy()
            p_exp = float(probs[1])
            
        fft_mag = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), d=1.0 / 22050.0)
        tot_e = np.sum(fft_mag[freqs >= 20.0] ** 2) + 1e-9
        lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
        hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
        crest = peak / max(rms_lin, 1e-9)
        
        sub_len = len(window) // 4
        sub_rms = [float(np.sqrt(np.mean(window[s*sub_len:(s+1)*sub_len]**2))) for s in range(4)]
        sub_db = [20.0 * np.log10(max(r, 1e-9)) for r in sub_rms]
        subframe_range = max(sub_db) - min(sub_db)
        
        # Spectral peaks and concentration
        band_mask = (freqs >= 35.0) & (freqs <= 1000.0)
        band_freqs = freqs[band_mask]
        band_mags = fft_mag[band_mask]
        top_idx = np.argmax(band_mags)
        top_hz = band_freqs[top_idx]
        
        # Energy share of top 3 Hz bin around the peak
        peak_bin_e = np.sum(fft_mag[(freqs >= top_hz - 2.0) & (freqs <= top_hz + 2.0)] ** 2)
        top1_share = peak_bin_e / tot_e
        
        pos_mag = np.maximum(fft_mag[freqs >= 35.0], 1e-9)
        flatness = np.exp(np.mean(np.log(pos_mag))) / np.mean(pos_mag)
        
        if p_exp > best_p:
            best_p = p_exp
            best_stats = {
                "win": i,
                "p_exp": p_exp,
                "filt_rms": filt_rms,
                "lf": lf_ratio,
                "hf": hf_ratio,
                "sub_rng": subframe_range,
                "crest": crest,
                "top_hz": top_hz,
                "top1_share": top1_share,
                "flatness": flatness
            }
            
    s = best_stats
    print(f"{grp:<18} {fpath.name:<28} {s['win']:<4} {s['p_exp']:6.3f} {s['filt_rms']:7.1f} {s['lf']:5.3f} {s['hf']:5.3f} {s['sub_rng']:6.2f} {s['crest']:5.2f} {s['top_hz']:6.1f}Hz {s['top1_share']*100:8.2f}% {s['flatness']:8.4f}")
