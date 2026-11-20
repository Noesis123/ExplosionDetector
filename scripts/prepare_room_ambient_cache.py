import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter

FILES = [
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]

OUT_FILE = ROOT / "training" / "data" / "phase24_room_ambient_cache.pt"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

ambient_records = []
total_windows = 0

for fpath in FILES:
    raw_audio, sr, dur, ch = load_audio_file(fpath)
    mono = preprocessor.to_mono(raw_audio)
    res = preprocessor.resample(mono, orig_sr=sr)
    # Also extract with 35 Hz highpass and without 35 Hz highpass to represent both conditions
    filt = hp_filter.process(res)
    
    win_len = 22050
    hop_len = int(0.25 * 22050)
    n_w = (len(filt) - win_len) // hop_len + 1
    
    file_windows = []
    for i in range(n_w):
        # Extract both filtered (streaming) and raw (file) windows
        w_filt = filt[i*hop_len : i*hop_len + win_len]
        w_raw = res[i*hop_len : i*hop_len + win_len]
        
        spec_filt = preprocessor.compute_log_mel_spectrogram(w_filt)
        spec_raw = preprocessor.compute_log_mel_spectrogram(w_raw)
        
        file_windows.append({
            "window_idx": i,
            "time_sec": i * 0.25,
            "spec_filt": torch.from_numpy(spec_filt).float(),
            "spec_raw": torch.from_numpy(spec_raw).float(),
        })
        total_windows += 2
        
    ambient_records.append({
        "filename": fpath.name,
        "source_group": "physical_room_ac_01",
        "category": "room_ambient_ac",
        "binary_label": 0,
        "windows": file_windows
    })

torch.save({
    "source_group": "physical_room_ac_01",
    "description": "Approved Phase 24 Room Ambient Recordings (Rec 7, 8, 9)",
    "file_records": ambient_records,
    "total_window_pairs": len(file_windows) * len(FILES)
}, OUT_FILE)

print(f"Saved Phase 24 Room Ambient Cache to {OUT_FILE}")
print(f"  Files: {len(ambient_records)}")
print(f"  Windows per file: {[len(r['windows']) for r in ambient_records]}")
print(f"  Total spectrogram windows: {total_windows}")
