import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter

manifest = pd.read_csv(ROOT / "data/manifests/dataset_manifest_phase21_expanded.csv")
pos = manifest[(manifest["label"] == "explosion") & (manifest["split"].isin(["val", "holdout", "test", "challenge"]))]

preprocessor = AudioPreprocessor(sample_rate=22050)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

print(f"{'Filename':<32} {'Peak RMS (dBFS)':<15} {'Mean RMS (dBFS)':<15} {'Duration'}")
print("-" * 75)

peak_rms_list = []
for idx, r in pos.iterrows():
    p = ROOT / r["file_path"]
    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    res = preprocessor.resample(mono, orig_sr=sr)
    filt = hp_filter.process(res)
    
    # 1.0s window with 0.25s hop
    win_len = 22050
    hop_len = int(0.25 * 22050)
    n_w = max(1, (len(filt) - win_len) // hop_len + 1)
    
    win_rms = []
    for i in range(n_w):
        w = filt[i*hop_len : i*hop_len + win_len]
        r_lin = float(np.sqrt(np.mean(w**2)))
        win_rms.append(20.0 * np.log10(max(r_lin, 1e-9)))
        
    p_rms = max(win_rms)
    m_rms = np.mean(win_rms)
    peak_rms_list.append((p.name, p_rms, m_rms, dur))
    print(f"{p.name:<32} {p_rms:12.1f} dBFS {m_rms:12.1f} dBFS {dur:6.2f}s")

print(f"\nMinimum Peak RMS across any benchmark explosion: {min(x[1] for x in peak_rms_list):.1f} dBFS ({[x for x in peak_rms_list if x[1] == min(y[1] for y in peak_rms_list)][0][0]})")
