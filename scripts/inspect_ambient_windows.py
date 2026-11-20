import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, torch
from audio.microphone_stream import PhysicalWaveInCapture, RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
import time; time.sleep(3.0)
cap.stop()

raw = np.concatenate(samples)
prep = engine.preprocessor
wins, ts = prep.create_sliding_windows(raw)

print(f"Total windows: {len(wins)}")
for i, (win, (t_start, t_end)) in enumerate(zip(wins, ts)):
    rms = np.sqrt(np.mean(win**2))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    peak = np.max(np.abs(win))
    
    spec = prep.compute_log_mel_spectrogram(win)
    with torch.no_grad():
        probs = torch.softmax(engine.model(torch.from_numpy(spec[None, None, :, :]).float()), dim=-1)[0]
    p_exp = probs[1].item()
    
    fft_mag = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), d=1.0/22050)
    tot_e = np.sum(fft_mag**2) + 1e-9
    lf_ratio = np.sum(fft_mag[freqs < 1000]**2) / tot_e
    hf_ratio = np.sum(fft_mag[freqs > 4000]**2) / tot_e
    
    trig = "TRIGGER!" if (p_exp >= 0.70 and rms_db >= -45.0) else ""
    print(f"Win {i:2d} ({t_start:.2f}-{t_end:.2f}s): RMS={rms_db:.1f}dB, Peak={peak:.4f}, p_exp={p_exp:.4f}, LF={lf_ratio:.3f}, HF={hf_ratio:.4f} {trig}")
