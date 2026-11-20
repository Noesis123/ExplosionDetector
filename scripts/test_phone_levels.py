import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.signal as signal
from audio.file_loader import load_audio_file
from audio.microphone_stream import RealtimeInferenceEngine

# Load alex_jauk
sig, sr, dur, ch = load_audio_file('data/positives/alex_jauk-slow-explosion-190430.mp3')
engine = RealtimeInferenceEngine()
engine.tracker.demo_threshold = 0.66
mono = engine.preprocessor.resample(engine.preprocessor.to_mono(sig), sr)

# Phone filter
sos = signal.butter(4, 250.0, btype='highpass', fs=22050, output='sos')
phone_sig = signal.sosfilt(sos, mono)
scaled = phone_sig * (10**(-18.0/20.0) / max(np.sqrt(np.mean(phone_sig**2)), 1e-9))

orig_proc = engine.tracker.process_window_with_acoustics
def hook(**kwargs):
    res = orig_proc(**kwargs)
    t = kwargs["time_sec"]
    rms = kwargs["rms_dbfs"]
    p = kwargs["p_exp"]
    lf = kwargs.get("lf_ratio")
    hf = kwargs.get("hf_ratio")
    sb = kwargs.get("sub_bass_ratio")
    srng = kwargs.get("subframe_range")
    act = res.get("action")
    rsn = res.get("reason")
    if p >= 0.35:
        print(f"t={t:.2f}s RMS={rms:.1f} p={p:.3f} LF={lf:.3f} HF={hf:.3f} SubBass={sb:.3f} SubRng={srng:.1f} act={act} rsn={rsn}")
    return res

engine.tracker.process_window_with_acoustics = hook
engine.inject_audio_data(scaled, 22050)
summ = engine.tracker.get_summary()
print(f"Summary: Alerts={summ['initial_alerts_count']}, Dups={summ['suppressed_duplicates_count']}")
