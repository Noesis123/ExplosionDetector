import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import scipy.signal as signal
from audio.file_loader import load_audio_file
from audio.microphone_stream import RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
p = Path("data/positives/soundreality-explosion-fx-343683.mp3")
sig, sr, dur, ch = load_audio_file(p)
mono = engine.preprocessor.to_mono(sig)
if sr != 22050:
    mono = engine.preprocessor.resample(mono, orig_sr=sr)
sos_phone = signal.butter(2, 250.0, btype="highpass", fs=22050, output="sos")
phone_mono = signal.sosfilt(sos_phone, mono).astype(np.float32)
phone_mono = phone_mono * (0.125 / (np.max(np.abs(phone_mono)) + 1e-6))

# Let's inspect tracker.process_window_with_acoustics
orig_proc = engine.tracker.process_window_with_acoustics

def debug_proc(time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak, lf_ratio, hf_ratio, crest=1.0, sub_bass_ratio=0.0, onset_surge=0.0, subframe_range=0.0):
    res = orig_proc(
        time_sec=time_sec,
        rms_dbfs=rms_dbfs,
        p_nt=p_nt,
        p_exp=p_exp,
        p_fw=p_fw,
        peak=peak,
        lf_ratio=lf_ratio,
        hf_ratio=hf_ratio,
        crest=crest,
        sub_bass_ratio=sub_bass_ratio,
        onset_surge=onset_surge,
        subframe_range=subframe_range,
    )
    if rms_dbfs >= -45.0:
        print(f"t={time_sec:.2f} | RMS={rms_dbfs:.1f} | p={p_exp:.3f} | LF={lf_ratio:.3f} | HF={hf_ratio:.3f} | Crest={crest:.2f} | Range={subframe_range:.1f} | Surge={onset_surge:.1f} | Action={res.get('action')} | Reason={res.get('reason')}")
    return res

engine.tracker.process_window_with_acoustics = debug_proc
engine.reset_state()
engine.inject_audio_data(phone_mono, 22050)
print("Total initial alerts:", len(engine.tracker.initial_alerts))
