import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.signal as signal
from audio.file_loader import load_audio_file
from audio.microphone_stream import RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
test_files = [
    'data/positives/alex_jauk-slow-explosion-190430.mp3',
    'data/positives/alban_gogh-dynamite-sound-effect-205859.mp3',
    'data/positives/11325622-big-cine-boom-sound-effect-245851.mp3',
]

# Simulate phone loudspeaker: 250 Hz 4th order Butterworth highpass filter
sos = signal.butter(4, 250.0, btype='highpass', fs=22050, output='sos')

orig_proc = engine.tracker.process_window_with_acoustics

for f in test_files:
    sig, sr, dur, ch = load_audio_file(f)
    print(f"\n=== Phone Speaker Simulation for {Path(f).name} ===")
    
    # Resample to 22050 mono
    mono = engine.preprocessor.resample(engine.preprocessor.to_mono(sig), sr)
    phone_sig = signal.sosfilt(sos, mono)
    
    # Scale to phone playback level at ~1 meter (-38 dBFS peak/RMS)
    cur_rms = np.sqrt(np.mean(phone_sig**2))
    target_rms = 10**(-38.0 / 20.0) # -38 dBFS
    scaled_sig = phone_sig * (target_rms / max(cur_rms, 1e-9))
    
    engine.reset_state()
    decisions = []
    def make_hook(dec_list):
        def hook(**kwargs):
            res = orig_proc(**kwargs)
            if kwargs['p_exp'] >= 0.5 or kwargs['rms_dbfs'] >= -45.0:
                dec_list.append((kwargs['time_sec'], kwargs['rms_dbfs'], kwargs['p_exp'], kwargs.get('lf_ratio'), kwargs.get('hf_ratio'), res['action'], res.get('reason')))
            return res
        return hook
    engine.tracker.process_window_with_acoustics = make_hook(decisions)
    engine.inject_audio_data(scaled_sig, 22050)
    summary = engine.tracker.get_summary()
    print(f"Initial alerts: {summary['initial_alerts_count']}, Dups suppressed: {summary['suppressed_duplicates_count']}")
    for d in decisions:
        if d[2] >= 0.5:
            print(f"  t={d[0]:.2f}s, RMS={d[1]:.1f}dB, p_exp={d[2]:.3f}, LF={d[3]:.2f}, HF={d[4]:.2f}, action={d[5]}, reason={d[6]}")
