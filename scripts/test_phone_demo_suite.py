import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.microphone_stream import RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
sos = signal.butter(2, 250.0, btype="highpass", fs=22050, output="sos")

files = [
    "data/positives/alban_gogh-dynamite-sound-effect-205859.mp3",
    "data/positives/11325622-big-cine-boom-sound-effect-245851.mp3",
    "data/positives/scottishperson-sound-effect-ww2-uk-bomb-blast-164351.mp3",
    "data/positives/ribhavagrawal-bomb-explosion-type-01-265510.mp3",
    "data/positives/soundreality-explosion-fx-343683.mp3",
    "data/positives/alex_jauk-slow-explosion-190430.mp3",
]

print("=" * 70)
print("PHONE PLAYBACK RECEPTION TEST AT -16 dBFS RMS (0.5m PHONE DISTANCE)")
print("=" * 70)

for f in files:
    sig, sr, dur, ch = load_audio_file(f)
    mono = engine.preprocessor.resample(engine.preprocessor.to_mono(sig), sr)
    phone = signal.sosfilt(sos, mono)
    scaled = phone * (10**(-16.0/20.0) / max(np.sqrt(np.mean(phone**2)), 1e-9))
    engine.reset_state()
    engine.inject_audio_data(scaled, 22050)
    summ = engine.tracker.get_summary()
    alerts = summ["initial_alerts_count"]
    dups = summ["suppressed_duplicates_count"]
    status = "DETECTED" if alerts >= 1 else "MISSED"
    print(f"  {Path(f).name[:38]:<38}: {status} (Alerts={alerts}, Dups={dups})")
print("=" * 70)
