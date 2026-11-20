import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.file_loader import load_audio_file
from audio.microphone_stream import RealtimeInferenceEngine

files_to_test = [
    ("Explosion Demolition 07", "data/positives/bbc_demolition_07022507.mp3", 1),
    ("Explosion Demolition 08", "data/positives/bbc_demolition_07022508.mp3", 1),
    ("Ambient Recording 7", "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec07.m4a", 0),
    ("Ambient Recording 8", "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec08.m4a", 0),
    ("Ambient Recording 9", "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec09.m4a", 0),
]

engine = RealtimeInferenceEngine()

all_passed = True
for name, fpath, expected_alerts in files_to_test:
    engine.reset_state()
    sig, sr, dur, ch = load_audio_file(fpath)
    engine.inject_audio_data(sig, sr)
    summary = engine.tracker.get_summary()
    alerts = summary['initial_alerts_count']
    passed = (alerts >= 1 if expected_alerts > 0 else alerts == 0)
    status_str = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status_str}] {name} ({fpath}): Expected {expected_alerts}, Got {alerts} alerts (Rejected: {engine.tracker.total_acoustically_rejected})")

print(f"\nOverall result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
