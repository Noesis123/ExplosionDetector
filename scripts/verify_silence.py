import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from audio.microphone_stream import RealtimeInferenceEngine

engine = RealtimeInferenceEngine()
engine.reset_state()
silence = np.zeros(22050 * 10, dtype=np.float32)
engine.inject_audio_data(silence, 22050)
summ = engine.tracker.get_summary()
p_exp = engine.latest_probabilities["explosion"]
print(f"Digital Silence Test (10s): Alerts={summ['initial_alerts_count']}, Latest p_exp={p_exp:.4f}")
assert summ['initial_alerts_count'] == 0, "FAIL: Alerts triggered on silence!"
assert p_exp == 0.0, "FAIL: Non-zero p_exp on silence!"
print("PASS: Exactly zero alerts on confirmed digital silence.")
