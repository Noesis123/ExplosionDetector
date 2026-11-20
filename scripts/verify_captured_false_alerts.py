import numpy as np
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.microphone_stream import RealtimeInferenceEngine, DEFAULT_CHECKPOINT

print("=" * 70)
print("VERIFICATION STEP 1: REPLAY CAPTURED FALSE-ALERT AUDIO")
print("=" * 70)

engine = RealtimeInferenceEngine(
    checkpoint_path=DEFAULT_CHECKPOINT,
    db_path=Path("reports/test_verification_alerts.db"),
    device_id=0,
    use_simulation=False,
)

# Test Sample 1: 4.5s live diagnostic sample
sample1 = np.load("scripts/live_mic_diagnostic_sample.npy")
print(f"Sample 1: {len(sample1)} samples ({len(sample1)/22050:.2f} s)")

alerts1 = []
engine.subscribe(lambda payload: alerts1.append(payload) if payload.get("active_alert") else None)
engine.state = "TESTING"

# Feed chunks of 2048 samples (similar to physical WaveIn chunk size)
chunk_size = 2048
for i in range(0, len(sample1), chunk_size):
    chunk = sample1[i : i + chunk_size]
    engine.on_audio_chunk(chunk, 22050)

print(f"Sample 1 Results: Dispatched Alerts = {engine.total_explosions_detected}")

# Test Sample 2: 15.0s ambient sample (which previously had 27 windows with p >= 0.70)
sample2 = np.load("scripts/live_mic_15s_ambient.npy")
print(f"\nSample 2: {len(sample2)} samples ({len(sample2)/22050:.2f} s)")

engine.reset_state()
engine.total_explosions_detected = 0
engine.state = "TESTING"

for i in range(0, len(sample2), chunk_size):
    chunk = sample2[i : i + chunk_size]
    engine.on_audio_chunk(chunk, 22050)

print(f"Sample 2 Results: Dispatched Alerts = {engine.total_explosions_detected}")
print(f"Total Windows Processed: {engine.windows_processed_count}")
print(f"Total Duplicates/Rejections Suppressed: {engine.total_duplicates_suppressed}")

if engine.total_explosions_detected == 0:
    print("\n>>> SUCCESS: Exact captured false-alert audio produced ZERO false alerts! <<<")
else:
    print(f"\n>>> FAILURE: {engine.total_explosions_detected} false alerts generated! <<<")

