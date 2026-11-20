import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.microphone_stream import (
    RealtimeInferenceEngine,
    DEFAULT_CHECKPOINT,
)

print("=" * 70)
print("TEST B: NORMAL CONVERSATION & SPEECH EVALUATION")
print("=" * 70)

manifest_path = ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
speech_df = manifest_df[manifest_df["category"].str.contains("speech|shout", case=False, na=False)]
print(f"Found {len(speech_df)} speech/conversation files in manifest")

test_db = Path("reports/test_speech_alerts.db")
if test_db.exists():
    test_db.unlink()

engine = RealtimeInferenceEngine(
    checkpoint_path=DEFAULT_CHECKPOINT,
    db_path=test_db,
    device_id=0,
    use_simulation=False,
)

chunk_size = 2048
speech_alerts = 0

for idx, r in speech_df.iterrows():
    p = ROOT / r["file_path"]
    if not p.exists():
        continue
    sig, sr, dur, ch = load_audio_file(p)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=-1)
    
    engine.reset_state()
    engine.state = "TESTING"
    
    for i in range(0, len(sig), chunk_size):
        chunk = sig[i : i + chunk_size]
        engine.on_audio_chunk(chunk, sr)
        
    if engine.total_explosions_detected > 0:
        speech_alerts += 1
        print(f"  [ALERT on Speech] File: {p.name} (alerts: {engine.total_explosions_detected})")

print(f"\nTEST B RESULTS:")
print(f"  Total Speech Files Evaluated: {len(speech_df)}")
print(f"  False Alerts on Speech: {speech_alerts}")
print(f"  Status: {'PASS (0 Speech Alerts)' if speech_alerts == 0 else 'FAIL'}")
print("=" * 70)

