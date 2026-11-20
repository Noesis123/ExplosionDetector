import time
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.microphone_stream import (
    RealtimeInferenceEngine,
    DEFAULT_CHECKPOINT,
)

print("=" * 70)
print("TEST A: 5-MINUTE CONTINUOUS LIVE PHYSICAL MICROPHONE MONITORING")
print("=" * 70)

test_db = Path("reports/live_test_5min_alerts.db")
if test_db.exists():
    test_db.unlink()

engine = RealtimeInferenceEngine(
    checkpoint_path=DEFAULT_CHECKPOINT,
    db_path=test_db,
    device_id=0,
    use_simulation=False,
)

alerts = []
def alert_callback(payload):
    if payload.get("active_alert"):
        alerts.append(payload)
        rec = payload.get("active_alert_record")
        print(f"\n[ALERT DETECTED] at {payload.get('active_alert_timestamp')}: {rec}")

engine.subscribe(alert_callback)

print("Starting physical microphone monitoring (Windows WaveIn native)...")
engine.start_monitoring()

total_duration_sec = 300  # 5 minutes
t_start = time.time()
last_report = t_start

try:
    while time.time() - t_start < total_duration_sec:
        time.sleep(1.0)
        now = time.time()
        elapsed = now - t_start
        if now - last_report >= 30.0:
            last_report = now
            st = engine.get_status()
            print(f"[{elapsed:5.1f}s / {total_duration_sec}s] "
                  f"Energy: {st['audio_energy_dbfs']:5.1f} dBFS | "
                  f"Windows: {st['windows_processed']:5d} | "
                  f"Conf: {st['latest_confidence']:.3f} ({st['latest_prediction']}) | "
                  f"Alerts: {st['total_explosions_detected']} | "
                  f"Latency: {st['latency_ms']:.1f}ms")
finally:
    engine.stop_monitoring()

total_elapsed = time.time() - t_start
final_status = engine.get_status()
print("\n" + "=" * 70)
print("TEST A COMPLETE — SUMMARY:")
print(f"  Total Duration: {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)")
print(f"  Windows Processed: {final_status['windows_processed']}")
print(f"  Total Dispatched Alerts: {final_status['total_explosions_detected']}")
print(f"  Duplicates / Noise Suppressed: {final_status['total_duplicates_suppressed']}")
print(f"  Status: {'PASS (0 False Alerts)' if final_status['total_explosions_detected'] == 0 else 'FAIL'}")
print("=" * 70)

