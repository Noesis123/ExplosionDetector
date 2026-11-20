import urllib.request
import json
import time

print("Monitoring live physical microphone for 30 seconds...")
for i in range(6):
    time.sleep(5.0)
    with urllib.request.urlopen("http://127.0.0.1:8767/api/status") as resp:
        r = json.loads(resp.read().decode())
    win = r.get("windows_processed")
    rms = r.get("audio_energy_dbfs")
    alerts = r.get("total_explosions_detected")
    pred = r.get("latest_prediction")
    conf = r.get("latest_confidence")
    print(f"T+{(i+1)*5:02d}s: Windows={win} RMS={rms:.1f} dBFS Alerts={alerts} Pred={pred} Conf={conf:.3f}")

with urllib.request.urlopen("http://127.0.0.1:8767/api/alerts") as resp:
    alerts_data = json.loads(resp.read().decode())
print(f"Final Alerts in DB: {alerts_data['total']}")
