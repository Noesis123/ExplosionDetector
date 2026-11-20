import urllib.request
import json
import time

def post(endpoint, data=None):
    url = f"http://127.0.0.1:8767{endpoint}"
    payload = json.dumps(data).encode("utf-8") if data else b""
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def get(endpoint):
    url = f"http://127.0.0.1:8767{endpoint}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())

print("1. Clearing alert database...")
post("/api/alerts/clear")

print("2. Checking initial status...")
st = get("/api/status")
print(f"   Initial Alerts: {st['total_explosions_detected']}")

ambient_files = [
    "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec07.m4a",
    "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec08.m4a",
    "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec09.m4a",
]

for af in ambient_files:
    print(f"3. Injecting ambient file: {af}...")
    res = post("/api/test/inject_audio", {"file_path": af})
    print(f"   Inject response: {res['status']}, samples={res.get('samples')}, dur={res.get('duration_sec')}s")
    # Wait for injection to complete
    time.sleep(2.0)
    alerts_data = get("/api/alerts")
    print(f"   Total alerts in DB after {af}: {alerts_data['total']}")

alerts_after_ambient = get("/api/alerts")["total"]
print(f"--> Total alerts after all 3 ambient recordings: {alerts_after_ambient} (Expected: 0)")

print("\n4. Injecting genuine demolition explosion: bbc_demolition_07022508.mp3...")
res = post("/api/test/inject_audio", {"file_path": "data/positives/bbc_demolition_07022508.mp3"})
print(f"   Inject response: {res['status']}, samples={res.get('samples')}, dur={res.get('duration_sec')}s")
time.sleep(2.5)
alerts_data = get("/api/alerts")
print(f"--> Total alerts after demolition blast: {alerts_data['total']} (Expected >= 1)")
for a in alerts_data["alerts"]:
    print(f"    Alert ID={a['id']} Type={a['alert_type']} Conf={a['confidence']:.3f} RMS={a['audio_energy_dbfs']:.1f}dBFS Time={a['timestamp']}")

if alerts_after_ambient == 0 and alerts_data['total'] >= 1:
    print("\n[SUCCESS] Server API test verified: 0 ambient alerts, genuine explosion successfully detected!")
else:
    print("\n[FAILURE] Criteria not met!")
