import sqlite3
import json
import urllib.request
from pathlib import Path

# 1. Check server API status
try:
    with urllib.request.urlopen("http://127.0.0.1:8767/api/status") as r:
        st = json.loads(r.read().decode("utf-8"))
        print("=== SERVER STATUS ===")
        for k in ["monitoring_active", "state", "audio_energy_dbfs", "windows_processed", "total_explosions_detected", "total_duplicates_suppressed", "settings"]:
            print(f"  {k}: {st.get(k)}")
except Exception as e:
    print("Server status err:", e)

# 2. Check API alerts
try:
    with urllib.request.urlopen("http://127.0.0.1:8767/api/alerts") as r:
        alerts = json.loads(r.read().decode("utf-8"))
        print(f"\n=== API ALERTS ({len(alerts)}) ===")
        for a in alerts[-10:]:
            print(" ", a)
except Exception as e:
    print("API alerts err:", e)

# 3. Check SQLite DB if exists
for db_path in Path(".").glob("**/*.db"):
    print("Found db:", db_path)
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        print("  Tables:", tables)
        for t in tables:
            tname = t[0]
            cur.execute(f"SELECT count(*) FROM {tname};")
            cnt = cur.fetchone()[0]
            print(f"    {tname} row count: {cnt}")
            if cnt > 0:
                cur.execute(f"SELECT * FROM {tname} ORDER BY rowid DESC LIMIT 5;")
                rows = cur.fetchall()
                for rw in rows:
                    print("      ", rw)
        conn.close()
    except Exception as e:
        print("  DB read error:", e)

