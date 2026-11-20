import sqlite3
import json

conn = sqlite3.connect("reports/alerts_history.db")
cur = conn.cursor()
cur.execute("SELECT id, timestamp, alert_type, confidence, category, audio_energy_dbfs, source, details_json FROM alerts WHERE timestamp >= '2026-11-19' ORDER BY id ASC;")
rows = cur.fetchall()
print(f"Total alerts today: {len(rows)}")
for r in rows:
    print(r)
conn.close()
