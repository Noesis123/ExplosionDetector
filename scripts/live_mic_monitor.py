import sys
import time
import json
import urllib.request
from datetime import datetime

def monitor_live_test(test_name: str, duration_sec: float = 6.0):
    print("=" * 60)
    print(f"LIVE TEST: {test_name}")
    print(f"Duration: {duration_sec}s | Monitoring http://127.0.0.1:8767/api/status")
    print("=" * 60)

    # Record initial alert count
    try:
        with urllib.request.urlopen("http://127.0.0.1:8767/api/alerts") as r:
            init_alerts_data = json.loads(r.read().decode())
            init_alert_count = init_alerts_data.get("total", len(init_alerts_data.get("alerts", [])))
    except Exception as e:
        print(f"Warning: could not read initial alerts: {e}")
        init_alert_count = 0

    t_start = time.time()
    rms_values = []
    p_exp_values = []
    new_alerts = []
    seen_alert_ids = set()

    while (time.time() - t_start) < duration_sec:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8767/api/status") as r:
                st = json.loads(r.read().decode())
                rms = st.get("audio_energy_dbfs", -99.9)
                probs = st.get("latest_probabilities", {})
                p_exp = probs.get("explosion", 0.0)
                rms_values.append(rms)
                p_exp_values.append(p_exp)

            # Check alerts
            with urllib.request.urlopen("http://127.0.0.1:8767/api/alerts") as r:
                al_data = json.loads(r.read().decode())
                for a in al_data.get("alerts", []):
                    aid = a.get("id")
                    if aid not in seen_alert_ids:
                        seen_alert_ids.add(aid)
                        # Check if alert was created after test start
                        new_alerts.append(a)

        except Exception as e:
            pass
        time.sleep(0.15)

    duration_actual = time.time() - t_start
    min_rms = min(rms_values) if rms_values else -99.9
    mean_rms = sum(rms_values) / len(rms_values) if rms_values else -99.9
    max_rms = max(rms_values) if rms_values else -99.9

    min_p = min(p_exp_values) if p_exp_values else 0.0
    mean_p = sum(p_exp_values) / len(p_exp_values) if p_exp_values else 0.0
    max_p = max(p_exp_values) if p_exp_values else 0.0

    print(f"\n--- Results for {test_name} ({duration_actual:.1f}s) ---")
    print(f"Microphone RMS: Mean={mean_rms:.1f} dBFS (Min={min_rms:.1f}, Max={max_rms:.1f})")
    print(f"CNN p_exp:      Mean={mean_p:.4f} (Min={min_p:.4f}, Max={max_p:.4f})")
    print(f"Alerts Fired:   {len(new_alerts)}")
    if new_alerts:
        for idx, a in enumerate(new_alerts, 1):
            print(f"  Alert #{idx}: Type={a.get('alert_type')}, Conf={a.get('confidence')}, Energy={a.get('audio_energy_dbfs')} dBFS, Source={a.get('source')}")
    else:
        print("  Status: Clean / No Alert Fired (PASS for non-explosion)")

    return {
        "test_name": test_name,
        "duration_sec": duration_actual,
        "mean_rms": round(mean_rms, 1),
        "min_rms": round(min_rms, 1),
        "max_rms": round(max_rms, 1),
        "mean_p_exp": round(mean_p, 4),
        "max_p_exp": round(max_p, 4),
        "alerts_count": len(new_alerts),
        "new_alerts": new_alerts,
    }

if __name__ == "__main__":
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "Ambient Microphone Test"
    dur_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    monitor_live_test(test_arg, dur_arg)
