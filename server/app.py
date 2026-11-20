"""Lightweight Real-Time Explosion Detection Backend Server.

Implemented using Python standard library http.server (zero third-party dependencies).
Provides:
- Static HTML5/CSS3/Vanilla-JS dashboard serving on GET /
- REST endpoints for start/stop monitoring, hardware device enumeration, and alert history
- Real-time Audio File Test Injection endpoint (POST /api/test/inject_audio)
- Server-Sent Events (SSE) stream for sub-second live dashboard updates on GET /api/stream
- SQLite alert database integration
- Strictly binary explosion detection: 'explosion' vs 'non_explosion'
"""

import argparse
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.microphone_stream import (
    RealtimeInferenceEngine,
    list_audio_input_devices,
    DEFAULT_CHECKPOINT,
)
from inference.alert_db import AlertDatabase

STATIC_DIR = Path(__file__).resolve().parent / "static"


class DetectionServerState:
    """Singleton server state containing the inference engine and active SSE clients."""

    def __init__(self, checkpoint_path: Optional[Path] = None, db_path: Optional[Path] = None):
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self.engine = RealtimeInferenceEngine(
            checkpoint_path=ckpt,
            db_path=db_path,
            device_id=0,
            use_simulation=False,
        )
        self.sse_queues: List[queue.Queue] = []
        self._lock = threading.RLock()

        # Hook engine notifications to broadcast into all SSE client queues
        self.engine.subscribe(self.broadcast_event)

    def broadcast_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            for q in list(self.sse_queues):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def add_sse_client(self) -> queue.Queue:
        q = queue.Queue(maxsize=30)
        with self._lock:
            self.sse_queues.append(q)
        return q

    def remove_sse_client(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self.sse_queues:
                self.sse_queues.remove(q)


GLOBAL_STATE: Optional[DetectionServerState] = None


class DetectionAPIRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST API, SSE streaming, and static UI assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    @property
    def state(self) -> DetectionServerState:
        global GLOBAL_STATE
        if GLOBAL_STATE is None:
            GLOBAL_STATE = DetectionServerState()
        return GLOBAL_STATE

    def log_message(self, format, *args):
        # Quiet standard logging to avoid console spam during SSE streaming
        pass

    def _send_json_response(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. API: System Status
        if path == "/api/status":
            self._send_json_response(self.state.engine.get_status())
            return

        # 2. API: Audio Devices
        if path == "/api/devices":
            devs = list_audio_input_devices()
            self._send_json_response({"devices": devs, "selected_id": self.state.engine.device_id})
            return

        # 3. API: Latest Prediction
        if path == "/api/prediction/latest":
            st = self.state.engine.get_status()
            self._send_json_response({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monitoring_active": st["monitoring_active"],
                "prediction": st["latest_prediction"],
                "confidence": st["latest_confidence"],
                "probabilities": st["latest_probabilities"],
                "audio_energy_dbfs": st["audio_energy_dbfs"],
                "active_alert": st["active_alert"],
                "active_alert_timestamp": st["active_alert_timestamp"],
                "device_name": st["device_name"],
                "latency_ms": st["latency_ms"],
                "model_name": st.get("model_name", "ExplosionBinaryCNN"),
                "model_version": st.get("model_version", "ExpF + AEV (Active Production)"),
                "model_checkpoint": st.get("model_checkpoint", "candidate_expF_accuracy_optimized_best.pt"),
                "waveform": st.get("waveform", []),
            })
            return

        # 4. API: Alerts History
        if path == "/api/alerts":
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])
            alert_type = query.get("alert_type", [None])[0]
            search = query.get("search", [None])[0]
            start_date = query.get("start_date", [None])[0]
            end_date = query.get("end_date", [None])[0]
            alerts = self.state.engine.alert_db.get_alerts(
                limit=limit,
                offset=offset,
                alert_type=alert_type,
                search=search,
                start_date=start_date,
                end_date=end_date,
            )
            total = self.state.engine.alert_db.count_alerts(
                alert_type=alert_type,
                search=search,
                start_date=start_date,
                end_date=end_date,
            )
            self._send_json_response({"total": total, "alerts": alerts})
            return

        # 4b. API: Export Alerts as CSV
        if path == "/api/alerts/export":
            import csv
            import io
            alert_type = query.get("alert_type", [None])[0]
            search = query.get("search", [None])[0]
            start_date = query.get("start_date", [None])[0]
            end_date = query.get("end_date", [None])[0]
            alerts = self.state.engine.alert_db.get_alerts(
                limit=10000,
                offset=0,
                alert_type=alert_type,
                search=search,
                start_date=start_date,
                end_date=end_date,
            )
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["ID", "Timestamp_UTC", "Alert_Type", "Confidence", "Audio_Energy_dBFS", "Category", "Source", "Device", "Status"])
            for a in alerts:
                writer.writerow([
                    a["id"], a["timestamp"], a["alert_type"], a["confidence"],
                    a["audio_energy_dbfs"], a["category"], a["source"], a.get("device", a["source"]), a["status"]
                ])
            csv_data = output.getvalue().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="alerts_history_export.csv"')
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data)
            return

        # 5. API: Settings
        if path == "/api/settings":
            st = self.state.engine.get_status()
            self._send_json_response(st["settings"])
            return

        # 6. API: Real-time Server-Sent Events (SSE) Stream
        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            client_q = self.state.add_sse_client()
            try:
                # Send immediate initial status event
                init_data = json.dumps(self.state.engine.get_status())
                self.wfile.write(f"data: {init_data}\n\n".encode("utf-8"))
                self.wfile.flush()

                while True:
                    try:
                        event_payload = client_q.get(timeout=1.0)
                        data_str = json.dumps(event_payload)
                        self.wfile.write(f"data: {data_str}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        # Heartbeat
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                self.state.remove_sse_client(client_q)
            return

        # 7. Static file delivery: default to index.html
        if path == "/" or path == "/index.html":
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Read JSON body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                body = json.loads(raw_body)
            except Exception:
                pass

        # 1. API: Start Monitoring
        if path == "/api/monitor/start":
            dev_id = body.get("device_id")
            sim = body.get("simulation")
            file_path = body.get("file_path")
            try:
                self.state.engine.start_monitoring(
                    device_id=dev_id,
                    use_simulation=sim,
                    simulation_file=file_path,
                )
                self._send_json_response({"status": "started", "engine": self.state.engine.get_status()})
            except Exception as e:
                self._send_json_response({"status": "error", "error": str(e)}, status=500)
            return

        # 2. API: Stop Monitoring
        if path == "/api/monitor/stop":
            self.state.engine.stop_monitoring()
            self._send_json_response({"status": "stopped", "engine": self.state.engine.get_status()})
            return

        # 3. API: Update Settings
        if path == "/api/settings":
            updated = self.state.engine.update_settings(
                explosion_threshold=body.get("explosion_threshold"),
                rms_squelch_dbfs=body.get("rms_squelch_dbfs"),
                config_preset=body.get("config_preset"),
            )
            self._send_json_response({"status": "updated", "settings": updated})
            return

        # 3b. API: Rollback / Preset Switch
        if path in ["/api/rollback", "/api/config_preset"]:
            preset = body.get("preset", body.get("config_preset", "Config C"))
            st = self.state.engine.set_config_preset(preset)
            self._send_json_response({
                "status": "success",
                "preset": self.state.engine.tracker.config_preset,
                "model_version": self.state.engine.model_version,
                "settings": st["settings"],
            })
            return

        # 4. API: Clear Alerts
        if path == "/api/alerts/clear":
            count = self.state.engine.alert_db.clear_alerts()
            self.state.engine.reset_state()
            self._send_json_response({"status": "cleared", "deleted_count": count})
            return

        # 5. API: Audio Test File Injection (Test A / Test C)
        if path == "/api/test/inject_audio":
            file_path = body.get("file_path")
            if not file_path:
                self._send_json_response({"status": "error", "error": "file_path parameter is required"}, status=400)
                return

            p = Path(file_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / file_path
            if not p.exists():
                self._send_json_response({"status": "error", "error": f"File not found: {p}"}, status=404)
                return

            if self.state.engine.state == "TESTING":
                self._send_json_response({"status": "busy", "error": "Another audio test injection is currently running. Please wait for completion."}, status=409)
                return

            try:
                from audio.file_loader import load_audio_file
                sig, sr, dur, ch = load_audio_file(p)
                sig_np = np.asarray(sig, dtype=np.float32)

                self.state.engine.state = "TESTING"

                # Run injection asynchronously in a background thread to not block HTTP response
                def run_injection():
                    self.state.engine.inject_audio_data(sig_np, native_sr=sr)

                th = threading.Thread(target=run_injection, daemon=True, name="TestInjectionThread")
                th.start()
                self._send_json_response({
                    "status": "injecting",
                    "file": p.name,
                    "samples": len(sig_np),
                    "duration_sec": round(dur, 2),
                    "native_sr": sr
                })
            except Exception as e:
                self._send_json_response({"status": "error", "error": str(e)}, status=500)
            return

        self._send_json_response({"error": "Endpoint not found"}, status=404)


class RobustThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        cls, val, tb = sys.exc_info()
        if cls in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            return
        super().handle_error(request, client_address)


def create_server(host: str = "127.0.0.1", port: int = 8767, state: Optional[DetectionServerState] = None) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer instance."""
    global GLOBAL_STATE
    if state is not None:
        GLOBAL_STATE = state
    server_address = (host, int(port))
    httpd = RobustThreadingHTTPServer(server_address, DetectionAPIRequestHandler)
    return httpd


def run_server(host: str = "127.0.0.1", port: int = 8767):
    httpd = create_server(host, port)
    print(f"[Explosion Detection Server] Running at http://{host}:{port}/")
    while True:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Explosion Detection Server] Stopping...")
            if GLOBAL_STATE:
                GLOBAL_STATE.engine.stop_monitoring()
            httpd.server_close()
            break
        except Exception as e:
            print(f"[Server Notice] Handled transient socket event: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explosion Detection Production Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host binding (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8767, help="Port binding (default: 8767)")
    args = parser.parse_args()
    run_server(args.host, args.port)
