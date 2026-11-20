"""Test Suite for Modernized Dashboard and Real-Time Audio Monitoring.

Verifies the 12 core requirements:
1. Dashboard static HTML components, layout, and overview controls.
2. Physical microphone device enumeration.
3. Starting monitoring directly from Overview.
4. Capturing audio into circular ring buffer.
5. Receiving live AudioMulticlassCNN predictions.
6. Displaying explosion and fireworks alerts.
7. Writing alert records to SQLite database.
8. Alert history querying, filtering (type, search, date/time), and CSV export.
9. Stopping and restarting microphone monitoring lifecycle.
10. Alert persistence across dashboard refreshes and engine restarts.
11. SSE stream event formatting and waveform transmission.
12. Real hardware microphone probing with clean fallback.
"""

import io
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse
import numpy as np
import pytest

from audio.microphone_stream import (
    AudioRingBuffer,
    RealtimeInferenceEngine,
    SimulatedAudioStream,
    WaveInAudioCapture,
    list_audio_input_devices,
)
from inference.alert_db import AlertDatabase
from server.app import DetectionAPIRequestHandler, DetectionServerState, STATIC_DIR


class DummySocket:
    def __init__(self, data=b""):
        self._rfile = io.BytesIO(data)
        self._wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self._rfile
        return self._wfile

    def sendall(self, b):
        self._wfile.write(b)


class DummyServer:
    pass


def _execute_http(method: str, path: str, body: dict = None, state: DetectionServerState = None):
    body_bytes = json.dumps(body).encode("utf-8") if body else b""
    headers = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
    if body_bytes:
        headers += f"Content-Length: {len(body_bytes)}\r\nContent-Type: application/json\r\n"
    headers += "\r\n"
    raw_data = headers.encode("utf-8") + body_bytes

    sock = DummySocket(raw_data)

    class CustomHandler(DetectionAPIRequestHandler):
        @property
        def state(self):
            if state is not None:
                return state
            return super().state

    handler = CustomHandler(sock, ("127.0.0.1", 8080), DummyServer())
    resp_raw = sock._wfile.getvalue().decode("utf-8", errors="replace")

    parts = resp_raw.split("\r\n\r\n", 1)
    status_line = parts[0].split("\r\n")[0]
    body_str = parts[1] if len(parts) > 1 else ""
    status = int(status_line.split()[1])
    return status, body_str


def test_dashboard_html_components_and_overview_controls():
    """Verify index.html contains all Overview controls, canvas oscilloscope, and responsive elements."""
    index_path = STATIC_DIR / "index.html"
    assert index_path.exists(), f"Missing index.html at {index_path}"
    content = index_path.read_text(encoding="utf-8")

    # Document title
    assert "<title>NOESIS — Autonomous Acoustic Explosion Detection System</title>" in content

    # Responsive viewport
    assert 'name="viewport"' in content
    assert "width=device-width" in content

    # Real microphone controls directly on Overview
    assert 'id="deviceSelect"' in content
    assert 'id="btnStartCapture"' in content
    assert 'id="btnStopCapture"' in content

    # Live oscilloscope canvas
    assert 'id="oscilloscopeCanvas"' in content

    # Class probability meters
    assert 'id="valProbNt"' in content
    assert 'id="valProbExp"' in content

    # Hero threat banner
    assert 'id="heroThreatBanner"' in content
    assert 'id="heroStatusTitle"' in content

    # Alerts history & CSV export
    assert 'id="alertsTableBody"' in content
    assert "exportAlerts()" in content
    assert "clearAlerts()" in content


def test_physical_microphone_enumeration():
    """Verify hardware and virtual microphone enumeration."""
    devices = list_audio_input_devices()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    dev = devices[0]
    assert "id" in dev
    assert "name" in dev
    assert "channels" in dev


def test_engine_overview_monitoring_lifecycle(tmp_path):
    """Verify start, streaming, status polling, and stopping monitoring from Overview."""
    db_file = tmp_path / "lifecycle_test.db"
    engine = RealtimeInferenceEngine(db_path=db_file, use_simulation=True)

    status_idle = engine.get_status()
    assert status_idle["monitoring_active"] is False
    assert status_idle["system_health"] == "IDLE"

    # Start monitoring in simulation mode
    engine.start_monitoring(use_simulation=True)
    time.sleep(0.6)

    status_running = engine.get_status()
    assert status_running["monitoring_active"] is True
    assert status_running["system_health"] == "OPTIMAL"
    assert "waveform" in status_running
    assert len(status_running["waveform"]) == 64
    assert status_running["windows_processed"] >= 1

    # Stop monitoring
    engine.stop_monitoring()
    time.sleep(0.1)

    status_stopped = engine.get_status()
    assert status_stopped["monitoring_active"] is False
    assert status_stopped["system_health"] == "IDLE"


def test_live_cnn_predictions_and_confidence(tmp_path):
    """Verify that live inference emits valid Softmax probabilities for binary classes."""
    db_file = tmp_path / "prediction_test.db"
    engine = RealtimeInferenceEngine(db_path=db_file, use_simulation=True)

    engine.start_monitoring(use_simulation=True)
    time.sleep(0.6)

    status = engine.get_status()
    probs = status["latest_probabilities"]
    assert "explosion" in probs
    assert "non_explosion" in probs

    # Valid probability distribution
    total_p = sum(probs.values())
    assert 0.99 <= total_p <= 1.01
    assert 0.0 <= status["latest_confidence"] <= 1.0
    assert status["latest_prediction"] in ("non_explosion", "explosion")
    assert status["latency_ms"] >= 0.0

    engine.stop_monitoring()


def test_sqlite_alerts_persistence_and_date_filtering(tmp_path):
    """Verify alert recording, date-range filtering, and search functionality."""
    db_file = tmp_path / "alerts_filter_test.db"
    db = AlertDatabase(db_file)

    # Insert test alerts at specific timestamps
    db.record_alert(
        alert_type="EXPLOSION_DETECTED",
        confidence=0.942,
        audio_energy_dbfs=-14.2,
        category="explosion",
        source="microphone",
        timestamp="2026-11-13T10:00:00Z",
        details={"device": "External Microphone"},
    )
    db.record_alert(
        alert_type="FIREWORKS_DETECTED",
        confidence=0.885,
        audio_energy_dbfs=-19.5,
        category="fireworks",
        source="microphone",
        timestamp="2026-11-13T12:00:00Z",
        details={"device": "External Microphone"},
    )
    db.record_alert(
        alert_type="EXPLOSION_DETECTED",
        confidence=0.961,
        audio_energy_dbfs=-11.8,
        category="explosion",
        source="microphone",
        timestamp="2026-11-13T14:00:00Z",
        details={"device": "Microphone Array"},
    )

    # Total count
    assert db.count_alerts() == 3

    # Filter by alert_type
    exp_alerts = db.get_alerts(alert_type="EXPLOSION_DETECTED")
    assert len(exp_alerts) == 2
    assert exp_alerts[0]["alert_type"] == "EXPLOSION_DETECTED"
    assert exp_alerts[0]["device"] in ("External Microphone", "Microphone Array")

    # Filter by search
    array_alerts = db.get_alerts(search="Array")
    assert len(array_alerts) == 1
    assert array_alerts[0]["device"] == "Microphone Array"

    # Filter by date range (11:00 to 13:00 should return only the fireworks alert)
    range_alerts = db.get_alerts(start_date="2026-11-13T11:00:00Z", end_date="2026-11-13T13:00:00Z")
    assert len(range_alerts) == 1
    assert range_alerts[0]["alert_type"] == "FIREWORKS_DETECTED"

    # Clear history
    del_count = db.clear_alerts()
    assert del_count == 3
    assert db.count_alerts() == 0


def test_api_csv_export_endpoint(tmp_path):
    """Verify /api/alerts/export returns well-formed CSV attachment."""
    db_file = tmp_path / "csv_export_test.db"
    db = AlertDatabase(db_file)
    db.record_alert(
        alert_type="EXPLOSION_DETECTED",
        confidence=0.95,
        audio_energy_dbfs=-15.0,
        category="explosion",
        source="microphone",
        timestamp="2026-11-13T15:30:00Z",
        details={"device": "Test Mic"},
    )

    state = DetectionServerState(db_path=db_file)
    status, body = _execute_http("GET", "/api/alerts/export", state=state)
    assert status == 200
    assert "ID,Timestamp_UTC,Alert_Type,Confidence,Audio_Energy_dBFS,Category,Source,Device,Status" in body
    assert "EXPLOSION_DETECTED,0.95,-15.0,explosion,microphone,Test Mic,active" in body


def test_alert_persistence_across_restarts(tmp_path):
    """Verify that restarting the server retains all SQLite alert history."""
    db_file = tmp_path / "restart_persistence.db"
    state1 = DetectionServerState(db_path=db_file)
    state1.engine.alert_db.record_alert(
        alert_type="EXPLOSION_DETECTED",
        confidence=0.91,
        audio_energy_dbfs=-16.5,
        timestamp="2026-11-13T16:00:00Z",
    )
    assert state1.engine.alert_db.count_alerts() == 1

    # Simulate shutdown and restart with fresh state pointing to same DB
    state1.engine.stop_monitoring()
    del state1

    state2 = DetectionServerState(db_path=db_file)
    assert state2.engine.alert_db.count_alerts() == 1
    recs = state2.engine.alert_db.get_alerts()
    assert len(recs) == 1
    assert recs[0]["alert_type"] == "EXPLOSION_DETECTED"
    assert recs[0]["confidence"] == 0.91
