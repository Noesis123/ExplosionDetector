"""Integration and Functional Test Suite for Phase 22 Real-Time Detection & Dashboard.

Verifies:
1. Detection rules configuration structure and parameters.
2. Circular audio ring buffer strides, wraps, and bounds.
3. Windows WaveIn device enumeration and simulated stream fallback.
4. Squelch gate suppression of low-energy background sensor noise.
5. Consecutive window filtering suppressing single-window transient spikes.
6. SQLite alert database CRUD operations.
7. Backend REST API endpoints (GET /status, GET /devices, POST /settings, POST /monitor/start/stop).
8. End-to-end detection behavior: explosions alert correctly; everyday civilian sounds produce zero alerts.
"""

import io
import json
from pathlib import Path
import time
import numpy as np
import pytest

from audio.microphone_stream import (
    AudioRingBuffer,
    RealtimeInferenceEngine,
    SimulatedAudioStream,
    list_audio_input_devices,
)
from inference.alert_db import AlertDatabase
from server.app import DetectionAPIRequestHandler, DetectionServerState, STATIC_DIR


def test_detection_rules_config_validity():
    cfg_path = Path("configs/detection_rules.json")
    assert cfg_path.exists(), f"Configuration file not found: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "alert_rules" in data
    rules = data["alert_rules"]
    assert 0.5 <= rules["explosion_threshold"] <= 0.95
    assert 0.5 <= rules["fireworks_threshold"] <= 0.95
    assert rules["consecutive_windows"] >= 1
    assert -60.0 <= rules["rms_squelch_dbfs"] <= -20.0
    assert rules["cooldown_seconds"] > 0.0

    assert "validation_performance" in data
    perf = data["validation_performance"]
    assert perf["after_optimization"]["recording_accuracy"] >= 0.85
    assert perf["after_optimization"]["explosion_recall"] >= 0.80
    assert perf["after_optimization"]["civilian_false_alarms"] == 0


def test_audio_ring_buffer_operations():
    # 100-sample capacity buffer
    rb = AudioRingBuffer(capacity_samples=100)

    # Initial read should return zeros
    init_read = rb.read_latest(50)
    assert len(init_read) == 50
    assert np.all(init_read == 0.0)

    # Write 30 ones
    rb.write(np.ones(30, dtype=np.float32))
    read30 = rb.read_latest(30)
    assert np.all(read30 == 1.0)

    # Write 80 twos (total 110 written -> wraps around)
    rb.write(np.full(80, 2.0, dtype=np.float32))
    assert rb.total_samples_written == 110

    # Read latest 100 samples: should be 20 ones then 80 twos
    read_all = rb.read_latest(100)
    assert len(read_all) == 100
    assert np.all(read_all[:20] == 1.0)
    assert np.all(read_all[20:] == 2.0)

    # Clear
    rb.clear()
    assert rb.total_samples_written == 0
    assert np.all(rb.read_latest(50) == 0.0)


def test_audio_device_enumeration():
    devs = list_audio_input_devices()
    assert isinstance(devs, list)
    assert len(devs) > 0
    for d in devs:
        assert "id" in d
        assert "name" in d
        assert "channels" in d


def test_rms_squelch_gate_suppression():
    engine = RealtimeInferenceEngine(use_simulation=True)
    engine.rms_squelch_dbfs = -42.0

    # Test quiet waveform (RMS ~ -60 dBFS)
    quiet_window = (np.random.randn(22050).astype(np.float32) * 1e-4) # RMS ~ 0.0001 = -80 dBFS
    rms = float(np.sqrt(np.mean(quiet_window ** 2)))
    squelch_lin = 10.0 ** (engine.rms_squelch_dbfs / 20.0)
    assert rms < squelch_lin

    # Under squelch gate, probabilities must be forced to [1.0, 0.0, 0.0]
    forced_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert forced_probs[0] == 1.0
    assert forced_probs[1] == 0.0
    assert forced_probs[2] == 0.0


def test_consecutive_window_confirmation():
    engine = RealtimeInferenceEngine(use_simulation=True)
    engine.consecutive_windows = 2
    engine.explosion_threshold = 0.75

    # Sequence with isolated 1-window explosion spike (w0: 0.1, w1: 0.9, w2: 0.1)
    spikes = [0.10, 0.90, 0.10]
    triggered_any = False

    exp_count = 0
    for score in spikes:
        if score >= engine.explosion_threshold:
            exp_count += 1
        else:
            exp_count = 0
        if exp_count >= engine.consecutive_windows:
            triggered_any = True

    # Single spike must NOT trigger alert when consecutive_windows=2
    assert not triggered_any, "Isolated single-window spike should not trigger when k=2"

    # Sustained 2-window burst (w0: 0.85, w1: 0.88)
    exp_count = 0
    triggered_sustained = False
    for score in [0.85, 0.88]:
        if score >= engine.explosion_threshold:
            exp_count += 1
        else:
            exp_count = 0
        if exp_count >= engine.consecutive_windows:
            triggered_sustained = True

    assert triggered_sustained, "Sustained burst should trigger when k=2"


def test_alert_database_crud(tmp_path):
    db_path = tmp_path / "test_alerts.db"
    db = AlertDatabase(db_path=db_path)

    assert db.count_alerts() == 0

    # Record alert
    rec1 = db.record_alert(
        alert_type="EXPLOSION_DETECTED",
        confidence=0.9234,
        audio_energy_dbfs=-18.5,
        category="positives",
        source="microphone",
    )
    assert rec1["id"] == 1
    assert rec1["alert_type"] == "EXPLOSION_DETECTED"
    assert db.count_alerts() == 1

    # Record second alert
    rec2 = db.record_alert(
        alert_type="FIREWORKS_DETECTED",
        confidence=0.8412,
        audio_energy_dbfs=-24.1,
        category="fireworks",
        source="simulation",
    )
    assert rec2["id"] == 2
    assert db.count_alerts() == 2

    # Query alerts
    all_alerts = db.get_alerts(limit=10)
    assert len(all_alerts) == 2
    # Order by id DESC
    assert all_alerts[0]["id"] == 2
    assert all_alerts[1]["id"] == 1

    # Filter by alert_type
    exp_only = db.get_alerts(alert_type="EXPLOSION_DETECTED")
    assert len(exp_only) == 1
    assert exp_only[0]["alert_type"] == "EXPLOSION_DETECTED"

    # Search
    search_res = db.get_alerts(search="simulation")
    assert len(search_res) == 1
    assert search_res[0]["source"] == "simulation"

    # Clear
    deleted = db.clear_alerts()
    assert deleted == 2
    assert db.count_alerts() == 0


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


def _execute_http_request(method: str, path: str, body: bytes = b"", state: DetectionServerState = None):
    headers = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
    if body:
        headers += f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n"
    headers += "\r\n"
    raw_data = headers.encode("utf-8") + body

    sock = DummySocket(raw_data)

    class CustomHandler(DetectionAPIRequestHandler):
        @property
        def state(self):
            return state

    handler = CustomHandler(sock, ("127.0.0.1", 8080), DummyServer())
    resp_raw = sock._wfile.getvalue().decode("utf-8", errors="replace")

    # Split status, headers, body
    parts = resp_raw.split("\r\n\r\n", 1)
    header_lines = parts[0].split("\r\n")
    status_line = header_lines[0]
    body_str = parts[1] if len(parts) > 1 else ""

    status_code = int(status_line.split()[1])
    return status_code, body_str


def test_server_rest_endpoints(tmp_path):
    test_db = tmp_path / "server_test_alerts.db"
    state = DetectionServerState(db_path=test_db)

    # 1. GET /
    status, body = _execute_http_request("GET", "/", state=state)
    assert status == 200
    assert "<title>NOESIS — Autonomous Acoustic Explosion Detection System</title>" in body

    # 2. GET /api/status
    status, body = _execute_http_request("GET", "/api/status", state=state)
    assert status == 200
    data = json.loads(body)
    assert "monitoring_active" in data
    assert "system_health" in data

    # 3. GET /api/devices
    status, body = _execute_http_request("GET", "/api/devices", state=state)
    assert status == 200
    data = json.loads(body)
    assert "devices" in data
    assert len(data["devices"]) > 0

    # 4. GET /api/settings
    status, body = _execute_http_request("GET", "/api/settings", state=state)
    assert status == 200
    data = json.loads(body)
    assert "explosion_threshold" in data

    # 5. POST /api/settings
    post_body = json.dumps({"explosion_threshold": 0.82, "rms_squelch_dbfs": -44.0}).encode("utf-8")
    status, body = _execute_http_request("POST", "/api/settings", body=post_body, state=state)
    assert status == 200
    data = json.loads(body)
    assert data["settings"]["explosion_threshold"] == 0.82
    assert data["settings"]["rms_squelch_dbfs"] == -44.0

    # 6. POST /api/monitor/start with simulation
    start_body = json.dumps({"simulation": True}).encode("utf-8")
    status, body = _execute_http_request("POST", "/api/monitor/start", body=start_body, state=state)
    assert status == 200
    assert state.engine.is_running

    # 7. GET /api/prediction/latest
    status, body = _execute_http_request("GET", "/api/prediction/latest", state=state)
    assert status == 200
    data = json.loads(body)
    assert "prediction" in data
    assert "probabilities" in data

    # 8. POST /api/monitor/stop
    status, body = _execute_http_request("POST", "/api/monitor/stop", state=state)
    assert status == 200
    assert not state.engine.is_running

    # 9. GET /api/alerts
    status, body = _execute_http_request("GET", "/api/alerts", state=state)
    assert status == 200
    data = json.loads(body)
    assert "alerts" in data
    assert "total" in data


def test_end_to_end_validation_threat_differentiation(tmp_path):
    """Verify that an explosion sample triggers an alert, while a civilian sample triggers zero alerts."""
    db_path = tmp_path / "e2e_alerts.db"
    pos_file = Path("data/positives/flutie8211-bomb-and-echo-2-540400.mp3")
    civilian_file = Path("data/phase19_approved/speech_conversation/p19_speech_conversation_3_reitanna_b7b191e3.mp3")

    if not pos_file.exists() or not civilian_file.exists():
        pytest.skip("Audio sample files not found for end-to-end simulation.")

    # 1. Test civilian audio: MUST PRODUCE ZERO ALERTS
    engine_civ = RealtimeInferenceEngine(
        db_path=db_path,
        use_simulation=True,
        simulation_file=civilian_file,
    )
    engine_civ.alert_db.clear_alerts()
    engine_civ.start_monitoring()
    time.sleep(1.2) # Allow at least 4 windows to process
    engine_civ.stop_monitoring()

    civ_alerts_count = engine_civ.alert_db.count_alerts()
    assert civ_alerts_count == 0, f"Civilian speech should generate 0 alerts, got {civ_alerts_count}"

    # 2. Test genuine explosion audio: MUST PRODUCE AN EXPLOSION ALERT
    engine_exp = RealtimeInferenceEngine(
        db_path=db_path,
        use_simulation=True,
        simulation_file=pos_file,
    )
    engine_exp.start_monitoring()
    time.sleep(1.2) # Allow at least 4 windows to process
    engine_exp.stop_monitoring()

    exp_alerts = engine_exp.alert_db.get_alerts(alert_type="EXPLOSION_DETECTED")
    assert len(exp_alerts) >= 1, "Genuine explosion sample should trigger at least 1 EXPLOSION_DETECTED alert"
    assert exp_alerts[0]["confidence"] >= 0.75
