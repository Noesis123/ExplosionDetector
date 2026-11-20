"""Real-Time Audio Capture, Ring Buffer, and Inference Engine.

Supports:
- Native Windows audio capture using ctypes.windll.winmm WaveIn API (zero pip/third-party dependencies).
- Fallback SimulatedAudioStream for automated tests, recorded file streaming, and headless CI.
- Real-time sliding window inference (1.0s window, 0.25s hop) with AudioMulticlassCNN.
- RMS squelch gate (-42 dBFS default) to suppress sensor noise floor and silence false alarms.
- Consecutive-window confirmation, cooldown refractory period, and primary alert conflict resolution.
- SQLite persistence of threat alert events.
"""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ctypes
from ctypes import wintypes
import math
import os
import threading
import time
import numpy as np

from audio.preprocessing import AudioPreprocessor
from inference.alert_db import AlertDatabase
from inference.predictor_interface import MulticlassModelPredictor

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_WINDOW_DURATION = 1.0
DEFAULT_HOP_DURATION = 0.25
DEFAULT_RING_BUFFER_DURATION = 5.0
DEFAULT_RMS_SQUELCH_DBFS = -58.0
DEFAULT_EXPLOSION_THRESHOLD = 0.75
DEFAULT_FIREWORKS_THRESHOLD = 0.75
DEFAULT_CONSECUTIVE_WINDOWS = 1
DEFAULT_COOLDOWN_SECONDS = 2.5


# --- Windows WaveIn Native Structures ---
class _WAVEINCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", ctypes.c_uint),
        ("szPname", ctypes.c_wchar * 32),
        ("dwFormats", ctypes.c_uint),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
    ]


class _WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class _WAVEHDR(ctypes.Structure):
    pass


_WAVEHDR._fields_ = [
    ("lpData", ctypes.c_char_p),
    ("dwBufferLength", wintypes.DWORD),
    ("dwBytesRecorded", wintypes.DWORD),
    ("dwUser", ctypes.c_void_p),
    ("dwFlags", wintypes.DWORD),
    ("dwLoops", wintypes.DWORD),
    ("lpNext", ctypes.POINTER(_WAVEHDR)),
    ("reserved", ctypes.c_void_p),
]


def list_audio_input_devices() -> List[Dict[str, Any]]:
    """Enumerate physical and virtual audio recording devices on Windows."""
    devices: List[Dict[str, Any]] = []
    if os.name != "nt":
        devices.append({"id": -1, "name": "Simulated Stream Only (Non-Windows)", "channels": 1, "is_default": True})
        return devices

    try:
        winmm = ctypes.windll.winmm
        num_devs = winmm.waveInGetNumDevs()
        for i in range(num_devs):
            caps = _WAVEINCAPSW()
            res = winmm.waveInGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps))
            if res == 0:
                devices.append({
                    "id": i,
                    "name": str(caps.szPname).strip(),
                    "channels": int(caps.wChannels),
                    "is_default": (i == 0),
                })
    except Exception as e:
        devices.append({"id": -1, "name": f"Enumeration Error: {e}", "channels": 1, "is_default": False})

    if not devices:
        devices.append({"id": 0, "name": "Default Audio Input", "channels": 1, "is_default": True})
    return devices


class AudioRingBuffer:
    """Thread-safe circular audio ring buffer for sliding-window continuous inference."""

    def __init__(self, capacity_samples: int):
        self.capacity = max(1024, int(capacity_samples))
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_pos = 0
        self.total_samples_written = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Append float32 audio samples into the circular buffer."""
        arr = np.asarray(samples, dtype=np.float32)
        if len(arr) == 0:
            return

        with self._lock:
            n = len(arr)
            if n >= self.capacity:
                self.buffer[:] = arr[-self.capacity:]
                self.write_pos = 0
            else:
                end_pos = self.write_pos + n
                if end_pos <= self.capacity:
                    self.buffer[self.write_pos:end_pos] = arr
                else:
                    first_chunk = self.capacity - self.write_pos
                    second_chunk = n - first_chunk
                    self.buffer[self.write_pos:] = arr[:first_chunk]
                    self.buffer[:second_chunk] = arr[first_chunk:]
                self.write_pos = (self.write_pos + n) % self.capacity
            self.total_samples_written += n

    def read_latest(self, num_samples: int) -> np.ndarray:
        """Extract the most recent `num_samples` from the circular buffer in chronological order."""
        num_samples = min(num_samples, self.capacity)
        with self._lock:
            if self.total_samples_written < num_samples:
                # Buffer has not filled up yet; zero-pad on left
                available = min(self.total_samples_written, self.capacity)
                out = np.zeros(num_samples, dtype=np.float32)
                if available > 0:
                    start_pos = (self.write_pos - available) % self.capacity
                    if start_pos + available <= self.capacity:
                        out[-available:] = self.buffer[start_pos:start_pos + available]
                    else:
                        first_chunk = self.capacity - start_pos
                        second_chunk = available - first_chunk
                        out[-available:-second_chunk] = self.buffer[start_pos:]
                        out[-second_chunk:] = self.buffer[:second_chunk]
                return out

            start_pos = (self.write_pos - num_samples) % self.capacity
            if start_pos + num_samples <= self.capacity:
                return self.buffer[start_pos:start_pos + num_samples].copy()
            else:
                first_chunk = self.capacity - start_pos
                second_chunk = num_samples - first_chunk
                out = np.empty(num_samples, dtype=np.float32)
                out[:first_chunk] = self.buffer[start_pos:]
                out[first_chunk:] = self.buffer[:second_chunk]
                return out

    def clear(self) -> None:
        with self._lock:
            self.buffer.fill(0.0)
            self.write_pos = 0
            self.total_samples_written = 0


class WaveInAudioCapture:
    """Hardware microphone capture using Windows WaveIn native multimedia API."""

    def __init__(self, device_id: int = 0, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.device_id = int(device_id)
        self.sample_rate = int(sample_rate)
        self.is_capturing = False
        self.ring_buffer = AudioRingBuffer(self.sample_rate * 5)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.device_name = f"WaveIn Device {self.device_id}"
        self.capture_error: Optional[str] = None

        # Resolve device name
        devs = list_audio_input_devices()
        for d in devs:
            if d.get("id") == self.device_id:
                self.device_name = d.get("name", self.device_name)
                break

    def start(self) -> None:
        if self.is_capturing:
            return
        self._stop_event.clear()
        self.is_capturing = True
        self.capture_error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="WaveInCaptureThread")
        self._thread.start()

    def stop(self) -> None:
        self.is_capturing = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _capture_loop(self) -> None:
        if os.name != "nt":
            self._simulate_fallback_loop()
            return

        winmm = ctypes.windll.winmm
        bits = 16
        h_wavein = wintypes.HANDLE()
        res = -1
        opened_channels = 1

        # Attempt to open device: requested ID (1 ch, then 2 ch), default 0 (1 ch, then 2 ch), WAVE_MAPPER (-1)
        candidates = [self.device_id]
        if 0 not in candidates:
            candidates.append(0)
        candidates.append(ctypes.c_uint(-1).value)

        for dev_try in candidates:
            for ch in (1, 2):
                wfx = _WAVEFORMATEX()
                wfx.wFormatTag = 1 # PCM
                wfx.nChannels = ch
                wfx.nSamplesPerSec = self.sample_rate
                wfx.wBitsPerSample = bits
                wfx.nBlockAlign = ch * (bits // 8)
                wfx.nAvgBytesPerSec = self.sample_rate * wfx.nBlockAlign
                wfx.cbSize = 0
                res = winmm.waveInOpen(ctypes.byref(h_wavein), dev_try, ctypes.byref(wfx), 0, 0, 0)
                if res == 0:
                    opened_channels = ch
                    self.capture_error = None
                    break
            if res == 0:
                break

        if res != 0:
            self.capture_error = f"Audio capture device unavailable (WinMM error {res})"
            self._simulate_fallback_loop()
            return

        # Double/triple buffer: 250ms chunks (hop size)
        chunk_samples = int(self.sample_rate * 0.25)
        buf_size = chunk_samples * opened_channels * (bits // 8)
        buffers = []
        hdrs = []

        for _ in range(3):
            buf = (ctypes.c_char * buf_size)()
            hdr = _WAVEHDR()
            hdr.lpData = ctypes.cast(buf, ctypes.c_char_p)
            hdr.dwBufferLength = buf_size
            hdr.dwFlags = 0
            winmm.waveInPrepareHeader(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
            winmm.waveInAddBuffer(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
            buffers.append(buf)
            hdrs.append(hdr)

        winmm.waveInStart(h_wavein)

        try:
            curr_idx = 0
            while not self._stop_event.is_set():
                hdr = hdrs[curr_idx]
                # Check WHDR_DONE (0x01)
                if hdr.dwFlags & 0x01:
                    bytes_rec = hdr.dwBytesRecorded
                    if bytes_rec > 0:
                        raw_data = bytes(buffers[curr_idx])[:bytes_rec]
                        pcm_s16 = np.frombuffer(raw_data, dtype=np.int16)
                        pcm_f32 = pcm_s16.astype(np.float32) / 32768.0
                        if opened_channels == 2:
                            # Stereo downmix to mono
                            pcm_f32 = (pcm_f32[0::2] + pcm_f32[1::2]) * 0.5
                        self.ring_buffer.write(pcm_f32)

                    hdr.dwFlags &= ~0x01
                    winmm.waveInAddBuffer(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
                    curr_idx = (curr_idx + 1) % len(hdrs)
                else:
                    time.sleep(0.02)
        finally:
            winmm.waveInStop(h_wavein)
            winmm.waveInReset(h_wavein)
            for hdr in hdrs:
                winmm.waveInUnprepareHeader(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
            winmm.waveInClose(h_wavein)

    def _simulate_fallback_loop(self) -> None:
        chunk_samples = int(self.sample_rate * 0.25)
        while not self._stop_event.is_set():
            noise = (np.random.randn(chunk_samples).astype(np.float32) * 1e-4)
            self.ring_buffer.write(noise)
            time.sleep(0.25)


class SimulatedAudioStream:
    """Streams audio from a file or synthetic waveform into the ring buffer for testing."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None, sample_rate: int = DEFAULT_SAMPLE_RATE, real_time: bool = True):
        self.sample_rate = int(sample_rate)
        self.real_time = bool(real_time)
        self.is_capturing = False
        self.ring_buffer = AudioRingBuffer(self.sample_rate * 10)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.device_name = f"Simulation Stream [{Path(file_path).name if file_path else 'Synthetic Noise'}]"

        self.audio_data = np.zeros(self.sample_rate * 2, dtype=np.float32)
        if file_path and Path(file_path).exists():
            from audio.file_loader import load_audio_file
            sig, sr, dur, ch = load_audio_file(file_path)
            pre = AudioPreprocessor(sample_rate=self.sample_rate)
            mono = pre.to_mono(sig)
            self.audio_data = pre.resample(mono, sr)

    def start(self) -> None:
        if self.is_capturing:
            return
        self._stop_event.clear()
        self.is_capturing = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name="SimulatedStreamThread")
        self._thread.start()

    def stop(self) -> None:
        self.is_capturing = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _stream_loop(self) -> None:
        chunk_samples = int(self.sample_rate * 0.25)
        pos = 0
        total_len = len(self.audio_data)

        while not self._stop_event.is_set():
            if pos + chunk_samples <= total_len:
                chunk = self.audio_data[pos:pos + chunk_samples]
                pos += chunk_samples
            else:
                # Loop simulation or pad
                first = self.audio_data[pos:]
                second = self.audio_data[:chunk_samples - len(first)]
                chunk = np.concatenate([first, second])
                pos = len(second)

            self.ring_buffer.write(chunk)
            if self.real_time:
                time.sleep(0.25)


class RealtimeInferenceEngine:
    """Coordinates microphone capture, sliding-window CNN evaluation, alerting, and persistence."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "explosion_detection/models/checkpoints/multiclass_phase21_best_model.pt",
        db_path: Optional[Path] = None,
        device_id: int = 0, # Default physical microphone
        use_simulation: bool = False,
        simulation_file: Optional[Union[str, Path]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.alert_db = AlertDatabase(db_path)
        self.preprocessor = AudioPreprocessor(sample_rate=DEFAULT_SAMPLE_RATE)
        self.predictor = MulticlassModelPredictor(self.checkpoint_path)

        self.device_id = int(device_id)
        self.use_simulation = bool(use_simulation)
        self.simulation_file = simulation_file

        # Active configuration
        self.explosion_threshold = DEFAULT_EXPLOSION_THRESHOLD
        self.fireworks_threshold = DEFAULT_FIREWORKS_THRESHOLD
        self.consecutive_windows = DEFAULT_CONSECUTIVE_WINDOWS
        self.rms_squelch_dbfs = DEFAULT_RMS_SQUELCH_DBFS
        self.cooldown_seconds = DEFAULT_COOLDOWN_SECONDS

        rules_path = Path("configs/detection_rules.json")
        if rules_path.exists():
            try:
                import json
                rules = json.loads(rules_path.read_text(encoding="utf-8"))
                ar = rules.get("alert_rules", {})
                if "explosion_threshold" in ar:
                    self.explosion_threshold = float(ar["explosion_threshold"])
                if "fireworks_threshold" in ar:
                    self.fireworks_threshold = float(ar["fireworks_threshold"])
                if "consecutive_windows" in ar:
                    self.consecutive_windows = int(ar["consecutive_windows"])
                if "rms_squelch_dbfs" in ar:
                    self.rms_squelch_dbfs = float(ar["rms_squelch_dbfs"])
                if "cooldown_seconds" in ar:
                    self.cooldown_seconds = float(ar["cooldown_seconds"])
            except Exception:
                pass

        # Operational state
        self.is_running = False
        self.active_alert: Optional[str] = None
        self.active_alert_timestamp: Optional[str] = None
        self.last_alert_time: float = 0.0
        self.consecutive_exp_count = 0
        self.consecutive_fw_count = 0

        self.latest_probabilities: Dict[str, float] = {"non_target": 1.0, "explosion": 0.0, "fireworks": 0.0}
        self.latest_prediction: str = "non_target"
        self.latest_confidence: float = 1.0
        self.latest_rms_dbfs: float = -99.9
        self.latest_latency_ms: float = 0.0
        self.latest_waveform: List[float] = [0.0] * 64
        self.windows_processed_count: int = 0
        self.total_explosions_detected: int = 0
        self.total_fireworks_detected: int = 0

        self.capture_stream: Optional[Union[WaveInAudioCapture, SimulatedAudioStream]] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        """Return full operational snapshot for UI Overview and health checks."""
        with self._lock:
            device_name = self.capture_stream.device_name if self.capture_stream else "None"
            cap_err = getattr(self.capture_stream, "capture_error", None) if self.capture_stream else None
            health = "ERROR" if cap_err else ("OPTIMAL" if self.is_running else "IDLE")
            return {
                "monitoring_active": self.is_running,
                "selected_device_id": self.device_id,
                "device_name": device_name,
                "is_simulation": self.use_simulation,
                "latest_prediction": self.latest_prediction,
                "latest_confidence": round(self.latest_confidence, 4),
                "latest_probabilities": {k: round(v, 4) for k, v in self.latest_probabilities.items()},
                "audio_energy_dbfs": round(self.latest_rms_dbfs, 1),
                "active_alert": self.active_alert,
                "active_alert_timestamp": self.active_alert_timestamp,
                "windows_processed": self.windows_processed_count,
                "total_explosions_detected": self.total_explosions_detected,
                "total_fireworks_detected": self.total_fireworks_detected,
                "latency_ms": round(self.latest_latency_ms, 2),
                "model_name": "AudioMulticlassCNN",
                "model_checkpoint": self.checkpoint_path.name,
                "model_version": "Phase 21 (Active Production)",
                "model_parameters": getattr(self.predictor, "checkpoint_metadata", {}).get("model_parameters", 133571),
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "waveform": list(self.latest_waveform),
                "system_health": health,
                "error": cap_err,
                "settings": {
                    "explosion_threshold": self.explosion_threshold,
                    "fireworks_threshold": self.fireworks_threshold,
                    "consecutive_windows": self.consecutive_windows,
                    "rms_squelch_dbfs": self.rms_squelch_dbfs,
                    "cooldown_seconds": self.cooldown_seconds,
                },
            }

    def update_settings(
        self,
        explosion_threshold: Optional[float] = None,
        fireworks_threshold: Optional[float] = None,
        consecutive_windows: Optional[int] = None,
        rms_squelch_dbfs: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Dynamically update alert thresholds and squelch gating."""
        with self._lock:
            if explosion_threshold is not None:
                self.explosion_threshold = max(0.1, min(0.99, float(explosion_threshold)))
            if fireworks_threshold is not None:
                self.fireworks_threshold = max(0.1, min(0.99, float(fireworks_threshold)))
            if consecutive_windows is not None:
                self.consecutive_windows = max(1, min(10, int(consecutive_windows)))
            if rms_squelch_dbfs is not None:
                self.rms_squelch_dbfs = max(-90.0, min(-10.0, float(rms_squelch_dbfs)))
            if cooldown_seconds is not None:
                self.cooldown_seconds = max(0.5, min(10.0, float(cooldown_seconds)))

            return {
                "explosion_threshold": self.explosion_threshold,
                "fireworks_threshold": self.fireworks_threshold,
                "consecutive_windows": self.consecutive_windows,
                "rms_squelch_dbfs": self.rms_squelch_dbfs,
                "cooldown_seconds": self.cooldown_seconds,
            }

    def start_monitoring(
        self,
        device_id: Optional[int] = None,
        use_simulation: Optional[bool] = None,
        simulation_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Start audio capture and the real-time inference loop."""
        if self.is_running:
            self.stop_monitoring()

        with self._lock:
            if device_id is not None:
                self.device_id = int(device_id)
            if use_simulation is not None:
                self.use_simulation = bool(use_simulation)
            if simulation_file is not None:
                self.simulation_file = simulation_file

            if self.use_simulation:
                self.capture_stream = SimulatedAudioStream(
                    file_path=self.simulation_file,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    real_time=True,
                )
            else:
                self.capture_stream = WaveInAudioCapture(
                    device_id=self.device_id,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                )

            self.capture_stream.start()
            self._stop_event.clear()
            self.is_running = True
            self._engine_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="RealtimeInferenceThread",
            )
            self._engine_thread.start()

    def stop_monitoring(self) -> None:
        """Stop audio capture and background inference."""
        self.is_running = False
        self._stop_event.set()
        if self.capture_stream:
            self.capture_stream.stop()
            self.capture_stream = None
        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=1.0)
        self._engine_thread = None

        with self._lock:
            self.active_alert = None
            self.active_alert_timestamp = None

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to real-time event updates (e.g. for Server-Sent Events)."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify_subscribers(self, payload: Dict[str, Any]) -> None:
        for cb in list(self._subscribers):
            try:
                cb(payload)
            except Exception:
                pass

    def _inference_loop(self) -> None:
        window_samples = int(DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_DURATION) # 22050
        hop_duration = DEFAULT_HOP_DURATION # 0.25s
        squelch_linear = 10.0 ** (self.rms_squelch_dbfs / 20.0)

        # Allow buffer to initially receive at least one hop
        time.sleep(0.1)

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            if not self.capture_stream:
                break

            # Read latest 1.0-second audio window
            window_audio = self.capture_stream.ring_buffer.read_latest(window_samples)
            if len(window_audio) < window_samples:
                time.sleep(hop_duration)
                continue

            # Compute raw waveform RMS
            rms = float(np.sqrt(np.mean(window_audio ** 2)))
            rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))

            now_iso = datetime.now(timezone.utc).isoformat()
            squelch_linear = 10.0 ** (self.rms_squelch_dbfs / 20.0)

            # Evaluate squelch gate
            if rms < squelch_linear:
                # Squelch active: silence / near-silence forced to non_target with 0.0 target scores
                probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                # Compute Log-Mel spectrogram
                spec = self.preprocessor.compute_log_mel_spectrogram(window_audio)
                # Model inference
                batch_spec = spec[np.newaxis, ...] # (1, 64, 44)
                probs = self.predictor.predict_batch(batch_spec)[0] # [p_nt, p_exp, p_fw]

            p_nt, p_exp, p_fw = float(probs[0]), float(probs[1]), float(probs[2])

            # Consecutive window tracking
            if p_exp >= self.explosion_threshold:
                self.consecutive_exp_count += 1
            else:
                self.consecutive_exp_count = 0

            if p_fw >= self.fireworks_threshold:
                self.consecutive_fw_count += 1
            else:
                self.consecutive_fw_count = 0

            exp_triggered = self.consecutive_exp_count >= self.consecutive_windows
            fw_triggered = self.consecutive_fw_count >= self.consecutive_windows

            # Alert conflict resolution & cooldown
            current_time = time.time()
            cooldown_active = (current_time - self.last_alert_time) < self.cooldown_seconds

            new_alert: Optional[str] = None
            alert_conf = 0.0
            category = "threat"

            if (exp_triggered or fw_triggered) and not cooldown_active:
                if exp_triggered and fw_triggered:
                    # Conflict: higher confidence wins
                    if p_exp >= p_fw:
                        new_alert = "EXPLOSION_DETECTED"
                        alert_conf = p_exp
                        category = "explosion"
                    else:
                        new_alert = "FIREWORKS_DETECTED"
                        alert_conf = p_fw
                        category = "fireworks"
                elif exp_triggered:
                    new_alert = "EXPLOSION_DETECTED"
                    alert_conf = p_exp
                    category = "explosion"
                elif fw_triggered:
                    new_alert = "FIREWORKS_DETECTED"
                    alert_conf = p_fw
                    category = "fireworks"

                if new_alert:
                    self.last_alert_time = current_time
                    if new_alert == "EXPLOSION_DETECTED":
                        self.total_explosions_detected += 1
                    else:
                        self.total_fireworks_detected += 1

                    # Persist alert
                    src = "simulation" if self.use_simulation else "microphone"
                    self.alert_db.record_alert(
                        alert_type=new_alert,
                        confidence=alert_conf,
                        audio_energy_dbfs=rms_dbfs,
                        category=category,
                        source=src,
                        timestamp=now_iso,
                        details={"probs": [p_nt, p_exp, p_fw], "device": self.capture_stream.device_name},
                    )

            # Determine dominant model prediction (argmax across classes)
            class_labels = ["non_target", "explosion", "fireworks"]
            max_idx = int(np.argmax(probs))
            pred_class = class_labels[max_idx]
            pred_conf = float(probs[max_idx])

            # Active alert hold (display for 2.0s after trigger)
            if new_alert:
                self.active_alert = new_alert
                self.active_alert_timestamp = now_iso
            elif (current_time - self.last_alert_time) > 2.0:
                self.active_alert = None
                self.active_alert_timestamp = None

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0

            # Downsample window_audio to 64 normalized amplitude samples for real-time oscilloscope
            stride = max(1, len(window_audio) // 64)
            waveform_peaks = [round(float(window_audio[i]), 3) for i in range(0, min(len(window_audio), 64 * stride), stride)][:64]

            with self._lock:
                self.latest_probabilities = {
                    "non_target": p_nt,
                    "explosion": p_exp,
                    "fireworks": p_fw,
                }
                self.latest_prediction = pred_class
                self.latest_confidence = pred_conf
                self.latest_rms_dbfs = rms_dbfs
                self.latest_latency_ms = latency_ms
                self.latest_waveform = waveform_peaks
                self.windows_processed_count += 1

            # Dispatch full status notification including monitoring_active and device info
            payload = self.get_status()
            payload.update({
                "timestamp": now_iso,
                "prediction": pred_class,
                "confidence": round(pred_conf, 4),
                "probabilities": {k: round(v, 4) for k, v in self.latest_probabilities.items()},
                "total_explosions": self.total_explosions_detected,
                "total_fireworks": self.total_fireworks_detected,
                "latency_ms": round(latency_ms, 2),
                "waveform": waveform_peaks,
            })
            self._notify_subscribers(payload)

            # Sleep to match hop stride
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.01, hop_duration - elapsed)
            time.sleep(sleep_time)

