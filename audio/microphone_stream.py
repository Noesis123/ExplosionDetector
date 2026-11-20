"""Real-Time Audio Capture, Ring Buffer, and Binary Explosion Inference Engine.

Integrates:
- Physical Windows WaveIn audio capture with multi-sample-rate support (48kHz, 44.1kHz, 22.05kHz),
  stateful streaming anti-aliased resampling to 22,050 Hz mono, and bounded circular buffering.
- Dedicated binary ExplosionBinaryCNN (130,146 parameters, ExpF + AEV).
- AcousticVerifiedEventTracker with causal zero-lookahead acoustic physics verification (Config C):
  * LF energy ratio >= 0.35 (suppressing high-frequency impacts/horns)
  * HF harshness ratio <= 0.15 (suppressing glass/metallic clatter)
  * Near-squelch fluctuation suppression
  * Immediate initial alert dispatch (0 ms lookahead delay)
  * Duplicate suppression for ongoing continuous events
  * Independent blast separation across silence (>0.75s) or re-onset (>6dB surge)
- RMS squelch gate (-45.0 dBFS) to ignore silence and sensor noise floor.
- Audio file injection interface for testing without hardware dependencies.
- SQLite persistence of threat alert events.
- Strictly two classes: 'non_explosion' and 'explosion'. Fireworks treated as non_explosion.
"""

from collections import deque
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ctypes
from ctypes import wintypes
import numpy as np
import scipy.signal as signal

from audio.preprocessing import AudioPreprocessor
from audio.streaming_resampler import StatefulStreamingResampler, RobustLiveAudioBuffer

class AudioRingBuffer:
    """Thread-safe circular audio ring buffer."""

    def __init__(self, capacity_samples: int = 220500):
        self.capacity = capacity_samples
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_pos = 0
        self.total_samples_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        n = len(data)
        with self._lock:
            if n >= self.capacity:
                self.buffer[:] = data[-self.capacity:]
                self.write_pos = 0
            else:
                end = self.write_pos + n
                if end <= self.capacity:
                    self.buffer[self.write_pos:end] = data
                else:
                    first = self.capacity - self.write_pos
                    self.buffer[self.write_pos:] = data[:first]
                    self.buffer[:n - first] = data[first:]
                self.write_pos = (self.write_pos + n) % self.capacity
            self.total_samples_written += n

    def read_latest(self, n: int) -> np.ndarray:
        with self._lock:
            if n > self.capacity:
                n = self.capacity
            if self.total_samples_written == 0:
                return np.zeros(n, dtype=np.float32)
            start = (self.write_pos - n) % self.capacity
            if start + n <= self.capacity:
                return self.buffer[start:start + n].copy()
            part1 = self.buffer[start:]
            part2 = self.buffer[:(start + n) % self.capacity]
            return np.concatenate([part1, part2])

    def clear(self) -> None:
        with self._lock:
            self.buffer.fill(0.0)
            self.write_pos = 0
            self.total_samples_written = 0


class SimulatedAudioStream:
    """Legacy simulated audio stream stub."""
    pass


class StreamingHighpassFilter:
    """Stateful 2nd-order Butterworth highpass filter for real-time acoustic streams.
    Attenuates hardware DC offset and infrasonic chassis vibrations (< 20 Hz)
    while fully preserving genuine acoustic explosion rumble (20-100 Hz) and phone playback transients.
    """

    def __init__(self, cutoff_hz: float = 35.0, sample_rate: int = 22050):
        self.cutoff_hz = float(cutoff_hz)
        self.sample_rate = int(sample_rate)
        self.sos = signal.butter(2, self.cutoff_hz, btype="highpass", fs=self.sample_rate, output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

    def reset(self) -> None:
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return chunk.astype(np.float32)
        filtered, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        return filtered.astype(np.float32)
from inference.alert_db import AlertDatabase
from inference.tracker_verified import AcousticVerifiedEventTracker
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
import torch

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_WINDOW_DURATION = 1.0
DEFAULT_HOP_DURATION = 0.25
DEFAULT_RMS_SQUELCH_DBFS = -34.0
DEFAULT_EXPLOSION_THRESHOLD = 0.80
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"


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
        return [{"id": -1, "name": "Non-Windows Host (Simulation Only)", "channels": 1, "is_default": False}]

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


class PhysicalWaveInCapture:
    """Hardware microphone capture using native Windows WaveIn multimedia API.
    Supports native hardware sample rates (22050, 44100, 48000 Hz) and multi-channel downmixing.
    """

    def __init__(
        self,
        device_id: int = 0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        self.device_id = int(device_id)
        self.sample_rate = int(sample_rate)
        self.chunk_callback = chunk_callback
        self.is_capturing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.device_name = f"WaveIn Device {self.device_id}"
        self.capture_error: Optional[str] = None
        self.opened_channels: int = 1
        self.opened_sample_rate: int = self.sample_rate

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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="PhysicalWaveInThread")
        self._thread.start()

    def stop(self) -> None:
        self.is_capturing = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _capture_loop(self) -> None:
        if os.name != "nt":
            self.capture_error = "WaveIn is only available on Windows."
            self.is_capturing = False
            return

        winmm = ctypes.windll.winmm
        bits = 16
        h_wavein = wintypes.HANDLE()
        res = -1
        opened_channels = 1
        opened_sr = self.sample_rate

        # Try requested device_id, then default 0, across common sample rates and channels
        candidates = [self.device_id]
        if 0 not in candidates:
            candidates.append(0)

        for dev_try in candidates:
            for sr in (self.sample_rate, 44100, 48000):
                for ch in (2, 1):
                    wfx = _WAVEFORMATEX()
                    wfx.wFormatTag = 1  # PCM
                    wfx.nChannels = ch
                    wfx.nSamplesPerSec = sr
                    wfx.wBitsPerSample = bits
                    wfx.nBlockAlign = ch * (bits // 8)
                    wfx.nAvgBytesPerSec = sr * wfx.nBlockAlign
                    wfx.cbSize = 0
                    res = winmm.waveInOpen(ctypes.byref(h_wavein), dev_try, ctypes.byref(wfx), 0, 0, 0)
                    if res == 0:
                        opened_channels = ch
                        opened_sr = sr
                        self.opened_channels = ch
                        self.opened_sample_rate = sr
                        self.capture_error = None
                        break
                if res == 0:
                    break
            if res == 0:
                break

        if res != 0:
            self.capture_error = f"Physical audio capture unavailable (WinMM error code {res})"
            self.is_capturing = False
            return

        # Triple buffer: 250ms chunks (hop size)
        chunk_samples = int(opened_sr * 0.25)
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
                if hdr.dwFlags & 0x01:  # WHDR_DONE
                    bytes_rec = hdr.dwBytesRecorded
                    if bytes_rec > 0:
                        raw_data = bytes(buffers[curr_idx])[:bytes_rec]
                        pcm_s16 = np.frombuffer(raw_data, dtype=np.int16)
                        pcm_f32 = pcm_s16.astype(np.float32) / 32768.0
                        if opened_channels == 2:
                            # Stereo interleaved -> downmix to mono
                            pcm_f32 = (pcm_f32[0::2] + pcm_f32[1::2]) * 0.5

                        if self.chunk_callback:
                            self.chunk_callback(pcm_f32, opened_sr)

                    hdr.dwFlags &= ~0x01
                    winmm.waveInAddBuffer(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
                    curr_idx = (curr_idx + 1) % len(hdrs)
                else:
                    time.sleep(0.015)
        except Exception as e:
            self.capture_error = f"Capture loop exception: {e}"
        finally:
            try:
                winmm.waveInStop(h_wavein)
                winmm.waveInReset(h_wavein)
                for hdr in hdrs:
                    winmm.waveInUnprepareHeader(h_wavein, ctypes.byref(hdr), ctypes.sizeof(hdr))
                winmm.waveInClose(h_wavein)
            except Exception:
                pass


WaveInAudioCapture = PhysicalWaveInCapture


class RealtimeInferenceEngine:
    """Production Real-Time Inference Engine for Binary Explosion Detection.
    Connects physical WaveIn microphone or audio injector to preprocessing,
    ExplosionBinaryCNN (130,146 parameters), AcousticVerifiedEventTracker, and SQLite DB.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        db_path: Optional[Path] = None,
        device_id: int = 0,
        use_simulation: bool = False,
        simulation_file: Optional[Union[str, Path]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            cand = PROJECT_ROOT / self.checkpoint_path
            if cand.exists():
                self.checkpoint_path = cand

        self.alert_db = AlertDatabase(db_path)
        self.preprocessor = AudioPreprocessor(sample_rate=DEFAULT_SAMPLE_RATE)
        self.device_id = int(device_id)
        self.use_simulation = bool(use_simulation)
        self.simulation_file = simulation_file

        # Active configuration
        self.explosion_threshold = DEFAULT_EXPLOSION_THRESHOLD
        self.rms_squelch_dbfs = DEFAULT_RMS_SQUELCH_DBFS
        self.model_version = "ExpF + AEV Config E (Production)"

        # 1. Load ExplosionBinaryCNN (130,146 params)
        self.model = ExplosionBinaryCNN(freeze_bn=True)
        if self.checkpoint_path.exists():
            sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            self.model.load_state_dict(sd)
        self.model.eval()

        # 2. AcousticVerifiedEventTracker (Config E + Physics Blast Verification)
        self.tracker = AcousticVerifiedEventTracker(
            squelch_dbfs=self.rms_squelch_dbfs,
            conf_threshold=self.explosion_threshold,
            max_event_duration_sec=6.0,
            silence_closure_sec=0.75,
            reonset_delta_db=6.0,
            min_lf_ratio=0.40,
            max_hf_ratio=0.08,
            near_squelch_margin_db=3.0,
            min_crest_factor=2.5,
            max_crest_factor=12.0,
            adaptive_lf_threshold=0.65,
            demo_threshold=0.70,
        )

        # 3. Streaming Resamplers by input sample rate
        self.resamplers: Dict[int, StatefulStreamingResampler] = {}

        # 4. Stateful DC-block and Infrasonic/Chassis Vibration Filter (35 Hz 2nd-order Butterworth)
        self.streaming_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=DEFAULT_SAMPLE_RATE)

        # 5. Sliding Window Buffer
        self.buffer = RobustLiveAudioBuffer(sample_rate=DEFAULT_SAMPLE_RATE)

        # 6. Capture Stream
        self.capture_stream: Optional[PhysicalWaveInCapture] = None

        # 6. Operational State
        self.is_running = False
        self.state = "IDLE"  # IDLE, CAPTURING, INJECTING, ERROR
        self.active_alert: Optional[str] = None
        self.active_alert_timestamp: Optional[str] = None
        self.active_alert_record: Optional[Dict[str, Any]] = None
        self.last_alert_time: float = 0.0

        self.latest_probabilities: Dict[str, float] = {"non_explosion": 1.0, "explosion": 0.0}
        self.latest_prediction: str = "non_explosion"
        self.latest_confidence: float = 1.0
        self.latest_rms_dbfs: float = -99.9
        self.latest_latency_ms: float = 0.0
        self.latest_waveform: List[float] = [0.0] * 64

        self.windows_processed_count: int = 0
        self.total_explosions_detected: int = 0
        self.total_events_detected: int = 0
        self.total_duplicates_suppressed: int = 0

        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.RLock()

    def get_resampler(self, in_sr: int) -> StatefulStreamingResampler:
        if in_sr not in self.resamplers:
            self.resamplers[in_sr] = StatefulStreamingResampler(in_sr=in_sr, out_sr=DEFAULT_SAMPLE_RATE)
        return self.resamplers[in_sr]

    def reset_state(self):
        with self._lock:
            self.buffer.reset()
            self.tracker.reset()
            self.resamplers.clear()
            self.streaming_filter.reset()
            self.active_alert = None
            self.active_alert_timestamp = None
            self.active_alert_record = None
            self.total_explosions_detected = 0
            self.total_duplicates_suppressed = 0
            self.latest_probabilities = {"non_explosion": 1.0, "explosion": 0.0}
            self.latest_prediction = "non_explosion"
            self.latest_confidence = 1.0
            self.latest_rms_dbfs = -99.9
            self.latest_waveform = [0.0] * 64

    def get_status(self) -> Dict[str, Any]:
        """Return full operational snapshot for UI Overview and health checks."""
        with self._lock:
            device_name = self.capture_stream.device_name if self.capture_stream else "None"
            cap_err = getattr(self.capture_stream, "capture_error", None) if self.capture_stream else None
            health = "ERROR" if cap_err else ("OPTIMAL" if self.is_running else "IDLE")
            return {
                "monitoring_active": self.is_running,
                "state": self.state,
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
                "total_duplicates_suppressed": self.total_duplicates_suppressed,
                "latency_ms": round(self.latest_latency_ms, 2),
                "model_name": "ExplosionBinaryCNN",
                "model_checkpoint": self.checkpoint_path.name,
                "model_version": self.model_version,
                "model_parameters": 130146,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "waveform": list(self.latest_waveform),
                "system_health": health,
                "error": cap_err,
                "settings": {
                    "config_preset": getattr(self.tracker, "config_preset", "Config E"),
                    "explosion_threshold": self.explosion_threshold,
                    "rms_squelch_dbfs": self.rms_squelch_dbfs,
                    "min_lf_ratio": self.tracker.min_lf_ratio,
                    "max_hf_ratio": self.tracker.max_hf_ratio,
                    "highpass_cutoff_hz": self.streaming_filter.cutoff_hz,
                },
            }

    def set_config_preset(self, preset: str) -> Dict[str, Any]:
        """Completely and reversibly switch between Config C and Config E presets."""
        with self._lock:
            self.tracker.set_config_preset(preset)
            self.explosion_threshold = self.tracker.conf_threshold
            self.rms_squelch_dbfs = self.tracker.squelch_dbfs
            if self.tracker.config_preset == "Config C":
                self.model_version = "ExpF + AEV Config C (Rolled Back Reference)"
            else:
                self.model_version = "ExpF + AEV Config E (Production)"
            return self.get_status()

    def update_settings(
        self,
        explosion_threshold: Optional[float] = None,
        rms_squelch_dbfs: Optional[float] = None,
        config_preset: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Dynamically update alert thresholds, squelch gating, and AEV presets."""
        with self._lock:
            if config_preset is not None:
                self.tracker.set_config_preset(config_preset)
                self.explosion_threshold = self.tracker.conf_threshold
                self.rms_squelch_dbfs = self.tracker.squelch_dbfs
                if self.tracker.config_preset == "Config C":
                    self.model_version = "ExpF + AEV Config C (Rolled Back Reference)"
                else:
                    self.model_version = "ExpF + AEV Config E (Production)"
            if explosion_threshold is not None:
                self.explosion_threshold = max(0.1, min(0.99, float(explosion_threshold)))
                self.tracker.conf_threshold = self.explosion_threshold
            if rms_squelch_dbfs is not None:
                self.rms_squelch_dbfs = max(-90.0, min(-10.0, float(rms_squelch_dbfs)))
                self.tracker.squelch_dbfs = self.rms_squelch_dbfs

            return {
                "config_preset": getattr(self.tracker, "config_preset", "Config E"),
                "explosion_threshold": self.explosion_threshold,
                "rms_squelch_dbfs": self.rms_squelch_dbfs,
                "min_lf_ratio": self.tracker.min_lf_ratio,
                "max_hf_ratio": self.tracker.max_hf_ratio,
                "model_version": self.model_version,
            }

    def start_monitoring(
        self,
        device_id: Optional[int] = None,
        use_simulation: Optional[bool] = None,
        simulation_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Start physical audio capture or simulation stream."""
        if self.is_running:
            self.stop_monitoring()

        with self._lock:
            if device_id is not None:
                self.device_id = int(device_id)
            if use_simulation is not None:
                self.use_simulation = bool(use_simulation)
            if simulation_file is not None:
                self.simulation_file = simulation_file

            self.reset_state()

            if self.use_simulation:
                # Simulation mode
                self.is_running = True
                self.state = "SIMULATION"
                if self.simulation_file:
                    th = threading.Thread(
                        target=self._run_file_simulation,
                        args=(self.simulation_file,),
                        daemon=True,
                        name="FileSimulationThread",
                    )
                else:
                    th = threading.Thread(
                        target=self._run_synthetic_simulation,
                        daemon=True,
                        name="SyntheticSimulationThread",
                    )
                th.start()
            else:
                # Physical microphone mode
                self.capture_stream = PhysicalWaveInCapture(
                    device_id=self.device_id,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    chunk_callback=self.on_audio_chunk,
                )
                self.capture_stream.start()
                time.sleep(0.08)
                if self.capture_stream.capture_error:
                    self.is_running = False
                    self.state = "ERROR"
                    raise RuntimeError(self.capture_stream.capture_error)

                self.is_running = True
                self.state = "CAPTURING"

        self._notify_subscribers(self.get_status())

    def stop_monitoring(self) -> None:
        """Stop audio capture."""
        self.is_running = False
        self.state = "IDLE"
        if self.capture_stream:
            self.capture_stream.stop()
            self.capture_stream = None

        with self._lock:
            self.active_alert = None
            self.active_alert_timestamp = None
            self.latest_waveform = [0.0] * 64

        self._notify_subscribers(self.get_status())

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

    def on_audio_chunk(self, raw_chunk: np.ndarray, native_sr: int) -> None:
        """Callback receiving streaming audio chunks from physical WaveIn or test injector."""
        if not self.is_running and self.state != "TESTING":
            return

        # 1. Downmix channels if multi-channel
        if raw_chunk.ndim > 1:
            raw_chunk = np.mean(raw_chunk, axis=-1)

        # 2. Sanitize NaNs/Infs
        if not np.all(np.isfinite(raw_chunk)):
            raw_chunk = np.nan_to_num(raw_chunk, nan=0.0, posinf=1.0, neginf=-1.0)

        raw_chunk = raw_chunk.astype(np.float32)

        # 3. Anti-alias resample to 22,050 Hz
        resampler = self.get_resampler(native_sr)
        resampled_chunk = resampler.process_chunk(raw_chunk)

        # 4. Stateful DC-block and Infrasonic/Chassis Vibration Filter (90 Hz 2nd-order Butterworth)
        filtered_chunk = self.streaming_filter.process(resampled_chunk)

        # 5. Push into sliding window buffer
        windows = self.buffer.push_samples(filtered_chunk)

        # 6. Process each 1.0-second window
        for win_time, win_audio in windows:
            self._process_single_window(win_time, win_audio)

    def _process_single_window(self, win_time: float, win_audio: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.windows_processed_count += 1

        # Waveform preview (64 points) for oscilloscope
        step = max(1, len(win_audio) // 64)
        waveform_preview = [round(float(x), 4) for x in win_audio[::step][:64]]

        # Raw RMS calculation
        rms = float(np.sqrt(np.mean(win_audio ** 2)))
        rms_dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
        peak = float(np.max(np.abs(win_audio)))

        # Squelch Gate (-45.0 dBFS)
        if rms_dbfs < self.rms_squelch_dbfs:
            p_not_exp = 1.0
            p_exp = 0.0
            decision = self.tracker.process_window_with_acoustics(
                time_sec=win_time,
                rms_dbfs=rms_dbfs,
                p_nt=1.0,
                p_exp=0.0,
                p_fw=0.0,
                peak=peak,
                lf_ratio=1.0,
                hf_ratio=0.0,
            )
        else:
            # 64x44 normalized Log-Mel spectrogram
            spec = self.preprocessor.compute_log_mel_spectrogram(win_audio)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
            with torch.no_grad():
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            p_not_exp = float(probs[0])
            p_exp = float(probs[1])

            # Causal physical acoustic energy ratios (zero lookahead)
            fft_mag = np.abs(np.fft.rfft(win_audio))
            freqs = np.fft.rfftfreq(len(win_audio), d=1.0 / DEFAULT_SAMPLE_RATE)
            audible_mask = (freqs >= 20.0)
            tot_e = np.sum(fft_mag[audible_mask] ** 2) + 1e-9
            lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
            hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
            sub_bass_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
            sub20_80_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 80.0)] ** 2) / tot_e)
            mid_ratio = float(np.sum(fft_mag[(freqs >= 1000.0) & (freqs < 4000.0)] ** 2) / tot_e)
            crest = peak / max(rms, 1e-9)

            # Intra-window dynamic range (4 subframes of 250ms each)
            sub4 = [win_audio[k * 5512 : (k + 1) * 5512] for k in range(4)]
            sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
            subframe_range = max(sub4_rms) - min(sub4_rms)

            # Intra-window decay ratio (2nd half energy / 1st half energy)
            e_half1 = float(np.sum(win_audio[:11025] ** 2) + 1e-9)
            e_half2 = float(np.sum(win_audio[11025:] ** 2) + 1e-9)
            decay_ratio_in_win = float(e_half2 / e_half1)

            decision = self.tracker.process_window_with_acoustics(
                time_sec=win_time,
                rms_dbfs=rms_dbfs,
                p_nt=p_not_exp,
                p_exp=p_exp,
                p_fw=0.0,
                peak=peak,
                lf_ratio=lf_ratio,
                hf_ratio=hf_ratio,
                crest=crest,
                sub_bass_ratio=sub_bass_ratio,
                subframe_range=subframe_range,
                mid_ratio=mid_ratio,
                sub20_80_ratio=sub20_80_ratio,
                decay_ratio_in_win=decay_ratio_in_win,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Classification label
        pred_class = "explosion" if p_exp >= self.explosion_threshold else "non_explosion"
        pred_conf = p_exp if pred_class == "explosion" else p_not_exp

        # Alert evaluation
        action = decision.get("action", "none")
        now_iso = datetime.now(timezone.utc).isoformat()
        current_time = time.time()
        new_alert_record = None

        if action == "initial_alert" and decision.get("alert") is not None:
            raw_alert = decision["alert"]
            self.total_explosions_detected += 1
            self.total_events_detected += 1
            self.last_alert_time = current_time

            dev_name = self.capture_stream.device_name if self.capture_stream else "Test Injection"
            src = "injection" if self.state == "TESTING" else ("simulation" if self.use_simulation else "microphone")

            new_alert_record = {
                "id": self.total_events_detected,
                "event_id": raw_alert["event_id"],
                "alert_type": "EXPLOSION_DETECTED",
                "confidence": round(p_exp, 4),
                "audio_energy_dbfs": round(rms_dbfs, 1),
                "peak_amplitude": round(peak, 4),
                "category": "explosion",
                "source": src,
                "device": dev_name,
                "timestamp": now_iso,
                "status": "ACTIVE",
                "details": {
                    "p_exp": round(p_exp, 4),
                    "p_not_exp": round(p_not_exp, 4),
                    "lf_ratio": round(lf_ratio, 3) if 'lf_ratio' in locals() else 1.0,
                    "hf_ratio": round(hf_ratio, 3) if 'hf_ratio' in locals() else 0.0,
                },
            }

            # Record in SQLite Alert Database
            self.alert_db.record_alert(
                alert_type="EXPLOSION_DETECTED",
                confidence=p_exp,
                audio_energy_dbfs=rms_dbfs,
                category="explosion",
                source=src,
                timestamp=now_iso,
                details=new_alert_record["details"],
            )

            with self._lock:
                self.active_alert = "EXPLOSION_DETECTED"
                self.active_alert_timestamp = now_iso
                self.active_alert_record = new_alert_record

        elif action == "suppressed_duplicate":
            with self._lock:
                self.total_duplicates_suppressed += 1

        # Reset active alert banner after 2.5 seconds
        if self.active_alert and (current_time - self.last_alert_time) > 2.5:
            with self._lock:
                self.active_alert = None
                self.active_alert_timestamp = None
                self.active_alert_record = None

        with self._lock:
            self.latest_probabilities = {"non_explosion": p_not_exp, "explosion": p_exp}
            self.latest_prediction = pred_class
            self.latest_confidence = pred_conf
            self.latest_rms_dbfs = rms_dbfs
            self.latest_latency_ms = latency_ms
            self.latest_waveform = waveform_preview

        # Broadcast SSE notification
        payload = self.get_status()
        payload.update({
            "timestamp": now_iso,
            "prediction": pred_class,
            "confidence": round(pred_conf, 4),
            "probabilities": {k: round(v, 4) for k, v in self.latest_probabilities.items()},
            "latency_ms": round(latency_ms, 2),
            "waveform": waveform_preview,
            "new_alert": new_alert_record,
        })
        self._notify_subscribers(payload)

    def inject_audio_data(self, audio_data: np.ndarray, native_sr: int = 22050) -> None:
        """Stream an audio array through the exact live capture, preprocessing, and inference pipeline."""
        prev_state = self.state
        self.reset_state()
        self.state = "TESTING"
        try:
            chunk_samples = int(native_sr * 0.25)  # 250 ms chunks
            total_samples = len(audio_data)
            pos = 0

            while pos < total_samples:
                chunk = audio_data[pos : pos + chunk_samples]
                pos += chunk_samples
                self.on_audio_chunk(chunk, native_sr)
                time.sleep(0.01)  # Brief yield for SSE broadcast

            # Flush any remaining buffer
            resampler = self.get_resampler(native_sr)
            remaining = resampler.flush()
            if len(remaining) > 0:
                filtered_remaining = self.streaming_filter.process(remaining)
                windows = self.buffer.push_samples(filtered_remaining)
                for win_time, win_audio in windows:
                    self._process_single_window(win_time, win_audio)

            self.tracker.finalize_stream()
        finally:
            if prev_state == "CAPTURING":
                self.state = "CAPTURING"
            else:
                self.state = "IDLE"

    def _run_file_simulation(self, file_path: Union[str, Path]) -> None:
        from audio.file_loader import load_audio_file
        sig, sr, dur, ch = load_audio_file(file_path)
        sig_np = np.asarray(sig, dtype=np.float32)

        chunk_samples = int(sr * 0.25)
        pos = 0
        total_samples = len(sig_np)

        while self.is_running and pos < total_samples:
            chunk = sig_np[pos : pos + chunk_samples]
            pos += chunk_samples
            self.on_audio_chunk(chunk, sr)
            time.sleep(0.25)  # Real-time hop pacing

        if self.is_running:
            self.tracker.finalize_stream()
            self.stop_monitoring()

    def _run_synthetic_simulation(self) -> None:
        """Stream synthetic background noise for headless unit testing."""
        chunk_samples = int(DEFAULT_SAMPLE_RATE * 0.1)
        while self.is_running:
            noise = (np.random.normal(0, 0.0002, chunk_samples)).astype(np.float32)
            self.on_audio_chunk(noise, DEFAULT_SAMPLE_RATE)
            time.sleep(0.05)
