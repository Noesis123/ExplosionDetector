# Developer Technical Handbook: Acoustic Threat Detection

A practical, hands-on guide for developers maintaining, extending, or customizing the Acoustic Explosion and Fireworks Detection application.

---

## 1. Quick Tour: Important Folders and Files

| File / Folder | Core Purpose | Key Classes & Functions to Know |
| :--- | :--- | :--- |
| `audio/microphone_stream.py` | Audio capture, circular buffer, live inference worker | `WaveInAudioCapture`, `SimulatedAudioStream`, `AudioRingBuffer`, `RealtimeInferenceEngine` |
| `audio/preprocessing.py` | Signal resampling, framing, Log-Mel spectrograms | `AudioPreprocessor`, `to_mono()`, `resample()`, `compute_log_mel_spectrogram()` |
| `audio/file_loader.py` | File decoding (WAV, MP3, FLAC, OGG) | `load_audio_file()`, `validate_audio_file()` |
| `inference/alert_db.py` | SQLite storage for alert events | `AlertDatabase`, `record_alert()`, `get_alerts()`, `count_alerts()` |
| `inference/predictor_interface.py`| Model wrappers and prediction contracts | `MulticlassModelPredictor`, `predict_batch()`, `predict_alerts()` |
| `explosion_detection/models/multiclass_cnn.py` | PyTorch neural network architecture | `AudioMulticlassCNN`, `build_multiclass_loss()` |
| `explosion_detection/multiclass_contract.py` | Threshold and consecutive alert evaluation | `evaluate_multiclass_alerts()`, `MULTICLASS_LABEL_MAP` |
| `configs/detection_rules.json` | Calibrated threshold & squelch values | JSON config read by engine and server |
| `server/app.py` | Lightweight HTTP & SSE server | `DetectionAPIRequestHandler`, `DetectionServerState`, `create_server()` |
| `server/static/index.html` | Front-end web dashboard | `initSSE()`, `updateDashboard()`, `startMonitoring()`, `saveSettings()` |
| `tests/` | Automated test suites | `test_phase22_integration.py`, `test_phase23_final_verification.py` |

---

## 2. Technology Choices and Why They Are Used

1. **Pure PyTorch / NumPy / SciPy / SoundFile (No Librosa / Numba):**
   - *Why:* Librosa depends on Numba's LLVM JIT compiler, which frequently fails under enterprise Windows Application Control policies. PyTorch STFT and SciPy polyphase sinc resampling deliver equivalent numerical fidelity with zero JIT compilation overhead.
2. **Windows WaveIn API via `ctypes.windll.winmm`:**
   - *Why:* Third-party audio libraries (e.g. `pyaudio`, `sounddevice`) require PortAudio DLL binaries or external C build tools that cannot be compiled or installed when `pip.exe` is blocked. `winmm` is built into every Windows installation since Windows NT.
3. **Python Standard Library `http.server` & Server-Sent Events (SSE):**
   - *Why:* Flask, FastAPI, or Django would introduce extra dependencies and complex ASGI servers. `ThreadingHTTPServer` handles static file delivery, REST endpoints, and sub-second streaming using native threads and standard library queues.
4. **SQLite (`sqlite3`):**
   - *Why:* Zero-configuration, serverless, single-file database (`reports/alerts_history.db`) that survives application restarts without managing database services.

---

## 3. How Audio Becomes a Prediction (Step-by-Step Code Walkthrough)

Understanding the lifecycle of a 1.0-second audio slice:

1. **Capture:** The background thread in `WaveInAudioCapture._capture_loop()` receives a 250 ms chunk of 16-bit PCM bytes from Windows multimedia buffers, converts it to float32 normalized to $[-1.0, 1.0]$, and calls `ring_buffer.write(pcm_f32)`.
2. **Framing:** Every 250 ms, `RealtimeInferenceEngine._inference_loop()` calls:
   ```python
   window_audio = self.capture_stream.ring_buffer.read_latest(22050)
   ```
   This extracts the most recent 1.0 second of audio (22,050 samples).
3. **Squelch Check:** The raw waveform RMS is computed:
   ```python
   rms = float(np.sqrt(np.mean(window_audio ** 2)))
   rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
   ```
   If `rms_dbfs < self.rms_squelch_dbfs` (default $-42.0\text{ dBFS}$), the audio is deemed silence or sensor noise. Inference is skipped, and probabilities are immediately set to `[1.0, 0.0, 0.0]`.
4. **Log-Mel Extraction:** If above squelch, `AudioPreprocessor.compute_log_mel_spectrogram(window_audio)` executes:
   - Centered STFT: Hann window, `n_fft=1024`, `hop_length=512`.
   - Mel projection: Dot product with 64-band Slaney Mel filter bank.
   - Decibel scale: Power to top-80 dB.
   - Normalization: Per-spectrogram zero-mean unit-variance $z = (x - \mu) / (\sigma + 10^{-6})$.
   - Output shape: `(64, 44)` (64 frequency bins, 44 time frames).
5. **Neural Network Forward Pass:**
   ```python
   tensor = torch.from_numpy(spec[np.newaxis, np.newaxis, ...]) # Shape: (1, 1, 64, 44)
   logits = self.model(tensor)
   probs = torch.softmax(logits, dim=-1).numpy()[0]
   # probs = [p_non_target, p_explosion, p_fireworks]
   ```

---

## 4. How Predictions Become Alerts

Alert decisions are governed by temporal logic in `RealtimeInferenceEngine._inference_loop()`:

```python
# 1. Consecutive confirmation tracking
if p_exp >= self.explosion_threshold:
    self.consecutive_exp_count += 1
else:
    self.consecutive_exp_count = 0

exp_triggered = self.consecutive_exp_count >= self.consecutive_windows

# 2. Refractory cooldown gate
cooldown_active = (current_time - self.last_alert_time) < self.cooldown_seconds

# 3. Conflict resolution and dispatch
if (exp_triggered or fw_triggered) and not cooldown_active:
    if exp_triggered and fw_triggered:
        new_alert = "EXPLOSION_DETECTED" if p_exp >= p_fw else "FIREWORKS_DETECTED"
    elif exp_triggered:
        new_alert = "EXPLOSION_DETECTED"
    elif fw_triggered:
        new_alert = "FIREWORKS_DETECTED"

    if new_alert:
        self.last_alert_time = current_time
        self.alert_db.record_alert(alert_type=new_alert, confidence=conf, ...)
```

---

## 5. How the Dashboard Receives Information

The web dashboard uses **Server-Sent Events (SSE)** rather than polling:

1. When `server/static/index.html` loads, JavaScript executes `new EventSource('/api/stream')`.
2. The server's `DetectionAPIRequestHandler.do_GET()` calls `state.add_sse_client()`, creating a client queue.
3. Every 250 ms, after the inference engine evaluates a window, it calls `_notify_subscribers(payload)`.
4. The server pushes the payload down the open HTTP connection as a text stream (`data: {"prediction": "non_target", ...}\n\n`).
5. The browser's `sseSource.onmessage` receives the event and immediately updates the gauges and banners.

---

## 6. How to Modify and Customize the System

### A. How to Change Detection Thresholds
- **Option 1 (Via Dashboard):** Navigate to the **Settings** tab in the browser, adjust the sliders for Explosion Threshold, Fireworks Threshold, Consecutive Windows, or RMS Squelch, and click **Save Configuration**.
- **Option 2 (Via Config File):** Edit `configs/detection_rules.json`:
  ```json
  "alert_rules": {
    "explosion_threshold": 0.75,
    "fireworks_threshold": 0.75,
    "consecutive_windows": 1,
    "rms_squelch_dbfs": -42.0,
    "cooldown_seconds": 2.5
  }
  ```
  Restart `server/app.py` to reload.

### B. How to Modify Audio Preprocessing
If you change STFT parameters (e.g. `n_mels=128` or `sample_rate=16000`) in `audio/preprocessing.py`:
- **Crucial Invariant:** You MUST update `AudioPreprocessor.__init__()` and ensure `AudioMulticlassCNN`'s input layer and linear classifier dimension match the new shape:
  $$\text{time\_frames} = 1 + \left\lfloor \frac{\text{sample\_rate} \times \text{window\_duration}}{\text{hop\_length}} \right\rfloor$$
- **Warning:** Any change to preprocessing parameters invalidates existing model weights (`multiclass_phase21_best_model.pt`) and requires full retraining.

### C. How to Add or Change Model Classes
To add a 4th class (e.g. `gunfire`):
1. In `explosion_detection/multiclass_contract.py`:
   - Update `MULTICLASS_CLASS_NAMES = ["non_target", "explosion", "fireworks", "gunfire"]`.
   - Update `MULTICLASS_NUM_CLASSES = 4`.
   - Update `MULTICLASS_ALERT_MAP` with the new alert tag.
2. In `explosion_detection/models/multiclass_cnn.py`:
   - Set `num_classes=4`.
3. In `server/static/index.html`:
   - Add a confidence meter and badge for the new class.
4. **Retraining Required:** Changing class count alters the final linear layer (`fc2 = nn.Linear(128, 4)`), requiring full model retraining with `--authorize_training`.

---

## 7. When Retraining is Required vs Not Required

| Change Description | Retraining Required? | What to Update Instead |
| :--- | :---: | :--- |
| Changing confidence thresholds ($\tau_{exp}, \tau_{fw}$) | **NO** | Update `configs/detection_rules.json` or dashboard settings. |
| Changing consecutive window count ($k$) | **NO** | Update `configs/detection_rules.json` or dashboard settings. |
| Adjusting RMS squelch gate level | **NO** | Update `configs/detection_rules.json` or dashboard settings. |
| Adjusting alert cooldown duration | **NO** | Update `configs/detection_rules.json` or dashboard settings. |
| Adding new audio training data | **YES** | Run `train_multiclass.py` with `--authorize_training`. |
| Changing sample rate, STFT n_fft, or n_mels | **YES** | Retraining required due to feature representation change. |
| Adding or removing model classes | **YES** | Retraining required due to classification head dimension change. |

---

## 8. Debugging and Testing Modifications

### Running Quick Unit & Integration Tests
```powershell
# Run the real-time inference and modernized dashboard tests
.\.venv312\Scripts\python.exe -m pytest tests/test_phase22_integration.py tests/test_phase23_final_verification.py tests/test_dashboard_modernization.py -v

# Run the complete project test suite
.\.venv312\Scripts\python.exe -m pytest tests explosion_detection/tests -q
```

### Simulating a Specific Audio File in the Engine
You can test the complete real-time engine against any audio file from Python:
```python
from audio.microphone_stream import RealtimeInferenceEngine
import time

engine = RealtimeInferenceEngine(
    use_simulation=True,
    simulation_file="data/positives/flutie8211-bomb-and-echo-2-540400.mp3"
)
engine.start_monitoring()
time.sleep(2.0)
print(engine.get_status())
engine.stop_monitoring()
```

### Inspecting Alerts Stored in SQLite
```python
from inference.alert_db import AlertDatabase
db = AlertDatabase()
for alert in db.get_alerts(limit=5):
    print(alert)
```

