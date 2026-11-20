# Architecture and Implementation Guide

This guide describes the complete architecture, data flows, and component interactions of the Acoustic Explosion and Fireworks Detection System.

---

## 1. System Overview and Core Pipeline

The system is a real-time and offline acoustic threat detection platform. It continuously captures audio from hardware microphones or files, processes audio into sliding-window Log-Mel spectrograms, evaluates a convolutional neural network (CNN), applies temporal alert policies and squelch gating, persists detections in an SQLite database, and streams live telemetry to a web dashboard.

```mermaid
flowchart TD
    subgraph AudioCapture["1. Audio Acquisition"]
        Mic["Hardware Microphone (ctypes.windll.winmm)"]
        Sim["Simulated File Stream (SimulatedAudioStream)"]
        RingBuf["AudioRingBuffer (5.0s Circular FIFO)"]
        Mic --> RingBuf
        Sim --> RingBuf
    end

    subgraph Preprocessing["2. Audio Preprocessing"]
        Window["Sliding Window Slicer (1.0s window, 0.25s hop)"]
        Squelch["RMS Squelch Gate (-42 dBFS Threshold)"]
        STFT["PyTorch STFT & Slaney Mel Bank"]
        ZNorm["Per-Spectrogram Z-Score Normalization"]
        RingBuf --> Window
        Window --> Squelch
        Squelch -->|RMS >= -42 dBFS| STFT
        Squelch -->|RMS < -42 dBFS (Silence)| Mute["Force [1.0, 0.0, 0.0]"]
        STFT --> ZNorm
    end

    subgraph Model["3. Neural Network Inference"]
        CNN["AudioMulticlassCNN (130,275 params)"]
        Softmax["Softmax Probabilities: [p_nt, p_exp, p_fw]"]
        ZNorm --> CNN
        CNN --> Softmax
    end

    subgraph AlertPolicy["4. Alert & Temporal Decision Engine"]
        Thresholds["Threshold Check (tau_exp >= 0.75, tau_fw >= 0.75)"]
        Consecutive["Consecutive Window Tracker (k >= 1 or k >= 2)"]
        Conflict["Conflict Resolution (Highest Confidence Wins)"]
        Cooldown["Refractory Cooldown Gate (2.5s suppression)"]
        Softmax --> Thresholds
        Mute --> Thresholds
        Thresholds --> Consecutive
        Consecutive --> Conflict
        Conflict --> Cooldown
    end

    subgraph Storage["5. Persistence & Backend"]
        AlertDB["SQLite Database (reports/alerts_history.db)"]
        Server["ThreadingHTTPServer (server/app.py)"]
        Cooldown -->|On Alert Trigger| AlertDB
        AlertDB --> Server
    end

    subgraph Presentation["6. Web Dashboard (server/static/index.html)"]
        SSE["Server-Sent Events Stream (4 Hz)"]
        REST["REST API (/status, /devices, /settings, /alerts)"]
        UI["Single-Page Dashboard (Overview, Live, Alerts, Settings)"]
        Server --> SSE
        Server --> REST
        SSE --> UI
        REST --> UI
    end
```

---

## 2. Component Directory Structure

```text
Explosion/
├── README.md                          # Quick start and operational guide
├── ARCHITECTURE.md                    # System architecture and design
├── PROJECT_TECHNICAL_HANDBOOK.md      # Developer guide for modifying components
├── requirements.txt                   # Verified Python 3.12 dependencies
├── audio/
│   ├── file_loader.py                 # Multi-format audio loader (SoundFile + stdlib PCM)
│   ├── preprocessing.py              # Torch STFT, Slaney Mel bank, polyphase resampler
│   ├── microphone_stream.py           # WaveIn capture, ring buffer, inference engine
│   └── duplicate_audit.py             # Acoustic cross-correlation auditing
├── configs/
│   ├── default.yaml                   # Baseline project configuration
│   └── detection_rules.json           # Calibrated and frozen alert thresholds
├── data/                              # Approved audio files and versioned splits
├── explosion_detection/
│   ├── config.py                      # Active dataset paths and constants
│   ├── multiclass_contract.py         # 3-class mapping and alert decision logic
│   ├── metrics_utils.py               # Recording and window evaluation metrics
│   ├── models/
│   │   ├── multiclass_cnn.py          # AudioMulticlassCNN architecture
│   │   └── checkpoints/               # Trained model checkpoints (.pt)
│   ├── scripts/                       # Training, manifest builders, audio review
│   └── reports/                       # Preserved audit ledgers and status report
├── inference/
│   ├── alert_db.py                    # Thread-safe SQLite alert storage
│   ├── predictor_interface.py        # BasePredictor and MulticlassModelPredictor
│   └── offline_engine.py              # Batch file/folder inference engine
├── server/
│   ├── app.py                         # Lightweight HTTP server with REST and SSE
│   └── static/
│       └── index.html                 # Single-page responsive dashboard UI
└── tests/                             # Full automated test suite (21 modules)
```

---

## 3. Data Flow and Processing Stages

### Stage 1: Audio Capture and Buffering
- **Module:** `audio/microphone_stream.py`
- **Classes:** `WaveInAudioCapture`, `SimulatedAudioStream`, `AudioRingBuffer`
- **Mechanism:** On Windows, `ctypes.windll.winmm` accesses the native multimedia subsystem (`waveInOpen`, `waveInPrepareHeader`, `waveInAddBuffer`, `waveInStart`). Incoming 16-bit PCM audio chunks (250 ms) are converted to float32 and pushed to an `AudioRingBuffer` maintaining a 5.0-second rolling window.
- **Sample Rate:** Standardized to 22,050 Hz mono.
- **Overflow Protection:** Circular array indexing with atomic write pointers ensures continuous non-blocking operation without memory reallocation.

### Stage 2: Feature Extraction and Squelch Gating
- **Module:** `audio/preprocessing.py`
- **Class:** `AudioPreprocessor`
- **Sliding Window:** 1.0-second slices (22,050 samples) with 0.25-second hop strides (5,512 samples), producing 4 evaluation cycles per second.
- **RMS Squelch Gate:** Computes raw waveform root-mean-square amplitude:
  $$\text{RMS}_{\text{dBFS}} = 20 \log_{10}\left(\sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}\right)$$
  If $\text{RMS}_{\text{dBFS}} < -42.0\text{ dBFS}$, the signal is classified as ambient sensor floor or silence, immediately forcing output probabilities to $[p_{nt}=1.0, p_{exp}=0.0, p_{fw}=0.0]$. Real explosions in the dataset exhibit peak window RMS between $-25.0\text{ dBFS}$ and $-3.5\text{ dBFS}$, guaranteeing a minimum $17\text{ dB}$ safety margin.
- **Spectrogram Extraction:** PyTorch centered STFT (`n_fft=1024`, `hop_length=512`, Hann window) projected onto an area-normalized 64-band Slaney Mel filter bank. Per-spectrogram zero-mean unit-variance normalization produces tensors of shape `(1, 64, 44)`.

### Stage 3: Neural Network Classification
- **Module:** `explosion_detection/models/multiclass_cnn.py`
- **Class:** `AudioMulticlassCNN` (130,275 parameters)
- **Architecture:** 4 convolutional blocks (Conv2d, BatchNorm2d, ReLU, MaxPool2d), AdaptiveAvgPool2d, Dropout (0.3), and a 3-logit linear projection head.
- **Output:** Softmax probabilities:
  - Class 0: `non_target`
  - Class 1: `explosion`
  - Class 2: `fireworks`

### Stage 4: Alert Aggregation and Temporal Policy
- **Module:** `explosion_detection/multiclass_contract.py` & `audio/microphone_stream.py`
- **Threshold Gating:** Independent thresholds evaluate target scores ($p_{exp} \ge 0.75$, $p_{fw} \ge 0.75$).
- **Consecutive Confirmation:** Requires $k$ consecutive windows meeting threshold ($k=1$ default, $k=2$ conservative).
- **Conflict Resolution:** If both explosion and fireworks qualify, the class with the higher probability wins:
  $$\text{PrimaryAlert} = \begin{cases} \text{EXPLOSION\_DETECTED} & \text{if } p_{exp} \ge p_{fw} \\ \text{FIREWORKS\_DETECTED} & \text{otherwise} \end{cases}$$
- **Refractory Cooldown:** An alert event triggers a 2.5-second refractory timer ($\Delta t$). Further alerts are suppressed during this window to prevent duplicate records from decaying blast reverberation.

### Stage 5: Persistence and Backend Delivery
- **Modules:** `inference/alert_db.py`, `server/app.py`
- **SQLite Database:** Thread-safe `AlertDatabase` inserts alert records (`timestamp`, `alert_type`, `confidence`, `category`, `audio_energy_dbfs`, `source`) into `reports/alerts_history.db`.
- **Server:** `ThreadingHTTPServer` dispatches REST responses and feeds active client queues for the Server-Sent Events (`GET /api/stream`) channel.

### Stage 6: Client Dashboard
- **Module:** `server/static/index.html`
- **Architecture:** Zero-dependency vanilla HTML5/CSS/JavaScript single-page application.
- **Real-Time Updates:** Receives 4 Hz event payloads from the SSE stream, dynamically updating meters, banners, telemetry, and feeds without page reloads.

---

## 4. Key Design Invariants

1. **Feature Consistency:** Training, offline batch evaluation, and real-time live monitoring share the identical `AudioPreprocessor` class and numerical constants.
2. **Deterministic Squelch:** Squelching occurs prior to spectrogram normalization, preventing low-amplitude quantization noise from being boosted into false-positive features.
3. **AppLocker & Sandbox Compliance:** All networking and audio capture use standard library primitives (`ctypes.windll.winmm`, `http.server`, `sqlite3`), eliminating third-party binary driver blocks.
4. **Frozen Holdout Isolation:** Evaluation on the 25-recording holdout split and 10-recording challenge split is conducted strictly post-freeze, with zero hyperparameter tuning against holdout data.
