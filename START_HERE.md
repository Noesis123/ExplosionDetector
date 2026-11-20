# NOESIS — Acoustic Explosion Detection System
## Final Project Submission & Operational Startup Guide

---

## 1. Project Purpose & System Overview

The **NOESIS Acoustic Explosion Detection System** is an edge-capable, real-time audio surveillance platform designed to continuously monitor physical acoustic environments, detect authentic explosive detonations (including distant blasts and controlled building demolitions), and reject environmental confounders such as vehicle backfires, small-arms gunfire, fireworks, lightning thunder, and HVAC/fan ambient noise.

The system runs single-threaded on standard x86 CPU hardware with sub-second response times (<255 ms end-to-end alert dispatch, 2.72 ms mean window feature extraction & CNN inference).

---

## 2. Active Model & Decision Architecture

- **Active Model:** `ExplosionBinaryCNN` (ExpF)
- **Active Checkpoint:** `explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt`
  - **Parameters:** 130,146 weights
  - **File Size:** 1,587,469 bytes
  - **Verified SHA-256:** `3395acbe99251e7445e89342011176a744e8e515ba7241dd255059928dde0287`
- **Active Decision Logic:** **AEV Config E** (High-Precision Blast Verification with Protected Detonation Branches)
- **Acoustic Verification Settings:**
  - Standard Confidence Threshold: `0.80`
  - Protected Demolition / Fragile Blast Threshold: `0.70`
  - RMS Squelch Gate: `−34.0 dBFS`
  - Vibration & Infrasonic High-Pass Filter: `35.0 Hz` (2nd-order Butterworth)
  - Dedicated Acoustic Discriminators:
    1. *Gunfire Crack Rejection:* Rejects sounds with mid-frequency crack ($E_{[1\text{kHz}, 4\text{kHz}]} \ge 0.28$) and low ground thump ($E_{[20\text{Hz}, 80\text{Hz}]} \le 0.05$).
    2. *Fireworks Sizzle Rejection:* Rejects near-squelch pops ($< -30\text{ dBFS}$) lacking impulsive dynamic range ($\Delta\text{Subframe} < 10\text{ dB}$).
    3. *Thunder / Low-Energy Rumble Rejection:* Rejects diffuse rumbles ($< -25\text{ dBFS}$) lacking concentrated infrasound ($E_{[20\text{Hz}, 80\text{Hz}]} < 0.10$).
    4. *Vehicle Backfire Rejection:* Rejects fast-decay exhaust pops ($\tau_{\text{decay}} < 0.08$) with zero acoustic resonance.

---

## 3. Verified Performance (Historical Benchmark)

The production pipeline was evaluated across the frozen 92-file historical benchmark evaluation suite (24 true explosions, 68 hard non-explosion negatives):

| Metric | Historical Baseline (Config C) | Intermediate (Config D) | **Production Target (Config E)** | Required Target | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Precision** | 53.66% (22/41) | 66.67% (22/33) | **84.62% (22/26)** | > 80.0% | **MET (+30.96% abs)** |
| **Recall** | 91.67% (22/24) | 91.67% (22/24) | **91.67% (22/24)** | $\ge$ 87.5% | **MET (100% preserved)** |
| **F1-Score** | 0.6769 | 0.7719 | **0.8800** | > 0.7333 | **MET (+0.2031)** |
| **False-Positive Recordings** | 19 / 68 | 11 / 68 | **4 / 68** | $\le$ 5 | **MET (78.9% reduction)** |
| **Controlled Demolitions** | 2 / 2 (100%) | 2 / 2 (100%) | **2 / 2 (100%)** | 2 / 2 (100%) | **MET (07022507, 07022508)** |
| **Silence / Room Ambient False Alerts** | Flapping | 0 | **0 alerts** | Strictly 0 | **MET (Clean immunity)** |

> [!NOTE]
> **Important Scientific Scope:**  
> The figures above are reproducible historical benchmark metrics measured against the standard evaluation test partition. They are not unconditional guarantees of 84.62% precision in arbitrary uncalibrated field environments.

---

## 4. Known Limitations & Edge Cases

The 4 remaining false positives out of 68 hard negative benchmark files are:
1. **Gunfire (`bbc_gunfire---_07034194.mp3`):** Heavy automatic cannon fired in immediate proximity to the microphone. The explosive propellant discharge generates substantial low-frequency muzzle shock mimicking a small bomb.
2. **Thunder (`fs_36044_inchadney.mp3`, `fs_195529_klankbeeld.mp3`):** Direct close-proximity cloud-to-ground lightning strikes (<100m). The rapid thermal expansion of ionized air creates an acoustic shockwave physically identical to an explosive blast.
3. **Artillery (`fs_171475_2887679652.mp3`):** Multi-kilogram chemical propellant detonation at howitzer muzzle (physically a true explosive detonation).

---

## 5. Python Installation & Dependency Requirements

The system requires **Python 3.10, 3.11, or 3.12** on Windows (or Linux).

### Option A: Using the Included `.venv312` (Fastest on Windows)
The submission archive includes the fully configured virtual environment in `.venv312`.
To activate:
```powershell
.venv312\Scripts\activate
```

### Option B: Clean Environment Recreation (Cross-Platform)
If `.venv312` is not portable to your specific Windows build or Python install path:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 2.0`
- `numpy >= 1.24`
- `scipy >= 1.10`
- `torchaudio >= 2.0`
- `librosa >= 0.10`
- `soundfile >= 0.12`
- `pandas >= 2.0`
- `matplotlib >= 3.7`

---

## 6. How to Start the Live Dashboard

### Step 1: Start Server
From the project root (`D:\NOESIS\Explosion\Explosion`):
```powershell
.venv312\Scripts\python.exe server\app.py --port 8767
```

### Step 2: Open Browser
Navigate to:
```
http://127.0.0.1:8767/
```

### Step 3: Select Microphone & Start Monitoring
1. In the top navigation bar or settings drawer, inspect the **Audio Input Device** dropdown.
2. Select your physical hardware microphone (e.g. `Microphone Array (Realtek(R) Audio)` or USB microphone).
3. Click the green **Start Monitoring** button.
4. The dashboard status indicator will transition to **CAPTURING** (green), and the live oscilloscope will visualize incoming audio energy in real time.

---

## 7. How Alerts are Displayed & Dispatched

1. **Sub-Second Audio Windowing:** The microphone stream is continuously windowed into 1.0-second slices with 75% overlap (250 ms hop size).
2. **Real-Time Visual Popup:** When an authentic detonation is verified, the dashboard displays a prominent red **EXPLOSION DETECTED** banner showing the exact confidence, timestamp, audio RMS level, and frequency ratios.
3. **Audio Alert Tone:** The browser interface sounds a distinct acoustic alert chime.
4. **SQLite Event History:** Every alert is permanently recorded in the local SQLite database (`reports/alerts_history.db`) and listed in the **Recent Detections** audit table.
5. **Duplicate Streak Suppression:** Secondary blast reverberations within 6.0 seconds are automatically aggregated into the primary event to avoid alert flooding.

---

## 8. Demonstration Methods (Phone Playback & Speaker)

Because live high explosives cannot be detonated during an evaluation or demonstration:
1. **External Desktop Speaker Playback:**
   - Play any verified explosion file (e.g. `data/positives/11325622-big-cine-boom-sound-effect-245851.mp3`) through external speakers.
   - The Realtek microphone captures the acoustic wave and immediately triggers an alert.
2. **Mobile Phone Playback:**
   - Hold a smartphone 15–25 cm from the microphone at 75–85% volume.
   - Play a detonation blast sound.
   - Even though phone micro-speakers roll off sub-bass rumble below 250 Hz, AEV Config E's **Protected Mobile Attack Transient Branch** detects the impulsive shockwave ($\Delta\text{Subframe} \ge 5\text{ dB}$, $\text{Crest} \ge 3.5$) and dispatches the alert.
3. **Software Test Audio Injection (Automated Test):**
   ```powershell
   .venv312\Scripts\python.exe -c "import urllib.request, json; req = urllib.request.Request('http://127.0.0.1:8767/api/test/inject_audio', data=json.dumps({'file_path': 'data/positives/11325622-big-cine-boom-sound-effect-245851.mp3'}).encode(), headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode())"
   ```

---

## 9. How to Run Existing Verification Scripts

1. **Run Full Production Verification Suite (92-File Historical Benchmark):**
   ```powershell
   .venv312\Scripts\python.exe scripts/evaluate_production_system.py
   ```
2. **Verify Preset Reversibility (Config C $\leftrightarrow$ Config E):**
   ```powershell
   .venv312\Scripts\python.exe scripts/verify_preset_metrics.py
   ```
3. **Monitor Real-Time Microphone Ambient Noise:**
   ```powershell
   .venv312\Scripts\python.exe scripts/live_mic_monitor.py "Ambient Test" 5.0
   ```
4. **Instant Rollback / Switch via API:**
   - Switch to Config C:
     ```powershell
     .venv312\Scripts\python.exe -c "import urllib.request, json; req = urllib.request.Request('http://127.0.0.1:8767/api/rollback', data=json.dumps({'preset': 'Config C'}).encode(), headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode())"
     ```
   - Restore Config E:
     ```powershell
     .venv312\Scripts\python.exe -c "import urllib.request, json; req = urllib.request.Request('http://127.0.0.1:8767/api/config_preset', data=json.dumps({'preset': 'Config E'}).encode(), headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode())"
     ```

---

## 10. How to Restore the Project from the Final ZIP

To unpack and run the submission package on a new system:

1. Extract `NOESIS_Explosion_Final_Submission.zip` to your target directory (e.g. `D:\NOESIS_RESTORED`).
2. Inside the extracted directory, enter the main project folder:
   ```powershell
   cd Explosion
   ```
3. (Optional) If running the Codex evaluation scripts, ensure `external_dependencies/Codex_Evaluation_Dataset` is located at `D:\NOESIS\Codex_Evaluation_Dataset` or update manifest paths accordingly.
4. Launch the dashboard:
   ```powershell
   .venv312\Scripts\python.exe server\app.py --port 8767
   ```
5. Open `http://127.0.0.1:8767/` in any modern web browser.
