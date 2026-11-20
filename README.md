# Acoustic Explosion Detection

A Python research project that classifies recorded sounds as **explosion** or **non_explosion**. It turns audio into small time-frequency images, applies a convolutional neural network (CNN), and combines predictions across a recording.

The project also contains a separate **three-class extension** for `non_target`, `explosion`, and `fireworks`. Its dataset, model architecture, predictor, metrics, and alert helpers exist, but a trained three-class checkpoint and an end-to-end multiclass training/evaluation CLI are not included.

**Read [ARCHITECTURE.md](ARCHITECTURE.md) for the complete implementation guide and diagrams.** This README explains how to navigate and use the project.

Documentation checked against the workspace on **12 November 2026**.

## Contents

- [What works today](#what-works-today)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Environment and installation](#environment-and-installation)
- [Run detection](#run-detection)
- [Train a binary model](#train-a-binary-model)
- [Evaluate a model](#evaluate-a-model)
- [Datasets and configuration](#datasets-and-configuration)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Suggested reading order](#suggested-reading-order)

## What works today

| Capability | Implementation status |
|---|---|
| Load WAV, MP3, FLAC, OGG, and OGA | Implemented (`soundfile` + stdlib WAV PCM fallback) |
| Convert audio to normalized Log-Mel spectrograms | Implemented with PyTorch STFT, NumPy Slaney Mel, and SciPy resampling (no Librosa/Numba) |
| Train/evaluate a binary CNN | Implemented; existing Phase 11 checkpoint included |
| Detect sounds in one recording or a folder | Implemented through `scripts/predict_file.py` |
| Export predictions and evaluation results | JSON and CSV |
| Validate manifests, paths, groups, and acoustic duplicates | Implemented |
| Three-class explosion/fireworks discrimination | Implemented; active production checkpoint `multiclass_phase21_best_model.pt` (Phase 24 candidate preserved) |
| Capture microphone audio continuously | Implemented via native Windows WaveIn API (`audio/microphone_stream.py`) with zero external C-dependencies |
| Web UI, API server, deployed alert service | Implemented: Standard library HTTP + SSE server (`server/app.py`) & Modernized Executive Dashboard (`server/static/index.html`) |

**Runtime status:** Validated on Python 3.12 (`.venv312`) using `torch`, `numpy`, `scipy`, `soundfile`, `pandas`, `scikit-learn`, and `pytest`. Legacy Librosa/Numba dependencies and older `.venv`/`.venv311` environments have been removed.

**Dataset limitation:** the default Phase 11 V1 manifest has a known acoustic duplicate across train and test. The corrected Phase 11 V2 dataset exists, but defaults and the included checkpoint still refer to V1. Use an explicit V2 manifest for new binary experiments and a new checkpoint filename.

## How it works

```text
Audio file
  -> decode waveform
  -> average channels to mono
  -> resample to 22,050 Hz
  -> overlapping 1-second windows, approximately 0.25 seconds apart
  -> normalized 64 x 44 Log-Mel spectrogram per window
  -> CNN produces a score for each window
  -> threshold + recording aggregation rule
  -> per-window timeline and recording result
```

A spectrogram represents frequency content over time. The CNN learns patterns in this representation rather than reading file names or descriptions. A recording can be declared positive from its highest-scoring window or from a run of consecutive positive windows.

The authoritative **manifest CSV** determines which recordings belong to training, validation, and test sets, and what their labels are. Folder names alone do not determine training labels. Adding audio to a folder does not automatically add it to training.

## Project layout

All paths and commands below are relative to this README's directory:

```text
Explosion/                         # Actual Python project root
├── README.md
├── ARCHITECTURE.md
├── requirements.txt               # Clean verified runtime & test dependencies
├── audio/                         # SoundFile decoding, Torch/SciPy/NumPy preprocessing, paths, audits
├── inference/                     # Predictor contract and offline inference engine
├── training/                      # Manifest validation and binary/multiclass PyTorch datasets
├── scripts/predict_file.py        # Offline prediction CLI
├── configs/default.yaml           # Descriptive baseline; not loaded by current CLIs
├── data/                          # Actual audio, previews, early audit manifests
│   ├── positives/
│   ├── hard_negatives/
│   ├── phase13_approved/
│   ├── manifests/
│   ├── previews/
│   └── raw/
├── data_staging/                  # Candidate downloads and staging catalog
├── tests/                         # Main test suite
└── explosion_detection/
    ├── config.py                  # Active manifest/checkpoint paths and constants
    ├── multiclass_contract.py     # Three-class mappings and independent alerts
    ├── metrics_utils.py           # Binary/multiclass metrics and aggregation
    ├── training_utils.py          # Training window cap and seed utilities
    ├── data/manifests/            # Versioned training/evaluation manifests
    ├── models/
    │   ├── baseline_cnn.py
    │   ├── multiclass_cnn.py
    │   └── checkpoints/           # Existing .pt models and training histories
    ├── scripts/                   # Train, evaluate, versioned dataset builders
    ├── tests/                     # Additional binary pipeline tests
    ├── reports/                   # Training, evaluation, integrity, expansion audits
    └── scratch/                   # Historical analysis and one-off experiments
```

There are two `data/manifests/` directories. Current model workflows use **`explosion_detection/data/manifests/`**. Audio itself lives in the top-level **`data/`** directory.

## Environment and installation

Use PowerShell from the actual project root. In this workspace:

```powershell
Set-Location 'D:\NOESIS\Explosion\Explosion'
```

The active virtual environment `.venv312` uses **Python 3.12.10** with PyTorch, NumPy, SciPy, SoundFile, pandas, scikit-learn, and pytest (Librosa and Numba are not required or installed). Calling its interpreter explicitly avoids activation-policy issues:

```powershell
.\.venv312\Scripts\python.exe --version
.\.venv312\Scripts\python.exe -c "import torch, numpy, scipy, soundfile, pandas, sklearn; print('Runtime packages ready:', torch.__version__)"
```

For a **fresh copy without a virtual environment**, create a Python 3.12 environment and install from `requirements.txt`:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

The workspace environment matches `requirements.txt` and has been validated end-to-end across audio decoding, preprocessing parity, dataset loading, model forward/backward steps, and offline inference.

## Real-Time Detection Dashboard & Microphone Monitoring

Launch the local real-time detection dashboard and live acoustic monitoring server:

```powershell
.\.venv312\Scripts\python.exe server/app.py --host 127.0.0.1 --port 8767
```

Then open your browser to **http://127.0.0.1:8767/**.

### Features
* **Overview Page as Primary Control Center:** Select physical microphone input (`WaveIn`), start/stop monitoring in real time, view live audio energy (RMS dBFS) against the calibrated squelch gate ($-42.0\text{ dBFS}$), and watch the HTML5 canvas oscilloscope trace the live audio signal.
* **Real-Time CNN Inference:** Evaluates sliding windows (1.0s window, 0.25s hop, 4 Hz) using `AudioMulticlassCNN` (`multiclass_phase21_best_model.pt`) with sub-millisecond latency.
* **Prominent Threat Alerts:** Immediate visual distinction between **CRITICAL EXPLOSION** detonations and **FIREWORKS** pyrotechnics, backed by cooldown refractory periods to prevent flood duplication.
* **Alerts History with SQLite Persistence:** Browse historical alerts (`reports/alerts_history.db`) with newest-first ordering, category filters, live search, date/time range query, and CSV export (`/api/alerts/export`).
* **Settings & Policy Management:** Adjust confidence thresholds ($\tau_{exp}, \tau_{fw}$), consecutive window count ($k$), RMS squelch gate level, and cooldown duration in real time.

## Run detection

### One recording with the included binary checkpoint

Replace the input path with your audio file:

```powershell
.\.venv312\Scripts\python.exe scripts/predict_file.py --input 'C:\audio\recording.wav' --model-checkpoint explosion_detection/models/checkpoints/phase11_best_model.pt --threshold 0.90 --aggregation-mode consecutive --output-dir reports/my_inference
```

This CLI's `consecutive` mode requires **2** consecutive windows. The historical Phase 11 evaluation used **3**, so the command above does not reproduce that report's decision policy.

For a peak-window decision, use `--aggregation-mode peak_window`. Any window at or above the threshold makes the recording positive.

### A folder

```powershell
.\.venv312\Scripts\python.exe scripts/predict_file.py --input 'C:\audio\recordings' --model-checkpoint explosion_detection/models/checkpoints/phase11_best_model.pt --threshold 0.90 --aggregation-mode consecutive --output-dir reports/my_batch
```

Subdirectories are searched by default. Add `--non-recursive` to search only the supplied folder.

### Pipeline-only mock run

Omitting `--model-checkpoint` selects `MockTestPredictor`, even though trained binary checkpoints exist:

```powershell
.\.venv312\Scripts\python.exe scripts/predict_file.py --input 'C:\audio\recording.wav' --output-dir reports/mock_check
```

**Mock scores are test fixtures and do not detect explosions.**

### Set exactly three consecutive windows in Python

The CLI does not expose the consecutive-window count. Run this example from the project root in Python:

```python
from audio.preprocessing import AudioPreprocessor
from inference.predictor_interface import PyTorchModelPredictor
from inference.offline_engine import OfflineInferenceEngine

predictor = PyTorchModelPredictor(
    'explosion_detection/models/checkpoints/phase11_best_model.pt',
    device='cpu',
)
engine = OfflineInferenceEngine(
    preprocessor=AudioPreprocessor(),
    predictor=predictor,
    threshold=0.90,
    aggregation_mode='consecutive',
    min_consecutive_windows=3,
)
result = engine.infer_file('C:/audio/recording.wav')
if not result.success:
    raise RuntimeError(result.error_message)
print(result.aggregated_prediction)
print(result.max_explosion_score)
```

### Outputs

| Invocation | Files written |
|---|---|
| Single file without `--output-dir` | Console output only |
| Single file with `--output-dir` | `<input_stem>_inference.json` |
| Folder | `batch_inference_summary.csv` and `batch_inference_details.json` |
| Folder without `--output-dir` | Same files under `reports/offline_inference_runs/` |

JSON includes per-window timestamps and scores. Results also record native sample rate, duration, peak/mean explosion score, positive-window count, predictor identity, `is_mock_predictor`, `success`, and errors.

Always inspect `success`: a failed file can carry `aggregated_prediction="non_explosion"` as a placeholder. Folder processing continues after individual failures; inspect `total_files_failed` and each result. Output filenames are fixed within a run directory, so use different directories to retain multiple runs.

## Train a binary model

For new binary experiments, explicitly use the corrected **Phase 11 V2** manifest and write a new checkpoint:

```powershell
.\.venv312\Scripts\python.exe explosion_detection/scripts/train.py --manifest_path explosion_detection/data/manifests/dataset_manifest_split_phase11_v2.csv --checkpoint_filename phase11_v2_experiment_best.pt --epochs 25 --batch_size 64 --device cpu
```

Training loads only training and validation recordings, precomputes spectrograms, applies training-only SpecAugment, and optimizes binary cross-entropy with AdamW. The default imbalance strategy weights positive examples using training-window counts. Early stopping uses validation loss with patience 5.

Outputs go to `explosion_detection/models/checkpoints/` unless `--checkpoints_dir` is supplied:

- `phase11_v2_experiment_best.pt`
- `phase11_v2_experiment_best_history.json`

Useful options:

| Option | Default / purpose |
|---|---|
| `--lr` | `0.001` |
| `--weight_decay` | `0.0001` |
| `--seed` | `42` |
| `--imbalance_strategy` | `weighted_loss`; also `window_cap`, `both`, `none` |
| `--max_windows_per_rec` | `50`, used when capping is enabled |
| `--metric_for_best` | `val_loss`; also `val_f1`, `val_pr_auc` (window metrics) |
| `--no_augment` | Disable training SpecAugment |
| `--device` | `auto` selects CUDA if available, otherwise CPU |
| `--dry_run` | One synthetic optimization step **after loading real train/val datasets** |

Running the trainer with no overrides uses V1 and writes `phase11_best_model.pt`. Use the explicit filename above to retain the included historical checkpoint. The script has no resume-training option.

## Evaluate a model

Validation with a specific decision rule:

```powershell
.\.venv312\Scripts\python.exe explosion_detection/scripts/evaluate.py --manifest_path explosion_detection/data/manifests/dataset_manifest_split_phase11_v2.csv --checkpoint_path explosion_detection/models/checkpoints/phase11_v2_experiment_best.pt --split val --threshold 0.90 --rule consecutive --consecutive_windows 3 --output_dir explosion_detection/reports/my_v2_validation
```

The checkpoint in this example is produced by the training command above; it is not included in the workspace.

Add `--calibrate_threshold` on validation to search thresholds from `0.10` through `0.90` in `0.01` steps for maximum **window-level macro F1**. This does not optimize the recording-level consecutive rule automatically and does not create a deployment policy file. Record the chosen threshold and rule explicitly.

For a deliberate legacy test evaluation, use `--split test --allow_test` with a fixed threshold and rule. Calibration on test is rejected. The existing test recordings have already been exposed in historical experiments; they are not a fresh independent benchmark. Phase 13 has a separate fresh holdout, but the current binary evaluator cannot consume that multiclass dataset or its `holdout` split.

The evaluator writes `predictions_<split>.csv` and `evaluation_<split>.json`, containing window metrics, recording metrics, and false positives by negative category. `--dry_run` uses synthetic evaluation data, but still imports PyTorch and constructs a model.

## Datasets and configuration

The versioned manifest directory is `explosion_detection/data/manifests/`:

| Manifest | Classes | Recordings | Train / val / test / holdout | Role |
|---|---|---:|---|---|
| `dataset_manifest_split.csv` | Binary | 200 | 160 / 20 / 20 / — | Historical Phase 6B |
| `dataset_manifest_split_phase11.csv` | Binary | 193 | 153 / 20 / 20 / — | Current default; known duplicate leakage |
| `dataset_manifest_split_phase11_v2.csv` | Binary | 192 | 152 / 20 / 20 / — | Corrected binary version |
| `dataset_manifest_phase12_multiclass.csv` | Three | 193 | 153 / 20 / 20 / — | V1 multiclass derivative |
| `dataset_manifest_phase12_multiclass_v2.csv` | Three | 192 | 152 / 20 / 20 / — | Corrected derivative; no fireworks in validation |
| `dataset_manifest_phase13_multiclass.csv` | Three | 242 | 179 / 27 / 20 / 16 | Expanded dataset with fresh holdout |
| `dataset_manifest_phase16_multiclass.csv` | Three | 250 (`247` approved) | 175 / 27 / 20 / 25 (`3` train pending) | Human-reviewed training-ready manifest |
| `dataset_manifest_phase19_multiclass.csv` | Three | 290 (`247` approved, `43` pending) | 198 / 37 / 20 / 25 / 10 (`challenge`) | Expanded everyday non-target dataset |
| `dataset_manifest_phase20_multiclass.csv` | Three | 290 (`286` approved, `4` pending/uncertain) | 194 / 37 / 20 / 25 / 10 (`challenge`) | Approved training manifest with everyday sounds |

Phase 20 contains **286 approved recordings across `non_target`, `explosion`, and `fireworks`** (with 4 unapproved/uncertain train rows excluded cleanly during approved training). Verify preflight or run a synthetic forward/backward step with:

```powershell
.\.venv312\Scripts\python.exe explosion_detection/scripts/train_multiclass.py --manifest_path explosion_detection/data/manifests/dataset_manifest_phase20_multiclass.csv --preflight_only
.\.venv312\Scripts\python.exe explosion_detection/scripts/train_multiclass.py --manifest_path explosion_detection/data/manifests/dataset_manifest_phase20_multiclass.csv --dry_run
```

Configuration is split across several places:

- `explosion_detection/config.py`: default manifest/checkpoint paths and shared constants.
- `EXPLOSION_MANIFEST_PATH`: environment override for the training/evaluation default manifest, read when config is imported.
- `AudioPreprocessor(...)`: actual feature-extraction defaults.
- CLI arguments: thresholds, aggregation, training options, and output paths.
- `configs/default.yaml`: descriptive baseline and planned settings; current entry points do **not** load it. Editing YAML will not change runtime behavior.

Inference defaults to threshold `0.80` and `peak_window`; evaluation defaults to `0.50` and `peak`. The included historical evaluation used threshold `0.90` and 3 consecutive windows. These settings are not interchangeable.

See the architecture guide for validation contracts, dataset derivation, and [historical evaluation caveats](ARCHITECTURE.md#10-historical-results-and-what-they-mean).

## Tests

Run the full project test suite in the validated `.venv312` Python 3.12 environment:

```powershell
.\.venv312\Scripts\python.exe -m pytest tests explosion_detection/tests -q
```

## Troubleshooting

| Symptom | Explanation and next step |
|---|---|
| `WinError 4551` on legacy Python 3.13 `.venv` | PyTorch and Numba binaries were blocked by Windows Application Control under the old `.venv`. Use `.venv312` (Python 3.12), where PyTorch loads natively and `librosa`/`numba` are no longer required. |
| CLI prints a mock warning despite bundled models | Supply `--model-checkpoint` explicitly. The CLI does not select a trained checkpoint automatically. Its additional text saying no trained model exists is stale. |
| Missing manifest during import | `config.py` validates its active manifest at import time. Restore it or correct/unset `EXPLOSION_MANIFEST_PATH`; there is no silent fallback. |
| Manifest path resolves outside the dataset | Dataset loading permits approved paths under `data/` and a specific historical prefix. Use a path such as `data/positives/clip.mp3` and retain the data layout. |
| Adding files does not change training | Update and validate a manifest with labels, groups, and split assignments. Training reads the manifest rather than scanning folders. |
| Multiclass manifest rejected by trainer | The current dataset/trainer is binary. A multiclass CSV alone cannot switch the model, labels, or loss function. |
| Historical script references an old drive/user | Several audit and scratch scripts embed old paths. Inspect and adapt them before use; use the documented current entry points for normal workflows. |
| Report metrics differ from a local inference run | Check checkpoint, manifest version, threshold, aggregation mode, and consecutive-window count. Also distinguish window metrics from recording metrics. |

## Suggested reading order

1. [Architecture overview and data flow](ARCHITECTURE.md#1-system-boundary-and-overall-design).
2. [Audio representation](ARCHITECTURE.md#3-audio-processing-and-shape-contracts).
3. [CNN architectures](ARCHITECTURE.md#4-neural-network-architectures).
4. [Dataset versions and integrity](ARCHITECTURE.md#5-dataset-contracts-and-version-history).
5. [Training, evaluation, and inference workflows](ARCHITECTURE.md#6-training-workflow).
6. [Consolidated Project Status Report](explosion_detection/reports/project_status_report.md).

Earlier reports describe the project at the time they were written. Their paths, default choices, and readiness claims may have been superseded; use the source and later integrity reports to interpret them.
