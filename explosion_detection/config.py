"""Configuration constants for the Explosion detection project.

All paths are resolved robustly relative to this configuration module,
making path resolution independent of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# Base directories
EXPLOSION_DETECTION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPLOSION_DETECTION_ROOT.parent

# Dataset root directory where audio recordings reside
DATA_ROOT = PROJECT_ROOT / "data"

# Manifest directory and files (historical Phase 6B vs corrected Phase 11)
MANIFEST_DIR = EXPLOSION_DETECTION_ROOT / "data" / "manifests"
HISTORICAL_PHASE6B_SPLIT_PATH = MANIFEST_DIR / "dataset_manifest_split.csv"
HISTORICAL_PHASE6B_CLEAN_PATH = MANIFEST_DIR / "dataset_manifest_clean.csv"
PHASE11_MANIFEST_SPLIT_PATH = MANIFEST_DIR / "dataset_manifest_split_phase11.csv"
PHASE11_MANIFEST_CLEAN_PATH = MANIFEST_DIR / "dataset_manifest_phase11.csv"
PHASE11_V2_MANIFEST_SPLIT_PATH = MANIFEST_DIR / "dataset_manifest_split_phase11_v2.csv"
PHASE11_V2_MANIFEST_CLEAN_PATH = MANIFEST_DIR / "dataset_manifest_phase11_v2.csv"
PHASE12_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase12_multiclass.csv"
PHASE12_V2_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase12_multiclass_v2.csv"
PHASE13_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase13_multiclass.csv"
PHASE14_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase14_multiclass.csv"
PHASE15_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase15_multiclass.csv"
PHASE16_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase16_multiclass.csv"
PHASE19_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase19_multiclass.csv"
PHASE20_MANIFEST_MULTICLASS_PATH = MANIFEST_DIR / "dataset_manifest_phase20_multiclass.csv"

# Active manifests default to the repaired, leakage-free Phase 11 binary dataset
MANIFEST_SPLIT_PATH = PHASE11_MANIFEST_SPLIT_PATH
MANIFEST_CLEAN_PATH = PHASE11_MANIFEST_CLEAN_PATH


def validate_manifest_path(manifest_path: Union[str, Path]) -> Path:
    """Validate an explicit manifest path without silent fallback to legacy manifests."""
    if manifest_path is None or not str(manifest_path).strip():
        raise ValueError("Manifest path cannot be empty.")

    raw = Path(manifest_path)
    if raw.is_absolute():
        candidate = raw.resolve()
    elif (Path.cwd() / raw).exists():
        candidate = (Path.cwd() / raw).resolve()
    elif (MANIFEST_DIR / raw).exists():
        candidate = (MANIFEST_DIR / raw).resolve()
    elif (PROJECT_ROOT / raw).exists():
        candidate = (PROJECT_ROOT / raw).resolve()
    else:
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path!r} "
            f"(checked cwd, '{MANIFEST_DIR}', and '{PROJECT_ROOT}')."
        )

    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest path is not a file: '{candidate}'")
    return candidate


def get_manifest_path(version: str = "phase11", split: bool = True) -> Path:
    """Return the validated manifest path for 'phase11' (default), 'phase11_v2', 'phase6b', 'phase12', 'phase12_v2', 'phase13', 'phase14', 'phase15', or 'phase16'."""
    key = version.lower().strip()
    mapping = {
        ("phase11", True): PHASE11_MANIFEST_SPLIT_PATH,
        ("phase11", False): PHASE11_MANIFEST_CLEAN_PATH,
        ("phase11_v2", True): PHASE11_V2_MANIFEST_SPLIT_PATH,
        ("phase11_v2", False): PHASE11_V2_MANIFEST_CLEAN_PATH,
        ("v2", True): PHASE11_V2_MANIFEST_SPLIT_PATH,
        ("v2", False): PHASE11_V2_MANIFEST_CLEAN_PATH,
        ("phase6b", True): HISTORICAL_PHASE6B_SPLIT_PATH,
        ("phase6b", False): HISTORICAL_PHASE6B_CLEAN_PATH,
        ("historical", True): HISTORICAL_PHASE6B_SPLIT_PATH,
        ("historical", False): HISTORICAL_PHASE6B_CLEAN_PATH,
        ("phase12", True): PHASE12_MANIFEST_MULTICLASS_PATH,
        ("phase12", False): PHASE12_MANIFEST_MULTICLASS_PATH,
        ("multiclass", True): PHASE12_MANIFEST_MULTICLASS_PATH,
        ("multiclass", False): PHASE12_MANIFEST_MULTICLASS_PATH,
        ("phase12_v2", True): PHASE12_V2_MANIFEST_MULTICLASS_PATH,
        ("phase12_v2", False): PHASE12_V2_MANIFEST_MULTICLASS_PATH,
        ("multiclass_v2", True): PHASE12_V2_MANIFEST_MULTICLASS_PATH,
        ("multiclass_v2", False): PHASE12_V2_MANIFEST_MULTICLASS_PATH,
        ("phase13", True): PHASE13_MANIFEST_MULTICLASS_PATH,
        ("phase13", False): PHASE13_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase13", True): PHASE13_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase13", False): PHASE13_MANIFEST_MULTICLASS_PATH,
        ("phase14", True): PHASE14_MANIFEST_MULTICLASS_PATH,
        ("phase14", False): PHASE14_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase14", True): PHASE14_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase14", False): PHASE14_MANIFEST_MULTICLASS_PATH,
        ("phase15", True): PHASE15_MANIFEST_MULTICLASS_PATH,
        ("phase15", False): PHASE15_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase15", True): PHASE15_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase15", False): PHASE15_MANIFEST_MULTICLASS_PATH,
        ("phase16", True): PHASE16_MANIFEST_MULTICLASS_PATH,
        ("phase16", False): PHASE16_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase16", True): PHASE16_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase16", False): PHASE16_MANIFEST_MULTICLASS_PATH,
        ("phase19", True): PHASE19_MANIFEST_MULTICLASS_PATH,
        ("phase19", False): PHASE19_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase19", True): PHASE19_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase19", False): PHASE19_MANIFEST_MULTICLASS_PATH,
        ("phase20", True): PHASE20_MANIFEST_MULTICLASS_PATH,
        ("phase20", False): PHASE20_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase20", True): PHASE20_MANIFEST_MULTICLASS_PATH,
        ("multiclass_phase20", False): PHASE20_MANIFEST_MULTICLASS_PATH,
    }
    if (key, split) not in mapping:
        raise ValueError(
            f"Unsupported manifest version {version!r}; expected 'phase11', 'phase11_v2', 'phase6b', 'phase12', 'phase12_v2', 'phase13', 'phase14', 'phase15', 'phase16', 'phase19', or 'phase20'."
        )
    return validate_manifest_path(mapping[(key, split)])


def resolve_active_manifest_path() -> str:
    """Resolve active manifest path from EXPLOSION_MANIFEST_PATH or Phase 11 default."""
    override = os.environ.get("EXPLOSION_MANIFEST_PATH", "").strip()
    if override:
        return str(validate_manifest_path(override))
    return str(validate_manifest_path(PHASE11_MANIFEST_SPLIT_PATH))


# Authoritative active manifest path for training and evaluation
MANIFEST_PATH = resolve_active_manifest_path()

# Models and checkpoints directory
MODELS_DIR = EXPLOSION_DETECTION_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
DEFAULT_CHECKPOINT_FILENAME = "phase11_best_model.pt"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / DEFAULT_CHECKPOINT_FILENAME
HISTORICAL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model.pt"
DEFAULT_MULTICLASS_CHECKPOINT_FILENAME = "multiclass_best_model.pt"
DEFAULT_MULTICLASS_CHECKPOINT_PATH = CHECKPOINTS_DIR / DEFAULT_MULTICLASS_CHECKPOINT_FILENAME

# Reports and outputs directory
REPORTS_DIR = EXPLOSION_DETECTION_ROOT / "reports"

# Default train/val/test split ratios (for audit and verification)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Audio preprocessing specifications (matching AudioPreprocessor)
SAMPLE_RATE = 22050
WINDOW_DURATION_SEC = 1.0
HOP_DURATION_SEC = 0.25
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
EXPECTED_TIME_FRAMES = 44
EXPECTED_INPUT_SHAPE = (1, 64, 44)

# Binary target label definitions (active Phase 11 default)
LABEL_MAP = {
    "non_explosion": 0,
    "explosion": 1
}
INV_LABEL_MAP = {0: "non_explosion", 1: "explosion"}
TARGET_CLASS = "explosion"
NEGATIVE_CLASS = "non_explosion"

# Phase 12 three-class multiclass definitions (explicit opt-in for future multiclass workflow)
try:
    from explosion_detection.multiclass_contract import (
        MULTICLASS_ALERT_MAP,
        MULTICLASS_CLASS_ALERT_MAP,
        MULTICLASS_CLASS_NAMES,
        MULTICLASS_INV_LABEL_MAP,
        MULTICLASS_LABEL_MAP,
    )
except ImportError:
    from multiclass_contract import (
        MULTICLASS_ALERT_MAP,
        MULTICLASS_CLASS_ALERT_MAP,
        MULTICLASS_CLASS_NAMES,
        MULTICLASS_INV_LABEL_MAP,
        MULTICLASS_LABEL_MAP,
    )
