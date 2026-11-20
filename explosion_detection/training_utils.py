"""Shared Training and Evaluation Utilities for Explosion Detection.

Provides:
- BalancedExplosionDataset: PyTorch Dataset extending ExplosionDataset with
  configurable per-recording window capping for training.
- Metric calculation utilities (Window-level and Recording-level).
- False positive category analysis.
- Deterministic seed setting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# Import base datasets and preprocessor
from training.dataset import ExplosionDataset, MulticlassExplosionDataset
from audio.preprocessing import AudioPreprocessor

try:
    from explosion_detection.metrics_utils import (
        aggregate_multiclass_recording_predictions,
        aggregate_recording_predictions,
        calibrate_threshold_on_val,
        compute_binary_metrics,
        compute_category_false_positives,
        compute_multiclass_metrics,
    )
    from explosion_detection.multiclass_contract import (
        compute_multiclass_class_weights,
        compute_recording_sampling_weights,
    )
except ImportError:
    from metrics_utils import (
        aggregate_multiclass_recording_predictions,
        aggregate_recording_predictions,
        calibrate_threshold_on_val,
        compute_binary_metrics,
        compute_category_false_positives,
        compute_multiclass_metrics,
    )
    from multiclass_contract import (
        compute_multiclass_class_weights,
        compute_recording_sampling_weights,
    )


def set_seed(seed: int = 42):
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cap_samples_per_recording(samples: List[Dict], max_windows: Optional[int]) -> List[Dict]:
    """Deterministically downsample windows per recording using evenly spaced indices."""
    if max_windows is None or max_windows <= 0:
        return samples
    from collections import defaultdict
    recording_samples = defaultdict(list)
    for s in samples:
        recording_samples[s["original_recording_id"]].append(s)

    capped_samples: List[Dict] = []
    for s_list in recording_samples.values():
        if len(s_list) <= max_windows:
            capped_samples.extend(s_list)
        else:
            indices = np.linspace(0, len(s_list) - 1, max_windows, dtype=int)
            capped_samples.extend(s_list[i] for i in indices)
    return capped_samples


class BalancedExplosionDataset(ExplosionDataset):
    """PyTorch Dataset extending ExplosionDataset with per-recording window capping.
    
    Prevents extremely long recordings (e.g. 388s audio generating 1,550 windows)
    from dominating the training optimization steps.
    Window capping is strictly active only when is_training=True.
    Validation and test datasets always retain 100% of their windows.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        is_training: bool = False,
        seed: int = 42,
        augment: bool = False,
        preload_into_memory: bool = True,
        max_windows_per_recording: Optional[int] = None,
        data_root: Optional[Union[str, Path]] = None,
    ):
        self.max_windows_per_recording = max_windows_per_recording
        
        super().__init__(
            manifest_path=manifest_path,
            split=split,
            preprocessor=preprocessor,
            is_training=is_training,
            seed=seed,
            augment=augment,
            preload_into_memory=preload_into_memory,
            data_root=data_root,
        )

    def _build_samples(self):
        """Build sliding window samples with optional per-recording capping in training."""
        super()._build_samples()
        if self.is_training:
            self.samples = _cap_samples_per_recording(self.samples, self.max_windows_per_recording)


class BalancedMulticlassExplosionDataset(MulticlassExplosionDataset):
    """Three-class Dataset with training-only per-recording window capping and approved filtering."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        is_training: bool = False,
        seed: int = 42,
        augment: bool = False,
        preload_into_memory: bool = True,
        max_windows_per_recording: Optional[int] = None,
        data_root: Optional[Union[str, Path]] = None,
        manifest_version: str = "auto",
        approved_only: bool = True,
    ):
        self.max_windows_per_recording = max_windows_per_recording
        super().__init__(
            manifest_path=manifest_path,
            split=split,
            preprocessor=preprocessor,
            is_training=is_training,
            seed=seed,
            augment=augment,
            preload_into_memory=preload_into_memory,
            data_root=data_root,
            manifest_version=manifest_version,
            approved_only=approved_only,
        )

    def _build_samples(self):
        """Build three-class sliding window samples with optional per-recording capping in training."""
        super()._build_samples()
        if self.is_training:
            self.samples = _cap_samples_per_recording(self.samples, self.max_windows_per_recording)


__all__ = [
    "set_seed",
    "compute_multiclass_class_weights",
    "compute_recording_sampling_weights",
    "BalancedExplosionDataset",
    "BalancedMulticlassExplosionDataset",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "aggregate_recording_predictions",
    "aggregate_multiclass_recording_predictions",
    "compute_category_false_positives",
    "calibrate_threshold_on_val",
]

