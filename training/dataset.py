"""Reproducible PyTorch Dataset for Acoustic Explosion Detection.

Implements recording-level train/val/test separation, overlapping window extraction,
Log-Mel feature extraction, deterministic seeding, and training augmentations.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from audio.file_loader import load_audio_file, AudioLoadError
from audio.preprocessing import AudioPreprocessor
from training.manifest_validator import (
    validate_and_load_manifest,
    validate_and_load_multiclass_manifest,
    validate_and_load_phase13_multiclass_manifest,
    validate_and_load_phase14_multiclass_manifest,
    validate_and_load_phase15_multiclass_manifest,
)


class ExplosionDataset(Dataset):
    """PyTorch Dataset for binary explosion sound classification."""

    LABEL_MAP = {
        "non_explosion": 0,
        "explosion": 1
    }
    INV_LABEL_MAP = {0: "non_explosion", 1: "explosion"}
    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        is_training: bool = False,
        seed: int = 42,
        augment: bool = False,
        preload_into_memory: bool = True,
        data_root: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            manifest_path: Path to dataset split manifest CSV
            split: One of VALID_SPLITS
            preprocessor: AudioPreprocessor instance (or None for default baseline)
            is_training: Whether this dataset is used for model training
            seed: Deterministic random seed
            augment: Whether to apply SpecAugment / gain augmentations (only active if is_training=True)
            preload_into_memory: Precompute and store spectrograms in memory for fast training
            data_root: Optional dataset root directory override for portable path resolution
        """
        if split not in self.VALID_SPLITS:
            if self.VALID_SPLITS == ("train", "val", "test"):
                raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.data_root = Path(data_root) if data_root is not None else None
        self.is_training = is_training
        self.seed = seed
        self.augment = augment and is_training
        self.preload_into_memory = preload_into_memory

        if preprocessor is None:
            self.preprocessor = AudioPreprocessor()
        else:
            self.preprocessor = preprocessor

        # Set deterministic seeds
        self._set_seed(self.seed)

        # 1. Validate manifest schema/integrity and resolve split recordings strictly
        self.validation_result = self._validate_split_manifest()
        self.records = list(self.validation_result.split_rows)

        # 2. Extract windowed dataset samples (fail loudly on any missing/corrupt recording)
        self.samples: List[Dict] = []
        self._build_samples()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _validate_split_manifest(self):
        """Validate the split manifest and return the resolved ManifestValidationResult."""
        return validate_and_load_manifest(
            manifest_path=self.manifest_path,
            split=self.split,
            data_root=self.data_root,
        )

    def _load_manifest_records(self) -> List[Dict]:
        """Validate manifest CSV and return resolved entries for the current split."""
        return list(self._validate_split_manifest().split_rows)

    @staticmethod
    def _parse_trim_bounds(record: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Parse optional traceable event trim bounds from a manifest record."""
        raw_start = str(record.get("trim_start_sec") or "").strip()
        raw_end = str(record.get("trim_end_sec") or "").strip()
        if not raw_start and not raw_end:
            return None, None
        if bool(raw_start) != bool(raw_end):
            raise ValueError(
                f"Manifest row {record.get('file_path')!r} must specify both trim_start_sec and trim_end_sec."
            )
        start_val = float(raw_start)
        end_val = float(raw_end)
        if start_val < 0.0 or end_val - start_val < 1.0:
            raise ValueError(
                f"Invalid trim bounds [{start_val}, {end_val}] for {record.get('file_path')!r}; "
                f"expected 0 <= trim_start_sec and trim_end_sec - trim_start_sec >= 1.0."
            )
        return start_val, end_val

    @staticmethod
    def _slice_audio_if_trimmed(
        audio: np.ndarray,
        sr: int,
        trim_start_sec: Optional[float],
        trim_end_sec: Optional[float],
    ) -> Tuple[np.ndarray, float]:
        """Slice waveform deterministically to optional traceable [trim_start_sec, trim_end_sec]."""
        if trim_start_sec is None or trim_end_sec is None:
            return audio, 0.0
        start_idx = max(0, int(round(trim_start_sec * sr)))
        end_idx = min(len(audio), int(round(trim_end_sec * sr)))
        if end_idx - start_idx < sr:
            raise AudioLoadError(
                f"Trimmed audio segment [{trim_start_sec:.2f}s, {trim_end_sec:.2f}s] has fewer than {sr} samples."
            )
        return audio[start_idx:end_idx], float(trim_start_sec)

    def _build_samples(self):
        """Process audio files and slice into sliding windows without silent skips."""
        for r in self.records:
            file_p = Path(r["resolved_file_path"])
            label_str = r["label"].lower().strip()
            if label_str not in self.LABEL_MAP:
                raise ValueError(f"Unknown label '{label_str}' in record {r}")

            label_int = self.LABEL_MAP[label_str]
            rec_id = r.get("original_recording_id", file_p.stem)
            src_ds = r.get("source_dataset", "unknown")
            # Authoritative manifest category takes precedence; do not infer label from directories.
            category = (r.get("category") or "").strip() or file_p.parent.name
            group_id = (r.get("group_id") or "").strip()
            trim_start, trim_end = self._parse_trim_bounds(r)

            audio, sr, dur, ch = load_audio_file(file_p)
            audio, offset_sec = self._slice_audio_if_trimmed(audio, sr, trim_start, trim_end)
            specs, timestamps = self.preprocessor.process_file_audio(audio, sr)
            if len(specs) == 0:
                raise AudioLoadError(
                    f"Audio preprocessing produced 0 windows for {file_p} in '{self.split}' split."
                )

            for i in range(len(specs)):
                spec = specs[i]  # Shape: (n_mels, expected_time_frames)
                t_start, t_end = timestamps[i]

                sample_data = {
                    "spec": spec if self.preload_into_memory else None,
                    "file_path": str(file_p),
                    "manifest_file_path": r["file_path"],
                    "window_index": i,
                    "label": label_int,
                    "label_str": label_str,
                    "category": category,
                    "group_id": group_id,
                    "trim_start_sec": trim_start,
                    "trim_end_sec": trim_end,
                    "start_sec": round(t_start + offset_sec, 3),
                    "end_sec": round(t_end + offset_sec, 3),
                    "original_recording_id": rec_id,
                    "source_dataset": src_ds
                }
                self.samples.append(sample_data)

        print(
            f"[Dataset] Loaded {len(self.records)} recordings / "
            f"{len(self.samples)} windows for split '{self.split}'."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_spec_augment(
        self,
        spec: np.ndarray,
        freq_mask_max: int = 8,
        time_mask_max: int = 6
    ) -> np.ndarray:
        """Apply SpecAugment time and frequency masking."""
        aug_spec = spec.copy()
        n_mels, n_frames = aug_spec.shape

        # Frequency mask
        if freq_mask_max > 0:
            f_width = random.randint(1, min(freq_mask_max, n_mels // 2))
            f_start = random.randint(0, n_mels - f_width)
            aug_spec[f_start : f_start + f_width, :] = 0.0

        # Time mask
        if time_mask_max > 0:
            t_width = random.randint(1, min(time_mask_max, n_frames // 2))
            t_start = random.randint(0, n_frames - t_width)
            aug_spec[:, t_start : t_start + t_width] = 0.0

        return aug_spec

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]

        if self.preload_into_memory and sample["spec"] is not None:
            spec = sample["spec"]
        else:
            # Recompute on the fly if memory caching disabled
            file_p = sample["file_path"]
            win_idx = sample["window_index"]
            audio, sr, _, _ = load_audio_file(file_p)
            audio, _ = self._slice_audio_if_trimmed(
                audio, sr, sample.get("trim_start_sec"), sample.get("trim_end_sec")
            )
            specs, _ = self.preprocessor.process_file_audio(audio, sr)
            spec = specs[win_idx]

        # Apply data augmentations if training
        if self.augment:
            spec = self._apply_spec_augment(spec)

        # Convert to PyTorch Tensor of shape (1, n_mels, time_frames)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        metadata = {
            "file_path": sample["file_path"],
            "original_recording_id": sample["original_recording_id"],
            "source_dataset": sample["source_dataset"],
            "label_str": sample["label_str"],
            "category": sample["category"],
            "group_id": sample["group_id"],
            "window_index": sample["window_index"],
            "start_sec": sample["start_sec"],
            "end_sec": sample["end_sec"]
        }

        return spec_tensor, label_tensor, metadata

    def get_class_counts(self) -> Dict[str, int]:
        """Return total window count per class in this split."""
        counts = {label_name: 0 for label_name in self.LABEL_MAP}
        for s in self.samples:
            counts[s["label_str"]] += 1
        return counts

    def get_source_groups(self) -> List[str]:
        """Return unique source recordings represented in this split."""
        return sorted(list(set(s["original_recording_id"] for s in self.samples)))


class MulticlassExplosionDataset(ExplosionDataset):
    """PyTorch Dataset for three-class acoustic event detection (non_target / explosion / fireworks)."""

    LABEL_MAP = {
        "non_target": 0,
        "explosion": 1,
        "fireworks": 2,
    }
    INV_LABEL_MAP = {0: "non_target", 1: "explosion", 2: "fireworks"}
    VALID_SPLITS = ("train", "val", "test", "holdout", "challenge")

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        is_training: bool = False,
        seed: int = 42,
        augment: bool = False,
        preload_into_memory: bool = True,
        data_root: Optional[Union[str, Path]] = None,
        manifest_version: str = "auto",
        approved_only: bool = True,
    ):
        """Initialize the three-class dataset using the Phase 12, 13, 14, or 15 manifest validator."""
        self.manifest_version = manifest_version.lower().strip()
        self.approved_only = bool(approved_only)
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

    def _detect_manifest_version(self) -> str:
        if self.manifest_version != "auto":
            return self.manifest_version
        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        cols = set(header)
        if "training_approved" in cols and "review_status" in cols:
            return "phase15"
        if "auditory_verification" in cols and "quality_confidence" in cols:
            return "phase14"
        if "exposure_status" in cols:
            return "phase13"
        return "phase12"

    def _validate_split_manifest(self):
        version = self._detect_manifest_version()
        if version == "phase15":
            return validate_and_load_phase15_multiclass_manifest(
                manifest_path=self.manifest_path,
                split=self.split,
                data_root=self.data_root,
                approved_only=self.approved_only,
            )
        if version == "phase14":
            return validate_and_load_phase14_multiclass_manifest(
                manifest_path=self.manifest_path,
                split=self.split,
                data_root=self.data_root,
            )
        if version == "phase13":
            return validate_and_load_phase13_multiclass_manifest(
                manifest_path=self.manifest_path,
                split=self.split,
                data_root=self.data_root,
            )
        return validate_and_load_multiclass_manifest(
            manifest_path=self.manifest_path,
            split=self.split,
            data_root=self.data_root,
        )

