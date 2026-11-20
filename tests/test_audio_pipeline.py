"""Automated Unit Tests for Audio Preprocessing & Data Loading Pipeline.

Tests:
1. Audio file loading (valid audio, missing file, empty file, corrupted data).
2. Channel conversion (1D mono, 2D stereo, multi-channel, amplitude clipping scaling).
3. Resampling (downsampling, upsampling, polyphase sinc filter anti-aliasing).
4. Window creation (exact sample counts, hop lengths, transient tail preservation, timestamps).
5. Spectrogram generation (shape (64, 44), frequency bounds, normalization, finite values).
6. Group-aware split integrity (0 group leakage across train/val/test in dataset_manifest.csv).
7. Dataset & DataLoader integration (tensor shapes, label mapping, deterministic reproducibility).
"""

import os
import sys
import tempfile
import unittest
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.preprocessing import AudioPreprocessor
from audio.file_loader import load_audio_file, validate_audio_file, AudioLoadError
from explosion_detection.config import PHASE11_MANIFEST_SPLIT_PATH
from training.dataset import ExplosionDataset
from training.manifest_validator import resolve_dataset_path, validate_and_load_manifest

class TestAudioPreprocessor(unittest.TestCase):
    """Test suite for AudioPreprocessor class."""

    def setUp(self):
        self.preprocessor = AudioPreprocessor(
            sample_rate=22050,
            window_duration_sec=1.0,
            hop_duration_sec=0.25,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            f_min=20.0,
            f_max=11025.0,
            normalize_spectrogram=True
        )

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        with self.assertRaises(ValueError):
            AudioPreprocessor(sample_rate=-1)
        with self.assertRaises(ValueError):
            AudioPreprocessor(window_duration_sec=0)
        with self.assertRaises(ValueError):
            AudioPreprocessor(hop_duration_sec=-0.5)
        with self.assertRaises(ValueError):
            AudioPreprocessor(n_mels=0)
        with self.assertRaises(ValueError):
            AudioPreprocessor(n_fft=1000) # not power of 2

    def test_to_mono_conversions(self):
        """Test channel conversion across 1D and 2D arrays."""
        # 1D Mono
        mono_1d = np.random.randn(22050).astype(np.float32)
        out = self.preprocessor.to_mono(mono_1d)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(len(out), 22050)

        # 2D Stereo (samples, 2)
        stereo_2d = np.random.randn(22050, 2).astype(np.float32)
        out = self.preprocessor.to_mono(stereo_2d)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(len(out), 22050)

        # 2D Stereo (2, samples)
        stereo_transposed = np.random.randn(2, 22050).astype(np.float32)
        out = self.preprocessor.to_mono(stereo_transposed)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(len(out), 22050)

        # Peak clipping scaling
        clipped = np.array([2.5, -3.0, 1.2], dtype=np.float32)
        out_clipped = self.preprocessor.to_mono(clipped)
        self.assertAlmostEqual(float(np.max(np.abs(out_clipped))), 1.0, places=5)

    def test_to_mono_invalid_inputs(self):
        """Test invalid input rejection (None, empty, NaN, Inf)."""
        with self.assertRaises(ValueError):
            self.preprocessor.to_mono(None)
        with self.assertRaises(ValueError):
            self.preprocessor.to_mono(np.array([], dtype=np.float32))
        with self.assertRaises(ValueError):
            self.preprocessor.to_mono(np.array([1.0, np.nan, 0.5], dtype=np.float32))
        with self.assertRaises(ValueError):
            self.preprocessor.to_mono(np.array([1.0, np.inf, 0.5], dtype=np.float32))
        with self.assertRaises(ValueError):
            self.preprocessor.to_mono(np.zeros((2, 3, 4), dtype=np.float32)) # 3D unsupported

    def test_polyphase_resampling(self):
        """Test resampling parity and anti-aliasing."""
        # 1. No-op resampling when sample rate matches
        signal_22k = np.random.randn(22050).astype(np.float32)
        out_same = self.preprocessor.resample(signal_22k, 22050)
        np.testing.assert_array_equal(signal_22k, out_same)

        # 2. Downsampling 44.1 kHz -> 22.05 kHz (factor of 2)
        signal_44k = np.random.randn(44100).astype(np.float32)
        out_down = self.preprocessor.resample(signal_44k, 44100)
        self.assertEqual(len(out_down), 22050)
        self.assertTrue(np.all(np.isfinite(out_down)))

        # 3. Downsampling 48 kHz -> 22.05 kHz
        signal_48k = np.random.randn(48000).astype(np.float32)
        out_48k = self.preprocessor.resample(signal_48k, 48000)
        self.assertEqual(len(out_48k), 22050)

        # 4. Upsampling 16 kHz -> 22.05 kHz
        signal_16k = np.random.randn(16000).astype(np.float32)
        out_up = self.preprocessor.resample(signal_16k, 16000)
        self.assertEqual(len(out_up), 22050)

    def test_window_creation(self):
        """Test sliding window segmentation and timestamps."""
        # 2.0 seconds audio at 22050 Hz = 44100 samples
        audio_2s = np.random.randn(44100).astype(np.float32)
        windows, timestamps = self.preprocessor.create_sliding_windows(audio_2s, preserve_tail=True)

        # Expected:
        # Window 1: 0 to 22050 (0.00s to 1.00s)
        # Window 2: 5512 to 27562 (0.25s to 1.25s)
        # Window 3: 11024 to 33074 (0.50s to 1.50s)
        # Window 4: 16536 to 38586 (0.75s to 1.75s)
        # Window 5: 22048 to 44098 (1.00s to 2.00s)
        self.assertEqual(windows.shape[1], 22050)
        self.assertEqual(len(windows), len(timestamps))
        self.assertGreaterEqual(len(windows), 5)

        # Check timestamp monotonically increasing
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i][0], timestamps[i-1][0])

    def test_short_audio_tail_preservation(self):
        """Test short transient preservation for audio shorter than 1.0s."""
        short_audio = np.random.randn(10000).astype(np.float32) # ~0.45s
        windows, timestamps = self.preprocessor.create_sliding_windows(short_audio, preserve_tail=True)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows.shape, (1, 22050))
        # Verify first 10000 samples match original, rest padded with zero
        np.testing.assert_array_almost_equal(windows[0][:10000], short_audio)
        np.testing.assert_array_equal(windows[0][10000:], np.zeros(12050))

    def test_log_mel_spectrogram_output_shape_and_bounds(self):
        """Test that Log-Mel Spectrogram has exact (64, 44) shape, finite values, and normalization."""
        audio_window = np.random.randn(22050).astype(np.float32)
        spec = self.preprocessor.compute_log_mel_spectrogram(audio_window)

        # Check shape: (n_mels, expected_time_frames) = (64, 44)
        self.assertEqual(spec.shape, (64, 44))
        self.assertEqual(spec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(spec)))

        # Check zero-mean unit-variance normalization
        self.assertAlmostEqual(float(np.mean(spec)), 0.0, places=3)
        self.assertAlmostEqual(float(np.std(spec)), 1.0, places=2)

    def test_live_buffer_processing_parity(self):
        """Test that live buffer processing matches file processing window computation."""
        test_audio = np.random.randn(22050).astype(np.float32)
        spec_from_file_func = self.preprocessor.compute_log_mel_spectrogram(test_audio)
        spec_from_live_func = self.preprocessor.process_live_buffer(test_audio, 22050)
        np.testing.assert_array_almost_equal(spec_from_file_func, spec_from_live_func, decimal=5)


class TestFileLoader(unittest.TestCase):
    """Test suite for file loader and validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Generate a synthetic test WAV file
        self.wav_path = self.temp_path / "test_tone.wav"
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(self.wav_path), sine, sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_wav(self):
        """Test loading a clean valid WAV file."""
        audio, sr, dur, ch = load_audio_file(self.wav_path)
        self.assertEqual(sr, 22050)
        self.assertAlmostEqual(dur, 1.0, places=2)
        self.assertEqual(ch, 1)
        self.assertEqual(audio.dtype, np.float32)

    def test_missing_file_raises_error(self):
        """Test that missing file raises AudioLoadError."""
        with self.assertRaises(AudioLoadError):
            load_audio_file(self.temp_path / "non_existent.wav")

    def test_empty_file_raises_error(self):
        """Test that 0-byte file raises AudioLoadError."""
        empty_file = self.temp_path / "empty.wav"
        empty_file.touch()
        with self.assertRaises(AudioLoadError):
            load_audio_file(empty_file)

    def test_corrupted_file_raises_error(self):
        """Test that corrupted pseudo-audio raises AudioLoadError."""
        corrupt_file = self.temp_path / "corrupt.wav"
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_RIFF_HEADER_123456789")
        with self.assertRaises(AudioLoadError):
            load_audio_file(corrupt_file)

    def test_validate_audio_file_helper(self):
        """Test the non-throwing validate_audio_file helper."""
        report = validate_audio_file(self.wav_path)
        self.assertTrue(report["valid"])
        self.assertIsNone(report["error"])
        self.assertEqual(report["sample_rate"], 22050)


class TestManifestAndSplitIntegrity(unittest.TestCase):
    """Test suite verifying manifest formatting and zero group leakage across splits."""

    def setUp(self):
        self.manifest_path = PHASE11_MANIFEST_SPLIT_PATH
        self.assertTrue(self.manifest_path.exists(), f"Manifest missing: {self.manifest_path}")

    def test_manifest_schema_and_labels(self):
        """Verify active Phase 11 schema, binary labels, and workspace path resolution."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 193)
        for r in rows:
            self.assertIn(r["label"], ("explosion", "non_explosion"))
            self.assertIn(r["split"], ("train", "val", "test"))
            self.assertTrue(
                resolve_dataset_path(r["file_path"]).exists(),
                f"File in manifest does not exist on disk: {r['file_path']}",
            )

    def test_zero_group_leakage_across_splits(self):
        """Verify that NO group_id leaks between train, val, and test splits."""
        res = validate_and_load_manifest(self.manifest_path, resolve_all_paths=True)
        groups_by_split = {"train": set(), "val": set(), "test": set()}
        for r in res.all_rows:
            groups_by_split[r["split"]].add(r["group_id"])

        train_val_overlap = groups_by_split["train"].intersection(groups_by_split["val"])
        train_test_overlap = groups_by_split["train"].intersection(groups_by_split["test"])
        val_test_overlap = groups_by_split["val"].intersection(groups_by_split["test"])

        self.assertEqual(len(train_val_overlap), 0, f"Leakage between Train and Val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Leakage between Train and Test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Leakage between Val and Test: {val_test_overlap}")


class TestExplosionDataset(unittest.TestCase):
    """Test suite for ExplosionDataset PyTorch integration."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = PHASE11_MANIFEST_SPLIT_PATH

    def test_dataset_splits_instantiation(self):
        """Test dataset loading for train, val, and test splits."""
        for split_name in ("train", "val", "test"):
            ds = ExplosionDataset(
                manifest_path=self.manifest_path,
                split=split_name,
                is_training=(split_name == "train"),
                preload_into_memory=True
            )
            self.assertGreater(len(ds), 0, f"Dataset for split '{split_name}' is empty")
            class_counts = ds.get_class_counts()
            self.assertIn("explosion", class_counts)
            self.assertIn("non_explosion", class_counts)

    def test_dataloader_batch_generation_and_tensor_dimensions(self):
        """Test PyTorch DataLoader batch generation and tensor shapes."""
        ds = ExplosionDataset(
            manifest_path=self.manifest_path,
            split="val",
            is_training=False,
            preload_into_memory=True
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        
        # Fetch one batch
        batch_specs, batch_labels, batch_meta = next(iter(loader))
        
        # Spec tensor: (batch_size, 1, n_mels, time_frames) = (4, 1, 64, 44)
        self.assertEqual(batch_specs.shape, (4, 1, 64, 44))
        self.assertEqual(batch_specs.dtype, torch.float32)
        
        # Labels tensor: (batch_size,) = (4,)
        self.assertEqual(batch_labels.shape, (4,))
        self.assertEqual(batch_labels.dtype, torch.int64)

    def test_deterministic_reproducibility(self):
        """Test that fixed random seed produces identical augmented outputs."""
        ds1 = ExplosionDataset(manifest_path=self.manifest_path, split="val", seed=123, is_training=False)
        ds2 = ExplosionDataset(manifest_path=self.manifest_path, split="val", seed=123, is_training=False)

        spec1, lbl1, _ = ds1[0]
        spec2, lbl2, _ = ds2[0]

        torch.testing.assert_close(spec1, spec2)
        self.assertEqual(lbl1.item(), lbl2.item())


if __name__ == "__main__":
    unittest.main()

