"""Automated Unit Tests for Offline Inference Engine.

Tests:
1. Mock predictor interface & explicit test labeling.
2. Single-file inference (output structure, fields, timing).
3. Folder and nested-folder recursive batch inference.
4. Timestamp and window-count calculations.
5. Result aggregation modes ('peak_window' vs 'consecutive').
6. Invalid, empty, corrupted, and unsupported audio handling.
7. CSV and JSON export verification.
8. Preservation invariant: verifies inference never modifies source audio or dataset manifests.
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.preprocessing import AudioPreprocessor
from inference.predictor_interface import BasePredictor, MockTestPredictor, PyTorchModelPredictor
from inference.offline_engine import OfflineInferenceEngine, FileInferenceResult, BatchInferenceResult

def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPredictorInterface(unittest.TestCase):
    """Test suite for predictor contracts and mock labeling."""

    def test_mock_predictor_contract_and_labeling(self):
        """Verify MockTestPredictor is explicitly marked as mock."""
        mock = MockTestPredictor()
        self.assertTrue(mock.is_mock())
        self.assertIn("MockTestPredictor", mock.name)
        self.assertIn("TEST_FIXTURE_ONLY", mock.name)

        # Output shape and valid probabilities
        dummy_spec = np.random.randn(64, 44).astype(np.float32)
        win_res = mock.predict_window(dummy_spec)
        self.assertIn("explosion", win_res)
        self.assertIn("non_explosion", win_res)
        self.assertAlmostEqual(win_res["explosion"] + win_res["non_explosion"], 1.0, places=5)

        dummy_batch = np.random.randn(5, 64, 44).astype(np.float32)
        batch_res = mock.predict_batch(dummy_batch)
        self.assertEqual(batch_res.shape, (5, 2))
        np.testing.assert_almost_equal(np.sum(batch_res, axis=1), np.ones(5), decimal=5)

    def test_pytorch_predictor_missing_checkpoint_fails_gracefully(self):
        """Verify PyTorchModelPredictor raises FileNotFoundError when no trained model exists."""
        non_existent = PROJECT_ROOT / "models" / "non_existent_checkpoint.pth"
        with self.assertRaises(FileNotFoundError):
            PyTorchModelPredictor(non_existent)


class TestOfflineInferenceEngine(unittest.TestCase):
    """Test suite for OfflineInferenceEngine."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Generate synthetic audio test files
        cls.sr = 22050
        
        # File 1: 2.0s clean tone
        cls.tone_file = cls.temp_path / "test_tone.wav"
        t = np.linspace(0, 2.0, int(cls.sr * 2.0), endpoint=False)
        tone_data = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(str(cls.tone_file), tone_data, cls.sr)

        # File 2: Nested subdirectory file (1.5s)
        cls.nested_dir = cls.temp_path / "subdir" / "subsubdir"
        cls.nested_dir.mkdir(parents=True, exist_ok=True)
        cls.nested_file = cls.nested_dir / "nested_tone.wav"
        t_nest = np.linspace(0, 1.5, int(cls.sr * 1.5), endpoint=False)
        nest_data = (0.3 * np.sin(2 * np.pi * 500 * t_nest)).astype(np.float32)
        sf.write(str(cls.nested_file), nest_data, cls.sr)

        # File 3: Empty file (0 bytes)
        cls.empty_file = cls.temp_path / "empty.wav"
        cls.empty_file.touch()

        # File 4: Corrupted file
        cls.corrupt_file = cls.temp_path / "corrupt.wav"
        with open(cls.corrupt_file, "wb") as f:
            f.write(b"CORRUPTED_NOT_A_WAV_HEADER")

        # File 5: Unsupported file
        cls.unsupported_file = cls.temp_path / "notes.txt"
        with open(cls.unsupported_file, "w") as f:
            f.write("text content")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)
        self.mock_predictor = MockTestPredictor(mode="deterministic_hash")
        self.engine = OfflineInferenceEngine(
            preprocessor=self.preprocessor,
            predictor=self.mock_predictor,
            threshold=0.80
        )

    def test_single_file_inference_structure(self):
        """Test inference on a single clean audio file."""
        res = self.engine.infer_file(self.tone_file)
        self.assertTrue(res.success)
        self.assertEqual(res.file_name, "test_tone.wav")
        self.assertAlmostEqual(res.duration_seconds, 2.0, places=2)
        self.assertEqual(res.sample_rate, 22050)
        self.assertGreaterEqual(res.num_windows, 5)
        self.assertEqual(len(res.windows), res.num_windows)
        self.assertTrue(res.is_mock_predictor)
        self.assertIn(res.aggregated_prediction, ("explosion", "non_explosion"))
        self.assertGreater(res.processing_time_ms, 0.0)

        # Check per-window timestamps
        for w in res.windows:
            self.assertGreater(w.end_time_s, w.start_time_s)
            self.assertIn(w.window_prediction, ("explosion", "non_explosion"))
            self.assertGreaterEqual(w.explosion_score, 0.0)
            self.assertLessEqual(w.explosion_score, 1.0)

    def test_timestamp_and_window_count_correctness(self):
        """Verify window timestamps match exactly with audio duration and hop size."""
        res = self.engine.infer_file(self.tone_file)
        w0 = res.windows[0]
        self.assertEqual(w0.start_time_s, 0.0)
        self.assertAlmostEqual(w0.end_time_s, 1.0, places=2)

        w1 = res.windows[1]
        self.assertAlmostEqual(w1.start_time_s, 0.25, places=2)
        self.assertAlmostEqual(w1.end_time_s, 1.25, places=2)

    def test_recursive_folder_batch_inference(self):
        """Verify batch folder inference processes nested audio files."""
        batch = self.engine.infer_folder(self.temp_path, recursive=True)
        self.assertIsInstance(batch, BatchInferenceResult)
        self.assertTrue(batch.is_mock_predictor)
        self.assertGreaterEqual(batch.total_files_discovered, 4) # tone, nested, empty, corrupt

        processed_names = [r.file_name for r in batch.results]
        self.assertIn("test_tone.wav", processed_names)
        self.assertIn("nested_tone.wav", processed_names)
        self.assertIn("empty.wav", processed_names)
        self.assertIn("corrupt.wav", processed_names)
        # Unsupported .txt must NOT be discovered as an audio file
        self.assertNotIn("notes.txt", processed_names)

    def test_invalid_and_unreadable_file_handling(self):
        """Verify that invalid audio files produce failure results without crashing."""
        # 1. Non-existent file
        res_missing = self.engine.infer_file(self.temp_path / "ghost.wav")
        self.assertFalse(res_missing.success)
        self.assertIn("not found", res_missing.error_message.lower())

        # 2. Empty file (0 bytes)
        res_empty = self.engine.infer_file(self.empty_file)
        self.assertFalse(res_empty.success)
        self.assertIn("empty", res_empty.error_message.lower())

        # 3. Corrupt header file
        res_corrupt = self.engine.infer_file(self.corrupt_file)
        self.assertFalse(res_corrupt.success)
        self.assertIn("error", res_corrupt.error_message.lower())

        # 4. Unsupported extension
        res_unsup = self.engine.infer_file(self.unsupported_file)
        self.assertFalse(res_unsup.success)
        self.assertIn("unsupported", res_unsup.error_message.lower())

    def test_aggregation_modes(self):
        """Verify peak_window vs consecutive aggregation modes."""
        # Using constant_zero predictor: should always predict non_explosion
        eng_zero = OfflineInferenceEngine(
            preprocessor=self.preprocessor,
            predictor=MockTestPredictor(mode="constant_zero"),
            threshold=0.50
        )
        res_zero = eng_zero.infer_file(self.tone_file)
        self.assertEqual(res_zero.aggregated_prediction, "non_explosion")
        self.assertEqual(res_zero.positive_windows_count, 0)

        # Using constant_high predictor: should predict explosion
        eng_high = OfflineInferenceEngine(
            preprocessor=self.preprocessor,
            predictor=MockTestPredictor(mode="constant_high"),
            threshold=0.50,
            aggregation_mode="consecutive",
            min_consecutive_windows=2
        )
        res_high = eng_high.infer_file(self.tone_file)
        self.assertEqual(res_high.aggregated_prediction, "explosion")
        self.assertEqual(res_high.positive_windows_count, res_high.num_windows)

    def test_csv_and_json_export(self):
        """Test exporting batch results to CSV and JSON."""
        out_export_dir = self.temp_path / "export_test"
        batch = self.engine.infer_folder(self.temp_path, recursive=True, output_dir=out_export_dir)

        csv_p = out_export_dir / "batch_inference_summary.csv"
        json_p = out_export_dir / "batch_inference_details.json"

        self.assertTrue(csv_p.exists(), "CSV export file was not created")
        self.assertTrue(json_p.exists(), "JSON export file was not created")

        # Verify CSV parsing
        with open(csv_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), len(batch.results))
            for r in rows:
                self.assertIn("file_name", r)
                self.assertIn("aggregated_prediction", r)
                self.assertIn("is_mock_predictor", r)

        # Verify JSON parsing
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["total_files_processed"], len(batch.results))
            self.assertTrue(data["is_mock_predictor"])
            self.assertEqual(len(data["results"]), len(batch.results))

    def test_source_audio_and_manifest_unmodified_invariant(self):
        """Strict verification: running inference must NEVER alter source files or dataset manifest."""
        manifest_p = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.csv"
        self.assertTrue(manifest_p.exists())

        # Checksum before inference
        manifest_md5_before = compute_md5(manifest_p)
        tone_md5_before = compute_md5(self.tone_file)

        # Run inference multiple times
        _ = self.engine.infer_file(self.tone_file)
        _ = self.engine.infer_folder(self.temp_path, recursive=False)

        # Checksum after inference
        manifest_md5_after = compute_md5(manifest_p)
        tone_md5_after = compute_md5(self.tone_file)

        self.assertEqual(manifest_md5_before, manifest_md5_after, "Manifest CSV was altered by inference!")
        self.assertEqual(tone_md5_before, tone_md5_after, "Source audio file was altered by inference!")


if __name__ == "__main__":
    unittest.main()

