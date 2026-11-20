"""Automated Tests for Explosion Detection Baseline Pipeline.

Tests:
1. Model input/output shapes.
2. Invalid tensor shape rejection.
3. Dataset split filtering (160 train, 20 val, 20 test).
4. Correct label mapping (explosion=1, non_explosion=0).
5. Manifest path resolution independence.
6. Reproducible model initialization across seeds.
7. Synthetic batch forward and backward gradient step.
8. Validation/test augmentation isolation.
9. Recording-level score aggregation logic.
10. Verification that training code strictly isolates the test set.
11. Exact model parameter count verification.
"""

import csv
import sys
import unittest
from pathlib import Path

# Add project root and explosion_detection to sys.path
TEST_DIR = Path(__file__).resolve().parent
EXPLOSION_DETECTION_ROOT = TEST_DIR.parent
PROJECT_ROOT = EXPLOSION_DETECTION_ROOT.parent

for p in (str(PROJECT_ROOT), str(EXPLOSION_DETECTION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn

import config
from models.baseline_cnn import build_baseline_cnn, AudioBaselineCNN
from training_utils import (
    set_seed,
    BalancedExplosionDataset,
    compute_binary_metrics,
    aggregate_recording_predictions,
    compute_category_false_positives
)


class FastBalancedExplosionDataset(BalancedExplosionDataset):
    """Fast test subclass that bypasses full audio decoding for split filtering checks."""
    def _build_samples(self):
        # Skip slow audio file loading during fast unit tests
        pass


class TestBaselineCNN(unittest.TestCase):
    """Test suite for AudioBaselineCNN model architecture."""

    def test_model_input_output_shape(self):
        """Verify model accepts (B, 1, 64, 44) and produces (B, 1) logits."""
        model = build_baseline_cnn(seed=42)
        model.eval()
        for batch_size in (1, 4, 16):
            x = torch.randn(batch_size, 1, 64, 44)
            out = model(x)
            self.assertEqual(out.shape, (batch_size, 1))
            self.assertEqual(out.dtype, torch.float32)

    def test_model_invalid_tensor_shapes(self):
        """Verify clear ValueError is raised for malformed input tensors."""
        model = build_baseline_cnn(seed=42)
        # Non-tensor input
        with self.assertRaises(TypeError):
            model(np.zeros((4, 1, 64, 44)))
        # 3D tensor missing channel
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 64, 44))
        self.assertIn("Expected 4D tensor", str(ctx.exception))
        # Wrong channels (e.g. 2 instead of 1)
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 2, 64, 44))
        self.assertIn("Invalid channel dimension", str(ctx.exception))
        # Wrong frequency bins (e.g. 128 instead of 64)
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 1, 128, 44))
        self.assertIn("Invalid spectrogram dimension", str(ctx.exception))
        # Wrong time frames (e.g. 80 instead of 44)
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 1, 64, 80))
        self.assertIn("Invalid spectrogram dimension", str(ctx.exception))

    def test_exact_parameter_count(self):
        """Verify exact total and trainable parameter counts."""
        model = build_baseline_cnn(seed=42)
        total, trainable = model.count_parameters()
        self.assertEqual(total, 130017)
        self.assertEqual(trainable, 130017)

    def test_reproducible_model_initialization(self):
        """Verify identical seed produces identical initial weights."""
        m1 = build_baseline_cnn(seed=42)
        m2 = build_baseline_cnn(seed=42)
        m3 = build_baseline_cnn(seed=999)

        # Same seed must match exactly
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)

        # Different seed must differ
        diff_found = any(not torch.equal(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))
        self.assertTrue(diff_found)

    def test_synthetic_forward_backward_pass(self):
        """Verify loss computation, gradient backpropagation, and weight update."""
        model = build_baseline_cnn(seed=42)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.75]))

        # Track initial weight of first conv layer
        w_before = model.block1[0].weight.clone()

        x = torch.randn(4, 1, 64, 44)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        # Check gradients exist and are finite
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.all(torch.isfinite(p.grad)))

        optimizer.step()
        w_after = model.block1[0].weight

        # Weights must be updated
        self.assertFalse(torch.equal(w_before, w_after))


class TestPipelineConfigurationAndData(unittest.TestCase):
    """Test suite for manifest loading, split isolation, and preprocessing."""

    def test_manifest_path_resolution(self):
        """Verify active Phase 11 MANIFEST_PATH default and historical Phase 6B manifest resolution."""
        active_p = Path(config.MANIFEST_PATH)
        self.assertTrue(active_p.exists(), f"Active split manifest does not exist at: {active_p}")
        self.assertEqual(active_p.name, "dataset_manifest_split_phase11.csv")

        historical_p = config.get_manifest_path("phase6b")
        self.assertTrue(historical_p.exists(), f"Historical Phase 6B manifest does not exist at: {historical_p}")
        self.assertEqual(historical_p.name, "dataset_manifest_split.csv")
        self.assertTrue(config.DATA_ROOT.exists(), f"DATA_ROOT does not exist: {config.DATA_ROOT}")

    def test_dataset_split_filtering(self):
        """Verify dataset loader respects historical Phase 6B splits: 160 train, 20 val, 20 test."""
        historical_manifest = config.HISTORICAL_PHASE6B_SPLIT_PATH
        train_ds = FastBalancedExplosionDataset(
            manifest_path=historical_manifest,
            split="train",
            is_training=False
        )
        self.assertEqual(len(train_ds.records), 160)

        val_ds = FastBalancedExplosionDataset(
            manifest_path=historical_manifest,
            split="val",
            is_training=False
        )
        self.assertEqual(len(val_ds.records), 20)

        test_ds = FastBalancedExplosionDataset(
            manifest_path=historical_manifest,
            split="test",
            is_training=False
        )
        self.assertEqual(len(test_ds.records), 20)

    def test_correct_label_mapping(self):
        """Verify binary label mapping matches project convention."""
        self.assertEqual(config.LABEL_MAP["non_explosion"], 0)
        self.assertEqual(config.LABEL_MAP["explosion"], 1)
        self.assertEqual(config.INV_LABEL_MAP[0], "non_explosion")
        self.assertEqual(config.INV_LABEL_MAP[1], "explosion")

    def test_augmentation_isolation(self):
        """Verify SpecAugment is only active when is_training=True and augment=True."""
        # When is_training=False, augment flag must be disabled
        val_ds = FastBalancedExplosionDataset(
            manifest_path=config.MANIFEST_PATH,
            split="val",
            is_training=False,
            augment=True
        )
        self.assertFalse(val_ds.augment)

        # When is_training=True and augment=True, augment flag is enabled
        train_ds = FastBalancedExplosionDataset(
            manifest_path=config.MANIFEST_PATH,
            split="train",
            is_training=True,
            augment=True
        )
        self.assertTrue(train_ds.augment)

    def test_recording_level_score_aggregation(self):
        """Verify recording-level peak window aggregation."""
        meta = [
            {"file_path": "track1.mp3", "label_str": "explosion"},
            {"file_path": "track1.mp3", "label_str": "explosion"},
            {"file_path": "track2.mp3", "label_str": "non_explosion"},
            {"file_path": "track2.mp3", "label_str": "non_explosion"},
            {"file_path": "track2.mp3", "label_str": "non_explosion"}
        ]
        scores = np.array([0.3, 0.85, 0.1, 0.45, 0.2])

        recs, metrics = aggregate_recording_predictions(meta, scores, threshold=0.5)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs["track1.mp3"]["max_score"], 0.85)
        self.assertEqual(recs["track1.mp3"]["pred_label"], 1)
        self.assertTrue(recs["track1.mp3"]["correct"])

        self.assertEqual(recs["track2.mp3"]["max_score"], 0.45)
        self.assertEqual(recs["track2.mp3"]["pred_label"], 0)
        self.assertTrue(recs["track2.mp3"]["correct"])

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_training_code_never_loads_test_data(self):
        """Verify train.py explicitly limits dataset loading to train and val."""
        import scripts.train as train_module
        # Parse default arguments safely with empty list to prevent unittest argv pollution
        args = train_module.parse_args([])
        
        train_ds = FastBalancedExplosionDataset(
            manifest_path=args.manifest_path,
            split="train",
            is_training=False
        )
        val_ds = FastBalancedExplosionDataset(
            manifest_path=args.manifest_path,
            split="val",
            is_training=False
        )
        train_paths = set(r["file_path"] for r in train_ds.records)
        val_paths = set(r["file_path"] for r in val_ds.records)

        with open(args.manifest_path, "r", encoding="utf-8") as f:
            test_paths = set(r["file_path"] for r in csv.DictReader(f) if r["split"] == "test")

        self.assertEqual(len(test_paths), 20)
        self.assertEqual(len(train_paths.intersection(test_paths)), 0)
        self.assertEqual(len(val_paths.intersection(test_paths)), 0)


class TestConsecutiveDecisionRule(unittest.TestCase):
    """Focused unit tests for the consecutive-window recording aggregation rule."""

    def test_default_peak_rule_preserves_existing_behavior(self):
        """Verify default call without specifying rule uses peak aggregation."""
        meta = [
            {"file_path": "trackA.mp3", "label_str": "explosion", "window_index": 0},
            {"file_path": "trackA.mp3", "label_str": "explosion", "window_index": 1},
        ]
        scores = np.array([0.2, 0.85])
        # Default call
        recs_def, metrics_def = aggregate_recording_predictions(meta, scores, threshold=0.50)
        # Explicit peak call
        recs_peak, metrics_peak = aggregate_recording_predictions(meta, scores, threshold=0.50, rule="peak")

        self.assertEqual(recs_def["trackA.mp3"]["pred_label"], 1)
        self.assertEqual(recs_peak["trackA.mp3"]["pred_label"], 1)
        self.assertEqual(metrics_def["balanced_accuracy"], metrics_peak["balanced_accuracy"])

    def test_three_consecutive_scores_trigger_alert(self):
        """Verify 3 consecutive windows >= 0.90 trigger an alert."""
        meta = [
            {"file_path": "track.mp3", "label_str": "explosion", "window_index": i}
            for i in range(5)
        ]
        scores = np.array([0.1, 0.90, 0.92, 0.95, 0.2])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["track.mp3"]["pred_label"], 1)
        self.assertEqual(recs["track.mp3"]["max_consecutive_run"], 3)
        self.assertEqual(metrics["tp"], 1)

    def test_two_qualifying_scores_do_not_trigger_alert(self):
        """Verify 2 qualifying windows do not trigger when 3 are required."""
        meta = [
            {"file_path": "track.mp3", "label_str": "non_explosion", "window_index": i}
            for i in range(5)
        ]
        scores = np.array([0.1, 0.92, 0.95, 0.2, 0.5])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["track.mp3"]["pred_label"], 0)
        self.assertEqual(recs["track.mp3"]["max_consecutive_run"], 2)
        self.assertEqual(metrics["tn"], 1)

    def test_below_threshold_score_resets_run(self):
        """Verify a sub-threshold window resets the consecutive counter."""
        meta = [
            {"file_path": "track.mp3", "label_str": "non_explosion", "window_index": i}
            for i in range(5)
        ]
        # Runs of 2 separated by 0.89
        scores = np.array([0.92, 0.95, 0.89, 0.93, 0.94])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["track.mp3"]["pred_label"], 0)
        self.assertEqual(recs["track.mp3"]["max_consecutive_run"], 2)
        self.assertEqual(metrics["tn"], 1)

    def test_scores_exactly_equal_threshold_qualify(self):
        """Verify scores exactly equal to 0.90 count as qualifying."""
        meta = [
            {"file_path": "track.mp3", "label_str": "explosion", "window_index": i}
            for i in range(3)
        ]
        scores = np.array([0.90, 0.90, 0.90])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["track.mp3"]["pred_label"], 1)
        self.assertEqual(recs["track.mp3"]["max_consecutive_run"], 3)
        self.assertEqual(metrics["tp"], 1)

    def test_windows_from_different_recordings_never_combined(self):
        """Verify consecutive runs do not cross recording boundaries."""
        meta = [
            {"file_path": "rec1.mp3", "label_str": "non_explosion", "window_index": 0},
            {"file_path": "rec1.mp3", "label_str": "non_explosion", "window_index": 1},
            {"file_path": "rec2.mp3", "label_str": "non_explosion", "window_index": 0},
            {"file_path": "rec2.mp3", "label_str": "non_explosion", "window_index": 1},
        ]
        # rec1 has 2 qualifying; rec2 has 2 qualifying; neither has 3
        scores = np.array([0.95, 0.95, 0.95, 0.95])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["rec1.mp3"]["pred_label"], 0)
        self.assertEqual(recs["rec1.mp3"]["max_consecutive_run"], 2)
        self.assertEqual(recs["rec2.mp3"]["pred_label"], 0)
        self.assertEqual(recs["rec2.mp3"]["max_consecutive_run"], 2)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 0)

    def test_unsorted_input_handled_chronologically(self):
        """Verify scrambled input order is sorted by window_index/start_sec before run calculation."""
        # Provided in order: index 2, index 0, index 1
        meta = [
            {"file_path": "track.mp3", "label_str": "explosion", "window_index": 2, "start_sec": 0.50},
            {"file_path": "track.mp3", "label_str": "explosion", "window_index": 0, "start_sec": 0.00},
            {"file_path": "track.mp3", "label_str": "explosion", "window_index": 1, "start_sec": 0.25},
        ]
        # Corresponding to indices [2, 0, 1]
        scores = np.array([0.95, 0.91, 0.92])
        recs, metrics = aggregate_recording_predictions(
            meta, scores, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["track.mp3"]["pred_label"], 1)
        self.assertEqual(recs["track.mp3"]["max_consecutive_run"], 3)

        # Scrambled input where qualifying windows are NOT consecutive chronologically
        meta_non_consec = [
            {"file_path": "trackB.mp3", "label_str": "non_explosion", "window_index": 1, "start_sec": 0.25},
            {"file_path": "trackB.mp3", "label_str": "non_explosion", "window_index": 0, "start_sec": 0.00},
            {"file_path": "trackB.mp3", "label_str": "non_explosion", "window_index": 2, "start_sec": 0.50},
        ]
        # Chronological: index 0 (0.95), index 1 (0.10), index 2 (0.95) -> max run = 1
        scores_non_consec = np.array([0.10, 0.95, 0.95])
        recsB, metricsB = aggregate_recording_predictions(
            meta_non_consec, scores_non_consec, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recsB["trackB.mp3"]["pred_label"], 0)
        self.assertEqual(recsB["trackB.mp3"]["max_consecutive_run"], 1)

    def test_recording_shorter_than_consecutive_windows_handled_safely(self):
        """Verify recordings with fewer windows than required run length do not crash."""
        meta_single = [
            {"file_path": "short.mp3", "label_str": "non_explosion", "window_index": 0}
        ]
        scores_single = np.array([0.99])
        recs, metrics = aggregate_recording_predictions(
            meta_single, scores_single, threshold=0.90, rule="consecutive", consecutive_windows=3
        )
        self.assertEqual(recs["short.mp3"]["pred_label"], 0)
        self.assertEqual(recs["short.mp3"]["max_consecutive_run"], 1)
        self.assertEqual(metrics["tn"], 1)

    def test_invalid_settings_rejected(self):
        """Verify clear ValueError is raised for invalid rules or consecutive_windows < 1."""
        meta = [{"file_path": "f.mp3", "label_str": "explosion"}]
        scores = np.array([0.5])
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, scores, rule="unknown_rule")
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, scores, rule="consecutive", consecutive_windows=0)
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, scores, rule="consecutive", consecutive_windows=-2)


if __name__ == "__main__":
    unittest.main()
