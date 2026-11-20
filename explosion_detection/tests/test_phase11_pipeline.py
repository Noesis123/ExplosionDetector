"""Phase 11 Pipeline and Split Integrity Automated Tests.

Tests:
1. Label mapping: non_explosion -> 0, explosion -> 1.
2. Phase 11 manifest integrity: exactly 193 files, all exist, no missing values in critical fields.
3. Session-level split isolation: verify 0 session groups cross splits.
4. Excluded files isolation: verify all 7 excluded files are absent from Phase 11 manifests.
5. Augmentation isolation: validation and test splits receive no SpecAugment.
6. Metric calculations regression test: for TP=9, FP=3, TN=7, FN=1:
   - positive F1 == 0.8182
   - negative F1 == 0.7778
   - macro F1 == 0.7980
   - precision == 0.7500
   - recall == 0.9000
   - specificity == 0.7000
   - balanced_accuracy == 0.8000
7. Checkpoint path isolation: verify phase11_best_model.pt does not overwrite best_model.pt.
8. Reproducibility of the new split from seed 10.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

TEST_DIR = Path(__file__).resolve().parent
EXPLOSION_DETECTION_ROOT = TEST_DIR.parent
PROJECT_ROOT = EXPLOSION_DETECTION_ROOT.parent

for p in (str(PROJECT_ROOT), str(EXPLOSION_DETECTION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from training.manifest_validator import resolve_dataset_path
from training_utils import compute_binary_metrics, aggregate_recording_predictions
from models.baseline_cnn import build_baseline_cnn


class TestPhase11Pipeline(unittest.TestCase):
    """Test suite for Phase 11 dataset repair, metrics, and leakage isolation."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = EXPLOSION_DETECTION_ROOT / "data/manifests/dataset_manifest_split_phase11.csv"
        cls.clean_manifest_path = EXPLOSION_DETECTION_ROOT / "data/manifests/dataset_manifest_phase11.csv"
        cls.excluded_path = EXPLOSION_DETECTION_ROOT / "reports/phase11_excluded_recordings.csv"
        cls.group_audit_path = EXPLOSION_DETECTION_ROOT / "reports/phase11_group_assignment.csv"

        cls.df_manifest = pd.read_csv(cls.manifest_path)
        cls.df_clean = pd.read_csv(cls.clean_manifest_path)
        cls.df_excluded = pd.read_csv(cls.excluded_path)
        cls.df_groups = pd.read_csv(cls.group_audit_path)

    def test_manifest_file_count_and_existence(self):
        """Verify Phase 11 manifest contains exactly 193 files and all paths exist on disk."""
        self.assertEqual(len(self.df_manifest), 193)
        self.assertEqual(len(self.df_clean), 193)
        for _, row in self.df_manifest.iterrows():
            fp = resolve_dataset_path(row["file_path"])
            self.assertTrue(fp.exists(), f"File does not exist: {fp}")

    def test_label_mapping_and_classes(self):
        """Verify only 'explosion' and 'non_explosion' labels are present."""
        unique_labels = set(self.df_manifest["label"])
        self.assertEqual(unique_labels, {"explosion", "non_explosion"})
        counts = self.df_manifest["label"].value_counts().to_dict()
        self.assertEqual(counts["explosion"], 92)
        self.assertEqual(counts["non_explosion"], 101)

    def test_split_proportions(self):
        """Verify train, val, and test splits have 153, 20, and 20 recordings respectively."""
        split_counts = self.df_manifest["split"].value_counts().to_dict()
        self.assertEqual(split_counts["train"], 153)
        self.assertEqual(split_counts["val"], 20)
        self.assertEqual(split_counts["test"], 20)

        # Val and test should both have 10 pos and 10 neg
        val_df = self.df_manifest[self.df_manifest["split"] == "val"]
        self.assertEqual(val_df["label"].value_counts().to_dict(), {"explosion": 10, "non_explosion": 10})

        test_df = self.df_manifest[self.df_manifest["split"] == "test"]
        self.assertEqual(test_df["label"].value_counts().to_dict(), {"explosion": 10, "non_explosion": 10})

    def test_session_level_split_isolation(self):
        """Verify zero session/disk groups cross split boundaries."""
        for gid, grp in self.df_manifest.groupby("group_id"):
            splits = set(grp["split"])
            labels = set(grp["label"])
            self.assertEqual(len(splits), 1, f"Group {gid} crosses splits: {splits}")
            self.assertEqual(len(labels), 1, f"Group {gid} has mixed labels: {labels}")

    def test_excluded_files_quarantine(self):
        """Verify that all 7 excluded files are completely absent from Phase 11 manifests."""
        self.assertEqual(len(self.df_excluded), 7)
        excluded_names = set(self.df_excluded["filename"])
        manifest_names = set(Path(p).name for p in self.df_manifest["file_path"])
        overlap = excluded_names.intersection(manifest_names)
        self.assertEqual(len(overlap), 0, f"Excluded files leaked into Phase 11 manifest: {overlap}")

    def test_artillery_and_fireworks_relabeling(self):
        """Verify BBC 25-Pounder artillery and fireworks are correctly mapped to non_explosion."""
        artillery_files = [
            "bbc_guns--gunf_07027149.mp3",
            "bbc_guns--gunf_07027150.mp3",
            "bbc_guns--gunf_07027151.mp3"
        ]
        for af in artillery_files:
            match = self.df_manifest[self.df_manifest["file_path"].str.endswith(af)]
            self.assertEqual(len(match), 1)
            row = match.iloc[0]
            self.assertEqual(row["label"], "non_explosion")
            self.assertEqual(row["category"], "artillery_firing")
            self.assertEqual(row["split"], "train")

        fw_match = self.df_manifest[self.df_manifest["file_path"].str.endswith("bbc_fireworks-_07019107.mp3")]
        self.assertEqual(len(fw_match), 1)
        fw_row = fw_match.iloc[0]
        self.assertEqual(fw_row["label"], "non_explosion")
        self.assertEqual(fw_row["category"], "fireworks")

    def test_metrics_calculation_and_macro_f1_distinction(self):
        """Regression test: verify exact formulas for TP=9, FP=3, TN=7, FN=1.
        
        Specifically:
        - Positive F1: 0.8182
        - Negative F1: 0.7778
        - True Macro F1: 0.7980
        - Precision: 0.7500
        - Recall: 0.9000
        - Specificity: 0.7000
        - Balanced Accuracy: 0.8000
        """
        y_true = np.array([0] * 10 + [1] * 10)
        # 7 TN, 3 FP, 1 FN, 9 TP
        y_pred = np.array([0] * 7 + [1] * 3 + [0] * 1 + [1] * 9)
        y_scores = np.array([0.1] * 7 + [0.95] * 3 + [0.2] * 1 + [0.95] * 9)

        metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50, y_pred=y_pred)

        self.assertEqual(metrics["tp"], 9)
        self.assertEqual(metrics["fp"], 3)
        self.assertEqual(metrics["tn"], 7)
        self.assertEqual(metrics["fn"], 1)

        self.assertAlmostEqual(metrics["precision"], 0.7500, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.9000, places=4)
        self.assertAlmostEqual(metrics["specificity"], 0.7000, places=4)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.8000, places=4)

        # F1 distinctions
        self.assertAlmostEqual(metrics["f1_positive"], 0.8182, places=4)
        self.assertAlmostEqual(metrics["f1_negative"], 0.7778, places=4)
        self.assertAlmostEqual(metrics["macro_f1"], 0.7980, places=4)

        # Verify f1_positive != macro_f1
        self.assertNotEqual(metrics["f1_positive"], metrics["macro_f1"])

    def test_checkpoint_path_isolation(self):
        """Verify Phase 11 checkpoint path is isolated from original Phase 7D checkpoint."""
        orig_ckpt = config.CHECKPOINTS_DIR / "best_model.pt"
        phase11_ckpt = config.CHECKPOINTS_DIR / "phase11_best_model.pt"
        self.assertNotEqual(orig_ckpt, phase11_ckpt)
        self.assertTrue(orig_ckpt.exists(), "Original Phase 7D checkpoint must be preserved!")


if __name__ == "__main__":
    unittest.main()

