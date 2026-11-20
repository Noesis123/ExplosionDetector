"""Non-PyTorch unit tests for Phase 6 manifest validation and loader error handling."""

import csv
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLOSION_DETECTION_ROOT = PROJECT_ROOT / "explosion_detection"
for p in (str(PROJECT_ROOT), str(EXPLOSION_DETECTION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from audio.path_utils import UnsafeDatasetPathError
from training.manifest_validator import validate_and_load_manifest


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestManifestValidator(unittest.TestCase):
    """Verify strict manifest schema, group leakage, duplicate, and missing-file checks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        (self.data_root / "positives").mkdir(parents=True)
        (self.data_root / "hard_negatives" / "gunfire").mkdir(parents=True)

        self.pos1 = self.data_root / "positives" / "exp_1.mp3"
        self.neg1 = self.data_root / "hard_negatives" / "gunfire" / "gun_1.mp3"
        self.neg2 = self.data_root / "hard_negatives" / "gunfire" / "gun_2.mp3"
        for p in (self.pos1, self.neg1, self.neg2):
            p.write_bytes(b"ID3")

        self.fieldnames = [
            "file_path",
            "label",
            "category",
            "group_id",
            "original_recording_id",
            "split",
            "validation_status",
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_synthetic_manifest_passes_and_preserves_category(self):
        manifest = self.root / "valid.csv"
        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": r"C:\Users\User\Desktop\CogniLearn\Explosion\data\positives\exp_1.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "g_exp",
                    "original_recording_id": "rec_1",
                    "split": "train",
                    "validation_status": "valid",
                },
                {
                    "file_path": "hard_negatives/gunfire/gun_1.mp3",
                    "label": "non_explosion",
                    "category": "artillery_firing",  # Deliberately distinct from parent dir name
                    "group_id": "g_neg",
                    "original_recording_id": "rec_2",
                    "split": "val",
                    "validation_status": "valid",
                },
            ],
        )

        res = validate_and_load_manifest(
            manifest, split="val", data_root=self.data_root, expected_total_count=2
        )
        self.assertEqual(len(res.split_rows), 1)
        self.assertEqual(res.split_rows[0]["category"], "artillery_firing")
        self.assertEqual(res.split_rows[0]["label"], "non_explosion")
        self.assertEqual(res.group_count, 2)
        self.assertEqual(Path(res.split_rows[0]["resolved_file_path"]), self.neg1.resolve())

    def test_missing_file_fails_loudly(self):
        manifest = self.root / "missing_audio.csv"
        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": "positives/does_not_exist.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "g1",
                    "original_recording_id": "r1",
                    "split": "train",
                    "validation_status": "valid",
                }
            ],
        )
        with self.assertRaises(FileNotFoundError):
            validate_and_load_manifest(manifest, split="train", data_root=self.data_root)

    def test_missing_required_columns_rejected(self):
        manifest = self.root / "bad_cols.csv"
        _write_csv(manifest, ["file_path", "split"], [{"file_path": "positives/exp_1.mp3", "split": "train"}])
        with self.assertRaises(ValueError):
            validate_and_load_manifest(manifest, data_root=self.data_root)

    def test_invalid_label_split_and_status_rejected(self):
        manifest = self.root / "bad_vals.csv"
        base_row = {
            "file_path": "positives/exp_1.mp3",
            "label": "explosion",
            "category": "positives",
            "group_id": "g1",
            "original_recording_id": "r1",
            "split": "train",
            "validation_status": "valid",
        }
        for col, bad_val in [("label", "firework"), ("split", "holdout"), ("validation_status", "excluded")]:
            row = dict(base_row, **{col: bad_val})
            _write_csv(manifest, self.fieldnames, [row])
            with self.subTest(col=col, bad_val=bad_val):
                with self.assertRaises(ValueError):
                    validate_and_load_manifest(manifest, data_root=self.data_root)

    def test_duplicate_entries_and_missing_group_id_rejected(self):
        manifest = self.root / "dups.csv"
        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": "positives/exp_1.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "g1",
                    "original_recording_id": "r1",
                    "split": "train",
                    "validation_status": "valid",
                },
                {
                    "file_path": "./positives/exp_1.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "g2",
                    "original_recording_id": "r2",
                    "split": "train",
                    "validation_status": "valid",
                },
            ],
        )
        with self.assertRaises(ValueError):
            validate_and_load_manifest(manifest, data_root=self.data_root, resolve_all_paths=True)

        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": "positives/exp_1.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "",
                    "original_recording_id": "r1",
                    "split": "train",
                    "validation_status": "valid",
                }
            ],
        )
        with self.assertRaises(ValueError):
            validate_and_load_manifest(manifest, data_root=self.data_root)

    def test_group_cross_split_and_mixed_label_leakage_rejected(self):
        manifest = self.root / "leak.csv"
        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": "hard_negatives/gunfire/gun_1.mp3",
                    "label": "non_explosion",
                    "category": "gunfire",
                    "group_id": "shared_g",
                    "original_recording_id": "r1",
                    "split": "train",
                    "validation_status": "valid",
                },
                {
                    "file_path": "hard_negatives/gunfire/gun_2.mp3",
                    "label": "non_explosion",
                    "category": "gunfire",
                    "group_id": "shared_g",
                    "original_recording_id": "r2",
                    "split": "val",
                    "validation_status": "valid",
                },
            ],
        )
        with self.assertRaises(ValueError):
            validate_and_load_manifest(manifest, data_root=self.data_root)

        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": "positives/exp_1.mp3",
                    "label": "explosion",
                    "category": "positives",
                    "group_id": "mixed_g",
                    "original_recording_id": "r1",
                    "split": "train",
                    "validation_status": "valid",
                },
                {
                    "file_path": "hard_negatives/gunfire/gun_1.mp3",
                    "label": "non_explosion",
                    "category": "gunfire",
                    "group_id": "mixed_g",
                    "original_recording_id": "r2",
                    "split": "train",
                    "validation_status": "valid",
                },
            ],
        )
        with self.assertRaises(ValueError):
            validate_and_load_manifest(manifest, data_root=self.data_root)

    def test_external_spray_path_rejected_without_touching_disk(self):
        manifest = self.root / "external.csv"
        _write_csv(
            manifest,
            self.fieldnames,
            [
                {
                    "file_path": r"C:\Users\User\Desktop\CogniLearn\Spray\backend\data\raw\door_slam_001.wav",
                    "label": "non_explosion",
                    "category": "impacts",
                    "group_id": "g_ext",
                    "original_recording_id": "r_ext",
                    "split": "train",
                    "validation_status": "valid",
                }
            ],
        )
        with self.assertRaises(UnsafeDatasetPathError):
            validate_and_load_manifest(manifest, split="train", data_root=self.data_root)

    def test_real_phase11_manifest_validates_and_flags_07049006_for_review(self):
        phase11_manifest = (
            EXPLOSION_DETECTION_ROOT / "data" / "manifests" / "dataset_manifest_split_phase11.csv"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = validate_and_load_manifest(
                phase11_manifest,
                require_group_id=True,
                require_category=True,
                expected_total_count=193,
                expected_split_counts={"train": 153, "val": 20, "test": 20},
                resolve_all_paths=True,
            )

        self.assertEqual(res.label_counts, {"explosion": 92, "non_explosion": 101})
        self.assertEqual(res.group_count, 84)
        self.assertEqual(len(res.resolved_paths), 193)
        self.assertTrue(
            any("bbc_industry--_07049006.mp3" in flag for flag in res.review_flags)
        )
        self.assertTrue(
            any("bbc_industry--_07049006.mp3" in str(w.message) for w in caught)
        )


if __name__ == "__main__":
    unittest.main()

