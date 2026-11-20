"""Non-PyTorch unit tests for dataset path portability and Phase 11 path resolution."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.path_utils import (
    AmbiguousDatasetPathError,
    UnsafeDatasetPathError,
    resolve_dataset_path,
    resolve_manifest_entries,
)


class TestDatasetPathPortability(unittest.TestCase):
    """Verify secure, drive-independent path resolution against a synthetic data root."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        (self.data_root / "positives").mkdir(parents=True)
        (self.data_root / "hard_negatives" / "gunfire").mkdir(parents=True)
        (self.data_root / "hard_negatives" / "thunder").mkdir(parents=True)

        self.outside_file = self.root / "outside_secret.wav"
        self.outside_file.write_bytes(b"RIFF")

        self.pos_file = self.data_root / "positives" / "boom_unique.wav"
        self.pos_file.write_bytes(b"RIFF")

        self.dup_a = self.data_root / "positives" / "dup_clip.mp3"
        self.dup_b = self.data_root / "hard_negatives" / "gunfire" / "dup_clip.mp3"
        self.dup_a.write_bytes(b"MP3A")
        self.dup_b.write_bytes(b"MP3B")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_legacy_windows_paths_with_mixed_separators(self):
        legacy_backslash = r"C:\Users\User\Desktop\CogniLearn\Explosion\data\positives\boom_unique.wav"
        legacy_forward = "C:/Users/User/Desktop/CogniLearn/Explosion/data/positives/boom_unique.wav"
        resolved_a = resolve_dataset_path(legacy_backslash, data_root=self.data_root)
        resolved_b = resolve_dataset_path(legacy_forward, data_root=self.data_root)
        self.assertEqual(resolved_a, self.pos_file.resolve())
        self.assertEqual(resolved_b, self.pos_file.resolve())

    def test_valid_relative_and_active_root_paths(self):
        self.assertEqual(
            resolve_dataset_path("positives/boom_unique.wav", data_root=self.data_root),
            self.pos_file.resolve(),
        )
        self.assertEqual(
            resolve_dataset_path(r"data\positives\boom_unique.wav", data_root=self.data_root),
            self.pos_file.resolve(),
        )
        self.assertEqual(
            resolve_dataset_path("./positives/boom_unique.wav", data_root=self.data_root),
            self.pos_file.resolve(),
        )
        # Unambiguous bare filename resolves when unique inside data_root
        self.assertEqual(
            resolve_dataset_path("boom_unique.wav", data_root=self.data_root),
            self.pos_file.resolve(),
        )
        # Active root path on any drive resolves cleanly
        self.assertEqual(
            resolve_dataset_path(self.pos_file, data_root=self.data_root),
            self.pos_file.resolve(),
        )

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path("positives/missing_file.wav", data_root=self.data_root)
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path("completely_missing.wav", data_root=self.data_root)

    def test_ambiguous_bare_filename_rejected(self):
        with self.assertRaises(AmbiguousDatasetPathError):
            resolve_dataset_path("dup_clip.mp3", data_root=self.data_root)
        # Explicit subpath resolves deterministically without ambiguity
        self.assertEqual(
            resolve_dataset_path("hard_negatives/gunfire/dup_clip.mp3", data_root=self.data_root),
            self.dup_b.resolve(),
        )

    def test_unsafe_traversal_rejected(self):
        for bad in (
            "../outside_secret.wav",
            "positives/../../outside_secret.wav",
            "positives/../positives/boom_unique.wav",
            r"C:\Users\User\Desktop\CogniLearn\Explosion\data\..\..\outside_secret.wav",
            "C:boom_unique.wav",
            r"\\server\share\boom_unique.wav",
            "",
            "   ",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(UnsafeDatasetPathError):
                    resolve_dataset_path(bad, data_root=self.data_root)

    def test_incorrect_dataset_roots_and_external_paths_rejected(self):
        for ext in (
            r"C:\Users\User\Desktop\CogniLearn\Spray\backend\data\raw\door_slam_001.wav",
            r"E:\AnotherProject\Explosion\data\positives\boom_unique.wav",
            str(self.outside_file),
            "/etc/passwd",
        ):
            with self.subTest(ext=ext):
                with self.assertRaises(UnsafeDatasetPathError):
                    resolve_dataset_path(ext, data_root=self.data_root)

    def test_all_193_phase11_manifest_paths_resolve_in_workspace(self):
        manifest_dir = PROJECT_ROOT / "explosion_detection" / "data" / "manifests"
        for name in ("dataset_manifest_phase11.csv", "dataset_manifest_split_phase11.csv"):
            manifest_path = manifest_dir / name
            with manifest_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 193)
            resolved = resolve_manifest_entries([r["file_path"] for r in rows])
            self.assertEqual(len(resolved), 193)
            self.assertTrue(all(p.is_file() for p in resolved))


if __name__ == "__main__":
    unittest.main()

