"""Non-PyTorch tests for evaluation metrics, recording aggregation, and threshold calibration."""

import ast
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLOSION_DETECTION_ROOT = PROJECT_ROOT / "explosion_detection"

for p in (str(PROJECT_ROOT), str(EXPLOSION_DETECTION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics_utils import (
    aggregate_recording_predictions,
    calibrate_threshold_on_val,
    compute_binary_metrics,
    compute_category_false_positives,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Verify exact metric calculations against Phase 8 and Phase 11 artifacts and edge cases."""

    @unittest.skip("Historical Phase 8 evaluation artifacts archived")
    def test_reproduces_phase8_historical_test_metrics(self):
        preds_csv = EXPLOSION_DETECTION_ROOT / "reports" / "phase8_test_evaluation" / "predictions_test.csv"
        eval_json = EXPLOSION_DETECTION_ROOT / "reports" / "phase8_test_evaluation" / "evaluation_test.json"
        with open(preds_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        expected = json.loads(eval_json.read_text(encoding="utf-8"))["recording_metrics"]

        y_true = np.array([int(r["true_label"]) for r in rows])
        y_scores = np.array([float(r["max_score"]) for r in rows])
        y_pred = np.array([int(r["pred_label"]) for r in rows])

        actual = compute_binary_metrics(y_true, y_scores, threshold=0.90, y_pred=y_pred)
        self.assertEqual(actual["tp"], expected["tp"])
        self.assertEqual(actual["fp"], expected["fp"])
        self.assertEqual(actual["tn"], expected["tn"])
        self.assertEqual(actual["fn"], expected["fn"])
        self.assertEqual(actual["precision"], expected["precision"])
        self.assertEqual(actual["recall"], expected["recall"])
        self.assertEqual(actual["f1"], expected["f1"])
        self.assertEqual(actual["specificity"], expected["specificity"])
        self.assertEqual(actual["balanced_accuracy"], expected["balanced_accuracy"])
        self.assertEqual(actual["roc_auc"], expected["roc_auc"])
        self.assertEqual(actual["pr_auc"], expected["pr_auc"])
        self.assertEqual(actual["f1_negative"], 0.7778)
        self.assertEqual(actual["macro_f1"], 0.7980)

    @unittest.skip("Historical Phase 11 evaluation artifacts archived")
    def test_reproduces_phase11_historical_test_metrics(self):
        preds_csv = EXPLOSION_DETECTION_ROOT / "reports" / "phase11_test_evaluation" / "predictions_test.csv"
        eval_json = EXPLOSION_DETECTION_ROOT / "reports" / "phase11_test_evaluation" / "evaluation_test.json"
        with open(preds_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        expected = json.loads(eval_json.read_text(encoding="utf-8"))["recording_metrics"]

        y_true = np.array([int(r["true_label"]) for r in rows])
        y_scores = np.array([float(r["max_score"]) for r in rows])
        y_pred = np.array([int(r["pred_label"]) for r in rows])

        actual = compute_binary_metrics(y_true, y_scores, threshold=0.90, y_pred=y_pred)
        for key in (
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1",
            "f1_positive",
            "f1_negative",
            "macro_f1",
            "specificity",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
        ):
            self.assertEqual(actual[key], expected[key], f"Mismatch on key '{key}'")

    def test_calibrate_threshold_maximizes_macro_f1_over_positive_f1(self):
        # At t=0.55: TN=5, FP=3, TP=4, FN=0 -> Pos F1=0.7273 (higher), Macro F1=0.7483
        # At t=0.56..0.85: TN=8, FP=0, TP=2, FN=2 -> Pos F1=0.6667, Macro F1=0.7778 (higher)
        y_true = np.array([0] * 8 + [1] * 4)
        y_scores = np.array([0.10] * 5 + [0.55] * 3 + [0.55] * 2 + [0.85] * 2)

        best_thresh, best_macro_f1 = calibrate_threshold_on_val(y_true, y_scores)
        self.assertGreater(best_thresh, 0.55)
        self.assertLessEqual(best_thresh, 0.85)
        self.assertAlmostEqual(best_thresh, 0.56, places=2)
        self.assertAlmostEqual(best_macro_f1, 0.7778, places=4)

    def test_calibrate_threshold_rejects_empty_arrays(self):
        with self.assertRaises(ValueError):
            calibrate_threshold_on_val(np.array([]), np.array([]))

    def test_single_class_and_empty_edge_cases(self):
        empty = compute_binary_metrics([], [], threshold=0.50)
        self.assertEqual(empty["total_samples"], 0)
        self.assertIsNone(empty["roc_auc"])
        self.assertIsNone(empty["pr_auc"])
        self.assertEqual(empty["confusion_matrix"], [[0, 0], [0, 0]])

        all_neg = compute_binary_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.50)
        self.assertEqual(all_neg["tn"], 3)
        self.assertEqual(all_neg["specificity"], 1.0)
        self.assertIsNone(all_neg["roc_auc"])
        self.assertIsNone(all_neg["pr_auc"])

        all_pos = compute_binary_metrics([1, 1, 1], [0.7, 0.8, 0.9], threshold=0.50)
        self.assertEqual(all_pos["tp"], 3)
        self.assertEqual(all_pos["recall"], 1.0)
        self.assertIsNone(all_pos["roc_auc"])
        self.assertIsNone(all_pos["pr_auc"])

    def test_validation_errors_on_bad_inputs(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 1], [0.2])
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 1], [0.2, float("nan")])
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 1], [float("inf"), 0.8])
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 2], [0.2, 0.8])

    def test_recording_aggregation_uses_authoritative_category_and_consecutive_rule(self):
        # Simulate file stored under raw/positives/ that Phase 11 reclassified to non-explosion artillery_firing,
        # plus an explosion with out-of-order window metadata and a broken 2+1 run vs true 3-window run.
        meta = [
            {
                "file_path": "data/raw/positives/07027149.mp3",
                "category": "artillery_firing",
                "label_str": "non_explosion",
                "window_index": 1,
            },
            {
                "file_path": "data/raw/positives/07027149.mp3",
                "category": "artillery_firing",
                "label_str": "non_explosion",
                "window_index": 0,
            },
            {
                "file_path": "data/raw/positives/07027149.mp3",
                "category": "artillery_firing",
                "label_str": "non_explosion",
                "window_index": 2,
            },
            {
                "file_path": "data/raw/positives/07027149.mp3",
                "category": "artillery_firing",
                "label_str": "non_explosion",
                "window_index": 3,
            },
            {
                "file_path": "data/raw/positives/explosion_a.mp3",
                "category": "positives",
                "label_str": "explosion",
                "window_index": 2,
            },
            {
                "file_path": "data/raw/positives/explosion_a.mp3",
                "category": "positives",
                "label_str": "explosion",
                "window_index": 0,
            },
            {
                "file_path": "data/raw/positives/explosion_a.mp3",
                "category": "positives",
                "label_str": "explosion",
                "window_index": 1,
            },
        ]
        # Artillery windows sorted: idx 0..3 -> [0.95, 0.92, 0.10, 0.96] (max run = 2 -> negative for C=3)
        # Explosion windows sorted: idx 0..2 -> [0.91, 0.93, 0.95] (max run = 3 -> positive for C=3)
        scores = np.array([0.92, 0.95, 0.10, 0.96, 0.95, 0.91, 0.93])

        recs, rec_metrics = aggregate_recording_predictions(
            meta,
            scores,
            threshold=0.90,
            rule="consecutive",
            consecutive_windows=3,
        )
        art = recs["data/raw/positives/07027149.mp3"]
        self.assertEqual(art["category"], "artillery_firing")
        self.assertEqual(art["max_consecutive_run"], 2)
        self.assertEqual(art["pred_label"], 0)
        self.assertTrue(art["correct"])

        exp = recs["data/raw/positives/explosion_a.mp3"]
        self.assertEqual(exp["category"], "positives")
        self.assertEqual(exp["max_consecutive_run"], 3)
        self.assertEqual(exp["pred_label"], 1)
        self.assertTrue(exp["correct"])

        self.assertEqual(rec_metrics["macro_f1"], 1.0)
        fp_report = compute_category_false_positives(recs)
        self.assertIn("artillery_firing", fp_report)
        self.assertEqual(fp_report["artillery_firing"]["false_positives"], 0)

    def test_recording_aggregation_fallback_category_and_input_validation(self):
        meta = [
            {"file_path": "test/gunfire/g1.mp3", "label_str": "non_explosion", "window_index": 0},
        ]
        recs, _ = aggregate_recording_predictions(meta, np.array([0.95]), threshold=0.90, rule="peak")
        self.assertEqual(recs["test/gunfire/g1.mp3"]["category"], "gunfire")
        self.assertEqual(recs["test/gunfire/g1.mp3"]["pred_label"], 1)

        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, np.array([0.5, 0.6]))
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, np.array([0.5]), rule="invalid")
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(meta, np.array([0.5]), rule="consecutive", consecutive_windows=0)
        with self.assertRaises(ValueError):
            aggregate_recording_predictions(
                [
                    {"file_path": "a.wav", "label_str": "explosion", "window_index": 0},
                    {"file_path": "a.wav", "label_str": "non_explosion", "window_index": 1},
                ],
                np.array([0.9, 0.1]),
            )

    @unittest.skip("Historical Phase evaluate.py contract archived")
    def test_evaluate_script_static_contract(self):
        eval_path = EXPLOSION_DETECTION_ROOT / "scripts" / "evaluate.py"
        source = eval_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Verify evaluate.py imports calibrate_threshold_on_val cleanly rather than duplicating it
        defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("calibrate_threshold_on_val", defined_funcs)

        # Verify every torch.load call in evaluate.py explicitly passes weights_only=True
        torch_load_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                torch_load_calls.append(node)
        self.assertGreaterEqual(len(torch_load_calls), 1)
        for call in torch_load_calls:
            kw_map = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn("weights_only", kw_map)
            self.assertIsInstance(kw_map["weights_only"], ast.Constant)
            self.assertIs(kw_map["weights_only"].value, True)


if __name__ == "__main__":
    unittest.main()

