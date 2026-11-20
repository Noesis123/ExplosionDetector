"""Pure-NumPy / scikit-learn Evaluation Metrics and Recording Aggregation Utilities.

Kept free of PyTorch and Numba dependencies so metric calculation, recording-level
aggregation, category false-positive analysis, and threshold calibration can be
tested and reused cleanly in any environment.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.50,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, List[List[int]], None]]:
    """Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_scores: Predicted probabilities in [0.0, 1.0].
        threshold: Decision threshold for positive prediction.
        y_pred: Optional explicit binary predictions (0 or 1). If None,
                computed as (y_scores >= threshold).astype(int).

    Returns:
        Dictionary of metrics including TP, FP, TN, FN, precision, recall,
        specificity, balanced accuracy, positive/negative/macro F1, ROC-AUC,
        and PR-AUC.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)

    if y_true_arr.ndim != 1 or y_scores_arr.ndim != 1:
        raise ValueError("y_true and y_scores must be 1-dimensional arrays.")
    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_scores)={len(y_scores_arr)}."
        )
    if len(y_scores_arr) > 0 and not np.all(np.isfinite(y_scores_arr)):
        raise ValueError("y_scores must contain only finite values (no NaN or Inf).")
    if len(y_true_arr) > 0 and not np.all(np.isin(y_true_arr, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if y_pred is None:
        y_pred_arr = (y_scores_arr >= threshold).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=int)
        if y_pred_arr.ndim != 1 or len(y_pred_arr) != len(y_true_arr):
            raise ValueError(
                f"Length mismatch: len(y_pred)={len(y_pred_arr)} != len(y_true)={len(y_true_arr)}."
            )
        if len(y_pred_arr) > 0 and not np.all(np.isin(y_pred_arr, [0, 1])):
            raise ValueError("y_pred must contain only binary predictions 0 and 1.")

    total = len(y_true_arr)
    if total == 0:
        return {
            "threshold": float(threshold),
            "total_samples": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_positive": 0.0,
            "f1_negative": 0.0,
            "macro_f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1_pos = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, average="binary", zero_division=0))
    f1_neg = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, average="binary", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # AUC metrics require both classes to be present in y_true
    has_both_classes = len(np.unique(y_true_arr)) > 1
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
    else:
        roc_auc = None
        pr_auc = None

    return {
        "threshold": float(threshold),
        "total_samples": int(total),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_pos, 4),
        "f1_positive": round(f1_pos, 4),
        "f1_negative": round(f1_neg, 4),
        "macro_f1": round(macro_f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }


def calibrate_threshold_on_val(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """Find the threshold in [0.10, 0.90] that maximizes Macro F1 on validation data."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    if len(y_true_arr) == 0:
        raise ValueError("Cannot calibrate threshold on empty validation data.")

    best_thresh = 0.50
    best_macro_f1 = -1.0
    for t in np.linspace(0.10, 0.90, 81):
        m = compute_binary_metrics(y_true_arr, y_scores_arr, threshold=float(t))
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(m["macro_f1"])
            best_thresh = float(round(t, 2))
    return best_thresh, best_macro_f1


def aggregate_recording_predictions(
    samples_metadata: List[Dict],
    window_scores: np.ndarray,
    threshold: float = 0.50,
    rule: str = "peak",
    consecutive_windows: int = 3,
) -> Tuple[Dict[str, Dict], Dict[str, Union[float, int, List[List[int]], None]]]:
    """Aggregate window-level prediction scores to recording-level decisions.

    Supported rules:
    - 'peak': Alert if maximum window score >= threshold.
    - 'consecutive': Alert if at least consecutive_windows contiguous windows
                     have score >= threshold.

    Returns:
        Tuple of (recording_details_dict, overall_metrics_dict).
    """
    if rule not in ("peak", "consecutive"):
        raise ValueError(f"Unknown aggregation rule '{rule}'. Must be 'peak' or 'consecutive'.")
    if consecutive_windows < 1:
        raise ValueError(f"consecutive_windows must be >= 1, got {consecutive_windows}.")

    scores_arr = np.asarray(window_scores, dtype=float)
    if scores_arr.ndim != 1:
        raise ValueError("window_scores must be a 1-dimensional array.")
    if len(samples_metadata) != len(scores_arr):
        raise ValueError(
            f"Length mismatch: len(samples_metadata)={len(samples_metadata)} != len(window_scores)={len(scores_arr)}."
        )
    if len(scores_arr) > 0 and not np.all(np.isfinite(scores_arr)):
        raise ValueError("window_scores must contain only finite values (no NaN or Inf).")

    rec_windows = defaultdict(list)
    rec_labels: Dict[str, int] = {}
    rec_cats: Dict[str, str] = {}

    for idx, meta in enumerate(samples_metadata):
        fp = meta["file_path"]
        sort_key = meta.get("window_index", meta.get("start_sec", idx))
        lbl = 1 if meta["label_str"] == "explosion" else 0
        if fp in rec_labels and rec_labels[fp] != lbl:
            raise ValueError(f"Conflicting labels in metadata for recording: {fp}")

        cat_val = str(meta.get("category") or "").strip()
        category = cat_val if cat_val else Path(fp).parent.name

        rec_windows[fp].append((sort_key, float(scores_arr[idx])))
        rec_labels[fp] = lbl
        if fp not in rec_cats:
            rec_cats[fp] = category

    records: Dict[str, Dict] = {}
    y_true_recs: List[int] = []
    y_scores_recs: List[float] = []
    y_pred_recs: List[int] = []

    for fp, items in sorted(rec_windows.items()):
        sorted_items = sorted(items, key=lambda x: x[0])
        scores = [score for _, score in sorted_items]

        max_score = float(np.max(scores)) if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        true_lbl = rec_labels[fp]

        max_consecutive_run = 0
        current_run = 0
        for s in scores:
            if s >= threshold:
                current_run += 1
                if current_run > max_consecutive_run:
                    max_consecutive_run = current_run
            else:
                current_run = 0

        if rule == "consecutive":
            pred_lbl = 1 if max_consecutive_run >= consecutive_windows else 0
        else:
            pred_lbl = 1 if max_score >= threshold else 0

        y_true_recs.append(true_lbl)
        y_scores_recs.append(max_score)
        y_pred_recs.append(pred_lbl)

        records[fp] = {
            "file_path": fp,
            "category": rec_cats[fp],
            "true_label": true_lbl,
            "true_label_str": "explosion" if true_lbl == 1 else "non_explosion",
            "pred_label": pred_lbl,
            "pred_label_str": "explosion" if pred_lbl == 1 else "non_explosion",
            "num_windows": len(scores),
            "max_score": round(max_score, 4),
            "mean_score": round(mean_score, 4),
            "max_consecutive_run": int(max_consecutive_run),
            "rule": rule,
            "consecutive_windows": int(consecutive_windows) if rule == "consecutive" else None,
            "correct": bool(true_lbl == pred_lbl),
        }

    rec_metrics = compute_binary_metrics(
        np.array(y_true_recs, dtype=int),
        np.array(y_scores_recs, dtype=float),
        threshold=threshold,
        y_pred=np.array(y_pred_recs, dtype=int),
    )

    return records, rec_metrics


def compute_category_false_positives(
    recording_records: Dict[str, Dict],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Analyze false positive counts and rates across non-explosion categories."""
    cat_counts = defaultdict(int)
    cat_fp = defaultdict(int)

    for _fp, rec in recording_records.items():
        if rec["true_label"] == 0:
            cat = rec["category"]
            cat_counts[cat] += 1
            if rec["pred_label"] == 1:
                cat_fp[cat] += 1

    report: Dict[str, Dict[str, Union[int, float]]] = {}
    for cat in sorted(cat_counts.keys()):
        total = cat_counts[cat]
        fps = cat_fp[cat]
        fpr = round(fps / total, 4) if total > 0 else 0.0
        report[cat] = {
            "total_recordings": total,
            "false_positives": fps,
            "false_positive_rate": fpr,
        }

    return report


MULTICLASS_DEFAULT_NAMES: Tuple[str, ...] = ("non_target", "explosion", "fireworks")


def compute_multiclass_metrics(
    y_true: Sequence[int],
    y_probs: Optional[np.ndarray] = None,
    y_pred: Optional[Sequence[int]] = None,
    class_names: Sequence[str] = MULTICLASS_DEFAULT_NAMES,
) -> Dict[str, Any]:
    """Compute 3-class confusion matrix, per-class P/R/F1, Macro F1, and alert confusion metrics."""
    names = tuple(class_names)
    num_classes = len(names)
    if num_classes < 2:
        raise ValueError(f"Expected at least 2 classes, got {num_classes}.")
    valid_ids = list(range(num_classes))

    y_true_arr = np.asarray(y_true, dtype=int)
    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got {y_true_arr.ndim}D.")
    total = len(y_true_arr)
    if total > 0 and not np.all(np.isin(y_true_arr, valid_ids)):
        raise ValueError(f"y_true must contain only class IDs in {valid_ids}.")

    if y_probs is not None:
        probs_arr = np.asarray(y_probs, dtype=float)
        if probs_arr.ndim != 2 or probs_arr.shape[1] != num_classes:
            raise ValueError(
                f"y_probs must have shape (N, {num_classes}), got {probs_arr.shape}."
            )
        if len(probs_arr) != total:
            raise ValueError(
                f"Length mismatch: len(y_true)={total} != len(y_probs)={len(probs_arr)}."
            )
        if len(probs_arr) > 0 and not np.all(np.isfinite(probs_arr)):
            raise ValueError("y_probs must contain only finite values (no NaN or Inf).")
    else:
        probs_arr = None

    if y_pred is not None:
        y_pred_arr = np.asarray(y_pred, dtype=int)
        if y_pred_arr.ndim != 1 or len(y_pred_arr) != total:
            raise ValueError(
                f"Length mismatch: len(y_true)={total} != len(y_pred)={len(y_pred_arr)}."
            )
        if total > 0 and not np.all(np.isin(y_pred_arr, valid_ids)):
            raise ValueError(f"y_pred must contain only class IDs in {valid_ids}.")
    elif probs_arr is not None:
        y_pred_arr = np.argmax(probs_arr, axis=1) if total > 0 else np.empty((0,), dtype=int)
    else:
        raise ValueError("Either y_probs or y_pred must be provided.")

    if total == 0:
        empty_cm = [[0] * num_classes for _ in range(num_classes)]
        empty_per_class = {
            name: {
                "class_id": idx,
                "support": 0,
                "predicted_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            for idx, name in enumerate(names)
        }
        return {
            "class_names": list(names),
            "total_samples": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": empty_cm,
            "per_class": empty_per_class,
            "false_positives_by_class": {name: 0 for name in names},
            "false_negatives_by_class": {name: 0 for name in names},
            "explosion_fireworks_confusion": {
                "true_explosion_pred_fireworks": 0,
                "true_fireworks_pred_explosion": 0,
                "total_cross_confusions": 0,
            },
            "non_target_false_alarms": {
                "non_target_pred_explosion": 0,
                "non_target_pred_fireworks": 0,
                "total_false_alarms_on_non_target": 0,
            },
            "missed_targets_as_non_target": {
                "explosion_pred_non_target": 0,
                "fireworks_pred_non_target": 0,
                "total_missed_targets": 0,
            },
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=valid_ids)
    per_class: Dict[str, Dict[str, Union[int, float]]] = {}
    f1_raw: List[float] = []
    supports: List[int] = []

    for idx, name in enumerate(names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = int(cm[idx, :].sum())
        predicted_count = int(cm[:, idx].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_val = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        f1_raw.append(f1_val)
        supports.append(support)
        per_class[name] = {
            "class_id": idx,
            "support": support,
            "predicted_count": predicted_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
        }

    accuracy = float(np.trace(cm) / total)
    macro_f1 = float(np.mean(f1_raw))
    weighted_f1 = float(sum(f1_raw[i] * supports[i] for i in range(num_classes)) / total)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    idx_map = {name: i for i, name in enumerate(names)}
    if {"non_target", "explosion", "fireworks"}.issubset(idx_map):
        nt_i = idx_map["non_target"]
        exp_i = idx_map["explosion"]
        fw_i = idx_map["fireworks"]
        exp_as_fw = int(cm[exp_i, fw_i])
        fw_as_exp = int(cm[fw_i, exp_i])
        nt_as_exp = int(cm[nt_i, exp_i])
        nt_as_fw = int(cm[nt_i, fw_i])
        exp_as_nt = int(cm[exp_i, nt_i])
        fw_as_nt = int(cm[fw_i, nt_i])
    else:
        exp_as_fw = fw_as_exp = nt_as_exp = nt_as_fw = exp_as_nt = fw_as_nt = 0

    return {
        "class_names": list(names),
        "total_samples": int(total),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "false_positives_by_class": {name: int(per_class[name]["fp"]) for name in names},
        "false_negatives_by_class": {name: int(per_class[name]["fn"]) for name in names},
        "explosion_fireworks_confusion": {
            "true_explosion_pred_fireworks": exp_as_fw,
            "true_fireworks_pred_explosion": fw_as_exp,
            "total_cross_confusions": exp_as_fw + fw_as_exp,
        },
        "non_target_false_alarms": {
            "non_target_pred_explosion": nt_as_exp,
            "non_target_pred_fireworks": nt_as_fw,
            "total_false_alarms_on_non_target": nt_as_exp + nt_as_fw,
        },
        "missed_targets_as_non_target": {
            "explosion_pred_non_target": exp_as_nt,
            "fireworks_pred_non_target": fw_as_nt,
            "total_missed_targets": exp_as_nt + fw_as_nt,
        },
    }


def aggregate_multiclass_recording_predictions(
    samples_metadata: List[Dict],
    window_probs: np.ndarray,
    explosion_threshold: float = 0.80,
    fireworks_threshold: float = 0.80,
    consecutive_windows: int = 1,
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """Aggregate window-level 3-class probabilities into recording-level metrics and independent alerts."""
    try:
        from explosion_detection.multiclass_contract import (
            MULTICLASS_INV_LABEL_MAP,
            MULTICLASS_LABEL_MAP,
            evaluate_multiclass_alerts,
        )
    except ImportError:
        from multiclass_contract import (
            MULTICLASS_INV_LABEL_MAP,
            MULTICLASS_LABEL_MAP,
            evaluate_multiclass_alerts,
        )

    probs_arr = np.asarray(window_probs, dtype=float)
    if probs_arr.ndim != 2 or probs_arr.shape[1] != 3:
        raise ValueError(f"window_probs must have shape (N, 3), got {probs_arr.shape}.")
    if len(samples_metadata) != len(probs_arr):
        raise ValueError(
            f"Length mismatch: len(samples_metadata)={len(samples_metadata)} != len(window_probs)={len(probs_arr)}."
        )
    if len(probs_arr) > 0 and not np.all(np.isfinite(probs_arr)):
        raise ValueError("window_probs must contain only finite values (no NaN or Inf).")

    rec_windows = defaultdict(list)
    rec_labels: Dict[str, int] = {}
    rec_cats: Dict[str, str] = {}

    for idx, meta in enumerate(samples_metadata):
        fp = meta["file_path"]
        sort_key = (meta.get("window_index", idx), meta.get("start_sec", idx), idx)
        lbl_str = str(meta["label_str"]).strip().lower()
        if lbl_str not in MULTICLASS_LABEL_MAP:
            raise ValueError(f"Unknown multiclass label_str {lbl_str!r} in metadata for {fp}.")
        lbl = MULTICLASS_LABEL_MAP[lbl_str]
        if fp in rec_labels and rec_labels[fp] != lbl:
            raise ValueError(f"Conflicting multiclass labels in metadata for recording: {fp}")
        rec_labels[fp] = lbl
        rec_cats[fp] = str(meta.get("category") or "").strip() or Path(fp).parent.name
        rec_windows[fp].append((sort_key, probs_arr[idx]))

    records: Dict[str, Dict] = {}
    y_true_recs: List[int] = []
    mean_probs_recs: List[np.ndarray] = []
    alert_eval_summary = {
        "explosion_alerts_triggered": 0,
        "fireworks_alerts_triggered": 0,
        "explosion_true_positives": 0,
        "fireworks_true_positives": 0,
        "non_target_false_explosion_alerts": 0,
        "non_target_false_fireworks_alerts": 0,
        "fireworks_false_explosion_alerts": 0,
        "explosion_false_fireworks_alerts": 0,
    }

    for fp, items in sorted(rec_windows.items()):
        sorted_items = sorted(items, key=lambda x: x[0])
        seq_probs = np.vstack([p for _, p in sorted_items])
        alert_res = evaluate_multiclass_alerts(
            seq_probs,
            explosion_threshold=explosion_threshold,
            fireworks_threshold=fireworks_threshold,
            consecutive_windows=consecutive_windows,
        )
        true_lbl = rec_labels[fp]
        true_str = MULTICLASS_INV_LABEL_MAP[true_lbl]
        mean_vec = np.mean(seq_probs, axis=0)

        y_true_recs.append(true_lbl)
        mean_probs_recs.append(mean_vec)

        exp_trig = bool(alert_res["explosion_triggered"])
        fw_trig = bool(alert_res["fireworks_triggered"])
        alert_eval_summary["explosion_alerts_triggered"] += int(exp_trig)
        alert_eval_summary["fireworks_alerts_triggered"] += int(fw_trig)
        if true_str == "explosion":
            alert_eval_summary["explosion_true_positives"] += int(exp_trig)
            alert_eval_summary["explosion_false_fireworks_alerts"] += int(fw_trig)
        elif true_str == "fireworks":
            alert_eval_summary["fireworks_true_positives"] += int(fw_trig)
            alert_eval_summary["fireworks_false_explosion_alerts"] += int(exp_trig)
        else:
            alert_eval_summary["non_target_false_explosion_alerts"] += int(exp_trig)
            alert_eval_summary["non_target_false_fireworks_alerts"] += int(fw_trig)

        records[fp] = {
            "file_path": fp,
            "category": rec_cats[fp],
            "true_label": true_lbl,
            "true_label_str": true_str,
            "pred_label": int(alert_res["predicted_class_id"]),
            "pred_label_str": str(alert_res["predicted_class"]),
            "num_windows": int(len(seq_probs)),
            "explosion_triggered": exp_trig,
            "fireworks_triggered": fw_trig,
            "alerts": list(alert_res["alerts"]),
            "primary_alert": alert_res["primary_alert"],
            "max_explosion_score": alert_res["max_explosion_score"],
            "max_fireworks_score": alert_res["max_fireworks_score"],
            "correct": bool(true_lbl == int(alert_res["predicted_class_id"])),
        }

    probs_matrix = (
        np.vstack(mean_probs_recs) if mean_probs_recs else np.empty((0, 3), dtype=float)
    )
    rec_metrics = compute_multiclass_metrics(
        y_true_recs,
        y_probs=probs_matrix,
        class_names=MULTICLASS_DEFAULT_NAMES,
    )
    rec_metrics["alert_summary"] = alert_eval_summary
    return records, rec_metrics


