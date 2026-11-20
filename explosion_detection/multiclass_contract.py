
"""Three-Class Multiclass Contract, Mapping, Softmax Helpers, and Alert Routing.

Kept free of PyTorch and Numba dependencies so class mappings, probability contracts,
and independent explosion/fireworks alert behavior can be imported and verified in any
runtime environment without triggering neural runtime imports.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

MULTICLASS_NUM_CLASSES = 3
MULTICLASS_CLASS_NAMES: Tuple[str, ...] = ("non_target", "explosion", "fireworks")

MULTICLASS_LABEL_MAP: Dict[str, int] = {
    "non_target": 0,
    "explosion": 1,
    "fireworks": 2,
}
MULTICLASS_INV_LABEL_MAP: Dict[int, str] = {
    0: "non_target",
    1: "explosion",
    2: "fireworks",
}

MULTICLASS_ALERT_MAP: Dict[int, Optional[str]] = {
    0: None,
    1: "EXPLOSION_DETECTED",
    2: "FIREWORKS_DETECTED",
}
MULTICLASS_CLASS_ALERT_MAP: Dict[str, Optional[str]] = {
    "non_target": None,
    "explosion": "EXPLOSION_DETECTED",
    "fireworks": "FIREWORKS_DETECTED",
}
MULTICLASS_CSV_ALERT_MAP: Dict[int, str] = {
    0: "None",
    1: "EXPLOSION_DETECTED",
    2: "FIREWORKS_DETECTED",
}

MULTICLASS_LOSS_SPEC = "CrossEntropyLoss"
MULTICLASS_INFERENCE_ACTIVATION = "softmax"


def resolve_alert_for_class(class_or_id: Union[int, str]) -> Optional[str]:
    """Return the independent alert string (or None) for a multiclass label name or ID."""
    if isinstance(class_or_id, bool):
        raise ValueError(f"Boolean is not a valid class identifier: {class_or_id!r}")
    if isinstance(class_or_id, int):
        if class_or_id not in MULTICLASS_ALERT_MAP:
            raise ValueError(
                f"Unknown class ID {class_or_id!r}; expected one of {sorted(MULTICLASS_ALERT_MAP)}."
            )
        return MULTICLASS_ALERT_MAP[class_or_id]
    if isinstance(class_or_id, str):
        key = class_or_id.strip().lower()
        if key not in MULTICLASS_CLASS_ALERT_MAP:
            raise ValueError(
                f"Unknown class name {class_or_id!r}; expected one of {list(MULTICLASS_CLASS_ALERT_MAP)}."
            )
        return MULTICLASS_CLASS_ALERT_MAP[key]
    raise TypeError(f"Expected int or str class identifier, got {type(class_or_id).__name__}.")


def map_phase11_record_to_multiclass(binary_label: str, category: str) -> Tuple[str, int, str]:
    """Map a verified Phase 11 binary record to Phase 12 three-class label metadata.

    Returns:
        Tuple of ``(label_name, label_id, csv_alert)``.
    """
    b_lbl = str(binary_label or "").strip().lower()
    cat = str(category or "").strip().lower()

    if b_lbl == "explosion":
        if cat == "fireworks":
            raise ValueError("Contradictory Phase 11 record: binary_label='explosion' with category='fireworks'.")
        cls_id = MULTICLASS_LABEL_MAP["explosion"]
        return "explosion", cls_id, MULTICLASS_CSV_ALERT_MAP[cls_id]

    if b_lbl == "non_explosion":
        if cat == "fireworks":
            cls_id = MULTICLASS_LABEL_MAP["fireworks"]
            return "fireworks", cls_id, MULTICLASS_CSV_ALERT_MAP[cls_id]
        cls_id = MULTICLASS_LABEL_MAP["non_target"]
        return "non_target", cls_id, MULTICLASS_CSV_ALERT_MAP[cls_id]

    raise ValueError(f"Unsupported binary_label {binary_label!r} (expected 'explosion' or 'non_explosion').")


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax to 1D ``(3,)`` or 2D ``(N, 3)`` class logits."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != MULTICLASS_NUM_CLASSES:
            raise ValueError(
                f"Expected {MULTICLASS_NUM_CLASSES} class logits, got shape {arr.shape}."
            )
    elif arr.ndim == 2:
        if arr.shape[1] != MULTICLASS_NUM_CLASSES:
            raise ValueError(
                f"Expected shape (N, {MULTICLASS_NUM_CLASSES}), got shape {arr.shape}."
            )
    else:
        raise ValueError(f"Expected 1D or 2D logits array, got {arr.ndim}D shape {arr.shape}.")

    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Logits must contain only finite values (no NaN or Inf).")
    if arr.size == 0:
        return np.empty(arr.shape, dtype=np.float32)

    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    return probs.astype(np.float32)


def _max_consecutive_run(scores: Sequence[float], threshold: float) -> int:
    best = 0
    cur = 0
    for s in scores:
        if s >= threshold:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def evaluate_multiclass_alerts(
    probabilities: np.ndarray,
    explosion_threshold: float = 0.80,
    fireworks_threshold: float = 0.80,
    consecutive_windows: int = 1,
) -> Dict[str, object]:
    """Evaluate independent explosion and fireworks alerts from class probabilities.

    Args:
        probabilities: Single window ``(3,)`` or chronological sequence ``(N, 3)``
            matching ``[non_target, explosion, fireworks]``.
        explosion_threshold: Independent score threshold for ``EXPLOSION_DETECTED``.
        fireworks_threshold: Independent score threshold for ``FIREWORKS_DETECTED``.
        consecutive_windows: Minimum consecutive qualifying windows required for alert.

    Returns:
        Dictionary with predicted class info, independent booleans, active alerts list,
        and ``primary_alert`` (or ``None`` when neither target class triggers).
    """
    if consecutive_windows < 1:
        raise ValueError(f"consecutive_windows must be >= 1, got {consecutive_windows}.")
    if not (0.0 <= explosion_threshold <= 1.0) or not (0.0 <= fireworks_threshold <= 1.0):
        raise ValueError("Thresholds must be in [0.0, 1.0].")

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        if probs.shape != (MULTICLASS_NUM_CLASSES,):
            raise ValueError(f"Expected shape (3,), got {probs.shape}.")
        probs = probs[np.newaxis, :]
    elif probs.ndim == 2:
        if probs.shape[0] == 0 or probs.shape[1] != MULTICLASS_NUM_CLASSES:
            raise ValueError(f"Expected non-empty (N, 3) probabilities, got shape {probs.shape}.")
    else:
        raise ValueError(f"Expected 1D or 2D probabilities array, got {probs.ndim}D shape {probs.shape}.")

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities must contain only finite values.")
    if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
        raise ValueError("Probabilities must lie in [0.0, 1.0].")
    if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3):
        raise ValueError("Class probability rows must sum to 1.0.")

    exp_scores = probs[:, MULTICLASS_LABEL_MAP["explosion"]]
    fw_scores = probs[:, MULTICLASS_LABEL_MAP["fireworks"]]
    nt_scores = probs[:, MULTICLASS_LABEL_MAP["non_target"]]

    max_exp = float(np.max(exp_scores))
    max_fw = float(np.max(fw_scores))
    mean_nt = float(np.mean(nt_scores))

    exp_run = _max_consecutive_run(exp_scores.tolist(), explosion_threshold)
    fw_run = _max_consecutive_run(fw_scores.tolist(), fireworks_threshold)

    exp_triggered = exp_run >= consecutive_windows
    fw_triggered = fw_run >= consecutive_windows

    alerts: List[str] = []
    if exp_triggered:
        alerts.append("EXPLOSION_DETECTED")
    if fw_triggered:
        alerts.append("FIREWORKS_DETECTED")

    if exp_triggered and fw_triggered:
        primary_alert: Optional[str] = (
            "EXPLOSION_DETECTED" if max_exp >= max_fw else "FIREWORKS_DETECTED"
        )
    elif exp_triggered:
        primary_alert = "EXPLOSION_DETECTED"
    elif fw_triggered:
        primary_alert = "FIREWORKS_DETECTED"
    else:
        primary_alert = None

    mean_probs = np.mean(probs, axis=0)
    argmax_id = int(np.argmax(mean_probs))
    return {
        "predicted_class_id": argmax_id,
        "predicted_class": MULTICLASS_INV_LABEL_MAP[argmax_id],
        "explosion_triggered": exp_triggered,
        "fireworks_triggered": fw_triggered,
        "explosion_alert": "EXPLOSION_DETECTED" if exp_triggered else None,
        "fireworks_alert": "FIREWORKS_DETECTED" if fw_triggered else None,
        "alerts": alerts,
        "primary_alert": primary_alert,
        "max_explosion_score": round(max_exp, 6),
        "max_fireworks_score": round(max_fw, 6),
        "mean_non_target_score": round(mean_nt, 6),
        "explosion_max_consecutive_run": exp_run,
        "fireworks_max_consecutive_run": fw_run,
    }


def compute_multiclass_class_weights(
    class_counts: Dict[str, int],
    num_classes: int = MULTICLASS_NUM_CLASSES,
) -> np.ndarray:
    """Compute normalized inverse-frequency class weights ordered ``[0..num_classes-1]``."""
    order = MULTICLASS_CLASS_NAMES if num_classes == MULTICLASS_NUM_CLASSES else tuple(class_counts.keys())
    counts = np.array([max(int(class_counts.get(name, 0)), 1) for name in order[:num_classes]], dtype=np.float32)
    raw = float(np.sum(counts)) / (float(num_classes) * counts)
    return (raw / float(np.mean(raw))).astype(np.float32)


def compute_recording_sampling_weights(
    samples: Sequence[Dict[str, object]],
    balance_classes: bool = True,
) -> np.ndarray:
    """Compute per-window sampling weights inversely proportional to recording window count (and class recordings)."""
    if not samples:
        return np.empty((0,), dtype=np.float64)

    rec_window_counts: Dict[str, int] = {}
    class_recordings: Dict[int, set] = {}
    for s in samples:
        rec_id = str(s["original_recording_id"])
        lbl = int(s["label"])
        rec_window_counts[rec_id] = rec_window_counts.get(rec_id, 0) + 1
        class_recordings.setdefault(lbl, set()).add(rec_id)

    weights = np.empty(len(samples), dtype=np.float64)
    for idx, s in enumerate(samples):
        rec_id = str(s["original_recording_id"])
        lbl = int(s["label"])
        w = 1.0 / float(rec_window_counts[rec_id])
        if balance_classes:
            w /= float(len(class_recordings[lbl]))
        weights[idx] = w

    mean_w = float(np.mean(weights))
    if mean_w > 0.0:
        weights /= mean_w
    return weights

