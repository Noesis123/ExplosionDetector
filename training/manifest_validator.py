"""Pure-Python dataset manifest validation and loading utilities.

Validates manifest schema, labels, splits, group isolation, duplicate entries,
expected counts, and resolves recording paths through ``audio.path_utils``
without requiring PyTorch or Numba at import time.
"""

from __future__ import annotations

import csv
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from audio.path_utils import (
    DEFAULT_LEGACY_DATA_PREFIXES,
    resolve_dataset_path,
)

VALID_LABELS = frozenset(("explosion", "non_explosion"))
VALID_SPLITS = frozenset(("train", "val", "test"))
PHASE13_VALID_SPLITS = frozenset(("train", "val", "test", "holdout", "challenge"))
BASE_REQUIRED_COLUMNS: Tuple[str, ...] = ("file_path", "label", "split")

MULTICLASS_LABEL_MAP: Dict[str, int] = {
    "non_target": 0,
    "explosion": 1,
    "fireworks": 2,
}
MULTICLASS_VALID_LABELS = frozenset(MULTICLASS_LABEL_MAP.keys())
MULTICLASS_CSV_ALERT_MAP: Dict[str, str] = {
    "non_target": "None",
    "explosion": "EXPLOSION_DETECTED",
    "fireworks": "FIREWORKS_DETECTED",
}
MULTICLASS_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "file_path",
    "label",
    "label_id",
    "alert",
    "category",
    "group_id",
    "split",
)
PHASE13_REQUIRED_COLUMNS: Tuple[str, ...] = MULTICLASS_REQUIRED_COLUMNS + (
    "exposure_status",
    "source_dataset",
    "source_url",
    "license",
)
PHASE14_REQUIRED_COLUMNS: Tuple[str, ...] = PHASE13_REQUIRED_COLUMNS + (
    "auditory_verification",
    "quality_confidence",
)
PHASE15_REQUIRED_COLUMNS: Tuple[str, ...] = PHASE14_REQUIRED_COLUMNS + (
    "training_approved",
    "review_status",
)
PHASE14_VALID_AUDITORY_STATUSES = frozenset(("legacy_curated", "pending", "human_verified"))
PHASE14_REJECTED_CONFIDENCE_VALUES = frozenset(("excluded", "low_uncertain", "ambiguous", "rejected"))
PHASE15_VALID_APPROVAL_FLAGS = frozenset(("true", "false"))
PHASE15_VALID_REVIEW_STATUSES = frozenset(("verified", "manual_review_required"))


@dataclass(frozen=True)
class ManifestValidationResult:
    """Validated manifest summary and resolved split rows."""

    manifest_path: Path
    all_rows: Tuple[Dict[str, str], ...]
    split_rows: Tuple[Dict[str, str], ...]
    resolved_paths: Tuple[Path, ...]
    label_counts: Dict[str, int]
    split_counts: Dict[str, int]
    group_count: Optional[int]
    review_flags: Tuple[str, ...]


def validate_and_load_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    require_group_id: bool = False,
    require_category: bool = False,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = False,
) -> ManifestValidationResult:
    """Validate manifest integrity and return resolved records for ``split`` (or all rows).

    Raises explicit ValueError / FileNotFoundError / UnsafeDatasetPathError /
    AmbiguousDatasetPathError instead of silently skipping malformed or missing rows.
    """
    mpath = Path(manifest_path)
    if not mpath.exists() or not mpath.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {mpath}")

    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    with mpath.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Manifest '{mpath}' has no CSV header columns.")

    missing_cols = [c for c in BASE_REQUIRED_COLUMNS if c not in fieldnames]
    if require_group_id and "group_id" not in fieldnames:
        missing_cols.append("group_id")
    if require_category and "category" not in fieldnames:
        missing_cols.append("category")
    if missing_cols:
        raise ValueError(f"Manifest '{mpath}' is missing required columns: {missing_cols}")

    if not rows:
        raise ValueError(f"Manifest '{mpath}' contains zero recording rows.")

    has_group_col = "group_id" in fieldnames
    has_cat_col = "category" in fieldnames
    has_rec_id_col = "original_recording_id" in fieldnames
    has_status_col = "validation_status" in fieldnames

    seen_raw_paths = set()
    seen_rec_ids = set()
    group_splits: Dict[str, set] = defaultdict(set)
    group_labels: Dict[str, set] = defaultdict(set)
    label_counts: Counter = Counter()
    split_counts: Counter = Counter()
    review_flags: List[str] = []

    for row_idx, row in enumerate(rows, start=2):
        raw_fp = str(row.get("file_path") or "").strip()
        if not raw_fp:
            raise ValueError(f"Row {row_idx} in '{mpath}' has empty file_path.")
        if raw_fp in seen_raw_paths:
            raise ValueError(f"Duplicate file_path entry at row {row_idx} in '{mpath}': {raw_fp!r}")
        seen_raw_paths.add(raw_fp)

        label = str(row.get("label") or "").strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label {row.get('label')!r} at row {row_idx} in '{mpath}'.")
        row["label"] = label
        label_counts[label] += 1

        row_split = str(row.get("split") or "").strip().lower()
        if row_split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {row.get('split')!r} at row {row_idx} in '{mpath}'.")
        row["split"] = row_split
        split_counts[row_split] += 1

        if has_status_col:
            status = str(row.get("validation_status") or "").strip().lower()
            if status and status != "valid":
                raise ValueError(
                    f"Non-valid recording status {status!r} at row {row_idx} in '{mpath}' ({raw_fp})."
                )

        if has_rec_id_col:
            rec_id = str(row.get("original_recording_id") or "").strip()
            if not rec_id:
                raise ValueError(f"Row {row_idx} in '{mpath}' has empty original_recording_id.")
            if rec_id in seen_rec_ids:
                raise ValueError(
                    f"Duplicate original_recording_id {rec_id!r} at row {row_idx} in '{mpath}'."
                )
            seen_rec_ids.add(rec_id)

        if has_cat_col:
            cat = str(row.get("category") or "").strip()
            if not cat:
                raise ValueError(f"Row {row_idx} in '{mpath}' has empty category.")
            if Path(raw_fp.replace("\\", "/")).name == "bbc_industry--_07049006.mp3" and cat == "gunfire":
                flag = (
                    "Manual review required: 'bbc_industry--_07049006.mp3' retains category='gunfire' "
                    "in manifest despite Phase 10 assembly-shop machinery evidence."
                )
                review_flags.append(flag)
                warnings.warn(flag, UserWarning, stacklevel=2)

        if has_group_col:
            gid = str(row.get("group_id") or "").strip()
            if not gid:
                raise ValueError(f"Row {row_idx} in '{mpath}' has empty group_id.")
            group_splits[gid].add(row_split)
            group_labels[gid].add(label)

    return _finalize_manifest_validation(
        mpath=mpath,
        rows=rows,
        split=split,
        has_group_col=has_group_col,
        group_splits=group_splits,
        group_labels=group_labels,
        label_counts=label_counts,
        split_counts=split_counts,
        review_flags=review_flags,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
    )


def _finalize_manifest_validation(
    *,
    mpath: Path,
    rows: List[Dict[str, str]],
    split: Optional[str],
    has_group_col: bool,
    group_splits: Dict[str, set],
    group_labels: Dict[str, set],
    label_counts: Counter,
    split_counts: Counter,
    review_flags: List[str],
    expected_total_count: Optional[int],
    expected_split_counts: Optional[Mapping[str, int]],
    resolve_all_paths: bool,
    data_root: Optional[Union[str, Path]],
    legacy_prefixes: Sequence[str],
) -> ManifestValidationResult:
    """Enforce group isolation, count expectations, and physical path resolution."""
    for gid, splits_for_group in group_splits.items():
        if len(splits_for_group) > 1:
            raise ValueError(
                f"Session group leakage in '{mpath}': group '{gid}' crosses splits {sorted(splits_for_group)}."
            )
    for gid, labels_for_group in group_labels.items():
        if len(labels_for_group) > 1:
            raise ValueError(
                f"Mixed class labels in group '{gid}' in '{mpath}': {sorted(labels_for_group)}."
            )

    if expected_total_count is not None and len(rows) != expected_total_count:
        raise ValueError(
            f"Expected {expected_total_count} manifest rows in '{mpath}', found {len(rows)}."
        )

    if expected_split_counts is not None:
        for sp_name, exp_cnt in expected_split_counts.items():
            actual_cnt = split_counts.get(sp_name, 0)
            if actual_cnt != exp_cnt:
                raise ValueError(
                    f"Expected {exp_cnt} rows for split '{sp_name}' in '{mpath}', found {actual_cnt}."
                )

    selected_rows = [r for r in rows if (split is None or r["split"] == split)]
    if not selected_rows:
        raise ValueError(f"No records found for split '{split}' in '{mpath}'.")

    rows_to_resolve = rows if resolve_all_paths else selected_rows
    resolved_all: List[Path] = []
    seen_resolved = set()
    for row in rows_to_resolve:
        resolved_p = resolve_dataset_path(
            row["file_path"], data_root=data_root, legacy_prefixes=legacy_prefixes
        )
        if resolved_p in seen_resolved:
            raise ValueError(
                f"Multiple manifest entries in '{mpath}' resolve to the same physical file: '{resolved_p}'."
            )
        seen_resolved.add(resolved_p)
        resolved_all.append(resolved_p)

    if resolve_all_paths and split is not None:
        resolved_selected = tuple(
            p for r, p in zip(rows, resolved_all) if r["split"] == split
        )
    else:
        resolved_selected = tuple(resolved_all)

    selected_with_resolved: List[Dict[str, str]] = []
    for row, res_p in zip(selected_rows, resolved_selected):
        enriched = dict(row)
        enriched["resolved_file_path"] = str(res_p)
        selected_with_resolved.append(enriched)

    return ManifestValidationResult(
        manifest_path=mpath.resolve(),
        all_rows=tuple(rows),
        split_rows=tuple(selected_with_resolved),
        resolved_paths=resolved_selected,
        label_counts=dict(label_counts),
        split_counts=dict(split_counts),
        group_count=len(group_splits) if has_group_col else None,
        review_flags=tuple(review_flags),
    )


def validate_and_load_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
    allowed_splits: Iterable[str] = VALID_SPLITS,
    required_columns: Tuple[str, ...] = MULTICLASS_REQUIRED_COLUMNS,
) -> ManifestValidationResult:
    """Validate the Phase 12/13 three-class manifest (``non_target``, ``explosion``, ``fireworks``)."""
    valid_splits = frozenset(str(s).strip().lower() for s in allowed_splits)
    mpath = Path(manifest_path)
    if not mpath.exists() or not mpath.is_file():
        raise FileNotFoundError(f"Multiclass manifest not found: {mpath}")
    if split is not None and split not in valid_splits:
        raise ValueError(f"split must be one of {sorted(valid_splits)}, got {split!r}")

    with mpath.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Manifest '{mpath}' has no CSV header columns.")
    missing_cols = [c for c in required_columns if c not in fieldnames]
    if missing_cols:
        raise ValueError(f"Multiclass manifest '{mpath}' is missing required columns: {missing_cols}")
    if not rows:
        raise ValueError(f"Multiclass manifest '{mpath}' contains zero recording rows.")

    has_rec_id_col = "original_recording_id" in fieldnames
    has_status_col = "validation_status" in fieldnames
    seen_raw_paths = set()
    seen_rec_ids = set()
    group_splits: Dict[str, set] = defaultdict(set)
    group_labels: Dict[str, set] = defaultdict(set)
    label_counts: Counter = Counter()
    split_counts: Counter = Counter()
    review_flags: List[str] = []

    for row_idx, row in enumerate(rows, start=2):
        raw_fp = str(row.get("file_path") or "").strip()
        if not raw_fp:
            raise ValueError(f"Row {row_idx} in '{mpath}' has empty file_path.")
        if raw_fp in seen_raw_paths:
            raise ValueError(f"Duplicate file_path entry at row {row_idx} in '{mpath}': {raw_fp!r}")
        seen_raw_paths.add(raw_fp)

        label = str(row.get("label") or "").strip().lower()
        if label not in MULTICLASS_VALID_LABELS:
            raise ValueError(
                f"Invalid multiclass label {row.get('label')!r} at row {row_idx} in '{mpath}'."
            )
        row["label"] = label
        label_counts[label] += 1

        expected_id = str(MULTICLASS_LABEL_MAP[label])
        actual_id = str(row.get("label_id") or "").strip()
        if actual_id != expected_id:
            raise ValueError(
                f"Mismatched label_id {actual_id!r} for label {label!r} (expected {expected_id!r}) "
                f"at row {row_idx} in '{mpath}'."
            )

        expected_alert = MULTICLASS_CSV_ALERT_MAP[label]
        actual_alert = str(row.get("alert") or "").strip()
        if actual_alert != expected_alert:
            raise ValueError(
                f"Mismatched alert {actual_alert!r} for label {label!r} (expected {expected_alert!r}) "
                f"at row {row_idx} in '{mpath}'."
            )

        row_split = str(row.get("split") or "").strip().lower()
        if row_split not in valid_splits:
            raise ValueError(f"Invalid split {row.get('split')!r} at row {row_idx} in '{mpath}'.")
        row["split"] = row_split
        split_counts[row_split] += 1

        if has_status_col:
            status = str(row.get("validation_status") or "").strip().lower()
            if status != "valid":
                raise ValueError(
                    f"Non-valid recording status {status!r} at row {row_idx} in '{mpath}' ({raw_fp})."
                )

        if has_rec_id_col:
            rec_id = str(row.get("original_recording_id") or "").strip()
            if not rec_id:
                raise ValueError(f"Row {row_idx} in '{mpath}' has empty original_recording_id.")
            if rec_id in seen_rec_ids:
                raise ValueError(
                    f"Duplicate original_recording_id {rec_id!r} at row {row_idx} in '{mpath}'."
                )
            seen_rec_ids.add(rec_id)

        cat = str(row.get("category") or "").strip()
        if not cat:
            raise ValueError(f"Row {row_idx} in '{mpath}' has empty category.")
        if Path(raw_fp.replace("\\", "/")).name == "bbc_industry--_07049006.mp3" and cat == "gunfire":
            flag = (
                "Manual review required: 'bbc_industry--_07049006.mp3' retains category='gunfire' "
                "in manifest despite Phase 10 assembly-shop machinery evidence."
            )
            review_flags.append(flag)
            warnings.warn(flag, UserWarning, stacklevel=2)

        gid = str(row.get("group_id") or "").strip()
        if not gid:
            raise ValueError(f"Row {row_idx} in '{mpath}' has empty group_id.")
        group_splits[gid].add(row_split)
        group_labels[gid].add(label)

    return _finalize_manifest_validation(
        mpath=mpath,
        rows=rows,
        split=split,
        has_group_col=True,
        group_splits=group_splits,
        group_labels=group_labels,
        label_counts=label_counts,
        split_counts=split_counts,
        review_flags=review_flags,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
    )


def _validate_exposure_and_provenance(rows: Sequence[Mapping[str, str]]) -> None:
    """Validate provenance and test/holdout isolation fields shared by Phase 13, 14, 15, and 19 manifests."""
    valid_exposure_tags = frozenset((
        "legacy_phase11_v2_train_val",
        "prev_phase11_test_exposed",
        "unexposed_phase13_train_val",
        "unexposed_phase15_train_val",
        "unexposed_phase19_train_val",
        "unexposed_phase19_challenge",
        "unexposed_holdout",
    ))
    for row in rows:
        exp_status = str(row.get("exposure_status") or "").strip()
        if exp_status not in valid_exposure_tags:
            raise ValueError(
                f"Invalid exposure_status {exp_status!r} for {row.get('file_path')!r}."
            )
        if row["split"] == "test" and exp_status != "prev_phase11_test_exposed":
            raise ValueError(
                f"Legacy test split row must carry exposure_status='prev_phase11_test_exposed', got {exp_status!r}."
            )
        if row["split"] == "holdout" and exp_status != "unexposed_holdout":
            raise ValueError(
                f"Phase 13/14 holdout split row must carry exposure_status='unexposed_holdout', got {exp_status!r}."
            )
        if exp_status == "unexposed_holdout" and row["split"] != "holdout":
            raise ValueError(
                f"Row with exposure_status='unexposed_holdout' must belong to split='holdout', got {row['split']!r}."
            )
        if not str(row.get("source_dataset") or "").strip():
            raise ValueError(f"Empty source_dataset for {row.get('file_path')!r}.")
        if not str(row.get("source_url") or "").strip():
            raise ValueError(f"Empty source_url for {row.get('file_path')!r}.")
        if not str(row.get("license") or "").strip():
            raise ValueError(f"Empty license for {row.get('file_path')!r}.")


def validate_and_load_phase13_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
) -> ManifestValidationResult:
    """Validate a Phase 13 expanded multiclass manifest with exposed legacy test vs unexposed holdout separation."""
    result = validate_and_load_multiclass_manifest(
        manifest_path=manifest_path,
        split=split,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        allowed_splits=PHASE13_VALID_SPLITS,
        required_columns=PHASE13_REQUIRED_COLUMNS,
    )
    _validate_exposure_and_provenance(result.all_rows)
    return result


def validate_and_load_phase14_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
) -> ManifestValidationResult:
    """Validate a Phase 14 verified multiclass manifest excluding uncertain recordings and tracking auditory status."""
    result = validate_and_load_multiclass_manifest(
        manifest_path=manifest_path,
        split=split,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        allowed_splits=PHASE13_VALID_SPLITS,
        required_columns=PHASE14_REQUIRED_COLUMNS,
    )
    _validate_exposure_and_provenance(result.all_rows)

    for row in result.all_rows:
        aud_status = str(row.get("auditory_verification") or "").strip().lower()
        if aud_status not in PHASE14_VALID_AUDITORY_STATUSES:
            raise ValueError(
                f"Invalid Phase 14 auditory_verification {row.get('auditory_verification')!r} for {row.get('file_path')!r}."
            )
        q_conf = str(row.get("quality_confidence") or "").strip().lower()
        if not q_conf or q_conf in PHASE14_REJECTED_CONFIDENCE_VALUES:
            raise ValueError(
                f"Uncertain/excluded quality_confidence {row.get('quality_confidence')!r} cannot appear in Phase 14 manifest: {row.get('file_path')!r}."
            )
    return result


def _validate_auditory_and_approval_fields(rows: Sequence[Mapping[str, str]]) -> None:
    """Validate Phase 15 auditory verification, confidence, review_status, and training_approved alignment."""
    for row in rows:
        aud_status = str(row.get("auditory_verification") or "").strip().lower()
        if aud_status not in PHASE14_VALID_AUDITORY_STATUSES:
            raise ValueError(
                f"Invalid Phase 15 auditory_verification {row.get('auditory_verification')!r} for {row.get('file_path')!r}."
            )
        q_conf = str(row.get("quality_confidence") or "").strip().lower()
        if not q_conf or q_conf in PHASE14_REJECTED_CONFIDENCE_VALUES:
            raise ValueError(
                f"Uncertain/excluded quality_confidence {row.get('quality_confidence')!r} cannot appear in Phase 15 manifest: {row.get('file_path')!r}."
            )
        rev_status = str(row.get("review_status") or "").strip().lower()
        if rev_status not in PHASE15_VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid Phase 15 review_status {row.get('review_status')!r} for {row.get('file_path')!r}."
            )
        t_app = str(row.get("training_approved") or "").strip().lower()
        if t_app not in PHASE15_VALID_APPROVAL_FLAGS:
            raise ValueError(
                f"Invalid Phase 15 training_approved flag {row.get('training_approved')!r} for {row.get('file_path')!r}."
            )
        if aud_status == "pending" and t_app == "true":
            raise ValueError(
                f"Unverified recording (auditory_verification='pending') cannot have training_approved='true': {row.get('file_path')!r}."
            )
        if t_app == "true" and aud_status not in ("legacy_curated", "human_verified"):
            raise ValueError(
                f"training_approved='true' requires legacy_curated or human_verified status: {row.get('file_path')!r}."
            )


def validate_and_load_phase15_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
    approved_only: bool = False,
) -> ManifestValidationResult:
    """Validate a Phase 15 multiclass manifest, optionally filtering to training_approved='true' rows."""
    result = validate_and_load_multiclass_manifest(
        manifest_path=manifest_path,
        split=None if approved_only else split,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        allowed_splits=PHASE13_VALID_SPLITS,
        required_columns=PHASE15_REQUIRED_COLUMNS,
    )
    _validate_exposure_and_provenance(result.all_rows)
    _validate_auditory_and_approval_fields(result.all_rows)

    if not approved_only:
        return result

    approved_all = tuple(
        r for r in result.all_rows if str(r.get("training_approved") or "").strip().lower() == "true"
    )
    if not approved_all:
        raise ValueError(f"No training_approved='true' recordings found in '{result.manifest_path}'.")

    selected_rows = [r for r in approved_all if (split is None or r["split"] == split)]
    if not selected_rows:
        raise ValueError(
            f"No training_approved='true' recordings found for split '{split}' in '{result.manifest_path}'."
        )

    rows_to_resolve = approved_all if resolve_all_paths else selected_rows
    resolved_all = [
        resolve_dataset_path(r["file_path"], data_root=data_root, legacy_prefixes=legacy_prefixes)
        for r in rows_to_resolve
    ]
    if resolve_all_paths and split is not None:
        resolved_selected = tuple(p for r, p in zip(approved_all, resolved_all) if r["split"] == split)
    else:
        resolved_selected = tuple(resolved_all)

    enriched_selected = []
    for r, res_p in zip(selected_rows, resolved_selected):
        enriched = dict(r)
        enriched["resolved_file_path"] = str(res_p)
        enriched_selected.append(enriched)

    return ManifestValidationResult(
        manifest_path=result.manifest_path,
        all_rows=approved_all,
        split_rows=tuple(enriched_selected),
        resolved_paths=resolved_selected,
        label_counts=dict(Counter(r["label"] for r in approved_all)),
        split_counts=dict(Counter(r["split"] for r in approved_all)),
        group_count=len({r["group_id"] for r in approved_all}),
        review_flags=result.review_flags,
    )


def build_phase12_multiclass_manifest(
    source_manifest_path: Union[str, Path],
    output_manifest_path: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    expected_total_count: Optional[int] = 193,
    expected_split_counts: Optional[Mapping[str, int]] = None,
) -> ManifestValidationResult:
    """Create a verified Phase 12 three-class manifest logically from a Phase 11 binary manifest."""
    if expected_split_counts is None and expected_total_count == 193:
        expected_split_counts = {"train": 153, "val": 20, "test": 20}

    phase11 = validate_and_load_manifest(
        source_manifest_path,
        data_root=data_root,
        require_group_id=True,
        require_category=True,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=True,
    )

    from audio.path_utils import DEFAULT_DATA_ROOT

    resolved_data_root = Path(data_root).resolve() if data_root is not None else DEFAULT_DATA_ROOT.resolve()
    workspace_root = resolved_data_root.parent

    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "label",
        "label_id",
        "alert",
        "binary_label",
        "category",
        "group_id",
        "source_dataset",
        "original_recording_id",
        "duration_seconds",
        "sample_rate",
        "channels",
        "split",
        "validation_status",
        "notes",
    ]

    out_rows: List[Dict[str, str]] = []
    for row in phase11.split_rows:
        resolved_p = Path(row["resolved_file_path"]).resolve()
        portable_path = resolved_p.relative_to(workspace_root).as_posix()

        b_lbl = row["label"]
        cat = row["category"].strip().lower()
        if b_lbl == "explosion":
            if cat == "fireworks":
                raise ValueError(f"Contradictory Phase 11 entry: {portable_path}")
            mc_label = "explosion"
        elif b_lbl == "non_explosion" and cat == "fireworks":
            mc_label = "fireworks"
        else:
            mc_label = "non_target"

        mc_id = str(MULTICLASS_LABEL_MAP[mc_label])
        mc_alert = MULTICLASS_CSV_ALERT_MAP[mc_label]
        orig_note = str(row.get("notes") or "").strip()
        mc_note = (
            f"Phase 12 multiclass: class={mc_label}, label_id={mc_id}, alert={mc_alert}"
            + (f" | {orig_note}" if orig_note else "")
        )

        out_rows.append({
            "file_path": portable_path,
            "label": mc_label,
            "label_id": mc_id,
            "alert": mc_alert,
            "binary_label": b_lbl,
            "category": row["category"],
            "group_id": row["group_id"],
            "source_dataset": row.get("source_dataset", ""),
            "original_recording_id": row.get("original_recording_id", ""),
            "duration_seconds": row.get("duration_seconds", ""),
            "sample_rate": row.get("sample_rate", ""),
            "channels": row.get("channels", ""),
            "split": row["split"],
            "validation_status": row.get("validation_status", "valid"),
            "notes": mc_note,
        })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return validate_and_load_multiclass_manifest(
        out_path,
        data_root=resolved_data_root,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=True,
    )


def validate_and_load_phase19_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
    approved_only: bool = False,
) -> ManifestValidationResult:
    """Validate a Phase 19 multiclass manifest, optionally filtering to training_approved='true' rows."""
    return validate_and_load_phase15_multiclass_manifest(
        manifest_path=manifest_path,
        split=split,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        approved_only=approved_only,
    )


def validate_and_load_phase20_multiclass_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
    expected_total_count: Optional[int] = None,
    expected_split_counts: Optional[Mapping[str, int]] = None,
    resolve_all_paths: bool = True,
    approved_only: bool = False,
) -> ManifestValidationResult:
    """Validate a Phase 20 multiclass manifest, optionally filtering to training_approved='true' rows."""
    return validate_and_load_phase15_multiclass_manifest(
        manifest_path=manifest_path,
        split=split,
        data_root=data_root,
        legacy_prefixes=legacy_prefixes,
        expected_total_count=expected_total_count,
        expected_split_counts=expected_split_counts,
        resolve_all_paths=resolve_all_paths,
        approved_only=approved_only,
    )


