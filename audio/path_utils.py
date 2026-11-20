"""Portable and secure dataset path resolution utilities.

Resolves recording paths recorded under legacy absolute Windows workspaces or
relative paths cleanly against the active workspace's ``data/`` directory,
without hardcoding drive letters or modifying historical manifests.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence, Tuple, Union

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data"

# Historical Windows workspace prefix recorded in Phase 5-11 manifests.
DEFAULT_LEGACY_DATA_PREFIXES: Tuple[str, ...] = (
    "c:/users/user/desktop/cognilearn/explosion/data/",
)

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:/")
_WINDOWS_DRIVE_REL_RE = re.compile(r"^[A-Za-z]:[^/]")


class UnsafeDatasetPathError(ValueError):
    """Raised when a path escapes the approved dataset root or uses an unsupported external location."""


class AmbiguousDatasetPathError(ValueError):
    """Raised when a relative or filename-only path matches multiple dataset files."""


def _normalize_raw_path(raw_path: Union[str, Path]) -> str:
    """Validate basic string properties and normalize separators to POSIX slashes."""
    if raw_path is None:
        raise UnsafeDatasetPathError("Dataset path cannot be None.")
    text = str(raw_path).strip()
    if not text:
        raise UnsafeDatasetPathError("Dataset path cannot be empty.")
    if "\x00" in text:
        raise UnsafeDatasetPathError(f"Dataset path contains null bytes: {raw_path!r}")

    norm = text.replace("\\", "/")
    if norm.startswith("//"):
        raise UnsafeDatasetPathError(f"UNC network paths are not permitted: {raw_path!r}")

    while "//" in norm:
        norm = norm.replace("//", "/")

    if _WINDOWS_DRIVE_REL_RE.match(norm):
        raise UnsafeDatasetPathError(
            f"Drive-relative Windows paths are not permitted: {raw_path!r}"
        )

    parts = norm.split("/")
    if ".." in parts:
        raise UnsafeDatasetPathError(
            f"Path traversal ('..') is not permitted outside dataset root: {raw_path!r}"
        )
    return norm


def _is_absolute_normalized(norm: str) -> bool:
    """Return True if normalized path is absolute (POSIX root or Windows drive root)."""
    return norm.startswith("/") or bool(_WINDOWS_ABS_RE.match(norm))


def _normalize_prefixes(
    approved_root: Path,
    legacy_prefixes: Sequence[str],
    raw_root: Optional[Path] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return active and legacy trailing-slash prefixes approved for ``approved_root``."""
    active = [approved_root.as_posix().rstrip("/") + "/"]
    if raw_root is not None:
        raw_abs = Path(os.path.abspath(raw_root)).as_posix().rstrip("/") + "/"
        if raw_abs not in active:
            active.append(raw_abs)
    legacy = tuple(
        _normalize_raw_path(prefix).rstrip("/") + "/"
        for prefix in legacy_prefixes
    )
    return tuple(active), legacy


def _strip_approved_prefix(
    norm_abs: str,
    active_prefixes: Tuple[str, ...],
    legacy_prefixes: Tuple[str, ...],
) -> Optional[str]:
    """Strip an approved root/legacy prefix from an absolute path if matched cleanly."""
    case_insensitive = os.name == "nt"
    norm_active = norm_abs.lower() if case_insensitive else norm_abs
    for prefix in active_prefixes:
        pref_cmp = prefix.lower() if case_insensitive else prefix
        if norm_active.startswith(pref_cmp):
            return norm_abs[len(prefix):].lstrip("/")

    # Historical Windows C:\... legacy prefixes always match case-insensitively.
    norm_lower = norm_abs.lower()
    for prefix in legacy_prefixes:
        if norm_lower.startswith(prefix.lower()):
            return norm_abs[len(prefix):].lstrip("/")
    return None


def _ensure_within_root(candidate: Path, approved_root: Path, raw_path: Union[str, Path]) -> Path:
    """Resolve symlinks and verify target exists as a file inside ``approved_root``."""
    resolved = candidate.resolve()
    if not resolved.is_relative_to(approved_root):
        raise UnsafeDatasetPathError(
            f"Resolved path escapes approved dataset root '{approved_root}': {raw_path!r} -> '{resolved}'"
        )
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset audio file not found at '{resolved}' (from {raw_path!r})"
        )
    return resolved


def _resolve_bare_filename(filename: str, approved_root: Path, raw_path: Union[str, Path]) -> Path:
    """Resolve an unambiguous bare filename inside ``approved_root``; fail on ambiguity or absence."""
    target_cmp = filename.lower() if os.name == "nt" else filename
    matches = []
    seen = set()
    for candidate in approved_root.rglob("*"):
        if not candidate.is_file():
            continue
        cand_name = candidate.name.lower() if os.name == "nt" else candidate.name
        if cand_name == target_cmp:
            resolved = candidate.resolve()
            if resolved not in seen and resolved.is_relative_to(approved_root):
                seen.add(resolved)
                matches.append(resolved)

    if len(matches) > 1:
        rel_matches = [str(m.relative_to(approved_root)) for m in sorted(matches)]
        raise AmbiguousDatasetPathError(
            f"Ambiguous dataset filename {raw_path!r} matches {len(matches)} files: {rel_matches}"
        )
    if not matches:
        raise FileNotFoundError(
            f"Dataset audio file {raw_path!r} does not exist under '{approved_root}'"
        )
    return matches[0]


def resolve_dataset_path(
    raw_path: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
) -> Path:
    """Resolve a manifest or user-provided dataset recording path safely under ``data_root``.

    Supports:
    - Absolute paths under the active ``data_root`` (any drive/mount point).
    - Historical Windows manifest paths matching ``legacy_prefixes`` mapped onto ``data_root``.
    - Relative paths inside ``data_root`` (e.g., ``positives/clip.mp3``) or ``data/``-prefixed
      project-relative paths (e.g., ``data/positives/clip.mp3``).
    - Unambiguous bare filenames if unique under ``data_root``.

    Rejects path traversal (``..``), external/unsupported absolute paths, ambiguous matches,
    and missing files by raising explicit exceptions rather than silently skipping.
    """
    raw_root_path = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    approved_root = raw_root_path.resolve()
    if not approved_root.exists() or not approved_root.is_dir():
        raise FileNotFoundError(f"Approved dataset root directory does not exist: '{approved_root}'")

    norm = _normalize_raw_path(raw_path)

    if _is_absolute_normalized(norm):
        active_prefixes, legacy_norms = _normalize_prefixes(
            approved_root, legacy_prefixes, raw_root=raw_root_path
        )
        rel_subpath = _strip_approved_prefix(norm, active_prefixes, legacy_norms)
        if rel_subpath is None:
            raise UnsafeDatasetPathError(
                f"External or incorrect dataset root in path {raw_path!r}; "
                f"expected path under '{approved_root}'."
            )
        if not rel_subpath:
            raise UnsafeDatasetPathError(f"Path points to dataset root directory, not a file: {raw_path!r}")
        return _ensure_within_root(approved_root / rel_subpath, approved_root, raw_path)

    # Clean leading './' segments for relative paths.
    rel = norm
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    if not rel or rel == ".":
        raise UnsafeDatasetPathError(f"Relative path must point to an audio file: {raw_path!r}")

    parts = PurePosixPath(rel).parts
    if len(parts) == 1:
        return _resolve_bare_filename(parts[0], approved_root, raw_path)

    candidates = [approved_root / rel]
    if parts[0].lower() == "data" and len(parts) > 1:
        candidates.append(approved_root / "/".join(parts[1:]))

    existing = []
    seen = set()
    for cand in candidates:
        res = cand.resolve()
        if not res.is_relative_to(approved_root):
            raise UnsafeDatasetPathError(
                f"Resolved path escapes approved dataset root '{approved_root}': {raw_path!r}"
            )
        if res.exists() and res.is_file() and res not in seen:
            seen.add(res)
            existing.append(res)

    if len(existing) > 1:
        rel_matches = [str(m.relative_to(approved_root)) for m in sorted(existing)]
        raise AmbiguousDatasetPathError(
            f"Ambiguous dataset relative path {raw_path!r} matches multiple files: {rel_matches}"
        )
    if len(existing) == 1:
        return existing[0]

    raise FileNotFoundError(
        f"Dataset audio file not found at '{(approved_root / rel).resolve()}' (from {raw_path!r})"
    )


def resolve_manifest_entries(
    file_paths: Iterable[Union[str, Path]],
    data_root: Optional[Union[str, Path]] = None,
    legacy_prefixes: Sequence[str] = DEFAULT_LEGACY_DATA_PREFIXES,
) -> Tuple[Path, ...]:
    """Resolve every recording path in a manifest; raise on the first unresolved/unsafe path."""
    return tuple(
        resolve_dataset_path(fp, data_root=data_root, legacy_prefixes=legacy_prefixes)
        for fp in file_paths
    )
