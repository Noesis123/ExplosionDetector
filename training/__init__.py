"""Training and dataset package for Acoustic Explosion Detection."""

from .manifest_validator import ManifestValidationResult, validate_and_load_manifest

__all__ = [
    "ExplosionDataset",
    "ManifestValidationResult",
    "validate_and_load_manifest",
]


def __getattr__(name: str):
    if name == "ExplosionDataset":
        from .dataset import ExplosionDataset
        return ExplosionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
