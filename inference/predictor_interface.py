"""Predictor Interface and Deterministic Mock for Pipeline Testing.

Defines the contract for model predictors and provides an explicitly labeled
mock predictor for testing data flows and offline inference before model training.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class BasePredictor(ABC):
    """Abstract base class for audio classification model predictors."""

    @abstractmethod
    def predict_window(self, spec: np.ndarray) -> Dict[str, float]:
        """Predict probability scores for a single 2D Log-Mel spectrogram.
        
        Args:
            spec: 2D array of shape (n_mels, time_frames), e.g. (64, 44).
            
        Returns:
            Dictionary mapping class name to predicted probability in [0.0, 1.0].
            e.g. {"explosion": 0.85, "non_explosion": 0.15}
        """
        pass

    @abstractmethod
    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        """Predict probability scores for a batch of 2D Log-Mel spectrograms.
        
        Args:
            specs: 3D array of shape (num_windows, n_mels, time_frames).
            
        Returns:
            2D array of shape (num_windows, num_classes) with class probabilities.
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of supported output class names in index order."""
        pass

    @abstractmethod
    def is_mock(self) -> bool:
        """Return True if this predictor is a mock/test placeholder, False for real models."""
        pass

    @property
    def name(self) -> str:
        """Human-readable predictor identifier."""
        return self.__class__.__name__


class MockTestPredictor(BasePredictor):
    """Deterministic Mock Predictor for Pipeline Testing ONLY.
    
    WARNING:
    This class is strictly a software test fixture designed to validate
    batch pipelining, timestamp calculations, and CSV/JSON serialization.
    It DOES NOT DETECT EXPLOSIONS and must NEVER be used as a real classifier.
    """

    CLASS_NAMES = ["non_explosion", "explosion"]

    def __init__(self, mode: str = "deterministic_hash"):
        """
        Args:
            mode: 'deterministic_hash' (predicts via deterministic hash of input),
                  'constant_zero' (always non_explosion),
                  'constant_high' (always explosion),
                  or 'energy_contrast' (heuristic spectral energy ratio for testing).
        """
        self.mode = mode

    def is_mock(self) -> bool:
        return True

    def get_class_names(self) -> List[str]:
        return list(self.CLASS_NAMES)

    @property
    def name(self) -> str:
        return f"MockTestPredictor[TEST_FIXTURE_ONLY, mode={self.mode}]"

    def predict_window(self, spec: np.ndarray) -> Dict[str, float]:
        """Compute deterministic mock probabilities for a single window."""
        batch_input = spec[np.newaxis, ...] # (1, n_mels, time_frames)
        probs = self.predict_batch(batch_input)[0]
        return {
            "non_explosion": float(probs[0]),
            "explosion": float(probs[1])
        }

    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        """Compute deterministic mock scores for a batch of windows."""
        num_windows = len(specs)
        if num_windows == 0:
            return np.empty((0, 2), dtype=np.float32)

        probs = np.zeros((num_windows, 2), dtype=np.float32)

        if self.mode == "constant_zero":
            probs[:, 0] = 1.0 # non_explosion
            probs[:, 1] = 0.0 # explosion

        elif self.mode == "constant_high":
            probs[:, 0] = 0.1
            probs[:, 1] = 0.9

        elif self.mode == "energy_contrast":
            # Simple contrast between low-frequency and high-frequency Mel bins for mock variation
            for i in range(num_windows):
                s = specs[i]
                low_e = np.mean(s[:16, :])
                high_e = np.mean(s[32:, :])
                p_exp = 1.0 / (1.0 + np.exp(-(low_e - high_e)))
                p_exp = float(np.clip(p_exp, 0.01, 0.99))
                probs[i, 1] = p_exp
                probs[i, 0] = 1.0 - p_exp

        else: # deterministic_hash (default)
            # Deterministic pseudo-score derived from spectrogram statistics for reproducible unit tests
            for i in range(num_windows):
                s = specs[i]
                val = float(np.sin(np.mean(s) * 10.0 + np.std(s) * 5.0))
                # Map to [0.05, 0.95]
                p_exp = 0.5 + 0.45 * val
                p_exp = float(np.clip(p_exp, 0.01, 0.99))
                probs[i, 1] = p_exp
                probs[i, 0] = 1.0 - p_exp

        return probs


class PyTorchModelPredictor(BasePredictor):
    """Predictor wrapper for trained binary AudioBaselineCNN checkpoints.

    Loads checkpoints safely using ``weights_only=True``, validates parameter compatibility
    against ``AudioBaselineCNN``, runs evaluation without gradient tracking, and converts
    the single raw logit output into ``[P(non_explosion), P(explosion)]`` via sigmoid.
    """

    DEFAULT_CLASS_NAMES = ["non_explosion", "explosion"]

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        class_names: Optional[List[str]] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.class_names = list(class_names) if class_names is not None else list(self.DEFAULT_CLASS_NAMES)
        if self.class_names != self.DEFAULT_CLASS_NAMES:
            raise ValueError(
                f"Binary PyTorchModelPredictor requires class_names={self.DEFAULT_CLASS_NAMES}, "
                f"got {self.class_names}."
            )

        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained model checkpoint not found: {self.checkpoint_path}. "
                f"Provide a valid .pt checkpoint path before running neural inference."
            )

        # Lazy import of torch to keep module lightweight
        import torch
        self.torch = torch
        self.checkpoint_metadata: Dict[str, object] = {}
        self.model = self._load_model()
        self.model.eval()

    @staticmethod
    def _import_model_class():
        """Import AudioBaselineCNN whether explosion_detection is on sys.path or not."""
        try:
            from models.baseline_cnn import AudioBaselineCNN
            return AudioBaselineCNN
        except ImportError:
            from explosion_detection.models.baseline_cnn import AudioBaselineCNN
            return AudioBaselineCNN

    def _load_model(self):
        """Safely load checkpoint weights with ``weights_only=True`` into AudioBaselineCNN."""
        checkpoint = self.torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Unsupported checkpoint payload type {type(checkpoint).__name__} "
                f"in {self.checkpoint_path}; expected dict state/metadata container."
            )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.checkpoint_metadata = {
                k: v for k, v in checkpoint.items()
                if k not in ("model_state_dict", "optimizer_state_dict")
            }
        else:
            state_dict = checkpoint
            self.checkpoint_metadata = {}

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Empty or invalid model_state_dict in {self.checkpoint_path}.")

        model_cls = self._import_model_class()
        model = model_cls()

        expected_total, _ = model.count_parameters()
        ckpt_param_count = self.checkpoint_metadata.get("model_parameters")
        if ckpt_param_count is not None and int(ckpt_param_count) != expected_total:
            raise ValueError(
                f"Checkpoint parameter count mismatch in {self.checkpoint_path}: "
                f"checkpoint reports {ckpt_param_count}, AudioBaselineCNN expects {expected_total}."
            )

        expected_keys = set(model.state_dict().keys())
        actual_keys = set(state_dict.keys())
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Incompatible state_dict keys in {self.checkpoint_path}: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        return model

    @property
    def name(self) -> str:
        return f"PyTorchModelPredictor[{self.checkpoint_path.name}]"

    def is_mock(self) -> bool:
        return False

    def get_class_names(self) -> List[str]:
        return list(self.class_names)

    def predict_window(self, spec: np.ndarray) -> Dict[str, float]:
        batch_input = np.asarray(spec, dtype=np.float32)[np.newaxis, ...]
        probs = self.predict_batch(batch_input)[0]
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        specs_arr = np.asarray(specs, dtype=np.float32)
        if len(specs_arr) == 0:
            return np.empty((0, len(self.class_names)), dtype=np.float32)
        if not np.all(np.isfinite(specs_arr)):
            raise ValueError("Input spectrograms contain NaN or Inf values.")

        if specs_arr.ndim == 3:
            # (N, 64, 44) -> (N, 1, 64, 44)
            specs_arr = specs_arr[:, np.newaxis, :, :]
        elif specs_arr.ndim != 4:
            raise ValueError(
                f"Expected 3D (N, 64, 44) or 4D (N, 1, 64, 44) spectrograms, "
                f"got {specs_arr.ndim}D array of shape {specs_arr.shape}."
            )

        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(specs_arr).to(self.device)
            outputs = self.model(tensor)
            if outputs.ndim != 2 or outputs.shape[1] != 1:
                raise ValueError(
                    f"Expected single-logit output of shape (N, 1), got {tuple(outputs.shape)}."
                )
            p_exp = torch.sigmoid(outputs.squeeze(1)).clamp(0.0, 1.0).cpu().numpy()

        probs = np.empty((len(specs_arr), 2), dtype=np.float32)
        probs[:, 1] = p_exp.astype(np.float32, copy=False)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs


class MulticlassModelPredictor(BasePredictor):
    """Predictor interface for future trained 3-class AudioMulticlassCNN checkpoints.

    Strictly forbids untrained inference: requires an explicit, validated 3-class
    checkpoint file (``weights_only=True``) matching ``AudioMulticlassCNN`` and classes
    ``['non_target', 'explosion', 'fireworks']``.
    """

    DEFAULT_CLASS_NAMES = ["non_target", "explosion", "fireworks"]
    ALERT_MAP: Dict[int, Optional[str]] = {
        0: None,
        1: "EXPLOSION_DETECTED",
        2: "FIREWORKS_DETECTED",
    }

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise ValueError(
                "MulticlassModelPredictor requires a trained checkpoint path; "
                "untrained inference is disabled."
            )
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.class_names = (
            list(class_names) if class_names is not None else list(self.DEFAULT_CLASS_NAMES)
        )
        if self.class_names != self.DEFAULT_CLASS_NAMES:
            raise ValueError(
                f"MulticlassModelPredictor requires class_names={self.DEFAULT_CLASS_NAMES}, "
                f"got {self.class_names}."
            )

        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained multiclass checkpoint not found: {self.checkpoint_path}. "
                f"Train a 3-class model before running multiclass inference."
            )

        import torch

        self.torch = torch
        self.checkpoint_metadata: Dict[str, object] = {}
        self.model = self._load_model()
        self.model.eval()

    @staticmethod
    def _import_multiclass_model_class():
        """Import AudioMulticlassCNN regardless of working directory sys.path setup."""
        try:
            from models.multiclass_cnn import AudioMulticlassCNN
            return AudioMulticlassCNN
        except ImportError:
            from explosion_detection.models.multiclass_cnn import AudioMulticlassCNN
            return AudioMulticlassCNN

    @staticmethod
    def _import_alert_evaluator():
        try:
            from multiclass_contract import evaluate_multiclass_alerts
            return evaluate_multiclass_alerts
        except ImportError:
            from explosion_detection.multiclass_contract import evaluate_multiclass_alerts
            return evaluate_multiclass_alerts

    def _load_model(self):
        checkpoint = self.torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Unsupported checkpoint payload type {type(checkpoint).__name__} "
                f"in {self.checkpoint_path}; expected dict state/metadata container."
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Multiclass checkpoint '{self.checkpoint_path}' must include 'model_state_dict' "
                f"and class metadata (unverified/raw checkpoints are rejected)."
            )

        state_dict = checkpoint["model_state_dict"]
        self.checkpoint_metadata = {
            k: v
            for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict")
        }

        ckpt_num_classes = self.checkpoint_metadata.get("num_classes")
        if ckpt_num_classes is None or int(ckpt_num_classes) != len(self.DEFAULT_CLASS_NAMES):
            raise ValueError(
                f"Checkpoint '{self.checkpoint_path}' is not a valid 3-class checkpoint "
                f"(num_classes={ckpt_num_classes!r}, expected 3)."
            )
        ckpt_classes = self.checkpoint_metadata.get("class_names")
        if ckpt_classes is not None and list(ckpt_classes) != self.DEFAULT_CLASS_NAMES:
            raise ValueError(
                f"Checkpoint class_names mismatch in '{self.checkpoint_path}': "
                f"got {ckpt_classes!r}, expected {self.DEFAULT_CLASS_NAMES}."
            )

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Empty or invalid model_state_dict in {self.checkpoint_path}.")

        model_cls = self._import_multiclass_model_class()
        model = model_cls(num_classes=len(self.DEFAULT_CLASS_NAMES))

        expected_total, _ = model.count_parameters()
        ckpt_param_count = self.checkpoint_metadata.get("model_parameters")
        if ckpt_param_count is not None and int(ckpt_param_count) != expected_total:
            raise ValueError(
                f"Checkpoint parameter count mismatch in {self.checkpoint_path}: "
                f"checkpoint reports {ckpt_param_count}, AudioMulticlassCNN expects {expected_total}."
            )

        expected_keys = set(model.state_dict().keys())
        actual_keys = set(state_dict.keys())
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Incompatible state_dict keys in {self.checkpoint_path}: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        return model

    @property
    def name(self) -> str:
        return f"MulticlassModelPredictor[{self.checkpoint_path.name}]"

    def is_mock(self) -> bool:
        return False

    def get_class_names(self) -> List[str]:
        return list(self.class_names)

    def predict_window(self, spec: np.ndarray) -> Dict[str, float]:
        batch_input = np.asarray(spec, dtype=np.float32)[np.newaxis, ...]
        probs = self.predict_batch(batch_input)[0]
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        specs_arr = np.asarray(specs, dtype=np.float32)
        num_classes = len(self.class_names)
        if len(specs_arr) == 0:
            return np.empty((0, num_classes), dtype=np.float32)
        if not np.all(np.isfinite(specs_arr)):
            raise ValueError("Input spectrograms contain NaN or Inf values.")

        if specs_arr.ndim == 3:
            specs_arr = specs_arr[:, np.newaxis, :, :]
        elif specs_arr.ndim != 4:
            raise ValueError(
                f"Expected 3D (N, 64, 44) or 4D (N, 1, 64, 44) spectrograms, "
                f"got {specs_arr.ndim}D array of shape {specs_arr.shape}."
            )

        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(specs_arr).to(self.device)
            logits = self.model(tensor)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"Expected {num_classes}-logit output of shape (N, {num_classes}), "
                    f"got {tuple(logits.shape)}."
                )
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32, copy=False)
        return probs

    def predict_alerts(
        self,
        specs: np.ndarray,
        explosion_threshold: float = 0.80,
        fireworks_threshold: float = 0.80,
        consecutive_windows: int = 1,
    ) -> Dict[str, object]:
        """Compute softmax class probabilities and independent explosion/fireworks alerts."""
        probs = self.predict_batch(specs)
        evaluator = self._import_alert_evaluator()
        result = evaluator(
            probs,
            explosion_threshold=explosion_threshold,
            fireworks_threshold=fireworks_threshold,
            consecutive_windows=consecutive_windows,
        )
        result["probabilities"] = probs
        return result


class BinaryExplosionPredictor(BasePredictor):
    """Predictor for ExplosionBinaryCNN (130,146 parameters, ExpF + AEV).

    Produces calibrated binary probabilities:
        Index 0: non_explosion
        Index 1: explosion
    """

    DEFAULT_CLASS_NAMES = ["non_explosion", "explosion"]

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt",
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.class_names = (
            list(class_names) if class_names is not None else list(self.DEFAULT_CLASS_NAMES)
        )
        if not self.checkpoint_path.exists():
            # Try relative to project root
            cand = Path("D:/NOESIS/Explosion/Explosion") / self.checkpoint_path
            if cand.exists():
                self.checkpoint_path = cand

        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Binary explosion checkpoint not found: {self.checkpoint_path}."
            )

        import torch
        self.torch = torch
        self.checkpoint_metadata: Dict[str, object] = {}
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        try:
            from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
        except ImportError:
            from models.binary_cnn import ExplosionBinaryCNN

        model = ExplosionBinaryCNN(freeze_bn=True)
        checkpoint = self.torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.checkpoint_metadata = {
                k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")
            }
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            self.checkpoint_metadata = {}
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        self.checkpoint_metadata["model_parameters"] = model.count_parameters()[0]
        return model

    @property
    def name(self) -> str:
        return f"BinaryExplosionPredictor[{self.checkpoint_path.name}]"

    def is_mock(self) -> bool:
        return False

    def get_class_names(self) -> List[str]:
        return list(self.class_names)

    def predict_window(self, spec: np.ndarray) -> Dict[str, float]:
        batch_input = np.asarray(spec, dtype=np.float32)[np.newaxis, ...]
        probs = self.predict_batch(batch_input)[0]
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        specs_arr = np.asarray(specs, dtype=np.float32)
        if len(specs_arr) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if not np.all(np.isfinite(specs_arr)):
            raise ValueError("Input spectrograms contain NaN or Inf values.")

        if specs_arr.ndim == 3:
            specs_arr = specs_arr[:, np.newaxis, :, :]
        elif specs_arr.ndim != 4:
            raise ValueError(
                f"Expected 3D (N, 64, 44) or 4D (N, 1, 64, 44) spectrograms, got {specs_arr.ndim}D array of shape {specs_arr.shape}."
            )

        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(specs_arr).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32, copy=False)
        return probs



