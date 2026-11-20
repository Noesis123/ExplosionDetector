"""Non-PyTorch unit tests for Phase 7 PyTorchModelPredictor contract and checkpoint metadata."""

import contextlib
import pickletools
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLOSION_DETECTION_ROOT = PROJECT_ROOT / "explosion_detection"
for p in (str(PROJECT_ROOT), str(EXPLOSION_DETECTION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.predictor_interface import PyTorchModelPredictor


class _FakeTensor:
    """Minimal NumPy-backed tensor shim to verify sigmoid contract without PyTorch DLLs."""

    def __init__(self, array: np.ndarray):
        self.arr = np.asarray(array, dtype=np.float32)

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def to(self, _device):
        return self

    def squeeze(self, axis: int):
        return _FakeTensor(np.squeeze(self.arr, axis=axis))

    def clamp(self, min_val: float, max_val: float):
        return _FakeTensor(np.clip(self.arr, min_val, max_val))

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


class _FakeCNN:
    """Minimal stand-in matching AudioBaselineCNN state_dict & forward contract."""

    EXPECTED_KEYS = (
        "block1.0.weight", "block1.1.weight", "block1.1.bias",
        "block1.1.running_mean", "block1.1.running_var", "block1.1.num_batches_tracked",
        "block2.0.weight", "block2.1.weight", "block2.1.bias",
        "block2.1.running_mean", "block2.1.running_var", "block2.1.num_batches_tracked",
        "block3.0.weight", "block3.1.weight", "block3.1.bias",
        "block3.1.running_mean", "block3.1.running_var", "block3.1.num_batches_tracked",
        "block4.0.weight", "block4.1.weight", "block4.1.bias",
        "block4.1.running_mean", "block4.1.running_var", "block4.1.num_batches_tracked",
        "head.2.weight", "head.2.bias",
    )

    def __init__(self):
        self.eval_called = False
        self.loaded_state = None
        self.last_input_shape = None
        self.to_device = None

    def count_parameters(self):
        return (130017, 130017)

    def state_dict(self):
        return {k: np.zeros(1, dtype=np.float32) for k in self.EXPECTED_KEYS}

    def load_state_dict(self, state_dict, strict=True):
        if strict and set(state_dict.keys()) != set(self.EXPECTED_KEYS):
            raise RuntimeError("State dict key mismatch")
        self.loaded_state = state_dict

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, tensor: _FakeTensor) -> _FakeTensor:
        self.last_input_shape = tensor.shape
        # Deterministic single-logit output derived from mean of each sample
        means = tensor.arr.mean(axis=(1, 2, 3), keepdims=False)
        logits = (means * 2.0)[:, np.newaxis]
        return _FakeTensor(logits)


def _make_fake_torch_module(payload):
    calls = []
    fake_torch = types.ModuleType("torch")

    def fake_load(path, map_location="cpu", weights_only=None):
        calls.append({"path": Path(path), "map_location": map_location, "weights_only": weights_only})
        return payload

    def fake_from_numpy(arr):
        return _FakeTensor(arr)

    def fake_sigmoid(tensor: _FakeTensor):
        return _FakeTensor(1.0 / (1.0 + np.exp(-tensor.arr)))

    @contextlib.contextmanager
    def fake_inference_mode():
        yield

    fake_torch.load = fake_load
    fake_torch.from_numpy = fake_from_numpy
    fake_torch.sigmoid = fake_sigmoid
    fake_torch.inference_mode = fake_inference_mode
    return fake_torch, calls


class TestPyTorchPredictorContract(unittest.TestCase):
    """Verify PyTorchModelPredictor safe loading, architecture checks, and sigmoid inference."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ckpt_file = Path(self.temp_dir.name) / "dummy.pt"
        self.ckpt_file.write_bytes(b"PK\x03\x04dummy")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_load_uses_weights_only_and_computes_sigmoid_probs(self):
        state = {k: np.zeros(1, dtype=np.float32) for k in _FakeCNN.EXPECTED_KEYS}
        payload = {"epoch": 17, "model_parameters": 130017, "model_state_dict": state}
        fake_torch, load_calls = _make_fake_torch_module(payload)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(PyTorchModelPredictor, "_import_model_class", return_value=_FakeCNN):
                pred = PyTorchModelPredictor(self.ckpt_file, device="cpu")

        self.assertEqual(len(load_calls), 1)
        self.assertTrue(load_calls[0]["weights_only"])
        self.assertTrue(pred.model.eval_called)
        self.assertFalse(pred.is_mock())
        self.assertEqual(pred.checkpoint_metadata["epoch"], 17)

        # Single-logit sigmoid check for 3D (N, 64, 44) and 4D (N, 1, 64, 44)
        spec_zero = np.zeros((2, 64, 44), dtype=np.float32)
        spec_pos = np.ones((1, 1, 64, 44), dtype=np.float32)
        out_3d = pred.predict_batch(spec_zero)
        self.assertEqual(pred.model.last_input_shape, (2, 1, 64, 44))
        self.assertEqual(out_3d.shape, (2, 2))
        np.testing.assert_allclose(out_3d, np.full((2, 2), 0.5, dtype=np.float32), atol=1e-6)

        out_4d = pred.predict_batch(spec_pos)
        expected_p_exp = 1.0 / (1.0 + np.exp(-2.0))
        self.assertAlmostEqual(float(out_4d[0, 1]), float(expected_p_exp), places=5)
        self.assertAlmostEqual(float(out_4d[0, 0]), float(1.0 - expected_p_exp), places=5)

        win_dict = pred.predict_window(np.ones((64, 44), dtype=np.float32))
        self.assertAlmostEqual(win_dict["explosion"], float(expected_p_exp), places=5)
        self.assertAlmostEqual(win_dict["non_explosion"], float(1.0 - expected_p_exp), places=5)

    def test_empty_invalid_shapes_and_incompatible_checkpoints_rejected(self):
        state = {k: np.zeros(1, dtype=np.float32) for k in _FakeCNN.EXPECTED_KEYS}
        fake_torch, _ = _make_fake_torch_module({"model_parameters": 130017, "model_state_dict": state})
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(PyTorchModelPredictor, "_import_model_class", return_value=_FakeCNN):
                pred = PyTorchModelPredictor(self.ckpt_file)

                empty_out = pred.predict_batch(np.empty((0, 64, 44), dtype=np.float32))
                self.assertEqual(empty_out.shape, (0, 2))

                with self.assertRaises(ValueError):
                    pred.predict_batch(np.zeros((64, 44), dtype=np.float32))  # 2D rejected

                bad_nan = np.zeros((1, 64, 44), dtype=np.float32)
                bad_nan[0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    pred.predict_batch(bad_nan)

        # Incompatible parameter count metadata rejected
        bad_param_torch, _ = _make_fake_torch_module({"model_parameters": 999, "model_state_dict": state})
        with mock.patch.dict(sys.modules, {"torch": bad_param_torch}):
            with mock.patch.object(PyTorchModelPredictor, "_import_model_class", return_value=_FakeCNN):
                with self.assertRaises(ValueError):
                    PyTorchModelPredictor(self.ckpt_file)

        # Missing state_dict keys rejected
        partial_state = {k: v for k, v in state.items() if k != "head.2.bias"}
        bad_keys_torch, _ = _make_fake_torch_module({"model_state_dict": partial_state})
        with mock.patch.dict(sys.modules, {"torch": bad_keys_torch}):
            with mock.patch.object(PyTorchModelPredictor, "_import_model_class", return_value=_FakeCNN):
                with self.assertRaises(ValueError):
                    PyTorchModelPredictor(self.ckpt_file)

    def test_saved_checkpoints_match_safe_pickle_globals_and_architecture_keys(self):
        allowed_globals = {
            "collections OrderedDict",
            "torch FloatStorage",
            "torch LongStorage",
            "torch._utils _rebuild_tensor_v2",
        }
        ckpt_dir = EXPLOSION_DETECTION_ROOT / "models" / "checkpoints"
        for ckpt_name in ("candidate_expF_accuracy_optimized_best.pt", "candidate_expE_hard_neg_refined_best.pt"):
            ckpt_path = ckpt_dir / ckpt_name
            self.assertTrue(ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}")
            with zipfile.ZipFile(ckpt_path) as zf:
                pkl_entry = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
                pkl_bytes = zf.read(pkl_entry)
            globals_seen = {
                str(arg)
                for op, arg, _ in pickletools.genops(pkl_bytes)
                if op.name in ("GLOBAL", "STACK_GLOBAL", "INST")
            }
            strings_seen = {
                arg
                for op, arg, _ in pickletools.genops(pkl_bytes)
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE") and isinstance(arg, str)
            }
            self.assertEqual(globals_seen, allowed_globals)
            self.assertIn("model_state_dict", strings_seen)
            for key in _FakeCNN.EXPECTED_KEYS:
                self.assertIn(key, strings_seen)


if __name__ == "__main__":
    unittest.main()

