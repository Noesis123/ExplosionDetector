"""Separate Three-Class 2D CNN Architecture for Explosion vs. Fireworks vs. Non-Target Audio.

Preserves AudioBaselineCNN (models/baseline_cnn.py) untouched for binary inference.
Designed for (batch_size, 1, 64, 44) normalized Log-Mel spectrograms and emits
3 unconstrained class logits ordered by:
    0: non_target (no alert)
    1: explosion  (EXPLOSION_DETECTED)
    2: fireworks  (FIREWORKS_DETECTED)

For future training, use CrossEntropyLoss on raw logits.
For future inference, apply Softmax(dim=-1) to obtain calibrated class probabilities.
"""

from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

try:
    from explosion_detection.multiclass_contract import (
        MULTICLASS_ALERT_MAP,
        MULTICLASS_CLASS_NAMES,
        MULTICLASS_INV_LABEL_MAP,
        MULTICLASS_LABEL_MAP,
        MULTICLASS_NUM_CLASSES,
    )
except ImportError:
    from multiclass_contract import (
        MULTICLASS_ALERT_MAP,
        MULTICLASS_CLASS_NAMES,
        MULTICLASS_INV_LABEL_MAP,
        MULTICLASS_LABEL_MAP,
        MULTICLASS_NUM_CLASSES,
    )


class AudioMulticlassCNN(nn.Module):
    """Lightweight regularized 2D CNN for three-class acoustic discrimination.

    Expected input shape: (batch_size, 1, 64, 44)
    Output shape: (batch_size, 3) unconstrained logits for CrossEntropyLoss.
    """

    EXPECTED_CHANNELS = 1
    EXPECTED_MELS = 64
    EXPECTED_TIME_FRAMES = 44
    NUM_CLASSES = MULTICLASS_NUM_CLASSES
    CLASS_NAMES = MULTICLASS_CLASS_NAMES
    LABEL_MAP = dict(MULTICLASS_LABEL_MAP)
    INV_LABEL_MAP = dict(MULTICLASS_INV_LABEL_MAP)
    ALERT_MAP = dict(MULTICLASS_ALERT_MAP)

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = MULTICLASS_NUM_CLASSES,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.30,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if num_classes != MULTICLASS_NUM_CLASSES:
            raise ValueError(
                f"AudioMulticlassCNN requires num_classes={MULTICLASS_NUM_CLASSES} "
                f"({MULTICLASS_CLASS_NAMES}), got {num_classes}."
            )

        if seed is not None:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv layers and Xavier normal for head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """Validate 4D spectrogram tensor shape ``(batch_size, 1, 64, 44)``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor of shape (batch_size, 1, 64, 44), "
                f"got {x.ndim}D tensor of shape {tuple(x.shape)}"
            )
        _, c, h, w = x.shape
        if c != self.EXPECTED_CHANNELS:
            raise ValueError(
                f"Invalid channel dimension: expected {self.EXPECTED_CHANNELS}, got {c}. "
                f"Full input shape: {tuple(x.shape)}"
            )
        if h != self.EXPECTED_MELS or w != self.EXPECTED_TIME_FRAMES:
            raise ValueError(
                f"Invalid spectrogram dimension: expected ({self.EXPECTED_MELS}, {self.EXPECTED_TIME_FRAMES}), "
                f"got ({h}, {w}). Full input shape: {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing 3 unconstrained logits per window for CrossEntropyLoss."""
        self.validate_input_shape(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Future inference helper returning Softmax class probabilities ``(batch_size, 3)``."""
        with torch.inference_mode():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> Tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_multiclass_loss(
    class_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.CrossEntropyLoss:
    """Construct ``nn.CrossEntropyLoss`` configured for future 3-class training."""
    if class_weights is None:
        return nn.CrossEntropyLoss()
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    if weight_tensor.shape != (MULTICLASS_NUM_CLASSES,):
        raise ValueError(
            f"class_weights must have shape ({MULTICLASS_NUM_CLASSES},), got {tuple(weight_tensor.shape)}."
        )
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def build_multiclass_cnn(
    num_classes: int = MULTICLASS_NUM_CLASSES,
    seed: int = 42,
) -> AudioMulticlassCNN:
    """Factory helper to construct AudioMulticlassCNN deterministically."""
    return AudioMulticlassCNN(num_classes=num_classes, seed=seed)

