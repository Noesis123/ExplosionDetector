"""Dedicated Explosion Binary CNN Architecture for Production.

Explosion-vs-Not-Explosion binary classifier adapted from pretrained AudioMulticlassCNN.
Input shape: (batch_size, 1, 64, 44) normalized Log-Mel spectrogram.
Output shape: (batch_size, 2) unconstrained logits:
    Index 0: not_explosion (Non-target + Fireworks)
    Index 1: explosion     (Explosion positive)
Total Parameters: 130,146
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ExplosionBinaryCNN(nn.Module):
    """Regularized 2D CNN for binary Explosion detection (130,146 parameters)."""

    EXPECTED_CHANNELS = 1
    EXPECTED_MELS = 64
    EXPECTED_TIME_FRAMES = 44
    NUM_CLASSES = 2
    CLASS_NAMES = ["not_explosion", "explosion"]

    def __init__(
        self,
        in_channels: int = 1,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.30,
        freeze_bn: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        self.num_classes = 2
        self.freeze_bn = freeze_bn

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
            nn.Linear(128, 2),
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

    def set_bn_eval(self) -> None:
        """Freeze BatchNorm running stats into evaluation mode to prevent running variance drift."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, mode: bool = True):
        """Override train to keep BatchNorm in eval mode if freeze_bn is True."""
        super().train(mode)
        if mode and self.freeze_bn:
            self.set_bn_eval()
        return self

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone convolutional layers."""
        for block in [self.block1, self.block2, self.block3, self.block4]:
            for p in block.parameters():
                p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch_size, 2) softmax probabilities [p_not_explosion, p_explosion]."""
        with torch.inference_mode():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict_explosion_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch_size,) Explosion confidence."""
        return self.predict_probabilities(x)[:, 1]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
