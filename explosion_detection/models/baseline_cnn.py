"""Baseline 2D Convolutional Neural Network for Acoustic Explosion Detection.

Designed specifically for small-dataset binary acoustic classification.
Operates on (batch_size, 1, 64, 44) normalized Log-Mel spectrograms.
Outputs 1 unconstrained binary logit per window for use with BCEWithLogitsLoss.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class AudioBaselineCNN(nn.Module):
    """Lightweight regularized 2D CNN for binary explosion detection.
    
    Expected input shape: (batch_size, 1, 64, 44)
    Output shape: (batch_size, 1) unconstrained logit.
    """

    EXPECTED_CHANNELS = 1
    EXPECTED_MELS = 64
    EXPECTED_TIME_FRAMES = 44

    def __init__(
        self,
        in_channels: int = 1,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.30,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        
        # Block 1: (B, 1, 64, 44) -> (B, 32, 32, 22)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 2: (B, 32, 32, 22) -> (B, 64, 16, 11)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 3: (B, 64, 16, 11) -> (B, 64, 8, 5)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 4: (B, 64, 8, 5) -> (B, 128, 1, 1)
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification Head: (B, 128) -> (B, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, 1)
        )

        # Deterministic weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """Strictly validate input tensor dimensions and shape.
        
        Raises ValueError with descriptive diagnostics if shape is invalid.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
            
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor of shape (batch_size, 1, 64, 44), "
                f"got {x.ndim}D tensor of shape {tuple(x.shape)}"
            )

        b, c, h, w = x.shape
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
        """Forward pass computing raw binary logit per window.
        
        Args:
            x: Tensor of shape (batch_size, 1, 64, 44)
            
        Returns:
            Tensor of shape (batch_size, 1) representing unconstrained logits.
            DO NOT apply sigmoid here; use BCEWithLogitsLoss during training.
        """
        self.validate_input_shape(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.head(x)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_baseline_cnn(seed: int = 42) -> AudioBaselineCNN:
    """Factory helper to construct the baseline CNN with deterministic initialization."""
    return AudioBaselineCNN(seed=seed)

