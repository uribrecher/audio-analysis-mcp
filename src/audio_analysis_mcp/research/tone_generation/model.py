"""Conditioned CNN for subtractive tone-generation MVP.

Architecture:
  mel-spec (1, 128, ~30) -> 3 conv blocks -> AdaptiveAvgPool2d((16, 1))  # pool over time
  -> flatten -> concat with 88-dim pitch multihot -> 2-layer MLP -> 3 heads.

The model is device-agnostic: device placement is the train script's
responsibility. Sigmoid heads clamp regression outputs to [0, 1] (load-bearing
for round-trip stability with schema_io.denormalize_predictions).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_PITCH_DIM = 88


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class ToneGenerationCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _ConvBlock(1, 32)
        self.block2 = _ConvBlock(32, 64)
        self.block3 = _ConvBlock(64, 128)
        # After 3 MaxPool2d(2): mel dim 128 -> 16; time dim ~30 -> ~3.
        # AdaptiveAvgPool2d((16, 1)) collapses time but preserves freq.
        self.time_pool = nn.AdaptiveAvgPool2d((16, 1))
        bottleneck_dim = 128 * 16  # channels * mel after pool
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim + _PITCH_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.head_shape = nn.Linear(256, 4)
        self.head_cutoff = nn.Linear(256, 1)
        self.head_resonance = nn.Linear(256, 1)

    def forward(
        self, mel: torch.Tensor, pitch_multihot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.block1(mel)
        h = self.block2(h)
        h = self.block3(h)
        h = self.time_pool(h)
        h = h.flatten(1)
        h = torch.cat([h, pitch_multihot], dim=1)
        h = self.mlp(h)
        return {
            "shape_logits": self.head_shape(h),
            "cutoff_norm": torch.sigmoid(self.head_cutoff(h)).squeeze(-1),
            "resonance": torch.sigmoid(self.head_resonance(h)).squeeze(-1),
        }
