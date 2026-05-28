"""
ResNet-style completion network with global residual.
Used as a control in the SIRM completion ablation (Table 14).
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ng = 1
        for g in (32, 16, 8, 4, 2, 1):
            if channels % g == 0:
                ng = g
                break
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(ng, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class ResNetCompletion(nn.Module):
    """Flat residual network (no encoder/decoder, no skips across scales)."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        h = self.body(h)
        out = self.tail(h) + x
        return torch.clamp(out, 0.0, 1.0)
