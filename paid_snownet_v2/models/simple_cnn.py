"""
Simple stack-of-conv completion network.
Used as a control in the SIRM completion ablation (Table 14).
"""

import torch
import torch.nn as nn


class SimpleCNNCompletion(nn.Module):
    """Plain feed-forward CNN: in_conv -> N residual conv blocks -> out_conv.

    Lacks U-shape skip connections and encoder/decoder structure, so it is
    weaker than the default U-Net but preserves the same input/output shape.
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_blocks=5):
        super().__init__()
        # Pick largest divisor of base_channels not exceeding 32.
        ng = 1
        for g in (32, 16, 8, 4, 2, 1):
            if base_channels % g == 0:
                ng = g
                break
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                nn.GroupNorm(ng, base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                nn.GroupNorm(ng, base_channels),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.in_conv(x)
        for blk in self.blocks:
            h = self.act(h + blk(h))
        out = self.out_conv(h) + x
        return torch.clamp(out, 0.0, 1.0)
