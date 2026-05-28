"""
Snow Physics Modeling (SPM) Module.

Multi-branch parallel processing with 3x3, 5x5, 7x7 convolutions (Fig. 2 of paper).
Outputs: scattering coefficient beta(x,y), transmission rate t(x,y),
         occlusion mask M(x,y), PSF sigma, depth map.

Supports component-level ablation (Table 13):
  - disable_mie:        disable Mie scattering branch
  - disable_rayleigh:   disable Rayleigh scattering branch
  - disable_occlusion:  disable occlusion mask estimation (fixed to 1)
  - fusion_mode:        'attention' (Eq. 8-9) | 'average' | 'concat'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_fusion import AttentionFusion


class ConvBranch(nn.Module):
    """Single convolutional branch with configurable kernel size.
    Each branch follows Conv -> ReLU -> BN (as shown in Fig. 2)."""

    def __init__(self, in_channels, out_channels, kernel_size, num_blocks=3):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            ]
        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


class AverageFusion(nn.Module):
    """Plain average fusion (for ablation against attention-based fusion)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, features):
        return sum(features) / len(features)


class ConcatFusion(nn.Module):
    """Concatenate features along channel dim then project back to C channels."""

    def __init__(self, num_branches, channels, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(num_branches * channels, channels, kernel_size=1)

    def forward(self, features):
        h = torch.cat(features, dim=1)
        return self.proj(h)


def _make_fusion(mode, num_branches, channels, hidden_dim):
    if mode == 'attention':
        return AttentionFusion(num_branches=num_branches, channels=channels,
                               hidden_dim=hidden_dim)
    elif mode == 'average':
        return AverageFusion()
    elif mode == 'concat':
        return ConcatFusion(num_branches=num_branches, channels=channels)
    raise ValueError(f"Unknown fusion mode: {mode}")


class SPMModule(nn.Module):
    """Snow Physics Modeling Module (Fig. 2).

    Three parallel branches with 3x3, 5x5, 7x7 receptive fields capture
    scattering features at different scales. An attention-based fusion
    aggregates them, then four heads produce the physical parameter maps.

    Args:
        in_channels:         input channel count (3 for RGB)
        branch_channels:     per-branch output channels
        kernel_sizes:        list of kernel sizes; one branch per entry
        attention_hidden_dim: hidden dim of attention fusion MLP
        disable_mie:         drop the Mie-scale branches (>=5x5) when True
        disable_rayleigh:    drop the Rayleigh-scale branch (3x3) when True
        disable_occlusion:   if True, occlusion mask is forced to 1 (no occlusion modeling)
        fusion_mode:         'attention' (default) | 'average' | 'concat'
    """

    def __init__(self,
                 in_channels=3,
                 branch_channels=64,
                 kernel_sizes=(3, 5, 7),
                 attention_hidden_dim=32,
                 disable_mie=False,
                 disable_rayleigh=False,
                 disable_occlusion=False,
                 fusion_mode='attention'):
        super().__init__()
        kernel_sizes = list(kernel_sizes)

        # Component ablation: drop scale branches if disabled.
        if disable_rayleigh:
            kernel_sizes = [k for k in kernel_sizes if k != 3]
        if disable_mie:
            kernel_sizes = [k for k in kernel_sizes if k < 5]
        if len(kernel_sizes) == 0:
            # Always keep at least one branch so the network is well-defined.
            kernel_sizes = [3]

        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.disable_occlusion = disable_occlusion
        self.fusion_mode = fusion_mode

        self.branches = nn.ModuleList([
            ConvBranch(in_channels, branch_channels, ks) for ks in kernel_sizes
        ])

        self.fusion = _make_fusion(
            mode=fusion_mode,
            num_branches=self.num_branches,
            channels=branch_channels,
            hidden_dim=attention_hidden_dim,
        )

        # Heads
        def head(out_ch=1):
            return nn.Sequential(
                nn.Conv2d(branch_channels, branch_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels // 2, out_ch, 1),
                nn.Sigmoid(),
            )

        self.scattering_head = head()
        self.transmission_head = head()
        self.occlusion_head = head()
        self.depth_head = head()

        self.psf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(branch_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        fused = self.fusion(feats)

        scattering = self.scattering_head(fused)
        transmission = self.transmission_head(fused)
        depth = self.depth_head(fused)
        psf_sigma = self.psf_head(fused)

        if self.disable_occlusion:
            occlusion = torch.ones_like(scattering)
        else:
            occlusion = self.occlusion_head(fused)

        return {
            'scattering_coeff': scattering,
            'transmission': transmission,
            'occlusion_mask': occlusion,
            'psf_sigma': psf_sigma,
            'depth_map': depth,
            'fused_features': fused,
        }
