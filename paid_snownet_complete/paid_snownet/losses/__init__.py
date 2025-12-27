"""
PAID-SnowNet Loss Functions

Multi-scale supervision loss functions for training:
- Charbonnier loss for robust pixel-wise comparison
- VGG perceptual loss for semantic similarity
- Edge loss for structure preservation
- SSIM loss for structural similarity
- Color consistency loss for perceptual uniformity
- Multi-scale loss combining all components

Reference: Eq(29-31) in paper
"""

from .multi_scale_loss import (
    MultiScaleLoss,
    CharbonnierLoss,
    VGGPerceptualLoss,
    EdgeLoss,
    SSIMLoss,
    ColorConsistencyLoss,
    SnowDenoiseMetric,
    create_loss_function
)

__all__ = [
    'MultiScaleLoss',
    'CharbonnierLoss',
    'VGGPerceptualLoss',
    'EdgeLoss',
    'SSIMLoss',
    'ColorConsistencyLoss',
    'SnowDenoiseMetric',
    'create_loss_function',
]
