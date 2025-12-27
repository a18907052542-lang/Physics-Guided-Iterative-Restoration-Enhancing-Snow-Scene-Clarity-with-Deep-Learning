"""
Snow Physics Modeling (SPM) Module for PAID-SnowNet
Implements Section 3.1 of the paper

The SPM module extracts physical parameters from degraded snow images:
- Scattering coefficient β(x,y)
- Transmission rate t(x,y)
- Occlusion mask M(x,y)
- Depth map d(x,y)
- PSF parameter σ

Multi-scale architecture captures different snow phenomena:
- 3×3: Local details, small-scale scattering
- 5×5: Regional features, medium snowflakes
- 7×7: Global context, atmospheric effects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from .attention_fusion import AdaptiveFusion, MultiScaleAttentionFusion


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None,
                 use_bn: bool = True, use_relu: bool = True):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride, padding, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction with parallel branches.
    Implements Figure 2 architecture from the paper.
    
    Three parallel branches with different kernel sizes:
    - Branch 1 (3×3): Local details, receptive field 3-9 pixels
    - Branch 2 (5×5): Regional features, receptive field 5-25 pixels  
    - Branch 3 (7×7): Global context, receptive field >49 pixels
    
    Table 3 ablation shows this combination achieves 30.85dB PSNR.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 kernel_sizes: List[int] = [3, 5, 7],
                 num_blocks: int = 4):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            base_channels: Base feature channels per branch
            kernel_sizes: Kernel sizes for each branch
            num_blocks: Number of residual blocks per branch
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        
        # Initial projection for each branch
        self.initial_convs = nn.ModuleList([
            ConvBlock(in_channels, base_channels, k)
            for k in kernel_sizes
        ])
        
        # Residual blocks for each branch
        self.branches = nn.ModuleList([
            nn.Sequential(*[
                ResidualBlock(base_channels, k)
                for _ in range(num_blocks)
            ])
            for k in kernel_sizes
        ])
        
        # Adaptive fusion (Equations 8-9)
        self.fusion = AdaptiveFusion(base_channels, self.num_branches)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Tuple of:
                - Fused features [B, C, H, W]
                - List of branch features [B, C, H, W] each
                - Attention weights [B, num_branches]
        """
        # Initial feature extraction
        branch_features = []
        for init_conv, branch in zip(self.initial_convs, self.branches):
            feat = init_conv(x)
            feat = branch(feat)
            branch_features.append(feat)
        
        # Adaptive fusion
        fused, attention_weights = self.fusion(branch_features)
        
        return fused, branch_features, attention_weights


class ScatteringEstimator(nn.Module):
    """
    Estimates spatially-varying scattering coefficient β(x,y).
    
    The scattering coefficient determines how much light is
    scattered by snow particles at each location.
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.estimator = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 3),
            ConvBlock(in_channels // 2, in_channels // 4, 3),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Softplus()  # Ensure positive values
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate scattering coefficient.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Scattering coefficient β [B, 1, H, W], positive values
        """
        return self.estimator(features)


class TransmissionEstimator(nn.Module):
    """
    Estimates transmission map t(x,y) = exp(-β·d).
    
    The transmission rate indicates how much of the original
    scene radiance reaches the camera through the snow medium.
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.estimator = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 3),
            ConvBlock(in_channels // 2, in_channels // 4, 3),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # Transmission in [0, 1]
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate transmission map.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Transmission map t [B, 1, H, W] in range [0, 1]
        """
        return self.estimator(features)


class OcclusionMaskEstimator(nn.Module):
    """
    Estimates soft occlusion mask M(x,y) ∈ [0, 1].
    
    The occlusion mask indicates which regions are covered
    by snow particles. Used in degradation model Eq(12).
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.estimator = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3),
            ConvBlock(in_channels, in_channels // 2, 3),
            ConvBlock(in_channels // 2, in_channels // 4, 3),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # Mask in [0, 1]
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate occlusion mask.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Occlusion mask M [B, 1, H, W] in range [0, 1]
        """
        return self.estimator(features)


class DepthEstimator(nn.Module):
    """
    Estimates relative depth map d(x,y).
    
    Depth information is used to compute attenuation
    in the degradation model.
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.estimator = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3),
            ConvBlock(in_channels, in_channels // 2, 3),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Softplus()  # Positive depth
        )
        
        # Scale factor for depth normalization
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth map.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Depth map d [B, 1, H, W], positive values
        """
        depth = self.estimator(features)
        return depth * self.scale.abs()


class PSFEstimator(nn.Module):
    """
    Estimates Point Spread Function parameter σ for blur.
    
    The PSF models the blur introduced by snow particles
    and atmospheric scattering.
    
    PSF: h(x,y) = (1/2πσ²) · exp(-(x²+y²)/(2σ²))
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        # Global estimation (image-level σ)
        self.global_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1),
            nn.Softplus()  # Positive σ
        )
        
        # Local estimation (spatially varying σ)
        self.local_estimator = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 3),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Softplus()
        )
        
        # Balance between global and local
        self.balance = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate PSF parameter.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Tuple of:
                - Global σ [B, 1]
                - Local σ map [B, 1, H, W]
        """
        sigma_global = self.global_estimator(features)  # [B, 1]
        sigma_local = self.local_estimator(features)     # [B, 1, H, W]
        
        return sigma_global, sigma_local


class SPMModule(nn.Module):
    """
    Snow Physics Modeling (SPM) Module.
    
    Complete implementation of Section 3.1, combining:
    - Multi-scale feature extraction (Figure 2)
    - Physical parameter estimation
    - Adaptive fusion (Equations 8-9)
    
    Outputs the unified degradation model parameters from Eq(12):
    y(x,y) = [(i₁+i₂)/(k²r²)]·A²M(x,y)x + A²[1-M(x,y)]I_snow + ε
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 kernel_sizes: List[int] = [3, 5, 7],
                 num_blocks: int = 4):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            base_channels: Base feature channels
            kernel_sizes: Kernel sizes for multi-scale branches
            num_blocks: Number of residual blocks per branch
        """
        super().__init__()
        
        # Multi-scale feature extraction
        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels, base_channels, kernel_sizes, num_blocks
        )
        
        # Physical parameter estimators
        self.scattering_estimator = ScatteringEstimator(base_channels)
        self.transmission_estimator = TransmissionEstimator(base_channels)
        self.occlusion_estimator = OcclusionMaskEstimator(base_channels)
        self.depth_estimator = DepthEstimator(base_channels)
        self.psf_estimator = PSFEstimator(base_channels)
        
        # Feature refinement
        self.refine = nn.Sequential(
            ConvBlock(base_channels, base_channels, 3),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract physical parameters from degraded image.
        
        Args:
            x: Degraded snow image [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'features': Refined features [B, C, H, W]
                - 'scattering': β(x,y) [B, 1, H, W]
                - 'transmission': t(x,y) [B, 1, H, W]
                - 'occlusion_mask': M(x,y) [B, 1, H, W]
                - 'depth': d(x,y) [B, 1, H, W]
                - 'psf_sigma': (global σ, local σ map)
                - 'attention_weights': Fusion weights [B, 3]
                - 'branch_features': List of branch features
        """
        # Multi-scale feature extraction
        fused, branch_features, attention_weights = self.feature_extractor(x)
        
        # Refine fused features
        features = self.refine(fused)
        
        # Estimate physical parameters
        scattering = self.scattering_estimator(features)
        transmission = self.transmission_estimator(features)
        occlusion_mask = self.occlusion_estimator(features)
        depth = self.depth_estimator(features)
        psf_global, psf_local = self.psf_estimator(features)
        
        return {
            'features': features,
            'scattering': scattering,
            'transmission': transmission,
            'occlusion_mask': occlusion_mask,
            'depth': depth,
            'psf_sigma': (psf_global, psf_local),
            'attention_weights': attention_weights,
            'branch_features': branch_features
        }


class LightweightSPM(nn.Module):
    """
    Lightweight SPM variant for faster inference.
    
    Reduces computational cost by:
    - Using fewer channels
    - Sharing parameters between estimators
    - Single-scale processing with dilated convolutions
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        # Efficient feature extraction with dilated convolutions
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, 3),
            ConvBlock(base_channels, base_channels, 3),
            nn.Conv2d(base_channels, base_channels, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Shared estimation head
        self.shared_head = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2, 3),
            ConvBlock(base_channels // 2, base_channels // 4, 1)
        )
        
        # Individual output heads
        self.scattering_head = nn.Sequential(
            nn.Conv2d(base_channels // 4, 1, 1),
            nn.Softplus()
        )
        self.transmission_head = nn.Sequential(
            nn.Conv2d(base_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(base_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Lightweight parameter estimation.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Dictionary with estimated parameters
        """
        features = self.encoder(x)
        shared = self.shared_head(features)
        
        return {
            'features': features,
            'scattering': self.scattering_head(shared),
            'transmission': self.transmission_head(shared),
            'occlusion_mask': self.occlusion_head(shared)
        }


# Initialize package
__all__ = [
    'SPMModule',
    'LightweightSPM',
    'MultiScaleFeatureExtractor',
    'ScatteringEstimator',
    'TransmissionEstimator',
    'OcclusionMaskEstimator',
    'DepthEstimator',
    'PSFEstimator'
]
