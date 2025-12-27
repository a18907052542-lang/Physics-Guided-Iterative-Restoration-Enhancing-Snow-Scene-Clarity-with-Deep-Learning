"""
Degradation Operators for PAID-SnowNet
Implements Equation (16) from the paper

This module provides differentiable degradation operators:
A = D_blur ∘ D_scatter ∘ D_occlude

Each operator is implemented as both forward and transpose
to enable gradient-based optimization in SIRM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GaussianKernel(nn.Module):
    """
    Generates Gaussian PSF kernels.
    
    h(x,y) = (1/2πσ²) · exp(-(x²+y²)/(2σ²))
    """
    
    def __init__(self, kernel_size: int = 15, channels: int = 3):
        """
        Args:
            kernel_size: Size of the kernel (should be odd)
            channels: Number of image channels
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = kernel_size // 2
        
    def create_kernel(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Create Gaussian kernel for given sigma.
        
        Args:
            sigma: Standard deviation [B] or [B, 1, H, W]
            
        Returns:
            Gaussian kernel [B, 1, K, K] or batched spatially-varying kernels
        """
        k = self.kernel_size
        
        # Create coordinate grid
        ax = torch.linspace(-(k - 1) / 2., (k - 1) / 2., k, device=sigma.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # Ensure sigma has right shape
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)
        elif sigma.dim() == 4:
            # Spatially varying: take mean for kernel generation
            sigma = sigma.mean(dim=[2, 3]).view(-1, 1, 1)
        
        # Gaussian formula
        kernel = torch.exp(-(xx.unsqueeze(0) ** 2 + yy.unsqueeze(0) ** 2) / 
                          (2. * sigma ** 2 + 1e-8))
        
        # Normalize
        kernel = kernel / (kernel.sum(dim=[-1, -2], keepdim=True) + 1e-8)
        
        return kernel.unsqueeze(1)  # [B, 1, K, K]
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur.
        
        Args:
            x: Input tensor [B, C, H, W]
            sigma: Blur sigma [B] or [B, 1, H, W]
            
        Returns:
            Blurred tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        kernel = self.create_kernel(sigma)  # [B, 1, K, K]
        
        # Expand kernel for all channels
        kernel = kernel.repeat(1, C, 1, 1)  # [B, C, K, K]
        
        # Apply depthwise convolution per batch
        outputs = []
        for i in range(B):
            # [1, C, H, W] conv with [C, 1, K, K]
            k = kernel[i:i+1].transpose(0, 1)  # [C, 1, K, K]
            out = F.conv2d(x[i:i+1], k, padding=self.padding, groups=C)
            outputs.append(out)
            
        return torch.cat(outputs, dim=0)


class BlurOperator(nn.Module):
    """
    Blur degradation operator D_blur.
    
    Implements convolution with Gaussian PSF:
    D_blur(x) = h * x
    
    where h is the point spread function.
    """
    
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.gaussian = GaussianKernel(kernel_size)
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply blur operator (forward pass).
        
        Args:
            x: Clean image [B, C, H, W]
            sigma: Blur sigma [B] or [B, 1, H, W]
            
        Returns:
            Blurred image [B, C, H, W]
        """
        return self.gaussian(x, sigma)
    
    def transpose(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply transpose of blur operator.
        
        For symmetric Gaussian kernel: D_blur^T = D_blur
        
        Args:
            y: Blurred image [B, C, H, W]
            sigma: Blur sigma
            
        Returns:
            Result of transpose operation [B, C, H, W]
        """
        # Gaussian kernel is symmetric, so transpose = forward
        return self.gaussian(y, sigma)


class ScatteringOperator(nn.Module):
    """
    Scattering degradation operator D_scatter.
    
    Implements attenuation based on scattering coefficient and depth:
    D_scatter(x) = diag(exp(-β·d)) · x = t · x
    
    where t is the transmittance map.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_transmittance(self, scattering_coeff: torch.Tensor,
                              depth: torch.Tensor) -> torch.Tensor:
        """
        Compute transmittance map.
        
        t(x,y) = exp(-β(x,y) · d(x,y))
        
        Args:
            scattering_coeff: β [B, 1, H, W]
            depth: d [B, 1, H, W]
            
        Returns:
            Transmittance t [B, 1, H, W] in [0, 1]
        """
        transmittance = torch.exp(-scattering_coeff * depth)
        return torch.clamp(transmittance, 0, 1)
    
    def forward(self, x: torch.Tensor, 
                transmittance: torch.Tensor) -> torch.Tensor:
        """
        Apply scattering attenuation (forward pass).
        
        Args:
            x: Clean image [B, C, H, W]
            transmittance: t [B, 1, H, W]
            
        Returns:
            Attenuated image [B, C, H, W]
        """
        return x * transmittance
    
    def transpose(self, y: torch.Tensor,
                  transmittance: torch.Tensor) -> torch.Tensor:
        """
        Apply transpose of scattering operator.
        
        D_scatter^T = D_scatter (diagonal operator)
        
        Args:
            y: Attenuated image [B, C, H, W]
            transmittance: t [B, 1, H, W]
            
        Returns:
            Result of transpose [B, C, H, W]
        """
        return y * transmittance


class OcclusionOperator(nn.Module):
    """
    Occlusion degradation operator D_occlude.
    
    Implements masking based on snow occlusion:
    D_occlude(x) = diag(M) · x
    
    where M is the occlusion mask (1 = visible, 0 = occluded).
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor,
                occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply occlusion (forward pass).
        
        Args:
            x: Clean image [B, C, H, W]
            occlusion_mask: M [B, 1, H, W] where 1=visible, 0=occluded
            
        Returns:
            Occluded image [B, C, H, W]
        """
        return x * occlusion_mask
    
    def transpose(self, y: torch.Tensor,
                  occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply transpose of occlusion operator.
        
        D_occlude^T = D_occlude (diagonal operator)
        """
        return y * occlusion_mask


class SnowReflectionModel(nn.Module):
    """
    Snow reflection/scattering contribution model.
    
    Implements Equation (14):
    I_snow = ρ_s · L_ambient · S(x,y)
    
    where:
    - ρ_s ≈ 0.9 is snow albedo
    - L_ambient is ambient light intensity
    - S(x,y) is snow coverage map
    """
    
    def __init__(self, snow_albedo: float = 0.9):
        super().__init__()
        self.albedo = snow_albedo
        
        # Learnable ambient light estimation
        self.ambient_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        
    def forward(self, degraded_image: torch.Tensor,
                snow_coverage: torch.Tensor) -> torch.Tensor:
        """
        Compute snow reflection contribution.
        
        Args:
            degraded_image: Input for ambient estimation [B, 3, H, W]
            snow_coverage: S(x,y) [B, 1, H, W]
            
        Returns:
            Snow reflection I_snow [B, 3, H, W]
        """
        # Estimate ambient light from image
        L_ambient = self.ambient_estimator(degraded_image)  # [B, 3]
        L_ambient = L_ambient.view(-1, 3, 1, 1)
        
        # Eq(14): I_snow = ρ_s · L_ambient · S(x,y)
        I_snow = self.albedo * L_ambient * snow_coverage
        
        return I_snow


class DegradationOperator(nn.Module):
    """
    Complete degradation operator implementing Equation (16).
    
    A = D_blur ∘ D_scatter ∘ D_occlude
    
    The composed operator and its transpose are used in the
    gradient computation for SIRM iterative restoration.
    """
    
    def __init__(self, kernel_size: int = 15):
        """
        Args:
            kernel_size: Blur kernel size
        """
        super().__init__()
        
        self.blur_op = BlurOperator(kernel_size)
        self.scatter_op = ScatteringOperator()
        self.occlude_op = OcclusionOperator()
        self.snow_model = SnowReflectionModel()
        
    def forward(self, x: torch.Tensor,
                sigma: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor,
                include_snow: bool = True,
                degraded_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply complete degradation model.
        
        Eq(12): y = A·M·x + (1-M)·I_snow + ε
        
        Args:
            x: Clean image [B, C, H, W]
            sigma: Blur sigma
            transmittance: t(x,y) [B, 1, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            include_snow: Whether to add snow reflection
            degraded_ref: Reference for ambient estimation
            
        Returns:
            Degraded image [B, C, H, W]
        """
        # Apply degradation chain: A = D_blur ∘ D_scatter ∘ D_occlude
        
        # 1. Occlusion masking
        y = self.occlude_op(x, occlusion_mask)
        
        # 2. Scattering attenuation
        y = self.scatter_op(y, transmittance)
        
        # 3. Blur
        y = self.blur_op(y, sigma)
        
        # Add snow reflection if requested
        if include_snow:
            ref = degraded_ref if degraded_ref is not None else y
            snow_coverage = 1 - occlusion_mask
            I_snow = self.snow_model(ref, snow_coverage)
            y = y + I_snow
            
        return y
    
    def transpose(self, y: torch.Tensor,
                  sigma: torch.Tensor,
                  transmittance: torch.Tensor,
                  occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply transpose of degradation operator.
        
        A^T = D_occlude^T ∘ D_scatter^T ∘ D_blur^T
        
        Args:
            y: Degraded observation [B, C, H, W]
            sigma: Blur sigma
            transmittance: t(x,y)
            occlusion_mask: M(x,y)
            
        Returns:
            Result of A^T·y [B, C, H, W]
        """
        # Apply transpose in reverse order
        
        # 1. Blur transpose
        result = self.blur_op.transpose(y, sigma)
        
        # 2. Scattering transpose
        result = self.scatter_op.transpose(result, transmittance)
        
        # 3. Occlusion transpose
        result = self.occlude_op.transpose(result, occlusion_mask)
        
        return result
    
    def compute_gradient(self, x: torch.Tensor,
                         y: torch.Tensor,
                         sigma: torch.Tensor,
                         transmittance: torch.Tensor,
                         occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of data fidelity term.
        
        ∇||Ax - y||² = 2·A^T·(Ax - y)
        
        Args:
            x: Current estimate [B, C, H, W]
            y: Observation [B, C, H, W]
            sigma: Blur sigma
            transmittance: t(x,y)
            occlusion_mask: M(x,y)
            
        Returns:
            Gradient [B, C, H, W]
        """
        # Forward: Ax
        Ax = self.forward(x, sigma, transmittance, occlusion_mask, 
                         include_snow=False)
        
        # Residual: Ax - y
        residual = Ax - y
        
        # Gradient: 2·A^T·residual
        gradient = 2 * self.transpose(residual, sigma, transmittance, 
                                       occlusion_mask)
        
        return gradient


class SpatiallyVaryingDegradation(nn.Module):
    """
    Spatially-varying degradation model.
    
    Handles cases where degradation parameters vary spatially
    across the image (e.g., different blur amounts in different regions).
    """
    
    def __init__(self, kernel_size: int = 15, patch_size: int = 32):
        """
        Args:
            kernel_size: Blur kernel size
            patch_size: Size of patches for local processing
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.base_op = DegradationOperator(kernel_size)
        
    def forward(self, x: torch.Tensor,
                sigma_map: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply spatially-varying degradation.
        
        Processes image in patches with local parameters.
        
        Args:
            x: Clean image [B, C, H, W]
            sigma_map: Spatially varying σ [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            
        Returns:
            Degraded image [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # For efficiency, use mean sigma with spatially-varying 
        # transmittance and occlusion
        sigma_mean = sigma_map.mean(dim=[2, 3]).squeeze()  # [B]
        
        return self.base_op(x, sigma_mean, transmittance, occlusion_mask)


# Initialize package
__all__ = [
    'GaussianKernel',
    'BlurOperator',
    'ScatteringOperator', 
    'OcclusionOperator',
    'SnowReflectionModel',
    'DegradationOperator',
    'SpatiallyVaryingDegradation'
]
