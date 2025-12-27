"""
PAID-SnowNet: Physics-Aware Iterative Denoising Network
Main model implementation combining SPM and SIRM modules

This is the complete network architecture as described in the paper,
integrating:
- Snow Physics Modeling (SPM) for parameter estimation
- Snow Iterative Restoration Module (SIRM) for image restoration

Model variants:
- Base: 45.2M params, 178ms inference, 30.85dB PSNR
- Lightweight: 18.6M params, 87ms inference, 29.42dB PSNR
- Deep: 78.4M params, 312ms inference, 30.91dB PSNR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .spm_module import SPMModule, LightweightSPM
from .sirm_module import SIRMModule, LightweightSIRM, DeepSIRM


class PAIDSnowNet(nn.Module):
    """
    PAID-SnowNet: Physics-Aware Iterative Denoising Network.
    
    Complete implementation combining SPM for physics parameter estimation
    and SIRM for iterative restoration.
    
    Architecture (Figure 1):
    1. SPM extracts: β(x,y), t(x,y), M(x,y), d(x,y), σ
    2. SIRM performs T iterations of gradient descent + U-Net refinement
    3. Multi-scale supervision at each iteration
    
    Default configuration (Table 7):
    - SPM: 3 branches (3×3, 5×5, 7×7), 64 channels
    - SIRM: 3 iterations, 4-layer U-Net, 64 channels
    - Total: 45.2M parameters, 178ms inference (RTX 4090)
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 base_channels: int = 64,
                 spm_kernel_sizes: List[int] = [3, 5, 7],
                 spm_num_blocks: int = 4,
                 sirm_iterations: int = 3,
                 sirm_num_layers: int = 4,
                 lambda_sparse: float = 0.01,
                 lambda_smooth: float = 0.001):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            base_channels: Base feature channels
            spm_kernel_sizes: Kernel sizes for SPM multi-scale branches
            spm_num_blocks: Number of residual blocks per SPM branch
            sirm_iterations: Number of SIRM iterations
            sirm_num_layers: U-Net depth in SIRM
            lambda_sparse: Sparsity regularization weight
            lambda_smooth: Smoothness regularization weight
        """
        super().__init__()
        
        # Snow Physics Modeling module
        self.spm = SPMModule(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_sizes=spm_kernel_sizes,
            num_blocks=spm_num_blocks
        )
        
        # Snow Iterative Restoration Module
        self.sirm = SIRMModule(
            num_iterations=sirm_iterations,
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=sirm_num_layers,
            lambda_sparse=lambda_sparse,
            lambda_smooth=lambda_smooth
        )
        
        # Global blur sigma estimation (backup if PSF fails)
        self.global_sigma = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, 
                return_physics: bool = False,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PAID-SnowNet.
        
        Args:
            x: Degraded snow image [B, 3, H, W] in range [0, 1]
            return_physics: Whether to return physics parameters
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - 'output': Restored image [B, 3, H, W]
                - 'physics': Physics parameters (if return_physics=True)
                - 'intermediates': Intermediate results (if return_intermediates=True)
        """
        # Stage 1: Physics parameter estimation via SPM
        physics = self.spm(x)
        
        # Extract parameters
        transmittance = physics['transmission']
        occlusion_mask = physics['occlusion_mask']
        
        # Get blur sigma (use global if local estimation is unstable)
        psf_global, psf_local = physics['psf_sigma']
        sigma = psf_global.squeeze()  # [B]
        
        # Ensure sigma is positive and reasonable
        sigma = torch.clamp(sigma, 0.5, 5.0)
        
        # Stage 2: Iterative restoration via SIRM
        sirm_output = self.sirm(
            y=x,
            sigma=sigma,
            transmittance=transmittance,
            occlusion_mask=occlusion_mask,
            return_intermediates=return_intermediates
        )
        
        # Prepare output
        result = {
            'output': sirm_output['output']
        }
        
        if return_physics:
            result['physics'] = physics
            
        if return_intermediates:
            result['intermediates'] = sirm_output['intermediates']
            result['gradient_norms'] = sirm_output.get('gradient_norms', [])
            
        return result
    
    def get_physics_parameters(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get only physics parameters without restoration.
        
        Useful for analysis and visualization.
        
        Args:
            x: Degraded image [B, 3, H, W]
            
        Returns:
            Dictionary with all physics parameters
        """
        return self.spm(x)
    
    def restore_with_custom_params(self, x: torch.Tensor,
                                   transmittance: torch.Tensor,
                                   occlusion_mask: torch.Tensor,
                                   sigma: float = 1.0) -> torch.Tensor:
        """
        Restore using custom physics parameters.
        
        Allows manual control over degradation parameters.
        
        Args:
            x: Degraded image [B, 3, H, W]
            transmittance: Custom transmittance [B, 1, H, W]
            occlusion_mask: Custom occlusion mask [B, 1, H, W]
            sigma: Blur sigma value
            
        Returns:
            Restored image [B, 3, H, W]
        """
        sigma_tensor = torch.tensor([sigma] * x.shape[0], device=x.device)
        
        output = self.sirm(
            y=x,
            sigma=sigma_tensor,
            transmittance=transmittance,
            occlusion_mask=occlusion_mask
        )
        
        return output['output']


class PAIDSnowNetLightweight(nn.Module):
    """
    Lightweight PAID-SnowNet for real-time applications.
    
    Optimized for speed with reduced model size:
    - 18.6M parameters
    - 87ms inference time
    - 29.42dB PSNR
    
    Trade-offs:
    - Fewer channels (32 vs 64)
    - Fewer iterations (2 vs 3)
    - Smaller receptive fields
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        self.spm = LightweightSPM(in_channels, base_channels)
        self.sirm = LightweightSIRM(num_iterations=2, base_channels=base_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Degraded image [B, 3, H, W]
            
        Returns:
            Dictionary with 'output' key
        """
        physics = self.spm(x)
        
        transmittance = physics['transmission']
        occlusion_mask = physics['occlusion_mask']
        sigma = torch.ones(x.shape[0], device=x.device)  # Fixed sigma
        
        output = self.sirm(x, sigma, transmittance, occlusion_mask)
        
        return {'output': output['output']}


class PAIDSnowNetDeep(nn.Module):
    """
    Deep PAID-SnowNet for maximum quality.
    
    Maximized capacity for best restoration quality:
    - 78.4M parameters
    - 312ms inference time  
    - 30.91dB PSNR
    
    Enhancements:
    - More channels (128 vs 64)
    - More iterations (4 vs 3)
    - Deeper U-Net (5 layers)
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 128):
        super().__init__()
        
        self.spm = SPMModule(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_sizes=[3, 5, 7, 9],  # Additional scale
            num_blocks=6
        )
        
        self.sirm = DeepSIRM(num_iterations=4, base_channels=base_channels)
        
    def forward(self, x: torch.Tensor,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Degraded image [B, 3, H, W]
            return_intermediates: Return intermediate results
            
        Returns:
            Dictionary with restoration results
        """
        physics = self.spm(x)
        
        transmittance = physics['transmission']
        occlusion_mask = physics['occlusion_mask']
        psf_global, _ = physics['psf_sigma']
        sigma = torch.clamp(psf_global.squeeze(), 0.5, 5.0)
        
        output = self.sirm(
            x, sigma, transmittance, occlusion_mask,
            return_intermediates=return_intermediates
        )
        
        result = {'output': output['output']}
        
        if return_intermediates:
            result['intermediates'] = output['intermediates']
            result['physics'] = physics
            
        return result


def create_model(model_type: str = 'base', **kwargs) -> nn.Module:
    """
    Factory function for creating PAID-SnowNet models.
    
    Args:
        model_type: One of 'base', 'lightweight', 'deep'
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        PAID-SnowNet model instance
    """
    models = {
        'base': PAIDSnowNet,
        'lightweight': PAIDSnowNetLightweight,
        'deep': PAIDSnowNetDeep
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> Dict[str, any]:
    """
    Get model summary including parameter count and architecture info.
    
    Args:
        model: PAID-SnowNet model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = count_parameters(model)
    
    # Count parameters by module
    module_params = {}
    for name, module in model.named_children():
        module_params[name] = count_parameters(module)
    
    return {
        'total_params': total_params,
        'total_params_m': total_params / 1e6,
        'module_params': module_params
    }


# Initialize package
__all__ = [
    'PAIDSnowNet',
    'PAIDSnowNetLightweight', 
    'PAIDSnowNetDeep',
    'create_model',
    'count_parameters',
    'model_summary'
]
