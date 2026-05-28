"""
Degradation operator decomposition for SIRM optimization.
A = D_blur ∘ D_scatter ∘ D_occlude  (Eq. 16)
File: models/degradation_ops.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_gaussian_kernel(sigma, kernel_size=None):
    """
    Create 2D Gaussian PSF kernel.
    h(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
    
    Args:
        sigma: (B, 1) or scalar blur parameter
        kernel_size: int, auto-computed if None
    Returns:
        kernel: (B, 1, K, K) normalized Gaussian kernel
    """
    if kernel_size is None:
        kernel_size = max(3, int(6 * sigma.max().item()) | 1)  # Ensure odd
        kernel_size = min(kernel_size, 31)
    
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=sigma.device) - half
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    grid = grid_x ** 2 + grid_y ** 2  # (K, K)
    
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    
    B = sigma.shape[0]
    grid = grid.unsqueeze(0).expand(B, -1, -1)  # (B, K, K)
    sigma_sq = (sigma.view(B, 1, 1) ** 2).clamp(min=1e-6)
    
    kernel = torch.exp(-grid / (2 * sigma_sq))
    kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)  # Normalize (Eq. 13)
    kernel = kernel.unsqueeze(1)  # (B, 1, K, K)
    
    return kernel


class DegradationOcclude(nn.Module):
    """
    Occlusion operation: D_occlude = diag(M(x,y))
    Applies soft occlusion mask to image.
    """
    
    def forward(self, x, occlusion_mask):
        """
        Args:
            x: (B, C, H, W) clean image
            occlusion_mask: (B, 1, H, W) soft mask M(x,y) in [0,1]
        Returns:
            occluded: (B, C, H, W)
        """
        return x * occlusion_mask


class DegradationScatter(nn.Module):
    """
    Scattering attenuation: D_scatter = diag(exp(-beta * d))
    """
    
    def forward(self, x, scattering_coeff, depth_map):
        """
        Args:
            x: (B, C, H, W)
            scattering_coeff: (B, 1, H, W) beta(x,y)
            depth_map: (B, 1, H, W) d(x,y)
        Returns:
            scattered: (B, C, H, W)
        """
        transmittance = torch.exp(-scattering_coeff * depth_map)
        return x * transmittance


class DegradationBlur(nn.Module):
    """
    Blur operation via convolution with estimated PSF (Eq. 10).
    I_observed = h * I_scene + n
    """
    
    def forward(self, x, psf_sigma):
        """
        Args:
            x: (B, C, H, W)
            psf_sigma: (B, 1) blur parameter
        Returns:
            blurred: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Use mean sigma across batch for efficiency
        mean_sigma = psf_sigma.mean()
        kernel = create_gaussian_kernel(mean_sigma.unsqueeze(0), kernel_size=11)
        # kernel: (1, 1, K, K)
        K = kernel.shape[-1]
        pad = K // 2
        
        # Expand kernel for depthwise conv across C channels
        kernel_c = kernel.squeeze(0).expand(C, 1, K, K)  # (C, 1, K, K)
        
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        blurred = F.conv2d(x_padded, kernel_c, groups=C)
        
        return blurred


class DegradationOperator(nn.Module):
    """
    Full degradation operator A = D_blur ∘ D_scatter ∘ D_occlude (Eq. 16).
    
    Also computes A^T(AX - Y) for gradient computation (Eq. 19).
    """
    
    def __init__(self):
        super(DegradationOperator, self).__init__()
        self.occlude = DegradationOcclude()
        self.scatter = DegradationScatter()
        self.blur = DegradationBlur()
    
    def forward(self, x, physics_params):
        """
        Apply degradation: y = A(x)
        
        Args:
            x: (B, C, H, W) clean image estimate
            physics_params: dict from SPM module
        Returns:
            degraded: (B, C, H, W)
        """
        M = physics_params['occlusion_mask']
        beta = physics_params['scattering_coeff']
        depth = physics_params['depth_map']
        sigma = physics_params['psf_sigma']
        
        h = self.occlude(x, M)
        h = self.scatter(h, beta, depth)
        h = self.blur(h, sigma)
        
        return h
    
    def compute_gradient(self, x, y, physics_params, lambda1=0.01, lambda2=0.001):
        """
        Compute gradient g_k (Equation 19).
        
        g_k = 2 * A^T(AX_k - Y) + lambda1 * grad_R_sparse + lambda2 * grad_R_smooth
        
        Args:
            x: (B, C, H, W) current estimate X_k
            y: (B, C, H, W) observed degraded image Y
            physics_params: dict from SPM
            lambda1: sparsity regularization weight (0.01)
            lambda2: smoothness regularization weight (0.001)
        Returns:
            gradient: (B, C, H, W)
        """
        # Data fidelity gradient: 2*A^T(AX - Y) 
        # Approximate A^T ≈ A for computational efficiency (common in unrolled networks)
        Ax = self.forward(x, physics_params)
        residual = Ax - y
        data_grad = 2.0 * residual  # Simplified A^T approximation
        
        # Sparsity regularization gradient (Eq. 17)
        # R_sparse = ||grad_X||_1
        # Subgradient: -div(sign(grad_X))
        grad_sparse = self._sparsity_gradient(x)
        
        # Smoothness regularization gradient (Eq. 18)
        # R_smooth with Charbonnier penalty
        grad_smooth = self._smoothness_gradient(x)
        
        gradient = data_grad + lambda1 * grad_sparse + lambda2 * grad_smooth
        
        return gradient
    
    def _sparsity_gradient(self, x, eps=1e-6):
        """
        Subgradient of L1 total variation: -div(sign(nabla X))
        """
        # Compute spatial gradients
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # Sign (subgradient of L1)
        sign_dx = torch.sign(dx)
        sign_dy = torch.sign(dy)
        
        # Negative divergence
        grad = torch.zeros_like(x)
        grad[:, :, :, :-1] -= sign_dx
        grad[:, :, :, 1:] += sign_dx
        grad[:, :, :-1, :] -= sign_dy
        grad[:, :, 1:, :] += sign_dy
        
        return grad
    
    def _smoothness_gradient(self, x, eps=1e-3):
        """
        Gradient of smoothness regularization with Charbonnier penalty (Eq. 18).
        phi(s) = sqrt(s^2 + epsilon^2)
        grad = -div(nabla_X / sqrt(|nabla_X|^2 + eps^2))
        """
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # Pad to original size
        dx_pad = F.pad(dx, [0, 1, 0, 0])
        dy_pad = F.pad(dy, [0, 0, 0, 1])
        
        grad_mag_sq = dx_pad ** 2 + dy_pad ** 2
        norm_factor = torch.sqrt(grad_mag_sq + eps ** 2)
        
        nx = dx_pad / norm_factor
        ny = dy_pad / norm_factor
        
        # Negative divergence
        div_x = torch.zeros_like(x)
        div_x[:, :, :, :-1] -= nx[:, :, :, :-1]
        div_x[:, :, :, 1:] += nx[:, :, :, :-1]
        
        div_y = torch.zeros_like(x)
        div_y[:, :, :-1, :] -= ny[:, :, :-1, :]
        div_y[:, :, 1:, :] += ny[:, :, :-1, :]
        
        return -(div_x + div_y)
