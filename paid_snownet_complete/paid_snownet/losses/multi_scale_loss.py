"""
Multi-Scale Loss Functions for PAID-SnowNet
Implements Equations (29)-(31) from the paper

This module provides training losses:
- Reconstruction loss (L1, L2, Charbonnier)
- Perceptual loss (VGG-19 features)
- Edge loss (gradient consistency)
- SSIM loss (structural similarity)
- Multi-scale supervision (weighted combination)

Multi-scale loss from Eq(29):
L_total = Σ w_t [L_rec(X^t, X_gt) + α·L_perc + β·L_edge]

Iteration weights from Eq(30):
w_t = 0.5^{T-t}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1).
    
    L = √(x² + ε²)
    
    More robust to outliers than L2, smoother than L1.
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.eps = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-19 features.
    
    L_perc = Σ ||φ_l(X) - φ_l(X_gt)||²
    
    where φ_l extracts features from layer l of VGG-19.
    
    Uses features from:
    - relu1_2: Low-level textures
    - relu2_2: Edges and patterns
    - relu3_4: Object parts
    - relu4_4: High-level semantics
    """
    
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None,
                 use_input_norm: bool = True):
        """
        Args:
            layer_weights: Weights for each VGG layer
            use_input_norm: Normalize input to VGG range
        """
        super().__init__()
        
        # Default layer weights
        if layer_weights is None:
            layer_weights = {
                'relu1_2': 0.1,
                'relu2_2': 0.2,
                'relu3_4': 0.4,
                'relu4_4': 0.3
            }
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        
        # Load pretrained VGG-19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract layers up to relu4_4
        self.slice1 = nn.Sequential()  # -> relu1_2
        self.slice2 = nn.Sequential()  # -> relu2_2
        self.slice3 = nn.Sequential()  # -> relu3_4
        self.slice4 = nn.Sequential()  # -> relu4_4
        
        # VGG layer indices
        # relu1_2: index 4
        # relu2_2: index 9
        # relu3_4: index 18
        # relu4_4: index 27
        
        for i in range(5):
            self.slice1.add_module(str(i), vgg[i])
        for i in range(5, 10):
            self.slice2.add_module(str(i), vgg[i])
        for i in range(10, 19):
            self.slice3.add_module(str(i), vgg[i])
        for i in range(19, 28):
            self.slice4.add_module(str(i), vgg[i])
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer('mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to VGG range."""
        if self.use_input_norm:
            return (x - self.mean) / self.std
        return x
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            
        Returns:
            Scalar perceptual loss
        """
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_relu1 = self.slice1(pred_norm)
        pred_relu2 = self.slice2(pred_relu1)
        pred_relu3 = self.slice3(pred_relu2)
        pred_relu4 = self.slice4(pred_relu3)
        
        with torch.no_grad():
            target_relu1 = self.slice1(target_norm)
            target_relu2 = self.slice2(target_relu1)
            target_relu3 = self.slice3(target_relu2)
            target_relu4 = self.slice4(target_relu3)
        
        # Compute loss for each layer
        loss = 0
        
        if 'relu1_2' in self.layer_weights:
            loss += self.layer_weights['relu1_2'] * \
                    F.mse_loss(pred_relu1, target_relu1)
        if 'relu2_2' in self.layer_weights:
            loss += self.layer_weights['relu2_2'] * \
                    F.mse_loss(pred_relu2, target_relu2)
        if 'relu3_4' in self.layer_weights:
            loss += self.layer_weights['relu3_4'] * \
                    F.mse_loss(pred_relu3, target_relu3)
        if 'relu4_4' in self.layer_weights:
            loss += self.layer_weights['relu4_4'] * \
                    F.mse_loss(pred_relu4, target_relu4)
                    
        return loss


class EdgeLoss(nn.Module):
    """
    Edge preservation loss using gradient consistency.
    
    L_edge = ||∇X - ∇X_gt||_1
    
    Ensures edges in restored image match ground truth.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients."""
        B, C, H, W = x.shape
        
        # Expand for all channels
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        return grad_x, grad_y
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth [B, C, H, W]
            
        Returns:
            Scalar edge loss
        """
        pred_grad_x, pred_grad_y = self.compute_gradient(pred)
        target_grad_x, target_grad_y = self.compute_gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    
    SSIM measures structural similarity considering:
    - Luminance
    - Contrast
    - Structure
    
    Loss = 1 - SSIM (to minimize)
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
        """
        super().__init__()
        self.window_size = window_size
        
        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('window', window)
        
        # SSIM constants
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth [B, C, H, W]
            
        Returns:
            Scalar SSIM loss (1 - SSIM)
        """
        B, C, H, W = pred.shape
        
        # Expand window for all channels
        window = self.window.repeat(C, 1, 1, 1)
        padding = self.window_size // 2
        
        # Compute means
        mu_pred = F.conv2d(pred, window, padding=padding, groups=C)
        mu_target = F.conv2d(target, window, padding=padding, groups=C)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute variances and covariance
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=padding, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=padding, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=C) - mu_pred_target
        
        # SSIM formula
        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        
        return 1 - ssim.mean()


class ColorConsistencyLoss(nn.Module):
    """
    Color consistency loss to prevent color shifts.
    
    Computes loss in LAB color space for perceptually
    uniform color differences.
    """
    
    def __init__(self):
        super().__init__()
        
    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space (simplified)."""
        # Convert to XYZ first
        # Simplified linear conversion
        r, g, b = rgb.chunk(3, dim=1)
        
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        
        # Normalize by reference white
        x = x / 0.950456
        z = z / 1.088754
        
        # Convert to LAB
        def f(t):
            delta = 6/29
            return torch.where(t > delta**3,
                              t ** (1/3),
                              t / (3 * delta**2) + 4/29)
        
        L = 116 * f(y) - 16
        a = 500 * (f(x) - f(y))
        b = 200 * (f(y) - f(z))
        
        return torch.cat([L, a, b], dim=1)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute color consistency loss.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            
        Returns:
            Scalar color loss
        """
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        return F.l1_loss(pred_lab, target_lab)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale supervision loss.
    
    Implements Equations (29) and (30):
    
    Eq(29): L_total = Σ w_t [L_rec(X^t, X_gt) + α·L_perc + β·L_edge]
    Eq(30): w_t = 0.5^{T-t}
    
    Applies supervision at each iteration with decreasing weights
    for earlier iterations (final output gets highest weight).
    """
    
    def __init__(self, 
                 num_iterations: int = 3,
                 alpha: float = 0.5,
                 beta: float = 0.1,
                 use_ssim: bool = True,
                 use_color: bool = True):
        """
        Args:
            num_iterations: Number of SIRM iterations T
            alpha: Weight for perceptual loss
            beta: Weight for edge loss
            use_ssim: Include SSIM loss
            use_color: Include color consistency loss
        """
        super().__init__()
        
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        
        # Component losses
        self.reconstruction = CharbonnierLoss()
        self.perceptual = VGGPerceptualLoss()
        self.edge = EdgeLoss()
        
        if use_ssim:
            self.ssim = SSIMLoss()
        else:
            self.ssim = None
            
        if use_color:
            self.color = ColorConsistencyLoss()
        else:
            self.color = None
            
        # Compute iteration weights: w_t = 0.5^{T-t}
        weights = torch.tensor([0.5 ** (num_iterations - t - 1) 
                               for t in range(num_iterations)])
        weights = weights / weights.sum()  # Normalize
        self.register_buffer('iteration_weights', weights)
        
    def forward(self, 
                intermediates: List[torch.Tensor],
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale loss.
        
        Args:
            intermediates: List of T intermediate results [B, 3, H, W] each
            target: Ground truth image [B, 3, H, W]
            
        Returns:
            Dictionary with:
                - 'total': Total weighted loss
                - 'reconstruction': Sum of reconstruction losses
                - 'perceptual': Sum of perceptual losses
                - 'edge': Sum of edge losses
                - 'per_iteration': List of per-iteration losses
        """
        total_loss = 0
        total_rec = 0
        total_perc = 0
        total_edge = 0
        per_iteration = []
        
        for t, pred in enumerate(intermediates):
            w = self.iteration_weights[t] if t < len(self.iteration_weights) else \
                self.iteration_weights[-1]
            
            # Reconstruction loss
            rec_loss = self.reconstruction(pred, target)
            
            # Perceptual loss
            perc_loss = self.perceptual(pred, target)
            
            # Edge loss
            edge_loss = self.edge(pred, target)
            
            # Combined loss for this iteration: Eq(29)
            iter_loss = rec_loss + self.alpha * perc_loss + self.beta * edge_loss
            
            # Add optional losses
            if self.ssim is not None:
                ssim_loss = self.ssim(pred, target)
                iter_loss = iter_loss + 0.1 * ssim_loss
                
            if self.color is not None:
                color_loss = self.color(pred, target)
                iter_loss = iter_loss + 0.05 * color_loss
            
            # Weighted contribution
            total_loss = total_loss + w * iter_loss
            total_rec = total_rec + w * rec_loss
            total_perc = total_perc + w * perc_loss
            total_edge = total_edge + w * edge_loss
            per_iteration.append(iter_loss.item())
            
        return {
            'total': total_loss,
            'reconstruction': total_rec,
            'perceptual': total_perc,
            'edge': total_edge,
            'per_iteration': per_iteration
        }


class SnowDenoiseMetric(nn.Module):
    """
    Comprehensive snow denoising metric from Equation (31).
    
    Combines multiple quality metrics with weights from Table 9:
    - Pixel accuracy (α=0.3)
    - SSIM (β=0.25)
    - Perceptual quality (γ=0.15)
    - Edge preservation (δ=0.2)
    - Color consistency (η=0.1)
    """
    
    def __init__(self):
        super().__init__()
        
        # Weights from Table 9
        self.alpha = 0.3   # Pixel
        self.beta = 0.25   # SSIM
        self.gamma = 0.15  # Perceptual
        self.delta = 0.2   # Edge
        self.eta = 0.1     # Color
        
        self.ssim = SSIMLoss()
        self.perceptual = VGGPerceptualLoss()
        self.edge = EdgeLoss()
        self.color = ColorConsistencyLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive quality metric.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            
        Returns:
            Dictionary with individual and combined scores
        """
        # Pixel accuracy (PSNR-based)
        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        pixel_score = psnr / 50.0  # Normalize to ~[0, 1]
        
        # SSIM (already in [0, 1])
        ssim_val = 1 - self.ssim(pred, target)
        
        # Perceptual (normalize)
        perc_val = 1 - torch.clamp(self.perceptual(pred, target), 0, 1)
        
        # Edge preservation (normalize)
        edge_val = 1 - torch.clamp(self.edge(pred, target) / 0.5, 0, 1)
        
        # Color consistency (normalize)
        color_val = 1 - torch.clamp(self.color(pred, target) / 20, 0, 1)
        
        # Combined score: Eq(31)
        combined = self.alpha * pixel_score + \
                   self.beta * ssim_val + \
                   self.gamma * perc_val + \
                   self.delta * edge_val + \
                   self.eta * color_val
        
        return {
            'combined': combined.item(),
            'psnr': psnr.item(),
            'ssim': ssim_val.item(),
            'perceptual': perc_val.item(),
            'edge': edge_val.item(),
            'color': color_val.item()
        }


# Initialize package
__all__ = [
    'CharbonnierLoss',
    'VGGPerceptualLoss',
    'EdgeLoss',
    'SSIMLoss',
    'ColorConsistencyLoss',
    'MultiScaleLoss',
    'SnowDenoiseMetric'
]
