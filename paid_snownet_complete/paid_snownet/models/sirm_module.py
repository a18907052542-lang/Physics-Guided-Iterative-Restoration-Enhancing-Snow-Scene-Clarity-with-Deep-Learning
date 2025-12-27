"""
Snow Iterative Restoration Module (SIRM) for PAID-SnowNet
Implements Section 3.2 of the paper

The SIRM module performs iterative restoration combining:
- Gradient descent with physics-aware weighting (Equations 19-24)
- U-Net based refinement (Equation 25)
- Multi-scale supervision (Equations 29-30)

Algorithm from Equation (26):
X^{t+1} = U(X^{t} - η^{t}∇L(X^{t}); θ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .degradation_ops import DegradationOperator
from .physics_weight import AdaGradOptimizer, PhysicsAwareWeight


class ConvBlock(nn.Module):
    """Convolutional block with optional batch norm and activation."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 use_bn: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, 1, padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch)
        )
        
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
        
    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                                   align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for iterative refinement.
    
    Implements Equation (25):
    X_{k+1} = U(X_k^{1/2}; θ)
    
    Table 6 shows 4-layer U-Net with 64 base channels achieves
    optimal balance (30.85dB, 45.2M params).
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, num_layers: int = 4,
                 bilinear: bool = True):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output channels
            base_channels: Base feature channels
            num_layers: Number of encoder/decoder layers
            bilinear: Use bilinear upsampling
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_layers - 1):
            out_ch = min(in_ch * 2, 512)
            self.downs.append(Down(in_ch, out_ch))
            in_ch = out_ch
            
        # Decoder
        self.ups = nn.ModuleList()
        for i in range(num_layers - 1):
            out_ch = in_ch // 2
            factor = 2 if bilinear else 1
            self.ups.append(Up(in_ch + in_ch // factor, out_ch, bilinear))
            in_ch = out_ch
            
        # Output convolution
        self.outc = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        U-Net forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Refined output [B, C, H, W]
        """
        # Encoder with skip connections
        skips = []
        x = self.inc(x)
        skips.append(x)
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            
        # Decoder with skip connections
        skips = skips[:-1][::-1]  # Reverse, exclude bottleneck
        
        for up, skip in zip(self.ups, skips):
            x = up(x, skip)
            
        return self.outc(x)


class GradientComputation(nn.Module):
    """
    Gradient computation for iterative optimization.
    
    Implements Equation (19):
    g_k = 2A^T(AX_k - Y) + λ_1∇R_sparse + λ_2∇R_smooth
    
    Regularization terms:
    - R_sparse = ||∇X||_1 (Equation 17): Sparsity for edge preservation
    - R_smooth with Charbonnier penalty φ(s) = √(s + ε²) (Equation 18)
    """
    
    def __init__(self, lambda_sparse: float = 0.01, lambda_smooth: float = 0.001,
                 charbonnier_eps: float = 1e-3):
        """
        Args:
            lambda_sparse: Weight for sparsity regularization
            lambda_smooth: Weight for smoothness regularization
            charbonnier_eps: Charbonnier penalty parameter
        """
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.eps = charbonnier_eps
        
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3) / 4
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3) / 4
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def compute_image_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image gradients using Sobel filters.
        
        Args:
            x: Image tensor [B, C, H, W]
            
        Returns:
            Tuple of (grad_x, grad_y) each [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Expand Sobel filters for all channels
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        return grad_x, grad_y
    
    def sparsity_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of sparsity regularizer.
        
        R_sparse = ||∇X||_1 = Σ|∇_x X| + |∇_y X|
        ∇R_sparse = sign(∇X) convolved with Sobel transpose
        
        Args:
            x: Current estimate [B, C, H, W]
            
        Returns:
            Gradient of sparsity term [B, C, H, W]
        """
        grad_x, grad_y = self.compute_image_gradient(x)
        
        # Subgradient of L1 norm
        sign_x = torch.sign(grad_x)
        sign_y = torch.sign(grad_y)
        
        # Transpose Sobel (flip kernels)
        C = x.shape[1]
        sobel_x_t = torch.flip(self.sobel_x, [2, 3]).repeat(C, 1, 1, 1)
        sobel_y_t = torch.flip(self.sobel_y, [2, 3]).repeat(C, 1, 1, 1)
        
        # Apply transpose
        grad_sparse = F.conv2d(sign_x, sobel_x_t, padding=1, groups=C) + \
                      F.conv2d(sign_y, sobel_y_t, padding=1, groups=C)
        
        return grad_sparse
    
    def smoothness_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of smoothness regularizer with Charbonnier penalty.
        
        Eq(18): R_smooth uses φ(s) = √(s + ε²)
        ∇R_smooth = ∇X / √(|∇X|² + ε²)
        
        Args:
            x: Current estimate [B, C, H, W]
            
        Returns:
            Gradient of smoothness term [B, C, H, W]
        """
        grad_x, grad_y = self.compute_image_gradient(x)
        
        # Charbonnier penalty derivative
        grad_mag_sq = grad_x ** 2 + grad_y ** 2
        weight = 1.0 / torch.sqrt(grad_mag_sq + self.eps ** 2)
        
        weighted_grad_x = weight * grad_x
        weighted_grad_y = weight * grad_y
        
        # Apply transpose
        C = x.shape[1]
        sobel_x_t = torch.flip(self.sobel_x, [2, 3]).repeat(C, 1, 1, 1)
        sobel_y_t = torch.flip(self.sobel_y, [2, 3]).repeat(C, 1, 1, 1)
        
        grad_smooth = F.conv2d(weighted_grad_x, sobel_x_t, padding=1, groups=C) + \
                      F.conv2d(weighted_grad_y, sobel_y_t, padding=1, groups=C)
        
        return grad_smooth
    
    def forward(self, x: torch.Tensor, y: torch.Tensor,
                degradation_op: DegradationOperator,
                sigma: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute full gradient for optimization.
        
        Eq(19): g_k = 2A^T(AX_k - Y) + λ_1∇R_sparse + λ_2∇R_smooth
        
        Args:
            x: Current estimate X_k [B, C, H, W]
            y: Observation Y [B, C, H, W]
            degradation_op: Degradation operator A
            sigma: Blur sigma
            transmittance: t(x,y) [B, 1, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            
        Returns:
            Total gradient [B, C, H, W]
        """
        # Data fidelity gradient: 2A^T(Ax - y)
        data_grad = degradation_op.compute_gradient(
            x, y, sigma, transmittance, occlusion_mask
        )
        
        # Regularization gradients
        sparse_grad = self.sparsity_gradient(x)
        smooth_grad = self.smoothness_gradient(x)
        
        # Eq(19): Combined gradient
        total_grad = data_grad + \
                     self.lambda_sparse * sparse_grad + \
                     self.lambda_smooth * smooth_grad
        
        return total_grad


class SIRMModule(nn.Module):
    """
    Snow Iterative Restoration Module (SIRM).
    
    Complete implementation of Section 3.2, performing iterative
    restoration via Algorithm from Equation (26):
    
    1. Initialize X^{0} = Y (degraded input)
    2. For t = 0 to T-1:
       a. Compute gradient g_t
       b. AdaGrad step size η_t
       c. Physics constraint: X_t^{1/2} = X_t - η_t · g_t (Eq 24)
       d. U-Net refinement: X_{t+1} = U(X_t^{1/2}) (Eq 25)
    3. Return X^{T}
    
    Table 4 shows T=3 iterations is optimal (30.85dB, 138ms).
    """
    
    def __init__(self, num_iterations: int = 3, in_channels: int = 3,
                 base_channels: int = 64, num_layers: int = 4,
                 lambda_sparse: float = 0.01, lambda_smooth: float = 0.001):
        """
        Args:
            num_iterations: Number of restoration iterations T
            in_channels: Image channels
            base_channels: U-Net base channels
            num_layers: U-Net depth
            lambda_sparse: Sparsity regularization weight
            lambda_smooth: Smoothness regularization weight
        """
        super().__init__()
        self.num_iterations = num_iterations
        
        # U-Net for refinement
        self.unet = UNet(in_channels, in_channels, base_channels, num_layers)
        
        # Gradient computation
        self.grad_compute = GradientComputation(lambda_sparse, lambda_smooth)
        
        # AdaGrad optimizer
        self.adagrad = AdaGradOptimizer(eta_0=0.1)
        
        # Degradation operator
        self.degradation_op = DegradationOperator(kernel_size=15)
        
        # Learnable iteration-specific parameters
        self.iter_weights = nn.Parameter(torch.ones(num_iterations))
        
    def forward(self, y: torch.Tensor,
                sigma: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Iterative restoration.
        
        Args:
            y: Degraded observation [B, C, H, W]
            sigma: Blur sigma
            transmittance: t(x,y) [B, 1, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - 'output': Final restored image [B, C, H, W]
                - 'intermediates': List of intermediate results (if requested)
                - 'gradients': List of gradient norms
        """
        # Initialize: X^{0} = Y
        x = y.clone()
        
        intermediates = [x] if return_intermediates else []
        gradient_norms = []
        
        # Reset AdaGrad state
        self.adagrad.reset_state(y.shape, y.device)
        
        # Iterative restoration
        for t in range(self.num_iterations):
            # Compute gradient: Eq(19)
            gradient = self.grad_compute(
                x, y, self.degradation_op, sigma, transmittance, occlusion_mask
            )
            gradient_norms.append(gradient.norm().item())
            
            # AdaGrad step: Eq(21-22)
            step_size, scaled_grad = self.adagrad(
                gradient, occlusion_mask, transmittance, t
            )
            
            # Physics constraint step: Eq(24)
            # X_t^{1/2} = X_t - η_t · g_t
            x_half = x - scaled_grad
            
            # U-Net refinement: Eq(25)
            # X_{t+1} = U(X_t^{1/2}; θ)
            x = self.unet(x_half)
            
            # Clip to valid range
            x = torch.clamp(x, 0, 1)
            
            if return_intermediates:
                intermediates.append(x)
        
        return {
            'output': x,
            'intermediates': intermediates,
            'gradient_norms': gradient_norms
        }
    
    def get_iteration_weights(self) -> torch.Tensor:
        """
        Get normalized iteration weights for multi-scale loss.
        
        Eq(30): w_t = 0.5^{T-t}
        
        Returns:
            Normalized weights [num_iterations]
        """
        T = self.num_iterations
        weights = torch.tensor([0.5 ** (T - t - 1) for t in range(T)])
        return weights / weights.sum()


class LightweightSIRM(nn.Module):
    """
    Lightweight SIRM variant for faster inference.
    
    Uses smaller U-Net and fewer iterations.
    Table 4: T=2 achieves 29.6dB with 92ms inference.
    """
    
    def __init__(self, num_iterations: int = 2, base_channels: int = 32):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Lightweight U-Net
        self.unet = UNet(3, 3, base_channels, num_layers=3)
        
        # Simplified gradient
        self.grad_compute = GradientComputation(
            lambda_sparse=0.005, lambda_smooth=0.001
        )
        
        # Physics weighting
        self.physics_weight = PhysicsAwareWeight()
        
        # Fixed step size schedule
        self.step_sizes = nn.Parameter(torch.tensor([0.2, 0.1]))
        
        self.degradation_op = DegradationOperator(kernel_size=11)
        
    def forward(self, y: torch.Tensor,
                sigma: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Lightweight restoration."""
        x = y.clone()
        
        for t in range(self.num_iterations):
            gradient = self.grad_compute(
                x, y, self.degradation_op, sigma, transmittance, occlusion_mask
            )
            
            # Physics-aware step
            w = self.physics_weight(occlusion_mask, transmittance)
            step = self.step_sizes[t].abs() * w
            
            x_half = x - step * gradient
            x = self.unet(x_half)
            x = torch.clamp(x, 0, 1)
            
        return {'output': x}


class DeepSIRM(nn.Module):
    """
    Deep SIRM variant for maximum quality.
    
    Table 6: 5-layer, 128-channel achieves 30.91dB with 98.7M params.
    """
    
    def __init__(self, num_iterations: int = 4, base_channels: int = 128):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Deep U-Net
        self.unet = UNet(3, 3, base_channels, num_layers=5)
        
        # Full gradient computation
        self.grad_compute = GradientComputation(
            lambda_sparse=0.01, lambda_smooth=0.001
        )
        
        # AdaGrad with momentum
        from .physics_weight import MomentumAdaGrad
        self.optimizer = MomentumAdaGrad(eta_0=0.1, momentum=0.9)
        
        self.degradation_op = DegradationOperator(kernel_size=21)
        
    def forward(self, y: torch.Tensor,
                sigma: torch.Tensor,
                transmittance: torch.Tensor,
                occlusion_mask: torch.Tensor,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """Deep restoration."""
        x = y.clone()
        intermediates = [x] if return_intermediates else []
        
        self.optimizer.reset_state(y.shape, y.device)
        
        for t in range(self.num_iterations):
            gradient = self.grad_compute(
                x, y, self.degradation_op, sigma, transmittance, occlusion_mask
            )
            
            _, update = self.optimizer(gradient, occlusion_mask, transmittance)
            
            x_half = x - update
            x = self.unet(x_half)
            x = torch.clamp(x, 0, 1)
            
            if return_intermediates:
                intermediates.append(x)
                
        return {
            'output': x,
            'intermediates': intermediates
        }


# Initialize package
__all__ = [
    'SIRMModule',
    'LightweightSIRM',
    'DeepSIRM',
    'UNet',
    'GradientComputation'
]
