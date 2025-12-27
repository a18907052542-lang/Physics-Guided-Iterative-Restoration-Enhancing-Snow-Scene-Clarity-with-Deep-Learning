"""
Physics-Aware Weighting Module for PAID-SnowNet
Implements Equations (21)-(23) from the paper

This module provides physics-aware adaptive step sizes for the
iterative restoration process, accounting for:
- Local snow occlusion severity
- Transmittance variations
- Gradient magnitude history (AdaGrad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PhysicsAwareWeight(nn.Module):
    """
    Physics-aware weighting function.
    
    Implements Equation (23):
    w(x,y) = α_w·M(x,y) + β_w·t(x,y) + γ_w
    
    where:
    - M(x,y) is the occlusion mask (more occlusion → smaller weight)
    - t(x,y) is transmittance (lower transmittance → smaller weight)
    - α_w, β_w, γ_w are learnable parameters
    
    This ensures heavily degraded regions receive smaller step sizes
    to prevent over-correction.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            alpha: Weight for occlusion mask term
            beta: Weight for transmittance term
            gamma: Constant baseline weight
        """
        super().__init__()
        
        # Learnable parameters initialized with defaults
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
    def forward(self, occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-aware weight map.
        
        Eq(23): w(x,y) = α_w·M(x,y) + β_w·t(x,y) + γ_w
        
        Args:
            occlusion_mask: M(x,y) [B, 1, H, W] in [0, 1]
            transmittance: t(x,y) [B, 1, H, W] in [0, 1]
            
        Returns:
            Weight map w(x,y) [B, 1, H, W]
        """
        # Ensure positive weights through absolute values
        alpha = self.alpha.abs()
        beta = self.beta.abs()
        gamma = self.gamma.abs()
        
        # Eq(23)
        w = alpha * occlusion_mask + beta * transmittance + gamma
        
        # Normalize to reasonable range [0.1, 1.0]
        w = torch.clamp(w, 0.1, 1.0)
        
        return w


class AdaGradOptimizer(nn.Module):
    """
    AdaGrad optimizer with physics-aware weighting.
    
    Implements Equations (21) and (22):
    
    Eq(21): G_k = G_{k-1} + g_k²
    Eq(22): η_k(x,y) = η_0 / √(G_k + ε) · w(x,y)
    
    AdaGrad adapts learning rates based on gradient history,
    providing larger updates for infrequently updated parameters.
    
    Table 5 shows AdaGrad achieves 30.85dB with 3 iterations,
    outperforming SGD (29.89dB, 6 iter) and Adam (30.56dB, 4 iter).
    """
    
    def __init__(self, eta_0: float = 0.1, epsilon: float = 1e-8):
        """
        Args:
            eta_0: Initial learning rate
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.eta_0 = nn.Parameter(torch.tensor(eta_0))
        self.epsilon = epsilon
        
        # Physics-aware weighting
        self.physics_weight = PhysicsAwareWeight()
        
        # Accumulated gradient (not a parameter, reset each forward pass)
        self.register_buffer('G_accumulated', None)
        
    def reset_state(self, shape: Tuple[int, ...], device: torch.device):
        """
        Reset accumulated gradient state.
        
        Args:
            shape: Shape of gradient tensor
            device: Device for tensor allocation
        """
        self.G_accumulated = torch.zeros(shape, device=device)
        
    def forward(self, gradient: torch.Tensor,
                occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor,
                iteration: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive step size and apply update.
        
        Args:
            gradient: Current gradient g_k [B, C, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            iteration: Current iteration number
            
        Returns:
            Tuple of:
                - Adaptive step size η_k [B, 1, H, W]
                - Scaled gradient η_k · g_k [B, C, H, W]
        """
        # Initialize accumulated gradient if needed
        if self.G_accumulated is None or \
           self.G_accumulated.shape != gradient.shape:
            self.reset_state(gradient.shape, gradient.device)
        
        # Eq(21): Accumulate squared gradient
        # Average over channels for spatially-varying rate
        g_squared = (gradient ** 2).mean(dim=1, keepdim=True)
        self.G_accumulated = self.G_accumulated.mean(dim=1, keepdim=True) + g_squared
        
        # Physics-aware weight: Eq(23)
        w = self.physics_weight(occlusion_mask, transmittance)
        
        # Eq(22): Adaptive step size
        eta = self.eta_0.abs() / torch.sqrt(self.G_accumulated + self.epsilon) * w
        
        # Scale gradient
        scaled_gradient = gradient * eta
        
        return eta, scaled_gradient


class MomentumAdaGrad(nn.Module):
    """
    AdaGrad with momentum for faster convergence.
    
    Combines AdaGrad's adaptive learning rates with
    momentum for accelerated optimization.
    """
    
    def __init__(self, eta_0: float = 0.1, momentum: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Args:
            eta_0: Initial learning rate
            momentum: Momentum coefficient
            epsilon: Numerical stability constant
        """
        super().__init__()
        
        self.eta_0 = nn.Parameter(torch.tensor(eta_0))
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.physics_weight = PhysicsAwareWeight()
        
        self.register_buffer('G_accumulated', None)
        self.register_buffer('velocity', None)
        
    def reset_state(self, shape: Tuple[int, ...], device: torch.device):
        """Reset optimizer state."""
        self.G_accumulated = torch.zeros(shape, device=device)
        self.velocity = torch.zeros(shape, device=device)
        
    def forward(self, gradient: torch.Tensor,
                occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute momentum-accelerated AdaGrad update.
        
        Args:
            gradient: Current gradient [B, C, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            
        Returns:
            Tuple of (step_size, update direction)
        """
        if self.G_accumulated is None or \
           self.G_accumulated.shape != gradient.shape:
            self.reset_state(gradient.shape, gradient.device)
        
        # Accumulate squared gradient
        g_squared = (gradient ** 2).mean(dim=1, keepdim=True)
        self.G_accumulated = self.G_accumulated.mean(dim=1, keepdim=True) + g_squared
        
        # Physics-aware weight
        w = self.physics_weight(occlusion_mask, transmittance)
        
        # Adaptive step size
        eta = self.eta_0.abs() / torch.sqrt(self.G_accumulated + self.epsilon) * w
        
        # Momentum update
        self.velocity = self.momentum * self.velocity + eta * gradient
        
        return eta, self.velocity


class SpatiallyVaryingStepSize(nn.Module):
    """
    Learnable spatially-varying step size predictor.
    
    Uses a small CNN to predict optimal step sizes
    based on local image features.
    """
    
    def __init__(self, in_channels: int = 64, min_step: float = 0.01,
                 max_step: float = 0.5):
        """
        Args:
            in_channels: Number of feature channels
            min_step: Minimum step size
            max_step: Maximum step size
        """
        super().__init__()
        
        self.min_step = min_step
        self.max_step = max_step
        
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, features: torch.Tensor,
                occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor) -> torch.Tensor:
        """
        Predict spatially-varying step size.
        
        Args:
            features: Feature maps [B, C, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            
        Returns:
            Step size map [B, 1, H, W]
        """
        # Concatenate features with physics parameters
        combined = torch.cat([features, occlusion_mask, transmittance], dim=1)
        
        # Predict normalized step size
        step_normalized = self.predictor(combined)
        
        # Scale to [min_step, max_step]
        step_size = self.min_step + (self.max_step - self.min_step) * step_normalized
        
        return step_size


class IterationAwareStepSize(nn.Module):
    """
    Step size that adapts based on iteration number.
    
    Uses a schedule that starts larger and decreases,
    combined with physics-aware spatial weighting.
    """
    
    def __init__(self, initial_step: float = 0.2, decay_rate: float = 0.5,
                 min_step: float = 0.01):
        """
        Args:
            initial_step: Step size at iteration 0
            decay_rate: Decay factor per iteration
            min_step: Minimum step size
        """
        super().__init__()
        
        self.initial_step = nn.Parameter(torch.tensor(initial_step))
        self.decay_rate = decay_rate
        self.min_step = min_step
        
        self.physics_weight = PhysicsAwareWeight()
        
    def forward(self, iteration: int,
                occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor) -> torch.Tensor:
        """
        Compute iteration-aware step size.
        
        η_t = max(η_0 · γ^t, η_min) · w(x,y)
        
        Args:
            iteration: Current iteration number
            occlusion_mask: M(x,y) [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            
        Returns:
            Step size map [B, 1, H, W]
        """
        # Base step size with decay
        base_step = self.initial_step.abs() * (self.decay_rate ** iteration)
        base_step = max(base_step.item(), self.min_step)
        
        # Physics-aware modulation
        w = self.physics_weight(occlusion_mask, transmittance)
        
        return base_step * w


class GradientNormalization(nn.Module):
    """
    Gradient normalization for stable updates.
    
    Normalizes gradients to unit norm while preserving
    direction, scaled by physics-aware weights.
    """
    
    def __init__(self, clip_value: float = 1.0):
        """
        Args:
            clip_value: Maximum gradient norm
        """
        super().__init__()
        self.clip_value = clip_value
        self.physics_weight = PhysicsAwareWeight()
        
    def forward(self, gradient: torch.Tensor,
                occlusion_mask: torch.Tensor,
                transmittance: torch.Tensor,
                step_size: float = 0.1) -> torch.Tensor:
        """
        Normalize and scale gradient.
        
        Args:
            gradient: Raw gradient [B, C, H, W]
            occlusion_mask: M(x,y) [B, 1, H, W]
            transmittance: t(x,y) [B, 1, H, W]
            step_size: Base step size
            
        Returns:
            Normalized, scaled gradient [B, C, H, W]
        """
        # Compute gradient norm per sample
        norm = gradient.norm(p=2, dim=[1, 2, 3], keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        
        # Clip norm
        scale = torch.clamp(self.clip_value / norm, max=1.0)
        normalized = gradient * scale
        
        # Physics-aware weighting
        w = self.physics_weight(occlusion_mask, transmittance)
        
        return step_size * w * normalized


# Initialize package
__all__ = [
    'PhysicsAwareWeight',
    'AdaGradOptimizer',
    'MomentumAdaGrad',
    'SpatiallyVaryingStepSize',
    'IterationAwareStepSize',
    'GradientNormalization'
]
