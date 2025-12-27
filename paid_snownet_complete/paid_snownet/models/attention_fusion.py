"""
Attention Fusion Module for PAID-SnowNet
Implements Equations (8)-(9) from the paper

This module provides adaptive fusion mechanisms for combining
multi-scale features from different branches of the SPM module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AdaptiveFusion(nn.Module):
    """
    Adaptive Feature Fusion using channel attention.
    Implements Equations (8) and (9) from the paper.
    
    Eq(8): [α, β, γ] = Softmax(W_a · GAP([F_1, F_2, F_3]) + b_a)
    Eq(9): F_fused = α·F_1 + β·F_2 + γ·F_3
    
    Learns to adaptively weight features from different scale branches
    based on their global statistics.
    """
    
    def __init__(self, in_channels: int, num_branches: int = 3, 
                 reduction: int = 16):
        """
        Args:
            in_channels: Number of channels per branch
            num_branches: Number of branches to fuse (default: 3)
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Attention network: Eq(8)
        # W_a maps concatenated features to branch weights
        hidden_channels = max(in_channels * num_branches // reduction, 8)
        
        self.attention_fc = nn.Sequential(
            nn.Linear(in_channels * num_branches, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_branches),
            # Softmax applied in forward for numerical stability
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-branch features adaptively.
        
        Args:
            features: List of [B, C, H, W] tensors from each branch
            
        Returns:
            Tuple of:
                - Fused features [B, C, H, W]
                - Attention weights [B, num_branches]
        """
        assert len(features) == self.num_branches, \
            f"Expected {self.num_branches} branches, got {len(features)}"
        
        batch_size = features[0].shape[0]
        
        # Stack features [B, num_branches, C, H, W]
        stacked = torch.stack(features, dim=1)
        
        # Global Average Pooling: [B, num_branches, C, 1, 1]
        pooled = self.gap(stacked.view(-1, *stacked.shape[2:]))
        pooled = pooled.view(batch_size, self.num_branches, self.in_channels)
        
        # Concatenate: [B, num_branches * C]
        concat = pooled.view(batch_size, -1)
        
        # Eq(8): Compute attention weights
        weights = self.attention_fc(concat)  # [B, num_branches]
        weights = F.softmax(weights, dim=1)  # Normalize to sum to 1
        
        # Eq(9): Weighted combination
        # weights: [B, num_branches, 1, 1, 1]
        weights_expanded = weights.view(batch_size, self.num_branches, 1, 1, 1)
        
        # F_fused = Σ w_i * F_i
        fused = (stacked * weights_expanded).sum(dim=1)
        
        return fused, weights


class MultiScaleAttentionFusion(nn.Module):
    """
    Multi-scale attention fusion with spatial attention.
    
    Extends AdaptiveFusion with spatially-varying attention weights
    to handle local variations in snow density and particle size.
    """
    
    def __init__(self, in_channels: int, num_branches: int = 3):
        """
        Args:
            in_channels: Number of channels per branch
            num_branches: Number of branches to fuse
        """
        super().__init__()
        self.num_branches = num_branches
        
        # Channel attention (global)
        self.channel_attention = AdaptiveFusion(in_channels, num_branches)
        
        # Spatial attention (local)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_branches, 1),
            # Softmax applied per-pixel
        )
        
        # Fusion weight balancing
        self.balance = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse features with both channel and spatial attention.
        
        Args:
            features: List of [B, C, H, W] tensors
            
        Returns:
            Tuple of:
                - Fused features [B, C, H, W]
                - Combined attention weights [B, num_branches, H, W]
        """
        batch_size, channels, H, W = features[0].shape
        
        # Channel attention: [B, C, H, W], [B, num_branches]
        fused_channel, channel_weights = self.channel_attention(features)
        
        # Spatial attention
        concat = torch.cat(features, dim=1)  # [B, C*num_branches, H, W]
        spatial_weights = self.spatial_attention(concat)  # [B, num_branches, H, W]
        spatial_weights = F.softmax(spatial_weights, dim=1)
        
        # Combine channel and spatial attention
        # Expand channel weights: [B, num_branches, 1, 1]
        channel_weights_exp = channel_weights.view(batch_size, self.num_branches, 1, 1)
        
        # Balanced combination
        balance = torch.sigmoid(self.balance)
        combined_weights = balance * channel_weights_exp + (1 - balance) * spatial_weights
        
        # Apply combined attention
        stacked = torch.stack(features, dim=1)  # [B, num_branches, C, H, W]
        weights_exp = combined_weights.unsqueeze(2)  # [B, num_branches, 1, H, W]
        
        fused = (stacked * weights_exp).sum(dim=1)
        
        return fused, combined_weights


class ScaleAwareAttention(nn.Module):
    """
    Scale-aware attention module that explicitly encodes
    the expected receptive field of each branch.
    
    Uses learnable scale embeddings to guide attention based on
    the physical characteristics of snow at different scales.
    """
    
    def __init__(self, in_channels: int, num_branches: int = 3,
                 kernel_sizes: List[int] = [3, 5, 7]):
        """
        Args:
            in_channels: Number of channels per branch
            num_branches: Number of branches
            kernel_sizes: Kernel sizes for each branch
        """
        super().__init__()
        self.num_branches = num_branches
        self.kernel_sizes = kernel_sizes
        
        # Scale embeddings (learnable)
        # Initialized based on kernel size ratios
        scale_init = torch.tensor([k / max(kernel_sizes) for k in kernel_sizes])
        self.scale_embedding = nn.Parameter(scale_init.view(1, num_branches, 1, 1, 1))
        
        # Feature projection
        self.proj = nn.Conv2d(in_channels, in_channels // 4, 1)
        
        # Attention computation
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels // 4 * num_branches, num_branches, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale-aware feature fusion.
        
        Args:
            features: List of [B, C, H, W] tensors
            
        Returns:
            Tuple of:
                - Fused features [B, C, H, W]
                - Scale-aware weights [B, num_branches, H, W]
        """
        # Project features
        projected = [self.proj(f) for f in features]
        concat = torch.cat(projected, dim=1)
        
        # Compute base attention
        base_attention = self.attention(concat)  # [B, num_branches, H, W]
        
        # Modulate with scale embeddings
        scale_weights = F.softmax(self.scale_embedding.squeeze(-1).squeeze(-1), dim=1)
        attention = base_attention * scale_weights.expand_as(base_attention)
        
        # Normalize
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
        
        # Apply attention
        stacked = torch.stack(features, dim=1)
        weights_exp = attention.unsqueeze(2)
        fused = (stacked * weights_exp).sum(dim=1)
        
        return fused, attention


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for capturing interactions between
    features at different scales.
    
    Models the relationship between local details and global context
    through cross-attention mechanism.
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        """
        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, query_feat: torch.Tensor, 
                key_feat: torch.Tensor,
                value_feat: torch.Tensor) -> torch.Tensor:
        """
        Cross-scale attention.
        
        Args:
            query_feat: Query features [B, C, H, W]
            key_feat: Key features [B, C, H, W]
            value_feat: Value features [B, C, H, W]
            
        Returns:
            Attended features [B, C, H, W]
        """
        B, C, H, W = query_feat.shape
        
        # Project
        Q = self.q_proj(query_feat)  # [B, C, H, W]
        K = self.k_proj(key_feat)
        V = self.v_proj(value_feat)
        
        # Reshape for multi-head attention
        # [B, num_heads, head_dim, H*W]
        Q = Q.view(B, self.num_heads, self.head_dim, -1)
        K = K.view(B, self.num_heads, self.head_dim, -1)
        V = V.view(B, self.num_heads, self.head_dim, -1)
        
        # Attention: [B, num_heads, H*W, H*W]
        attn = torch.matmul(Q.transpose(-2, -1), K) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(V, attn.transpose(-2, -1))  # [B, num_heads, head_dim, H*W]
        out = out.view(B, C, H, W)
        
        return self.out_proj(out)


class HierarchicalFusion(nn.Module):
    """
    Hierarchical feature fusion for progressive integration
    of multi-scale features.
    
    Fuses features in a coarse-to-fine manner:
    F_12 = Fuse(F_1, F_2)
    F_final = Fuse(F_12, F_3)
    """
    
    def __init__(self, in_channels: int, num_branches: int = 3):
        """
        Args:
            in_channels: Number of channels per branch
            num_branches: Number of branches
        """
        super().__init__()
        self.num_branches = num_branches
        
        # Pairwise fusion modules
        self.fusions = nn.ModuleList([
            AdaptiveFusion(in_channels, 2)
            for _ in range(num_branches - 1)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Hierarchical fusion.
        
        Args:
            features: List of [B, C, H, W] tensors (ordered coarse to fine)
            
        Returns:
            Tuple of:
                - Final fused features [B, C, H, W]
                - List of intermediate attention weights
        """
        all_weights = []
        
        current = features[0]
        for i, fusion in enumerate(self.fusions):
            fused, weights = fusion([current, features[i + 1]])
            current = fused
            all_weights.append(weights)
            
        return current, all_weights


# Initialize package
__all__ = [
    'AdaptiveFusion',
    'MultiScaleAttentionFusion',
    'ScaleAwareAttention',
    'CrossScaleAttention',
    'HierarchicalFusion'
]
