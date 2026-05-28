"""
Attention-based fusion mechanism for multi-branch features.
Implements Equations 8-9 from the paper.
File: models/attention_fusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Adaptive weighting mechanism for multi-branch feature fusion.
    
    [alpha, beta, gamma] = Softmax(W_a * GAP([F1, F2, F3]) + b_a)  (Eq. 8)
    F_fused = alpha * F1 + beta * F2 + gamma * F3                   (Eq. 9)
    
    where GAP = Global Average Pooling, W_a in R^{3x3C}, b_a in R^3
    """
    
    def __init__(self, num_branches=3, channels=64, hidden_dim=32):
        """
        Args:
            num_branches: Number of feature branches (default 3)
            channels: Channel dimension C per branch
            hidden_dim: Attention hidden dimension (Table 8: 32)
        """
        super(AttentionFusion, self).__init__()
        
        self.num_branches = num_branches
        self.channels = channels
        
        # W_a in R^{3 x 3C} and b_a in R^3
        # Using a small MLP for better gradient flow
        self.attention_fc = nn.Sequential(
            nn.Linear(num_branches * channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_branches),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.attention_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """
        Args:
            features: list of tensors [F1, F2, F3], each (B, C, H, W)
        Returns:
            F_fused: (B, C, H, W) fused feature map
        """
        assert len(features) == self.num_branches
        
        B, C, H, W = features[0].shape
        
        # GAP([F1, F2, F3]): concatenate then global average pool
        # Each branch -> (B, C) after GAP -> concat -> (B, 3C)
        gap_features = []
        for f in features:
            gap = F.adaptive_avg_pool2d(f, 1).view(B, -1)  # (B, C)
            gap_features.append(gap)
        gap_concat = torch.cat(gap_features, dim=1)  # (B, 3C)
        
        # Compute attention weights: (B, 3)
        weights = self.attention_fc(gap_concat)  # (B, num_branches)
        weights = F.softmax(weights, dim=1)      # (B, num_branches)
        
        # Weighted fusion: F_fused = alpha*F1 + beta*F2 + gamma*F3
        fused = torch.zeros_like(features[0])
        for i in range(self.num_branches):
            w = weights[:, i].view(B, 1, 1, 1)  # (B, 1, 1, 1)
            fused = fused + w * features[i]
        
        return fused
