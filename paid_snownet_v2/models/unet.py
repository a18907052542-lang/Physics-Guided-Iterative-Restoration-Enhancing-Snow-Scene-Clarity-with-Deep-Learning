"""
U-Net completion network used within the SIRM module.
Architecture: 4 layers, 64 base channels (Table 6 default).
X_{k+1} = U(X_k^{1/2}; theta) = Dec(Enc(X_k^{1/2}; theta_enc); theta_dec)  (Eq. 25)
File: models/unet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Use GroupNorm instead of BatchNorm for robustness with
        # small spatial sizes and batch_size=1 during inference
        # GroupNorm requires num_groups to divide num_channels. Pick the
        # largest divisor of out_channels that is <= 32.
        num_groups = 1
        for g in (32, 16, 8, 4, 2, 1):
            if out_channels % g == 0:
                num_groups = g
                break
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Concat skip -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CompletionUNet(nn.Module):
    """
    U-Net completion network for the SIRM module.
    
    Default configuration (Table 6):
        - 4 encoder-decoder layers
        - 64 base channels
        - 45.2M parameters
    
    Input: intermediate restored image X_k^{1/2} (B, 3, H, W)
    Output: refined image X_{k+1} (B, 3, H, W)
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_layers=4):
        super(CompletionUNet, self).__init__()
        
        self.num_layers = num_layers
        
        # Encoder
        self.enc_first = DoubleConv(in_channels, base_channels)
        
        self.encoders = nn.ModuleList()
        channels = base_channels
        for i in range(num_layers - 1):
            self.encoders.append(DownBlock(channels, channels * 2))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = DownBlock(channels, channels * 2)
        channels *= 2
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.decoders.append(UpBlock(channels, channels // 2))
            channels //= 2
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) intermediate restored image
        Returns:
            out: (B, 3, H, W) refined image
        """
        # Encoder path
        skips = []
        h = self.enc_first(x)
        skips.append(h)
        
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 1 - i
            h = decoder(h, skips[skip_idx])
        
        # Output with residual connection
        out = self.out_conv(h)
        out = out + x  # Residual learning
        out = torch.clamp(out, 0.0, 1.0)
        
        return out
