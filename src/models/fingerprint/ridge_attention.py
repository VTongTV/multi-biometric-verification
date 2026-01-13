"""
Ridge Attention Module for Fingerprint Recognition.

Specialized attention mechanism that focuses on fingerprint ridge patterns
using channel attention and spatial attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation variant).
    Focuses on important feature channels for ridge patterns.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        squeezed = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, squeezed, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        
        # Both average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = torch.clamp(out, min=-50.0, max=50.0)  # Prevent overflow in FP16
        
        return x * torch.sigmoid(out).to(dtype)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    Focuses on important spatial regions (ridge areas vs valleys).
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Convolve and apply sigmoid
        out = self.conv(out)
        out = torch.clamp(out, min=-50.0, max=50.0)
        
        return x * torch.sigmoid(out).to(dtype)


class RidgeAttention(nn.Module):
    """
    Combined Ridge Attention for fingerprint recognition.
    
    Applies channel attention followed by spatial attention to focus
    on ridge patterns and important spatial regions.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for squeeze operation
        spatial_kernel: Kernel size for spatial attention (3 or 7)
    """
    
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ridge attention.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Attention-weighted feature map
        """
        # Channel attention first (what to focus on)
        x = self.channel_attention(x)
        
        # Spatial attention second (where to focus)
        x = self.spatial_attention(x)
        
        return x

