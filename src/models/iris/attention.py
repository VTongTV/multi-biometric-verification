"""
Attention Modules for Iris Recognition.

Uses proven, stable implementations based on torchvision patterns.
- SEBlock: Classic Squeeze-and-Excitation (proven stable with mixed precision)
- ECA: Efficient Channel Attention with safe initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block using torchvision's proven pattern.
    Includes NaN guards for stability.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        squeezed = max(1, channels // reduction)
        
        self.fc1 = nn.Conv2d(channels, squeezed, kernel_size=1)
        self.fc2 = nn.Conv2d(squeezed, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype
        dtype = x.dtype
        
        # Squeeze: Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        
        # Guard: Check for NaNs in input/pooling
        if torch.isnan(scale).any():
            # If input is already bad, return identity or zero checks
            # But safer to just clamp and proceed
            scale = torch.nan_to_num(scale)
            
        # Excitation
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        
        # Guard: Clamp before sigmoid to prevent overflow/underflow
        # FP16 range is small, exp() can overflow > 88
        scale = torch.clamp(scale, min=-50.0, max=50.0)
        
        scale = torch.sigmoid(scale)
        
        # Scale
        return x * scale.to(dtype)


class ECA(nn.Module):
    """
    Efficient Channel Attention with safe implementation.
    Includes NaN guards.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        # Adaptive kernel size
        gamma, b = 2, 1
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        k = max(3, min(k, 7))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        
        nn.init.constant_(self.conv.weight, 1.0 / k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        
        # Squeeze
        y = self.avg_pool(x)
        
        # Guard
        if torch.isnan(y).any():
            y = torch.nan_to_num(y)
            
        # Reshape for 1D conv
        y = y.squeeze(-1).transpose(-1, -2)
        
        # Conv
        y = self.conv(y)
        
        # Reshape back
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # Guard: Clamp before sigmoid
        y = torch.clamp(y, min=-50.0, max=50.0)
        
        # Scale
        return x * torch.sigmoid(y).to(dtype)


class IrisAttention(nn.Module):
    """
    Combined attention module for iris recognition.
    
    Combines SE + ECA sequentially. Uses standard patterns proven
    to work with mixed precision training.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.eca = ECA(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.eca(x)
        return x

