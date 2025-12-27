"""
Attention Modules for Face Recognition.

Uses proven, stable implementations based on torchvision patterns.
Copied from iris model for consistency across modalities.
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
        dtype = x.dtype
        # Squeeze: Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        
        # Guard
        if torch.isnan(scale).any():
            scale = torch.nan_to_num(scale)
            
        # Excitation
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        
        # Guard: Clamp before sigmoid
        scale = torch.clamp(scale, min=-50.0, max=50.0)
        
        scale = torch.sigmoid(scale)
        return x * scale.to(dtype)


class ECA(nn.Module):
    """
    Efficient Channel Attention with safe implementation.
    Includes NaN guards.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        gamma, b = 2, 1
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        k = max(3, min(k, 7))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        
        nn.init.constant_(self.conv.weight, 1.0 / k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = self.avg_pool(x)
        
        if torch.isnan(y).any():
            y = torch.nan_to_num(y)
            
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        y = torch.clamp(y, min=-50.0, max=50.0)
        
        return x * torch.sigmoid(y).to(dtype)


class FaceAttention(nn.Module):
    """
    Combined attention module for face recognition.
    
    Combines SE + ECA sequentially for enhanced feature attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.eca = ECA(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.eca(x)
        return x
