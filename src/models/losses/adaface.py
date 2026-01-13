"""
AdaFace: Quality Adaptive Margin for Face Recognition.

CVPR 2022 - Adapts margin based on image quality (approximated by feature norm).
Paper: https://arxiv.org/abs/2204.00964

Key insight: Hard samples should be emphasized when quality is high,
but de-emphasized when quality is low (likely due to image quality issues).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceLoss(nn.Module):
    """
    AdaFace: Adaptive Margin based on Feature Norm (Image Quality Proxy).
    
    The margin is adjusted based on the norm of the feature vector,
    where higher norms indicate higher image quality.
    
    Args:
        embedding_dim: Dimension of embeddings
        num_classes: Number of identity classes  
        scale: Scaling factor (default: 64.0)
        margin: Target angular margin in radians (default: 0.4)
        h: Hyperparameter for adaptive margin (default: 0.333)
        t_alpha: EMA momentum for batch statistics (default: 0.01)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.h = h
        self.t_alpha = t_alpha
        
        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Running statistics for feature norms (quality proxy)
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)
        
        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
    
    def set_margin(self, margin: float):
        """Update margin dynamically (for compatibility with trainer)."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute AdaFace logits with quality-adaptive margin.
        
        Args:
            embeddings: (B, D) embeddings (NOT normalized)
            labels: (B,) class labels
            
        Returns:
            (B, C) scaled logits with adaptive angular margin
        """
        # Compute feature norms (quality indicator)
        norms = torch.norm(embeddings, 2, 1, keepdim=True)
        
        # Update running statistics
        if self.training:
            with torch.no_grad():
                self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * norms.mean()
                self.batch_std = (1 - self.t_alpha) * self.batch_std + self.t_alpha * norms.std()
        
        # Compute adaptive margin based on normalized feature norm
        # Higher quality (higher norm) -> harder margin
        # Lower quality (lower norm) -> easier margin
        # Clamp norms to avoid div by zero or extreme values
        safe_norms = torch.clamp(norms, min=0.001)
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + 1e-6)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1, 1)
        
        # Normalize embeddings and weights for cosine similarity
        # Use float32 for normalization and matrix multiplication to avoid FP16 overflow/underflow
        embeddings_float = embeddings.float()
        weight_float = self.weight.float()
        
        embeddings_norm = F.normalize(embeddings_float, p=2, dim=1, eps=1e-6)
        weight_norm = F.normalize(weight_float, p=2, dim=1, eps=1e-6)
        
        # Cosine similarity - ALWAYS keep in float32 to avoid FP16 overflow
        cosine = F.linear(embeddings_norm, weight_norm)
        cosine = torch.clamp(cosine, -1 + 1e-6, 1 - 1e-6)
        # DO NOT cast back to FP16 - keep in float32 for numerical stability
        
        # Compute sin(theta)
        # Clamp to avoid negative values inside sqrt due to precision
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
        
        # Adaptive angular margin: g_angular = -m * g
        # where g is the margin scaler based on quality
        # For high quality (g > 0): increase margin (harder)
        # For low quality (g < 0): decrease margin (easier)
        # Keep in float32 for numerical stability
        margin_scaler_float = margin_scaler.float()
        cos_m_adaptive = torch.cos(self.margin + self.margin * margin_scaler_float)
        sin_m_adaptive = torch.sin(self.margin + self.margin * margin_scaler_float)
        # DO NOT cast to FP16 - keep in float32
        
        # cos(theta + m_adaptive)
        phi = cosine * cos_m_adaptive - sine * sin_m_adaptive
        
        # One-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # CRITICAL: Always return float32 to prevent FP16 overflow with scale=64
        return output.float()


class AdaFaceWithNormRegularization(AdaFaceLoss):
    """
    AdaFace with additional L2 norm regularization on features.
    
    Encourages the model to produce higher-norm features for
    high-quality inputs, improving quality awareness.
    """
    
    def __init__(self, *args, norm_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_weight = norm_weight
        self.last_norm_loss = 0.0
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute norm regularization (encourage higher norms)
        norms = torch.norm(embeddings, 2, 1)
        # Target: norms should be high (around 25-30 for well-trained models)
        self.last_norm_loss = -self.norm_weight * norms.mean()  # Negative to maximize
        
        return super().forward(embeddings, labels)
    
    def get_norm_loss(self) -> torch.Tensor:
        """Get the norm regularization loss from the last forward pass."""
        return self.last_norm_loss
