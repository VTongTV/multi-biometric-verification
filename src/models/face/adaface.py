"""
AdaFace Loss - Quality-Adaptive Margin (CVPR 2022).

Contribution for face recognition that adapts to margin based on
image quality estimated from feature norms.

Reference:
Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022
https://arxiv.org/abs/2204.00964
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceLoss(nn.Module):
    """
    AdaFace: Adaptive Margin based on Image Quality.
    
    Key insight: Feature norm correlates with image quality.
    Low-quality samples get smaller margins (easier to learn).
    High-quality samples get larger margins (harder negatives).
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of identity classes
        scale: Logit scaling factor (default: 64.0)
        margin: Base margin value (default: 0.4)
        h: Quality-to-margin sensitivity factor (default: 0.333)
        t_alpha: Batch statistics smoothing factor (default: 0.01)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.h = h
        self.t_alpha = t_alpha
        
        # Learnable weight matrix W (class centers)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Running statistics for batch norm estimation
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quality-adaptive margin.
        
        Args:
            embeddings: (N, D) normalized embeddings
            labels: (N,) class labels
            
        Returns:
            logits: (N, C) scaled and margin-adjusted logits
        """
        # Store original dtype for output
        orig_dtype = embeddings.dtype
        
        # Convert to float32 for numerical stability (critical for arccos)
        embeddings = embeddings.float()
        weight = self.weight.float()
        
        # Compute feature norms (proxy for quality)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        
        # Normalize embeddings for cosine similarity
        embeddings_normed = F.normalize(embeddings, p=2, dim=1)
        weight_normed = F.normalize(weight, p=2, dim=1)
        
        # Cosine similarity: cos(theta)
        cosine = torch.mm(embeddings_normed, weight_normed.t())
        
        # Clamp cosine for numerical stability (tighter range for arccos)
        cosine = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Update running batch statistics (during training)
        if self.training:
            with torch.no_grad():
                batch_mean = norms.mean()
                batch_std = norms.std() + 1e-6  # Add eps for stability
                self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * batch_mean
                self.batch_std = (1 - self.t_alpha) * self.batch_std + self.t_alpha * batch_std
        
        # Compute quality indicator: g = (||z|| - mean) / std
        # Safe gradient: detach stats
        safe_std = torch.clamp(self.batch_std, min=0.01)
        g = (norms - self.batch_mean.detach()) / safe_std.detach()
        
        # Clamp g to reasonable range to prevent extreme margins
        g = torch.clamp(g, min=-2.0, max=2.0)
        
        # Adaptive margin: m = margin + h * g
        # g is per-sample, so margin varies per sample
        margin_adaptive = self.margin + self.h * g  # (N, 1)
        margin_adaptive = torch.clamp(margin_adaptive, min=0.0, max=0.5)
        
        # Get target cosine values
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        target_cosine = (cosine * one_hot).sum(dim=1, keepdim=True)  # (N, 1)
        
        # Clamp target cosine again for extra safety
        target_cosine = torch.clamp(target_cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # theta = arccos(cos(theta))
        theta = torch.acos(target_cosine)
        
        # Additive angular margin: cos(theta + m)
        # Clamp theta + margin to be < pi for valid cos
        theta_m = torch.clamp(theta + margin_adaptive, max=math.pi - 1e-6)
        m_cos = torch.cos(theta_m)
        
        # Replace target entries with margin-adjusted values
        diff = m_cos - target_cosine
        logits = cosine + one_hot * diff
        
        # Scale
        logits = self.scale * logits
        
        # CRITICAL: Always return float32 to prevent FP16 overflow with scale=64
        # Do NOT convert back to original dtype - float32 is required for stability
        return logits.float()
    
    def set_margin(self, margin: float):
        """Update the base margin value."""
        self.margin = margin
    
    def get_cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get raw cosine similarity logits WITHOUT margin adjustment.
        Use this for validation/inference accuracy calculation.
        
        Args:
            embeddings: (N, D) embeddings
            
        Returns:
            (N, C) scaled cosine similarity logits
        """
        orig_dtype = embeddings.dtype
        embeddings = embeddings.float()
        weight = self.weight.float()
        
        embeddings_normed = F.normalize(embeddings, p=2, dim=1)
        weight_normed = F.normalize(weight, p=2, dim=1)
        
        cosine = torch.mm(embeddings_normed, weight_normed.t())
        # CRITICAL: Always return float32 for stability with scale=64
        return (cosine * self.scale).float()
    
    def extra_repr(self) -> str:
        return (f'embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, '
                f'scale={self.scale}, margin={self.margin}, h={self.h}')
