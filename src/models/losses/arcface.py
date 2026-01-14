"""
ArcFace Angular Margin Loss for Deep Biometric Recognition.

Loss function for learning discriminative embeddings.
Includes SubCenter variant for handling intra-class variation (e.g., L/R eyes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto

class UpdateMarginStrategy(Enum):
    EPOCH_LINEAR = auto()  # Standard linear ramp over epochs
    PATIENCE_BASED = auto() # Increase margin only when validation loss plateaus


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    Paper: https://arxiv.org/abs/1801.07698
    
    Args:
        embedding_dim: Dimension of embeddings
        num_classes: Number of identity classes
        scale: Scaling factor (default: 30.0)
        margin: Angular margin in radians (default: 0.5)
        easy_margin: Use easy margin formula (default: False)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,  # Reduced from 64 for small datasets (<2000 classes)
        margin: float = 0.5,
        easy_margin: bool = False,
        initial_margin: float = 0.0,  # Start with 0 margin for easy convergence
        update_strategy: UpdateMarginStrategy = UpdateMarginStrategy.EPOCH_LINEAR
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)
        
        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # For margin annealing - store target margin
        self.target_margin = margin
        self.initial_margin = initial_margin
    
    def set_margin(self, margin: float):
        """Update margin dynamically during training."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def get_annealed_margin(self, current_epoch: int, warmup_epochs: int, total_epochs: int) -> float:
        """ Legacy wrapper for epoch-linear strategy """
        return self.get_dynamic_margin(
            strategy=UpdateMarginStrategy.EPOCH_LINEAR,
            current_epoch=current_epoch,
            total_epochs=total_epochs
        )

    def get_dynamic_margin(self, strategy: UpdateMarginStrategy, **kwargs) -> float:
        """
        Calculate margin based on selected strategy.
        
        Args:
            strategy: UpdateMarginStrategy
            **kwargs: required args for the strategy (e.g., current_epoch, current_val_acc)
            
        Returns:
            Target margin for this step
        """
        if strategy == UpdateMarginStrategy.EPOCH_LINEAR:
            current_epoch = kwargs.get('current_epoch', 0)
            total_epochs = kwargs.get('total_epochs', 100)
            progress = min(1.0, current_epoch / max(1, total_epochs))
            return self.initial_margin + (self.target_margin - self.initial_margin) * progress
            
        elif strategy == UpdateMarginStrategy.PATIENCE_BASED:
            # Logic: We hold the margin steady until the model "masters" the current difficulty.
            # Implemented in the training loop: set_margin is called explicitly.
            # This method acts as a getter if needed.
            return self.margin
            
        return self.margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace logits.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) class labels
            
        Returns:
            (B, C) scaled logits with angular margin
        """
        # Store original dtype for casting back (important for mixed precision)
        original_dtype = embeddings.dtype
        
        # Cast to float32 for numerical stability (prevents FP16 overflow in cos/sin/sqrt)
        embeddings = embeddings.float()
        weight = self.weight.float()
        
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7, max=1.0))
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # IMPORTANT: Keep output in FP32 for numerical stability in loss computation
        # Casting back to FP16 can cause NaN in downstream softmax operations
        return output
    
    def get_cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get raw cosine similarity logits WITHOUT margin adjustment.
        Use this for validation/inference accuracy calculation.
        
        Args:
            embeddings: (B, D) embeddings
            
        Returns:
            (B, C) scaled cosine similarity logits
        """
        original_dtype = embeddings.dtype
        embeddings = embeddings.float()
        weight = self.weight.float()
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        cosine = F.linear(embeddings, weight)
        return (cosine * self.scale).to(original_dtype)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing regularization.
    
    NOTE: All computation happens in FP32 to prevent NaN from log_softmax
    which is numerically unstable in FP16.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy.
        
        Args:
            pred: (B, C) logits
            target: (B,) class labels
            
        Returns:
            Scalar loss
        """
        # CRITICAL: Cast to FP32 for numerical stability
        # log_softmax is known to be unstable in FP16
        pred = pred.float()
        
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubCenterArcFaceLoss(nn.Module):
    """
    SubCenter ArcFace: Handling Intra-Class Variation for Face Recognition.
    
    Paper: https://arxiv.org/abs/2003.11922
    
    Uses K subcenters per class to handle within-class variations such as:
    - Left vs Right eye (iris)
    - Different poses/expressions (face)
    - Different finger positions (fingerprint)
    
    Only the dominant subcenter (highest similarity) is used for margin penalty,
    allowing the model to learn multiple "modes" per identity.
    
    Args:
        embedding_dim: Dimension of embeddings
        num_classes: Number of identity classes
        num_subcenters: Number of subcenters per class (default: 3)
        scale: Scaling factor (default: 45.0)
        margin: Angular margin in radians (default: 0.35)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        scale: float = 45.0,
        margin: float = 0.35,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class centers: (num_classes, K, embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_subcenters, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)
        
        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # For margin annealing
        self.target_margin = margin
    
    def set_margin(self, margin: float):
        """Update margin dynamically during training."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SubCenter ArcFace logits.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) class labels
            
        Returns:
            (B, C) scaled logits with angular margin on dominant subcenter
        """
        original_dtype = embeddings.dtype
        embeddings = embeddings.float()
        weight = self.weight.float()  # (C, K, D)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (B, D)
        
        # Normalize all subcenters
        weight_norm = F.normalize(weight, p=2, dim=2)  # (C, K, D)
        
        # Compute cosine similarity with all subcenters
        # Reshape for batch matmul: (C, K, D) @ (D, B) -> (C, K, B) -> (B, C, K)
        cos_all = torch.einsum('ckd,bd->bck', weight_norm, embeddings)  # (B, C, K)
        
        # Get max cosine across subcenters (dominant subcenter)
        cosine, _ = cos_all.max(dim=2)  # (B, C)
        
        # Apply angular margin to target class
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output
    
    def get_cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get raw cosine similarity logits WITHOUT margin (for validation)."""
        embeddings = embeddings.float()
        weight = self.weight.float()
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(weight, p=2, dim=2)
        
        cos_all = torch.einsum('ckd,bd->bck', weight_norm, embeddings)
        cosine, _ = cos_all.max(dim=2)
        
        return cosine * self.scale

