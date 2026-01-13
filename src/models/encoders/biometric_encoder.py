"""
Biometric Encoder with EfficientNetV2-S Backbone.

Uses pretrained EfficientNetV2-S for efficient training.
Includes RadialAttention and ECA modules for iris recognition.
Includes Stochastic Depth (DropPath) for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


from ..iris.attention import IrisAttention
from .iresnet import iresnet50, iresnet100
from loguru import logger


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) regularization for residual blocks.
    
    Paper: https://arxiv.org/abs/1603.09382
    "Deep Networks with Stochastic Depth"
    
    Randomly drops entire paths during training, which:
    - Reduces training time
    - Improves generalization
    - Acts as implicit ensemble of networks with varying depths
    
    Args:
        drop_prob: Probability of dropping the path (default: 0.0)
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Batch-wise drop: same mask for entire batch for efficiency
        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale up to maintain expected value
        return x.div(keep_prob) * random_tensor


def add_stochastic_depth_to_efficientnet(backbone, drop_path_rate: float = 0.2):
    """
    Add stochastic depth to EfficientNet backbone by wrapping MBConv blocks.
    
    The drop rate is linearly increased for deeper layers (0 for first block,
    drop_path_rate for last block) following the original paper.
    
    Args:
        backbone: EfficientNetV2-S backbone
        drop_path_rate: Maximum drop rate for deepest blocks
    """
    # Count total number of blocks
    num_blocks = sum(1 for stage in backbone.features[1:-1] for _ in stage)
    
    if num_blocks == 0:
        return
    
    block_idx = 0
    for stage in backbone.features[1:-1]:  # Skip first conv and last conv
        for i, block in enumerate(stage):
            # Linear increase in drop rate
            block_drop_rate = drop_path_rate * block_idx / (num_blocks - 1) if num_blocks > 1 else 0
            
            # Wrap the block's forward with stochastic depth
            if hasattr(block, 'stochastic_depth') and block.stochastic_depth is not None:
                # EfficientNetV2 already has stochastic depth, just update rate
                block.stochastic_depth = DropPath(block_drop_rate)
            else:
                # Add stochastic depth wrapper
                original_forward = block.forward
                drop_path = DropPath(block_drop_rate)
                
                def make_forward(orig_fwd, dp):
                    def new_forward(x):
                        return dp(orig_fwd(x))
                    return new_forward
                
                block.forward = make_forward(original_forward, drop_path)
            
            block_idx += 1
    
    logger.info(f"Added Stochastic Depth (DropPath) with max_rate={drop_path_rate} to {num_blocks} blocks")


class BiometricEncoder(nn.Module):
    """
    Biometric Encoder supporting EfficientNetV2-S and IResNet backbones.
    
    Features:
    - EfficientNetV2-S: ImageNet pretrained, efficient (21.5M params)
    - IResNet50: Face pretrained (MS1MV2), 43.6M params
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.4,
        use_attention: bool = True,
        backbone_type: str = 'efficientnet',  # 'efficientnet', 'iresnet50', 'iresnet100'
        pretrained_weights_path: str = '',  # Path to ONNX weights for IResNet
        drop_path_rate: float = 0.2  # Stochastic depth rate (0.2 = 20% max drop for deepest layers)
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_type = backbone_type
        self.use_attention = use_attention
        
        if backbone_type == 'efficientnet':
            # Load pretrained EfficientNetV2-S
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            
            # Get backbone feature dimension
            backbone_dim = self.backbone.classifier[1].in_features
            
            # Modify first conv for grayscale if needed
            if in_channels != 3:
                original_conv = self.backbone.features[0][0]
                new_conv = nn.Conv2d(
                    in_channels, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
                # CRITICAL FIX: Initialize from pretrained RGB weights by averaging
                # This preserves the pretrained feature hierarchy instead of random init
                with torch.no_grad():
                    # Average across RGB channels: (out_channels, 3, k, k) -> (out_channels, 1, k, k)
                    new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                self.backbone.features[0][0] = new_conv
                logger.info("Initialized grayscale conv from averaged RGB pretrained weights")
                
            # --- RECTANGULAR INPUT ADAPTATION ---
            # NOTE: Only apply for truly rectangular inputs like iris (64x512)
            # For square inputs like fingerprints (160x160), keep original stride (2,2)
            # This is now handled at runtime, not here, to support multiple modalities
            
            # Remove classification head
            self.backbone.classifier = nn.Identity()
            self.backbone.avgpool = nn.Identity()
            
            # Add Stochastic Depth (DropPath) for regularization
            if drop_path_rate > 0:
                add_stochastic_depth_to_efficientnet(self.backbone, drop_path_rate)
            
            if use_attention:
                self.attention = IrisAttention(backbone_dim, reduction=16)
                
            # Embedding head
            self.embedding_head = nn.Sequential(
                nn.Linear(backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
            
        elif backbone_type.startswith('iresnet'):
            # IResNet backbone (InsightFace)
            use_onnx_backbone = False
            
            # Try to load ONNX pretrained backbone directly (best option)
            if pretrained and pretrained_weights_path:
                from pathlib import Path
                weights_path = Path(pretrained_weights_path)
                if weights_path.exists() and weights_path.suffix == '.onnx':
                    try:
                        from .load_pretrained import load_onnx_as_backbone
                        self.backbone = load_onnx_as_backbone(weights_path)
                        use_onnx_backbone = True
                        logger.info(f"Using ONNX backbone directly from {weights_path}")
                        
                        # ONNX backbone outputs 512-dim embeddings directly
                        # Create a simple projection if embedding_dim differs
                        if embedding_dim != 512:
                            self.embedding_head = nn.Sequential(
                                nn.Linear(512, embedding_dim),
                                nn.BatchNorm1d(embedding_dim)
                            )
                        else:
                            self.embedding_head = nn.Identity()
                            
                    except Exception as e:
                        logger.warning(f"Failed to load ONNX backbone: {e}")
                        logger.info("Falling back to IResNet from scratch")
                        use_onnx_backbone = False
            
            # Fallback to our IResNet implementation
            if not use_onnx_backbone:
                model_fn = iresnet100 if backbone_type == 'iresnet100' else iresnet50
                self.backbone = model_fn(pretrained=False, num_features=embedding_dim, dropout=dropout)
                
                if pretrained:
                    logger.info("Training IResNet from scratch (no ONNX weights loaded)")
                    logger.info("For best results, provide pretrained_weights_path to ONNX file")
                
                # Modify first conv if grayscale
                if in_channels != 3:
                    original_conv = self.backbone.conv1
                    self.backbone.conv1 = nn.Conv2d(
                        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                    )
                
                # Extract head from IResNet for differential learning rates
                self.embedding_head = nn.Sequential(
                    self.backbone.fc,
                    self.backbone.features
                )
                
                # Replace head in backbone with Identity
                self.backbone.fc = nn.Identity()
                self.backbone.features = nn.Identity()
            
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract embeddings."""
        if self.backbone_type == 'efficientnet':
            features = self.backbone.features(x)
            if self.use_attention:
                features = self.attention(features)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            embeddings = self.embedding_head(features)
        else:
            # IResNet backbone (now with head removed)
            features = self.backbone(x)
            embeddings = self.embedding_head(features)
            
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without embedding head."""
        features = self.backbone.features(x)
        if self.use_attention:
            features = self.attention(features)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

