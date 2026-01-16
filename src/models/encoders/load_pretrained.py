"""
Pretrained Weight Loader for IResNet Face Recognition.

Supports loading InsightFace ONNX models directly as PyTorch modules.
Recommended models (IResNet50):
- Glint360K: Best demographic balance (99.8% LFW, 87% MR-ALL)  
- WebFace600K: Highest overall accuracy (99.8% LFW, 90.5% MR-ALL)

Download from: https://github.com/deepinsight/insightface/tree/master/model_zoo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
from loguru import logger
from typing import Optional

# Try to import onnx2torch for conversion
try:
    import onnx
    from onnx2torch import convert
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnx2torch not installed. Run: pip install onnx onnx2torch")


class ONNXBackboneWrapper(nn.Module):
    """
    Wrapper for ONNX-converted InsightFace model.
    
    Provides a consistent interface matching our IResNet backbone expectations.
    The converted ONNX model outputs 512-dim embeddings directly.
    """
    
    def __init__(self, onnx_path: Path):
        super().__init__()
        
        if not ONNX_AVAILABLE:
            raise ImportError("Install onnx and onnx2torch: pip install onnx onnx2torch")
        
        logger.info(f"Loading ONNX backbone from {onnx_path}")
        onnx_path_str = os.path.normpath(str(onnx_path))
        onnx_model = onnx.load(onnx_path_str)

        # Convert to PyTorch (this creates a traced GraphModule)
        self.backbone = convert(onnx_model)
        
        # InsightFace R50 outputs 512-dim embeddings
        self.output_dim = 512
        
        # Dummy modules to satisfy trainer expectations
        self.fc = nn.Identity()
        self.features = nn.Identity()
        
        logger.info(f"ONNX backbone loaded successfully (output_dim={self.output_dim})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ONNX backbone."""
        # ONNX model expects (B, 3, 112, 112) and outputs (B, 512)
        return self.backbone(x)


def load_onnx_as_backbone(onnx_path: Path) -> nn.Module:
    """
    Load InsightFace ONNX model as a PyTorch backbone.
    
    This directly uses the converted ONNX model instead of trying to
    map weights to our IResNet implementation.
    
    Args:
        onnx_path: Path to .onnx file (e.g., R50_Glint360K.onnx)
        
    Returns:
        PyTorch module that can be used as backbone
    """
    return ONNXBackboneWrapper(onnx_path)


def load_onnx_weights_to_iresnet(
    onnx_path: Path,
    iresnet_model: nn.Module,
    strict: bool = False
) -> nn.Module:
    """
    Load weights from InsightFace ONNX model into PyTorch IResNet.
    
    Note: Due to architecture differences, this uses shape-based matching
    which may not transfer all weights. For best results, use 
    load_onnx_as_backbone() instead.
    
    Args:
        onnx_path: Path to .onnx file
        iresnet_model: Target PyTorch IResNet model
        strict: If True, raise error on missing/unexpected keys
        
    Returns:
        IResNet model with loaded weights (partial transfer)
    """
    if not ONNX_AVAILABLE:
        raise ImportError("Install onnx and onnx2torch: pip install onnx onnx2torch")
    
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    logger.info(f"Loading ONNX model from {onnx_path}")
    onnx_path_str = os.path.normpath(str(onnx_path))

    # Convert ONNX to PyTorch
    onnx_model = onnx.load(onnx_path_str)
    pytorch_model = convert(onnx_model)
    
    # Extract state dict
    onnx_state = pytorch_model.state_dict()
    target_state = iresnet_model.state_dict()
    
    # Map by shape matching
    mapped_state = _map_by_shape(onnx_state, target_state)
    
    # Load with appropriate strictness
    missing, unexpected = iresnet_model.load_state_dict(mapped_state, strict=strict)
    
    loaded_count = len(mapped_state)
    total_count = len([k for k in target_state if 'running' not in k and 'num_batches' not in k])
    
    logger.info(f"Transferred {loaded_count}/{total_count} weight tensors from ONNX")
    if loaded_count < total_count * 0.5:
        logger.warning("Less than 50% weights transferred. Consider using load_onnx_as_backbone() instead.")
    
    return iresnet_model


def _map_by_shape(onnx_state: dict, target_state: dict) -> dict:
    """Map ONNX weights to target by shape matching."""
    mapped = {}
    
    # Group tensors by shape
    onnx_by_shape = {}
    for key, tensor in onnx_state.items():
        if 'running' in key or 'num_batches' in key:
            continue
        shape = tuple(tensor.shape)
        if shape not in onnx_by_shape:
            onnx_by_shape[shape] = []
        onnx_by_shape[shape].append((key, tensor))
    
    target_by_shape = {}
    for key, tensor in target_state.items():
        if 'running' in key or 'num_batches' in key:
            continue
        shape = tuple(tensor.shape)
        if shape not in target_by_shape:
            target_by_shape[shape] = []
        target_by_shape[shape].append(key)
    
    # Match by shape and order
    for shape, onnx_items in onnx_by_shape.items():
        if shape in target_by_shape:
            target_keys = target_by_shape[shape]
            for i, (_, tensor) in enumerate(onnx_items):
                if i < len(target_keys):
                    mapped[target_keys[i]] = tensor
    
    return mapped
