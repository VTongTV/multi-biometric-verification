"""
Hardware optimization utilities for RTX 3050 (4GB VRAM) and Ryzen 5.

Provides automatic mixed precision, gradient checkpointing, and
memory-efficient training configurations.
"""

import gc
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class HardwareConfig:
    """Hardware configuration for the training environment."""
    
    device: torch.device
    dtype: torch.dtype
    use_amp: bool  # Automatic Mixed Precision
    use_gradient_checkpointing: bool
    optimal_batch_size: int
    num_workers: int
    pin_memory: bool
    
    def __repr__(self) -> str:
        return (
            f"HardwareConfig(\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  use_amp={self.use_amp},\n"
            f"  use_gradient_checkpointing={self.use_gradient_checkpointing},\n"
            f"  optimal_batch_size={self.optimal_batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  pin_memory={self.pin_memory}\n"
            f")"
        )


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
        return device
    else:
        logger.warning("CUDA not available, using CPU (training will be slow)")
        return torch.device("cpu")


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (total_gb, used_gb, free_gb)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - reserved
    
    return (total, allocated, free)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU memory cache cleared")


def detect_hardware_config(
    target_batch_size: int = 16,
    force_cpu: bool = False
) -> HardwareConfig:
    """
    Detect and configure optimal hardware settings for RTX 3050.
    
    Args:
        target_batch_size: Desired batch size (will be adjusted based on memory)
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        HardwareConfig with optimal settings for detected hardware
    """
    device = torch.device("cpu") if force_cpu else get_device()
    
    if device.type == "cuda":
        total_vram, _, _ = get_gpu_memory_info()
        
        # RTX 3050 has 4GB VRAM - be conservative
        if total_vram <= 4.5:
            # Tight VRAM budget
            use_amp = True
            use_gradient_checkpointing = True
            optimal_batch_size = min(target_batch_size, 16)
            logger.info(f"Detected limited VRAM ({total_vram:.1f}GB). "
                       "Enabling aggressive memory optimizations.")
        elif total_vram <= 8.5:
            # Moderate VRAM
            use_amp = True
            use_gradient_checkpointing = True
            optimal_batch_size = min(target_batch_size, 32)
        else:
            # Generous VRAM
            use_amp = True
            use_gradient_checkpointing = False
            optimal_batch_size = target_batch_size
            
        dtype = torch.float16 if use_amp else torch.float32
        # Ryzen 5 has 6 cores - use 4 workers for dataloading
        num_workers = 4
        pin_memory = True
    else:
        use_amp = False
        use_gradient_checkpointing = False
        dtype = torch.float32
        optimal_batch_size = target_batch_size
        # CPU: fewer workers to avoid overhead
        num_workers = min(os.cpu_count() or 2, 4)
        pin_memory = False
    
    config = HardwareConfig(
        device=device,
        dtype=dtype,
        use_amp=use_amp,
        use_gradient_checkpointing=use_gradient_checkpointing,
        optimal_batch_size=optimal_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Hardware configuration:\n{config}")
    return config


def setup_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing on supported model components.
    
    This trades compute for memory by recomputing activations during backward pass.
    
    Args:
        model: PyTorch model to modify
        
    Returns:
        Modified model with gradient checkpointing enabled
    """
    checkpointed_modules = 0
    
    for module in model.modules():
        # Enable for common backbone modules
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
            checkpointed_modules += 1
        elif hasattr(module, 'set_grad_checkpointing'):
            module.set_grad_checkpointing(True)
            checkpointed_modules += 1
    
    if checkpointed_modules > 0:
        logger.info(f"Enabled gradient checkpointing on {checkpointed_modules} modules")
    else:
        logger.warning("No modules support built-in gradient checkpointing. "
                      "Consider using torch.utils.checkpoint manually.")
    
    return model


def estimate_model_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate memory requirements for a model during training.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (C, H, W)
        batch_size: Batch size
        dtype: Data type
        
    Returns:
        Estimated memory in GB
    """
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradient memory (same as parameters)
    grad_bytes = param_bytes
    
    # Optimizer states (assuming Adam: 2 states per param)
    optimizer_bytes = 2 * param_bytes
    
    # Rough activation memory estimate
    # This is very approximate - real usage depends on architecture
    input_bytes = (
        batch_size * 
        torch.tensor(input_shape).prod().item() * 
        torch.finfo(dtype).bits // 8
    )
    # Activations can be 2-4x input size for typical CNNs
    activation_bytes = 3 * input_bytes * 10  # Very rough estimate
    
    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    total_gb = total_bytes / (1024**3)
    
    return total_gb


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    max_batch_size: int = 64,
    target_memory_fraction: float = 0.8,
    dtype: torch.dtype = torch.float16
) -> int:
    """
    Find optimal batch size that fits in GPU memory.
    
    Uses binary search to find the largest batch size that doesn't OOM.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (C, H, W)
        max_batch_size: Maximum batch size to try
        target_memory_fraction: Fraction of GPU memory to use
        dtype: Data type for computation
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return max_batch_size
    
    total_vram, _, _ = get_gpu_memory_info()
    target_memory = total_vram * target_memory_fraction
    
    # Binary search for optimal batch size
    low, high = 1, max_batch_size
    optimal = 1
    
    while low <= high:
        mid = (low + high) // 2
        estimated = estimate_model_memory(model, input_shape, mid, dtype)
        
        if estimated <= target_memory:
            optimal = mid
            low = mid + 1
        else:
            high = mid - 1
    
    logger.info(f"Estimated optimal batch size: {optimal} "
               f"(target memory: {target_memory:.2f}GB)")
    
    return optimal


class MemoryTracker:
    """Context manager for tracking GPU memory usage."""
    
    def __init__(self, label: str = ""):
        self.label = label
        self.start_allocated = 0.0
        self.start_reserved = 0.0
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated() / (1024**3)
            self.start_reserved = torch.cuda.memory_reserved() / (1024**3)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated() / (1024**3)
            end_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            delta_allocated = end_allocated - self.start_allocated
            delta_reserved = end_reserved - self.start_reserved
            
            logger.debug(
                f"[{self.label}] Memory: "
                f"allocated={delta_allocated:+.3f}GB, "
                f"reserved={delta_reserved:+.3f}GB"
            )
        return False


def create_amp_context(enabled: bool = True):
    """
    Create automatic mixed precision context.
    
    Args:
        enabled: Whether to enable AMP
        
    Returns:
        torch.cuda.amp.autocast context manager
    """
    return torch.cuda.amp.autocast(enabled=enabled, dtype=torch.float16)


def create_grad_scaler(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    """
    Create gradient scaler for mixed precision training.
    
    Args:
        enabled: Whether to enable gradient scaling
        
    Returns:
        GradScaler instance
    """
    return torch.cuda.amp.GradScaler(enabled=enabled)
