
import os
from pathlib import Path
from loguru import logger

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Storage
GALLERY_DIR = PROJECT_ROOT / "src/app/static/gallery"
DATA_DIR = PROJECT_ROOT.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_onnx"  # Segmented iris ROIs
MODEL_PATH = PROJECT_ROOT.parent.parent / "outputs/ensemble/ensemble_model.pt"
WEIGHTS_DIR = PROJECT_ROOT.parent.parent / "weights"  # <project_root>/weights

# ============================================================================
# AGGRESSIVE GPU UTILIZATION
# ============================================================================
# Check for CUDA availability - use GPU if available
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Aggressive GPU settings for maximum performance
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal convolution algorithms
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ for faster matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere+ for faster convolutions
    # Set GPU memory growth to avoid OOM
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    logger.info(f"  - CUDA Version: {torch.version.cuda}")
    logger.info(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
else:
    DEVICE = "cpu"
    logger.warning("CUDA not available, using CPU (inference will be slower)")

# Model Config
FACE_WEIGHTS_PATH = WEIGHTS_DIR / "R50_Glint360K.onnx"
logger.info(f"Weights directory: {WEIGHTS_DIR}")
logger.info(f"Face weights path: {FACE_WEIGHTS_PATH}")

# API Config
API_V1_STR = "/api/v1"
PROJECT_NAME = "Biometric Security Platform"
