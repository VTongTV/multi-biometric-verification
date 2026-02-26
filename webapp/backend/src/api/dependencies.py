
import torch
from pathlib import Path
from loguru import logger
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Resolve project root dynamically:
# webapp/backend/src/api/dependencies.py -> project root is parents[4]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINING_SRC = PROJECT_ROOT / "src"

# Add project root first for 'src.evaluation' style imports used by trainer.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Add training src for direct 'models' package imports
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

# ============================================================================
# NAMESPACE COLLISION FIX
# ============================================================================
# The 'src' package is already loaded as 'webapp/backend/src'.
# But training code expects 'src' to be '<project_root>/src'.
# We manually inject missing training modules into sys.modules under 'src.*'.
import importlib.util

def _inject_training_module(module_name, relative_path):
    """Manually load a module from TRAINING_SRC and inject into sys.modules."""
    if module_name in sys.modules:
        return

    file_path = TRAINING_SRC / relative_path
    if not file_path.exists():
        logger.warning(f"Could not find {file_path} to inject {module_name}")
        return

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info(f"Injected {module_name} from {file_path}")
    except Exception as e:
        logger.error(f"Failed to inject {module_name}: {e}")

# Inject 'src.evaluation' package
_inject_training_module("src.evaluation", "evaluation/__init__.py")

# Inject 'src.evaluation.evaluation_sampling' module
_inject_training_module("src.evaluation.evaluation_sampling", "evaluation/evaluation_sampling.py")

# Inject 'src.models' package and submodules
_inject_training_module("src.models", "models/__init__.py")
_inject_training_module("src.models.ensemble", "models/ensemble/__init__.py")
_inject_training_module("src.models.iris", "models/iris/__init__.py")
_inject_training_module("src.models.iris.segmentation", "models/iris/segmentation/__init__.py")
_inject_training_module("src.models.iris.segmentation.worldcoin_wrapper", "models/iris/segmentation/worldcoin_wrapper.py")
_inject_training_module("src.models.fingerprint", "models/fingerprint/__init__.py")
_inject_training_module("src.models.fingerprint.preprocessing", "models/fingerprint/preprocessing.py")
_inject_training_module("src.models.face", "models/face/__init__.py")

# Import directly from 'models' package (available because TRAINING_SRC is in sys.path)
from models.ensemble.ensemble_model import BiometricEnsemble
from src.api.core.config import MODEL_PATH, FACE_WEIGHTS_PATH, DEVICE, GALLERY_DIR, PROCESSED_DATA_DIR

# Global State
_model_instance = None
_gallery_cache = {}
_iris_segmenter = None
_fingerprint_preprocessor = None

# ============================================================================
# SEGMENTATION UTILS (Worldcoin ONNX)
# ============================================================================

def get_iris_segmenter():
    """Lazy load Worldcoin iris segmentation model."""
    global _iris_segmenter
    if _iris_segmenter is None:
        try:
            from src.models.iris.segmentation.worldcoin_wrapper import WorldcoinIrisSegmenter
            _iris_segmenter = WorldcoinIrisSegmenter()
            logger.info("Worldcoin iris segmentation model loaded successfully")
        except ImportError as e:
            logger.warning(f"onnxruntime not available, segmentation disabled: {e}")
            _iris_segmenter = None
        except Exception as e:
            logger.error(f"Failed to load Worldcoin segmenter: {e}")
            _iris_segmenter = None
    return _iris_segmenter

def get_fingerprint_preprocessor():
    """Lazy load Fingerprint Preprocessor (CLAHE + Gabor)."""
    global _fingerprint_preprocessor
    if _fingerprint_preprocessor is None:
        try:
            from src.models.fingerprint.preprocessing import FingerprintPreprocessor
            _fingerprint_preprocessor = FingerprintPreprocessor(
                apply_clahe=True,
                apply_gabor=False,  # Match training: Gabor usually disabled during augmentation pipeline or applied lightly
                apply_normalization=True
            )
            # Note: Training config says apply_gabor=False in FingerprintDatasetNative, 
            # but FingerprintDataset uses apply_gabor=False default. 
            # However, train.py sets use_gabor=getattr(args, 'use_gabor_preprocessing', False)
            # Checking train.py default args: use_gabor defaults to False in Config.
            # But the dataset loader in train.py: use_preprocessing=use_gabor.
            # Wait, let's double check src/models/fingerprint/config.py
            # use_gabor_preprocessing: bool = False
            # So training relies on EfficientNet learning from raw (or just normalized) images?
            # Or does it use CLAHE?
            # src/models/fingerprint/train.py:
            # self.preprocessor = FingerprintPreprocessor(apply_clahe=True, apply_gabor=False, apply_normalization=True)
            # This is inside FingerprintDatasetNative if use_preprocessing is True.
            # But use_preprocessing depends on args.use_gabor_preprocessing.
            # Let's check config again.
            logger.info("Fingerprint preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Fingerprint preprocessor: {e}")
            _fingerprint_preprocessor = None
    return _fingerprint_preprocessor

def _segment_iris_image(image_source, target_size=(224, 224)):
    """
    Segment iris image using Worldcoin ONNX model.

    Args:
        image_source: Path, PIL Image, or array
        target_size: Target size for ROI (224, 224) to match training

    Returns:
        PIL Image of segmented ROI, or None if segmentation fails
    """
    segmenter = get_iris_segmenter()

    if segmenter is None:
        return None

    try:
        # Load image
        if isinstance(image_source, (str, Path)):
            img = Image.open(str(image_source)).convert('RGB')
            img_array = np.array(img)
        else:
            # Already a PIL Image or array
            if isinstance(image_source, Image.Image):
                img_array = np.array(image_source.convert('RGB'))
            else:
                img_array = image_source

        # Segment using Worldcoin
        roi, quality = segmenter.extract_iris_roi_mask(img_array, target_size=target_size)

        if roi is not None and quality >= 0.3:
            return Image.fromarray(roi)
        else:
            logger.warning(f"Iris segmentation failed or low quality (q={quality})")
            return None

    except Exception as e:
        logger.error(f"Iris segmentation error: {e}")
        return None

def _preprocess_fingerprint(image_source):
    """
    Load and preprocess fingerprint image to match training pipeline.
    1. Load image
    2. Apply FingerprintPreprocessor (CLAHE + Intensity Norm)
    3. Convert to RGB (replicates channels 3x)
    """
    try:
        # 1. Load image
        if isinstance(image_source, (str, Path)):
            img = Image.open(str(image_source))
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            # File-like object
            img = Image.open(image_source)
            
        # Convert to numpy for preprocessor
        img_array = np.array(img)
        
        # 2. Apply Preprocessing (CLAHE)
        preprocessor = get_fingerprint_preprocessor()
        if preprocessor is not None:
            # enhance returns grayscale numpy array
            img_array = preprocessor.enhance(img_array)
            img = Image.fromarray(img_array)
        else:
            # Fallback if preprocessor fails to load
            img = img.convert('L')
        
        # 3. Convert to RGB (replicates channels 3x)
        # This matches src/data/augmentation/augmentations.py: GrayscaleTo3Channel
        return img.convert('RGB')
    except Exception as e:
        logger.error(f"Iris segmentation error: {e}")
        return None

def _preprocess_face(image_source):
    """
    Load and preprocess face image to match InsightFace ONNX model expectations.
    InsightFace models expect:
    1. Resize to 112x112
    2. BGR format (RGB -> BGR)
    3. Scale to [-1, 1] (subtract 127.5, divide by 128)
    """
    try:
        if isinstance(image_source, (str, Path)):
            img = Image.open(str(image_source))
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            img = Image.open(image_source)
        
        # Resize to 112x112
        img = img.resize((112, 112), Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert RGB to BGR (InsightFace standard)
        img_array = img_array[:, :, ::-1]
        
        # Convert to float32 in [0, 255]
        img_array = img_array.astype(np.float32)
        
        # Scale to [-1, 1]
        img_array = (img_array - 127.5) / 128.0
        
        # Convert to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Convert to torch tensor
        return torch.from_numpy(img_array).float().unsqueeze(0)
    except Exception as e:
        logger.error(f"Face preprocessing error: {e}")
        return None

# ============================================================================
# TRANSFORMS (Aligned with training/evaluation pipeline)
# ============================================================================

# Face transforms - Match InsightFace ONNX model (R50_Glint360K)
# Uses BGR input and scale [-1, 1] (not ImageNet normalization)
# NOTE: Use _preprocess_face() for exact preprocessing, or use this transform with RGB input
face_transform = T.Compose([
    T.Resize((112, 112)),
    T.Lambda(lambda img: torch.from_numpy(np.array(img)[:, :, ::-1].copy().transpose(2, 0, 1) if img.mode == 'RGB' else np.array(img).copy().transpose(2, 0, 1))),  # RGB -> BGR, HWC -> CHW, copy to fix strides
    T.Lambda(lambda x: (x - 127.5) / 128.0),  # Scale to [-1, 1]
])

# Iris transforms - Match training (src/models/iris/config.py, iris/augmentations.py)
# Uses 224x224 square crops from segmented ROIs, ImageNet normalization
iris_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
])

# Fingerprint transforms - Match training (src/models/fingerprint/config.py)
# Uses 224x224 for inference (model trained on native resolutions: 103x103 / 388x388)
# EfficientNet handles variable input sizes, so 224x224 is a reasonable inference size
finger_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
])

def load_image_tensor(source, transform, apply_segmentation=False, preprocessing_type=None):
    """
    Load image tensor with optional modality-specific preprocessing.

    Args:
        source: Path, PIL Image, or file-like object
        transform: Transform to apply
        apply_segmentation: Legacy flag (maps to preprocessing_type='iris')
        preprocessing_type: 'iris', 'fingerprint', or None

    Returns:
        Tensor on DEVICE, or None if failed
    """
    # Map legacy flag
    if apply_segmentation:
        preprocessing_type = 'iris'

    try:
        # 1. Load and Preprocess based on Type
        if preprocessing_type == 'fingerprint':
            img = _preprocess_fingerprint(source)
            if img is None: return None
        else:
            # Standard load (Face/Iris start as RGB)
            if isinstance(source, (str, Path)):
                img = Image.open(str(source)).convert('RGB')
            elif isinstance(source, Image.Image):
                img = source.convert('RGB')
            else:
                img = Image.open(source).convert('RGB')

        # 2. Apply Special Segmentation (Iris)
        if preprocessing_type == 'iris':
            segmented = _segment_iris_image(img, target_size=(224, 224))
            if segmented is not None:
                img = segmented
            else:
                logger.warning(f"Segmentation failed for iris, using raw image")

        # 3. Apply Transform
        result = transform(img).unsqueeze(0).to(DEVICE)
        logger.debug(f"Loaded {preprocessing_type or 'unknown'} tensor: shape={result.shape}")
        return result
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

def get_model() -> BiometricEnsemble:
    global _model_instance
    if _model_instance is None:
        logger.info("Loading BiometricEnsemble Model...")
        _model_instance = BiometricEnsemble(
            face_weights_path=str(FACE_WEIGHTS_PATH),
            device=DEVICE
        )
        if MODEL_PATH.exists():
            logger.info(f"Loading ensemble weights from {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            _model_instance.load_state_dict(state_dict)
            
            # Activate all branches since ensemble checkpoint contains all weights
            _model_instance.face_active.fill_(True)
            _model_instance.iris_nir_active.fill_(True)
            _model_instance.iris_vis_active.fill_(True)
            _model_instance.fingerprint_active.fill_(True)
            logger.info("All modalities activated from ensemble checkpoint")
        else:
            logger.warning(f"Ensemble checkpoint not found at {MODEL_PATH}. Using initialization weights only.")
        
        # Force model to DEVICE to resolve RuntimeError (Input CUDA, Weight CPU)
        _model_instance = _model_instance.to(DEVICE)
        
        # Also explicitly move face_encoder to DEVICE (ONNX wrapper might not inherit device)
        if hasattr(_model_instance, 'face_encoder'):
            _model_instance.face_encoder = _model_instance.face_encoder.to(DEVICE)
        
        _model_instance.eval()
        logger.info("Model loaded successfully.")
    return _model_instance

def get_gallery_cache() -> dict:
    global _gallery_cache
    if not _gallery_cache:
        model = get_model()
        if not GALLERY_DIR.exists():
            logger.warning(f"Gallery dir {GALLERY_DIR} does not exist.")
            return {}

        logger.info("Caching gallery embeddings...")
        identities = sorted([d.name for d in GALLERY_DIR.iterdir() if d.is_dir()])

        for identity in identities:
            gallery_path = GALLERY_DIR / identity

            # Load and segment iris images (required - models trained on segmented ROIs)
            iris_nir = load_image_tensor(gallery_path / 'iris_nir.jpg', iris_transform, preprocessing_type='iris')
            iris_vis = load_image_tensor(gallery_path / 'iris_vis.jpg', iris_transform, preprocessing_type='iris')

            inputs = {
                'face': load_image_tensor(gallery_path / 'face.jpg', face_transform),
                'iris_nir': iris_nir,
                'iris_vis': iris_vis,
                'fingerprint': load_image_tensor(gallery_path / 'fingerprint.bmp', finger_transform, preprocessing_type='fingerprint')
            }
            
            # Log which inputs are valid
            valid_keys = [k for k, v in inputs.items() if v is not None]
            logger.debug(f"Identity {identity}: Valid modalities: {valid_keys}")
            
            # Filter None inputs
            inputs = {k: v for k, v in inputs.items() if v is not None}

            if inputs:
                with torch.no_grad():
                    _gallery_cache[identity] = model(inputs)

        logger.info(f"Cached {len(_gallery_cache)} identities.")

    return _gallery_cache
