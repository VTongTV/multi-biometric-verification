"""
Face Recognition Models.

Modular SOTA face recognition training with:
- LFW and CelebA dataset support
- ArcFace loss with scale=64, margin=0.3
- MixUp augmentation
"""

from .config import TrainingConfig, get_data_dirs
from .dataset import FaceDataset, TransformSubset
from .augmentations import get_face_transforms, mixup_data, mixup_criterion
from .trainer import FaceTrainer

__all__ = [
    'TrainingConfig',
    'get_data_dirs',
    'FaceDataset', 
    'TransformSubset',
    'get_face_transforms',
    'mixup_data',
    'mixup_criterion',
    'FaceTrainer',
]
