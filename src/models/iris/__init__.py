"""
Iris Model Package.
SOTA iris recognition with modular training components.
"""

from .config import TrainingConfig, get_data_dirs
from .dataset import IrisDataset, TransformSubset
from .augmentations import get_iris_transforms, mixup_data, mixup_criterion, Cutout
from .trainer import IrisTrainer

__all__ = [
    'TrainingConfig',
    'get_data_dirs',
    'IrisDataset',
    'TransformSubset',
    'get_iris_transforms',
    'mixup_data',
    'mixup_criterion',
    'Cutout',
    'IrisTrainer',
]
