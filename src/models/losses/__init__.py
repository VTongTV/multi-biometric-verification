# Loss functions
"""Custom loss functions including angular margin losses."""

from .arcface import (
    ArcFaceLoss,
    LabelSmoothingCrossEntropy
)

__all__ = [
    'ArcFaceLoss',
    'LabelSmoothingCrossEntropy'
]
