# Inference module
"""Production inference pipeline."""

from .predictor import BiometricPredictor, load_predictor

__all__ = ['BiometricPredictor', 'load_predictor']
