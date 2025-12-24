# Fusion modules
"""Cross-modal fusion architectures."""

from .fusion_modules import (
    GatedFusion,
    AttentionFusion,
    EyeFusionModule,
    TriModalFusionNet,
    SimpleScoreFusion,
    create_fusion_module
)

__all__ = [
    'GatedFusion',
    'AttentionFusion', 
    'EyeFusionModule',
    'TriModalFusionNet',
    'SimpleScoreFusion',
    'create_fusion_module'
]
