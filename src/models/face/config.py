"""
Face Training Configuration.
All hyperparameters and settings for face training.
Aligned with iris model for consistency.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Default CelebA attributes for multi-task learning
DEFAULT_ATTRIBUTES = [
    'Male', 'Young', 'Eyeglasses', 'Smiling', 'Blurry',
    'Wearing_Hat', 'Attractive', 'Heavy_Makeup', 'Bald', 'Mustache'
]


@dataclass
class TrainingConfig:
    """Configuration for face model training."""
    
    # Paths
    output_dir: str = 'outputs/face'
    
    # Training hyperparameters (optimized for face recognition - SGD based)
    epochs: int = 100
    batch_size: int = 64  # Reduced from 72 to fix OOM on 4GB VRAM
    lr: float = 0.1  # Standard for SGD + ArcFace/AdaFace
    weight_decay: float = 5e-4
    accumulation_steps: int = 8  # Effective batch = 64*8 = 512
    
    # IMPROVEMENT: Configurable LR multipliers (was hardcoded)
    backbone_lr_mult: float = 0.1   # Backbone learns slower (pretrained)
    head_lr_mult: float = 1.0       # Head learns at base LR
    loss_lr_mult: float = 1.0       # Loss module LR (was 2.0 - too aggressive)
    
    # Optimizer choice (SGD is standard for face recognition)
    use_sgd: bool = True  # False = AdamW, True = SGD with momentum
    sgd_momentum: float = 0.9
    
    
    # Model settings
    embedding_dim: int = 512
    dropout: float = 0.4  # Same as iris
    
    # ====== NOVEL CONTRIBUTION 3: IResNet Backbone ======
    # Switch to face-optimized backbone (IResNet50) vs general purpose (EfficientNet)
    backbone_type: str = 'iresnet50'  # 'iresnet50', 'iresnet100', 'efficientnet'
    use_pretrained_face: bool = True  # Use pretrained weights if available
    pretrained_weights_path: str = 'weights/R50_Glint360K.onnx'  # InsightFace pretrained
    
    # Data balancing
    use_balanced_sampling: bool = True  # Use inverse frequency weighted sampling
    
    # ====== NOVEL CONTRIBUTION 1: AdaFace Loss ======
    # Quality-adaptive margin based on feature norms (CVPR 2022)
    use_adaface: bool = True
    adaface_scale: float = 64.0
    adaface_margin: float = 0.4  # Slightly higher than ArcFace default
    adaface_h: float = 0.333  # Quality-to-margin sensitivity
    adaface_t_alpha: float = 0.01  # Batch statistics smoothing
    
    # Legacy ArcFace settings (used when use_adaface=False)
    arcface_scale: float = 64.0
    arcface_margin: float = 0.50  # INCREASED from 0.30 to match original ArcFace paper
    
    # IMPROVEMENT: Margin warmup to prevent early embedding collapse
    margin_warmup_epochs: int = 10  # Warm up margin from 0 to target
    margin_schedule: str = 'linear'  # 'linear', 'cosine', or 'step'
    
    # IMPROVEMENT: SubCenter ArcFace for pose/expression variation
    use_subcenter_arcface: bool = False  # Enable SubCenter ArcFace
    num_subcenters: int = 3  # Number of subcenters per class
    
    # ====== NOVEL CONTRIBUTION 2: Multi-Task Attribute Learning ======
    # CelebA's 40 attributes for regularization
    use_attributes: bool = True
    attribute_loss_weight: float = 0.1  # Weight for attribute prediction loss
    attribute_names: List[str] = field(default_factory=lambda: DEFAULT_ATTRIBUTES.copy())
    
    # Scheduler settings
    warmup_epochs: int = 5
    eta_min: float = 1e-6
    
    # Early stopping - INCREASED for better convergence
    early_stop_patience: int = 1  # Was 5
    
    # Augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    use_randaugment: bool = True
    use_cutmix: bool = False  # CutMix augmentation
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.3
    
    # Data loading (optimized for Windows)
    num_workers: int = 2
    sample_fraction: float = 1.0
    image_size: int = 112  # IResNet standard input size is 112x112
    min_samples_per_class: int = 2  # Face datasets often have fewer samples per identity
    
    # Exclusion for valid benchmarking (Open-Set LFW)
    exclude_sources: List[str] = field(default_factory=lambda: ['lfw'])

    # Quick test mode - for fast verification that all graphs generate correctly
    quick_test: bool = False
    quick_test_samples_per_source: int = 10  # Classes per dataset source
    quick_test_epochs: int = 3
    
    # AT&T Validation Path (Optional override)
    atnt_path: str = None
    
    # Evaluation Mode
    evaluate_only: bool = False
    checkpoint_path: str = None  # Path to specific checkpoint for evaluation
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


def get_data_dirs(project_root: Path) -> List[Path]:
    """Get list of face dataset directories."""
    return [
        # LFW - primary face dataset
        project_root / 'data/raw/lfw-deepfunneled/lfw-deepfunneled',
        project_root / 'data/raw/lfw-deepfunneled',
        project_root / 'data/raw/lfw',
        # CelebA - large-scale celebrity faces
        project_root / 'data/raw/celeba',
    ]
