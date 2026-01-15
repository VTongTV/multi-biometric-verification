"""
Fingerprint Training Configuration.
All hyperparameters and settings for fingerprint training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# Default progressive resolution stages (LEGACY - NOT USED IN NATIVE MODE)
# Format: (start_epoch, end_epoch, resolution, transform_mode, batch_size, accum_steps)
# DEFAULT_PROGRESSIVE_STAGES = (
#     (0, 20, 224, 'resize', 32, 1),           # Stage 1: Standard 224 resize
#     (20, 40, 224, 'crop_from_384', 32, 1),   # Stage 2: 224 crops from 384 (5-crop cycling)
#     (40, 50, 299, 'resize', 24, 2),          # Stage 3: Higher res 299
#     (50, 60, 336, 'resize', 16, 2),          # Stage 4: Final polish at 336
# )
DEFAULT_PROGRESSIVE_STAGES = ()



@dataclass
class TrainingConfig:
    """Configuration for fingerprint model training."""
    
    # Paths
    data_dir: str = 'data/raw'
    output_dir: str = 'outputs/fingerprint'
    
    # Training hyperparameters
    epochs: int = 60  # Reduced from 100 - progressive stages handle convergence
    batch_size: int = 32  # Base batch size (overridden per stage)
    
    # Stable default LRs
    lr_backbone: float = 0.0001 
    lr_head: float = 0.001 
    
    # Legacy LR
    lr: float = 0.001 
    
    weight_decay: float = 1e-4
    accumulation_steps: int = 1 # Increased to maintain effective BS of 32
    early_stopping_patience: int = 15  # INCREASED from 10 for better convergence
    freeze_backbone_epochs: int = 0
    
    # Model settings
    embedding_dim: int = 512
    dropout: float = 0.3
    
    # ArcFace settings
    arcface_scale: float = 64.0  # AdaFace standard scale
    arcface_margin: float = 0.4  # AdaFace standard margin
    
    # IMPROVEMENT: Margin warmup to prevent early embedding collapse
    margin_warmup_epochs: int = 5  # Warm up margin from 0 to target
    margin_schedule: str = 'linear'  # 'linear', 'cosine', or 'step'
    
    # IMPROVEMENT: SubCenter ArcFace for multi-finger datasets
    use_subcenter_arcface: bool = False  # Enable SubCenter ArcFace
    num_subcenters: int = 3  # Number of subcenters per class
    
    # Data loading
    num_workers: int = 2
    sample_fraction: float = 1.0  
    class_fraction: float = 1.0 
    image_size: int = 224 # Native EfficientNet resolution (Up from 160)
    
    # Sample Fraction Scheduling (disabled)
    use_sample_scheduling: bool = False
    sample_schedule: tuple = (
        (0, 1.0),
    )
    
    # Preprocessing
    use_gabor_preprocessing: bool = False  # Let EfficientNet learn features
    
    use_fast_decoding: bool = True  
    
    # Augmentation
    use_mixup: bool = False
    use_cutmix: bool = False  # Disabled - CutMix destroys fingerprint minutiae
    cutmix_alpha: float = 1.0
    
    # GridMask settings - REDUCED ratio to preserve minutiae
    gridmask_ratio: float = 0.2  # Was 0.4 - too aggressive for fingerprints
    gridmask_prob: float = 0.3
    
    # TTA settings
    use_tta: bool = True
    tta_num_aug: int = 5
    
    # Quick test mode - for fast verification
    quick_test: bool = False
    quick_test_epochs: int = 5
    quick_test_sample_fraction: float = 0.2
    quick_test_batch_size: int = 16
    
    # ====== NATIVE RESOLUTION TRAINING ======
    # Train at native resolutions without upscaling
    use_native_resolution: bool = True
    use_progressive_resolution: bool = False  # Disabled - use native instead

    # Resolution groups: each group has target size, batch settings, and path patterns
    native_resolution_groups: dict = field(default_factory=lambda: {
        'low': {
            'target': (103, 103),
            'batch_size': 128,
            'accum_steps': 1,
            'path_patterns': ['socofing'],
        },
        'high': {
            'target': (388, 388),
            'batch_size': 12,
            'accum_steps': 1,
            'path_patterns': ['fingerprint', 'fingerprints'],
        },
    })

    # Oversampling ratio for minority group (high-res)
    # 1.0 = fully balanced (50/50), 0.0 = natural distribution (~98/2)
    # Reduced to 0.5 to relieve VRAM pressure and speed up epochs
    oversample_minority_ratio: float = 0.1
    
    # If True, batches are sorted by resolution (Low -> High) within each epoch
    # If False, batches are randomly interleaved (Standard)
    train_low_res_first: bool = False

    # ====== PROGRESSIVE RESOLUTION TRAINING (LEGACY - DISABLED) ======
    # Kept for reference only, use native resolution instead
    # Stage schedule: (start_epoch, end_epoch, resolution, transform_mode, batch_size, accum_steps)
    # transform_mode: 'resize' = standard resize, 'crop_from_384' = random 224 crops from 384
    progressive_stages: Tuple = field(default_factory=lambda: DEFAULT_PROGRESSIVE_STAGES)

    # Early skip settings - skip to next stage if validation plateaus
    stage_skip_patience: int = 4  # Epochs without improvement to trigger skip
    lr_decay_on_skip: float = 0.5  # LR multiplied by this when skipping/transitioning stages

    # 5-crop cycling for 'crop_from_384' mode
    # Cycles through 5 deterministic crop positions every N batches
    crop_cycle_interval: int = 200  # Change crop position every 200 batches
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


def get_data_dirs(project_root: Path) -> List[Path]:
    """Get list of fingerprint dataset directories."""
    return [
        project_root / 'data/raw/socofing/SOCOFing',
        project_root / 'data/raw/fingerprint',
        project_root / 'data/raw/fingerprints',
    ]
