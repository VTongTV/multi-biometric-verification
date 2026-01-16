"""
Iris Training Configuration.
All hyperparameters and settings for iris training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Set


@dataclass
class TrainingConfig:
    """Configuration for iris model training."""
    
    # Paths
    output_dir: str = 'outputs/iris'
    
    # Training hyperparameters (optimized for 95%+ accuracy)
    epochs: int = 200  # Increased from 150 for better convergence
    batch_size: int = 32  # Adjusted from 24
    lr: float = 2.7e-4  # Scaled for effective batch size 64
    weight_decay: float = 5e-4
    accumulation_steps: int = 2
    
    # Learning rate multipliers for differential learning
    backbone_lr_mult: float = 0.3  # Increased from 0.1 for faster backbone adaptation
    arcface_lr_mult: float = 2.0   # ArcFace centers learn faster
    
    # Model settings
    embedding_dim: int = 512
    dropout: float = 0.4  # Increased from 0.3 for better regularization
    
    # ArcFace settings (balanced for FP16 stability)
    arcface_scale: float = 45.0  # Reduced from 64 - prevents FP16 overflow
    arcface_margin: float = 0.35  # Reduced from 0.50 - safer with FP16
    
    # IMPROVEMENT: Margin warmup to prevent early embedding collapse
    margin_warmup_epochs: int = 10  # Warm up margin from 0 to target over 10 epochs
    use_subcenter_arcface: bool = False  # Enable SubCenter ArcFace for L/R eye variation
    num_subcenters: int = 3  # Number of subcenters per class (if enabled)
    margin_schedule: str = 'linear'  # 'linear', 'cosine', or 'step'
    
    # Augmentation strength for domain-specific training
    augmentation_strength: str = 'medium'  # 'light', 'medium', 'heavy', 'very_heavy'
    
    # Scheduler settings
    warmup_epochs: int = 5
    eta_min: float = 1e-6
    
    # Early stopping - INCREASED for better convergence
    early_stop_patience: int = 1
    
    # Augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.7  # Higher probability for regularization
    use_cutmix: bool = False  # CutMix augmentation (better for small datasets)
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5  # Probability of CutMix when enabled
    
    # Data loading (optimized for Windows)
    num_workers: int = 2
    sample_fraction: float = 1.0
    image_size: Tuple[int, int] = (224, 224)  # Square ROI (was 512x64)
    min_samples_per_class: int = 4  # Increased from 3 for more reliable classes
    
    # New Preprocessing Options
    use_daugman: bool = False       # True=Old normalization, False=Square Crop
    quality_threshold: float = 0.3  # Reject samples with low quality
    allowed_datasets: Optional[Set[str]] = None    # If set, safeguards against loading wrong domain data
    
    # Quick test mode - for fast verification that all graphs generate correctly
    quick_test: bool = False
    quick_test_samples_per_source: int = 10  # Samples per dataset source
    quick_test_epochs: int = 3
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps
    
    def __post_init__(self):
        """Validate hyperparameters to prevent training instability."""
        # ArcFace safety checks - prevent FP16 overflow
        if self.arcface_scale > 50.0:
            import warnings
            warnings.warn(
                f"arcface_scale={self.arcface_scale} is high and may cause FP16 overflow. "
                f"Recommended: <= 50.0 for RTX 3050, <= 64.0 for higher-end GPUs.",
                UserWarning
            )
        
        if self.arcface_margin > 0.5:
            import warnings
            warnings.warn(
                f"arcface_margin={self.arcface_margin} is aggressive and may slow convergence "
                f"or cause NaN losses. Recommended: 0.3-0.5.",
                UserWarning
            )
        
        # Combined risk check
        if self.arcface_scale * self.arcface_margin > 25.0:
            import warnings
            warnings.warn(
                f"arcface_scale * arcface_margin = {self.arcface_scale * self.arcface_margin:.1f} "
                f"is high. Combined values > 25 increase risk of FP16 overflow.",
                UserWarning
            )


def get_data_dirs(project_root: Path) -> List[Path]:
    """Get list of iris dataset directories."""
    return [
        # UBIRIS2 dataset (actual images, not segmentation masks)
        project_root / 'data/raw/ubiris/ubiris2_1/CLASSES_400_300_Part1',
        project_root / 'data/raw/sbvpi/SBVPI_Sclera/Training/Images',
        project_root / 'data/raw/sbvpi/SBVPI_Sclera/Validation/Images',
        # CASIA datasets
        project_root / 'data/raw/CASIA-Iris-Thousand/CASIA-Iris-Thousand/CASIA-Iris-Thousand',
        project_root / 'data/raw/CASIA-Iris-Interval',
        project_root / 'data/raw/CASIA-Iris-Syn',
        project_root / 'data/raw/CASIA-Iris-Lamp',
        # MMU dataset
        project_root / 'data/raw/MMU-Iris-Database',
    ]
