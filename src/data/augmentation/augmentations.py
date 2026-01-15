"""
SOTA Data Augmentation Techniques.

Implements Cutout, Mixup, and comprehensive augmentation pipelines.
"""

import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter
try:
    import kornia.augmentation as K
except ImportError:
    K = None





# ============================================================================
# NEW ROBUSTNESS AUGMENTATIONS (2024 UPDATE)
# ============================================================================

class KorniaElasticTransform(nn.Module):
    """
    GPU-accelerated Elastic Transform using Kornia.
    Mimics skin plasticity (non-linear distortion).
    """
    def __init__(self, alpha=63.0, sigma=32.0, p=0.5):
        super().__init__()
        self.p = p
        if K is not None:
            # Kornia expects (B, C, H, W)
            self.aug = K.RandomElasticTransform(
                kernel_size=(63, 63),
                sigma=(sigma, sigma),
                alpha=(1.0, 1.0),
                p=1.0 # Applied conditionally in forward
            )
        else:
            self.aug = None
    
    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img
            
        if self.aug is None:
            return img
            
        # Ensure tensor is (C, H, W) -> (1, C, H, W) for Kornia
        if img.dim() == 3:
            return self.aug(img.unsqueeze(0)).squeeze(0)
        return self.aug(img)


class ISONoise(nn.Module):
    """
    Simulates high ISO noise (dry sensor/electronic noise).
    """
    def __init__(self, color_shift=0.05, intensity=0.1, p=0.2):
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img
            
        noise = torch.randn_like(img) * self.intensity
        return torch.clamp(img + noise, 0, 1)


class GaborFilterAugment(nn.Module):
    """
    Simulates biological ridge frequency variations using Gabor filtering.
    """
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
        
    def forward(self, img):
        # Full Gabor is computationally heavy; for now we rely on the robustness 
        # provided by Elastic + Noise to cover similar domain shifts.
        return img 


class Cutout:
    """
    Randomly mask out square regions of input for regularization.
    
    Paper: https://arxiv.org/abs/1708.04552
    
    Args:
        n_holes: Number of patches to cut out
        length: Length (in pixels) of each square patch
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to tensor image."""
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return img * mask


class GridMask:
    """
    GridMask data augmentation for improved robustness.
    
    Paper: https://arxiv.org/abs/2001.04086
    
    Systematically removes square regions in a grid pattern,
    improving model generalization for biometric recognition.
    
    Args:
        d1: Minimum grid size
        d2: Maximum grid size
        ratio: Ratio of mask to total grid area
        prob: Probability of applying GridMask
    """
    
    def __init__(self, d1: int = 96, d2: int = 224, ratio: float = 0.6, prob: float = 0.5):
        self.d1 = d1
        self.d2 = d2
        self.ratio = ratio
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply GridMask to tensor image."""
        if random.random() > self.prob:
            return img
        
        _, h, w = img.shape
        d = random.randint(self.d1, self.d2)
        
        # Grid size
        l = int(d * self.ratio + 0.5)
        
        # Create mesh grid
        mask = torch.ones_like(img)
        
        # Random start point
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        
        # Apply grid pattern
        for i in range(-1, h // d + 1):
            s = d * i + st_h
            t = s + l
            s = max(0, s)
            t = min(h, t)
            if s < t:
                mask[:, s:t, :] = 0
        
        for i in range(-1, w // d + 1):
            s = d * i + st_w
            t = s + l
            s = max(0, s)
            t = min(w, t)
            if s < t:
                mask[:, :, s:t] = 0
        
        # XOR to create grid pattern (invert the crosses)
        mask = 1 - mask
        return img * mask


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Apply Mixup augmentation to a batch.
    
    Paper: https://arxiv.org/abs/1710.09412
    
    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        alpha: Mixup interpolation strength
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply CutMix augmentation to a batch.
    
    Paper: https://arxiv.org/abs/1905.04899
    
    CutMix is proven superior to Cutout for fine-grained classification:
    - Replaces removed regions with patches from other images
    - Mixes labels proportionally to area
    - Achieves 2.28% lower top-1 error on ImageNet than Cutout
    
    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter (1.0 = uniform distribution)
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Image dimensions
    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box coordinates (clipped to image bounds)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    
    # Apply cutmix: paste patch from shuffled images
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to actual patch area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class GrayscaleTo3Channel:
    """Convert 1-channel grayscale to 3-channel by duplication. Picklable for Windows multiprocessing."""
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

class ElasticTransform:
    """
    Elastic deformation of images as described in [Simard2003].
    
    This is the "Gold Standard" augmentation for fingerprints/MNIST.
    Mimics skin deformation when pressing against a sensor.
    """
    def __init__(self, alpha=200, sigma=8, alpha_affine=None, random_state=None):
        """
        Args:
            alpha: Scipy scaling factor (intensity of deformation)
            sigma: Smoothing factor (elasticity)
        """
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor
        """
        is_pil = False
        if not isinstance(img, np.ndarray) and not isinstance(img, torch.Tensor):
            img = np.array(img)
            is_pil = True
            
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)

        shape = img.shape
        # Handle 2D or 3D images
        h, w = shape[:2]
        
        dx = gaussian_filter((self.random_state.rand(h, w) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((self.random_state.rand(h, w) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # Check for 3 channels (RGB)
        if len(shape) == 3:
            res = np.zeros_like(img)
            for i in range(shape[2]):
                res[:, :, i] = map_coordinates(img[:, :, i], indices, order=1, mode='reflect').reshape(shape[:2])
        else:
             res = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

        if is_pil:
            from PIL import Image
            return Image.fromarray(res)
            
        return res


def get_fingerprint_transforms(image_size: int = 224, is_train: bool = True):
    """Get gentler augmentation pipeline for fingerprints.
    
    IMPORTANT: Fingerprint minutiae are tiny (3-5px). Aggressive augmentations
    like Cutout and GridMask destroy these discriminative features.
    
    CRITICAL: We convert grayscale to 3-channel by duplication to leverage
    pretrained RGB weights from ImageNet (research strongly recommends this).
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            
            # 1. ELASTIC DEFORMATION
            transforms.RandomApply([
                transforms.ToTensor(), # Kornia needs Tensor
                KorniaElasticTransform(p=0.5),
                transforms.ToPILImage() # Convert back for text transforms
            ], p=0.5),

            # 2. SENSOR DEGRADATION (Minimal)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ], p=0.1),
            
            # 3. CUTOUT / GRIDMASK - REDUCED ratio to preserve fingerprint minutiae
            transforms.RandomApply([
                 transforms.ToTensor(),
                 GridMask(d1=96, d2=224, ratio=0.2, prob=1.0), # Reduced from 0.4 to 0.2
                 transforms.ToPILImage()
            ], p=0.3),

            transforms.RandomPerspective(distortion_scale=0.2, p=0.2), 
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.05, 0.05), 
                scale=(0.95, 1.05), 
                shear=5
            ),
            
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            
            # 4. FINAL NOISE (ISO)
            ISONoise(intensity=0.05, p=0.1),

            GrayscaleTo3Channel(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            GrayscaleTo3Channel(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_face_transforms(image_size: int = 224, is_train: bool = True):
    """Get SOTA augmentation pipeline for faces."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(n_holes=1, length=40),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_iris_transforms(image_size: int = 224, is_train: bool = True):
    """Get SOTA augmentation pipeline for iris."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ============================================================================
# PROGRESSIVE RESOLUTION TRANSFORMS FOR FINGERPRINT
# ============================================================================

class FiveCropCycling:
    """
    Cycles through 5 deterministic crop positions from a larger image.
    
    Used for progressive resolution training Stage 2: loads images at 384x384,
    extracts 224x224 crops from 5 positions (4 corners + center).
    
    The crop position cycles every N batches (configured externally).
    """
    
    # Crop positions: top-left, top-right, bottom-left, bottom-right, center
    POSITIONS = ['tl', 'tr', 'bl', 'br', 'center']
    
    def __init__(self, source_size: int = 384, crop_size: int = 224):
        self.source_size = source_size
        self.crop_size = crop_size
        self._current_position_idx = 0
    
    def set_position(self, position_idx: int):
        """Set the current crop position (0-4)."""
        self._current_position_idx = position_idx % 5
    
    def get_crop_coords(self) -> tuple:
        """Get (top, left) coordinates for current crop position."""
        s = self.source_size
        c = self.crop_size
        margin = s - c  # How much space we have
        
        pos = self.POSITIONS[self._current_position_idx]
        
        if pos == 'tl':
            return (0, 0)
        elif pos == 'tr':
            return (0, margin)
        elif pos == 'bl':
            return (margin, 0)
        elif pos == 'br':
            return (margin, margin)
        else:  # center
            return (margin // 2, margin // 2)
    
    def __call__(self, img):
        """Apply crop at current position."""
        top, left = self.get_crop_coords()
        return TF.crop(img, top, left, self.crop_size, self.crop_size)
    
    def __repr__(self):
        return f"FiveCropCycling(source={self.source_size}, crop={self.crop_size}, pos={self.POSITIONS[self._current_position_idx]})"


def get_fingerprint_transforms_progressive(
    image_size: int = 224,
    is_train: bool = True,
    transform_mode: str = 'resize',
    source_size: int = 384,
    crop_cycler = None  # Optional FiveCropCycling instance
):
    """
    Progressive resolution transforms for fingerprint training.

    Supports multiple modes for different training stages:
    - 'resize': Standard resize to image_size (Stage 1, 3, 4)
    - 'crop_from_384': Load at source_size, apply 5-crop cycling (Stage 2)

    Args:
        image_size: Target image size for the network
        is_train: Whether this is for training (applies augmentations)
        transform_mode: 'resize' or 'crop_from_384'
        source_size: Source image size for crop_from_384 mode
        crop_cycler: FiveCropCycling instance for position tracking

    Returns:
        transforms.Compose pipeline
    """
    if transform_mode == 'crop_from_384':
        # Stage 2: High-resolution crops
        if crop_cycler is None:
            crop_cycler = FiveCropCycling(source_size=source_size, crop_size=image_size)

        if is_train:
            return transforms.Compose([
                # Load at higher resolution
                transforms.Resize((source_size, source_size)),

                # Apply 5-crop cycling (position set externally per batch interval)
                crop_cycler,

                # Standard augmentations (applied to cropped region)
                transforms.RandomRotation(10),

                # Elastic deformation
                transforms.RandomApply([
                    transforms.ToTensor(),
                    KorniaElasticTransform(p=0.5),
                    transforms.ToPILImage()
                ], p=0.5),

                # Minimal sensor degradation
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                ], p=0.1),

                # GridMask (reduced ratio for fingerprints)
                transforms.RandomApply([
                    transforms.ToTensor(),
                    GridMask(d1=96, d2=image_size, ratio=0.2, prob=1.0),
                    transforms.ToPILImage()
                ], p=0.3),

                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    shear=5
                ),

                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),

                # Final noise
                ISONoise(intensity=0.05, p=0.1),

                GrayscaleTo3Channel(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            # Validation: center crop from high-res
            return transforms.Compose([
                transforms.Resize((source_size, source_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                GrayscaleTo3Channel(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    else:
        # Standard resize mode (Stages 1, 3, 4)
        return get_fingerprint_transforms(image_size, is_train)


def get_fingerprint_transforms_native(
    target_size: tuple = (103, 103),
    is_train: bool = True
) -> transforms.Compose:
    """
    Augmentation pipeline for native resolution training.

    NO UPSCALING - images are padded/cropped to target_size before this.
    Augmentation intensity scales with resolution to preserve minutiae.

    Args:
        target_size: (H, W) tuple - image already padded/cropped to this size
        is_train: Whether training mode

    Returns:
        transforms.Compose pipeline
    """
    h, w = target_size
    scale = min(h, w) / 224  # Scale factor relative to standard size

    # Scale GridMask parameters with resolution
    gm_d1 = max(16, int(48 * scale))
    gm_d2 = max(32, int(112 * scale))

    if is_train:
        return transforms.Compose([
            # NO Resize - already at native size
            transforms.RandomRotation(10),

            # Elastic (resolution-agnostic via Kornia)
            transforms.RandomApply([
                transforms.ToTensor(),
                KorniaElasticTransform(p=0.5),
                transforms.ToPILImage()
            ], p=0.5),

            # Blur
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ], p=0.1),

            # GridMask (scaled)
            transforms.RandomApply([
                transforms.ToTensor(),
                GridMask(d1=gm_d1, d2=gm_d2, ratio=0.2, prob=1.0),
                transforms.ToPILImage()
            ], p=0.3),

            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=4
            ),

            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            ISONoise(intensity=0.05, p=0.1),

            GrayscaleTo3Channel(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            GrayscaleTo3Channel(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
