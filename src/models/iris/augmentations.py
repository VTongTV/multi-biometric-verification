"""
Iris Data Augmentation Utilities.
Includes transforms, Cutout, and MixUp for training.
"""

import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, Union


class Cutout:
    """Randomly occlude a rectangular region in the image."""
    
    def __init__(self, n_holes: int = 1, length: int = 32):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


def get_iris_transforms(image_size: Union[int, Tuple[int, int]] = 224, is_train: bool = True):
    """
    Get augmentation pipeline for iris images.
    Includes rotation and occlusion robustness.
    """
    # Handle tuple input (Width, Height) or int
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        w, h = image_size  # Config stores (Width, Height)
    
    if is_train:
        # Data augmentation for improved accuracy
        # For rectangular images, we pad both dimensions for rotation safety
        pad_h, pad_w = 32, 32
        
        return transforms.Compose([
            transforms.Resize((h + pad_h, w + pad_w)),  # Resize (H, W) for TorchVision
            transforms.RandomRotation(15),  # Critical for tilt invariance
            transforms.RandomCrop((h, w)),  # Crop back to target size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Color augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.0)),  # Stronger occlusion
        ])
    else:
        return transforms.Compose([
            transforms.Resize((h, w)), # TorchVision Resize takes (H, W)
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    MixUp augmentation for improved generalization.
    Blends random pairs of images and labels.
    
    Returns:
        mixed_x: Blended images
        y_a, y_b: Original label pairs
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss as weighted combination of two label losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    CutMix augmentation - cuts and pastes patches between images.
    
    More effective than MixUp for small datasets as it forces the model
    to learn from partial object views.
    
    Paper: https://arxiv.org/abs/1905.04899
    
    Returns:
        mixed_x: Images with patches swapped
        y_a, y_b: Original label pairs
        lam: Actual mixing ratio based on bbox area
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get bounding box
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Uniform random center
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    # Bounding box coordinates (clipped to image)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Swap the patch
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual bbox area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_iris_transforms_by_strength(
    strength: str,
    image_size: Union[int, Tuple[int, int]] = 224,
    is_train: bool = True
):
    """
    Get iris augmentation pipeline based on strength level.
    
    Args:
        strength: One of 'light', 'medium', 'heavy', 'very_heavy'
        image_size: Target image size (int or tuple)
        is_train: Whether this is for training or validation
        
    Returns:
        torchvision.transforms.Compose pipeline
    """
    # Handle tuple input (Width, Height) or int
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        w, h = image_size  # Config stores (Width, Height)
    
    # Validation transforms are always the same
    if not is_train:
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    # Training transforms based on strength
    pad_h, pad_w = 32, 32
    
    if strength == 'light':
        return transforms.Compose([
            transforms.Resize((h + pad_h, w + pad_w)),
            transforms.RandomRotation(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0)),
        ])
    
    elif strength == 'medium':
        return transforms.Compose([
            transforms.Resize((h + pad_h, w + pad_w)),
            transforms.RandomRotation(15),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), ratio=(0.3, 3.0)),
        ])
    
    elif strength == 'heavy':
        return transforms.Compose([
            transforms.Resize((h + pad_h, w + pad_w)),
            transforms.RandomRotation(15),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.0)),
        ])
    
    elif strength == 'very_heavy':
        # Aggressive augmentation for small datasets (visible iris)
        # Helps prevent overfitting on limited data
        return transforms.Compose([
            transforms.Resize((h + pad_h, w + pad_w)),
            transforms.RandomRotation(20),  # More rotation
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Stronger color augmentation for visible light variance
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Blur for focus variance
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.30), ratio=(0.3, 3.0)),  # More occlusion
        ])
    
    else:
        # Default to standard (same as get_iris_transforms)
        return get_iris_transforms(image_size, is_train=True)
