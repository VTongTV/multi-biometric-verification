"""
Face Data Augmentation Utilities.
Includes transforms, Cutout, GridMask, and MixUp for training.
"""

import random
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple


class Cutout:
    """
    Randomly mask out square regions of input for regularization.
    Paper: https://arxiv.org/abs/1708.04552
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
        l = int(d * self.ratio + 0.5)
        mask = torch.ones_like(img)
        
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        
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
        
        mask = 1 - mask
        return img * mask





def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp augmentation for improved generalization.
    
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



def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute MixUp loss as weighted combination of two label losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation - cuts and pastes patches between images.
    
    More effective than MixUp for face recognition as it forces the model
    to learn from partial face views.
    
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


def get_face_transforms(image_size: int = 224, is_train: bool = True, use_randaugment: bool = True):
    """
    Get augmentation pipeline for face images.
    Includes SOTA augmentations for robust face recognition.
    """
    if is_train:
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if use_randaugment:
            # RandAugment with N=2, M=9 (standard for ImageNet-scale)
            # We exclude some operations in policy if needed, but standard is usually fine
            transforms_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        else:
            # Legacy manual augmentations
            transforms_list.extend([
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
            
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(n_holes=1, length=40),
            GridMask(d1=64, d2=128, ratio=0.5, prob=0.3),
        ])
        
        return transforms.Compose(transforms_list)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
