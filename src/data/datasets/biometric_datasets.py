"""
PyTorch dataset classes for biometric modalities.

Provides datasets for:
- LFW (Labeled Faces in the Wild) - Face verification
- Generic identity folder dataset
- Verification pair dataset for testing
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from loguru import logger


class IdentityFolderDataset(Dataset):
    """
    Generic dataset for identity verification.
    
    Expects directory structure:
    root/
        identity_001/
            image_001.jpg
            image_002.jpg
        identity_002/
            image_001.jpg
            ...
    
    Args:
        root_dir: Path to dataset root
        transform: Optional transform to apply to images
        min_samples: Minimum samples per identity (filters out identities with fewer)
        extensions: Valid image extensions
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        min_samples: int = 1,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        # Scan directory for identities
        self.samples: List[Tuple[Path, int]] = []  # (image_path, label)
        self.identity_to_label: Dict[str, int] = {}
        self.label_to_identity: Dict[int, str] = {}
        self.identity_indices: Dict[int, List[int]] = {}  # label -> [sample indices]
        
        self._scan_directory(min_samples)
        
        logger.info(
            f"IdentityFolderDataset: {len(self.samples)} images, "
            f"{len(self.identity_to_label)} identities from {self.root_dir}"
        )
    
    def _scan_directory(self, min_samples: int) -> None:
        """Scan directory for identity folders and images."""
        label = 0
        
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            
            # Find all images for this identity
            images = [
                f for f in identity_dir.iterdir()
                if f.suffix.lower() in self.extensions
            ]
            
            if len(images) < min_samples:
                continue
            
            identity_name = identity_dir.name
            self.identity_to_label[identity_name] = label
            self.label_to_identity[label] = identity_name
            self.identity_indices[label] = []
            
            for img_path in sorted(images):
                sample_idx = len(self.samples)
                self.samples.append((img_path, label))
                self.identity_indices[label].append(sample_idx)
            
            label += 1
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)
    
    def get_identity_samples(self, label: int) -> List[int]:
        """Get all sample indices for a given identity label."""
        return self.identity_indices.get(label, [])


class LFWDataset(IdentityFolderDataset):
    """
    LFW (Labeled Faces in the Wild) dataset.
    
    Specialized for face verification with LFW-specific structure.
    
    Args:
        root_dir: Path to LFW root (should contain identity folders)
        transform: Optional transform
        split: 'train', 'val', or 'test'
        train_ratio: Ratio for train split
        val_ratio: Ratio for validation split
        seed: Random seed for split
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        # Default transform for LFW faces
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # LFW requires at least 2 images per identity for verification
        super().__init__(root_dir, transform, min_samples=2)
        
        # Create train/val/test splits by identity
        self.split = split
        self._create_splits(train_ratio, val_ratio, seed)
    
    def _create_splits(
        self,
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> None:
        """Create train/val/test splits by identity."""
        random.seed(seed)
        
        all_labels = list(self.identity_to_label.values())
        random.shuffle(all_labels)
        
        n = len(all_labels)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if self.split == 'train':
            valid_labels = set(all_labels[:train_end])
        elif self.split == 'val':
            valid_labels = set(all_labels[train_end:val_end])
        else:  # test
            valid_labels = set(all_labels[val_end:])
        
        # Filter samples
        filtered_samples = []
        for img_path, label in self.samples:
            if label in valid_labels:
                filtered_samples.append((img_path, label))
        
        self.samples = filtered_samples
        
        # Update identity indices
        self.identity_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.identity_indices:
                self.identity_indices[label] = []
            self.identity_indices[label].append(idx)
        
        logger.info(f"LFW {self.split} split: {len(self.samples)} images")


class VerificationPairDataset(Dataset):
    """
    Dataset for verification pair testing.
    
    Generates positive and negative pairs for verification evaluation.
    
    Args:
        base_dataset: Underlying identity dataset
        num_positive_pairs: Number of positive pairs to generate
        num_negative_pairs: Number of negative pairs to generate
        seed: Random seed
    """
    
    def __init__(
        self,
        base_dataset: IdentityFolderDataset,
        num_positive_pairs: int = 3000,
        num_negative_pairs: int = 3000,
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.seed = seed
        
        random.seed(seed)
        
        # Generate pairs
        self.pairs: List[Tuple[int, int, int]] = []  # (idx1, idx2, is_same)
        
        self._generate_positive_pairs(num_positive_pairs)
        self._generate_negative_pairs(num_negative_pairs)
        
        # Shuffle pairs
        random.shuffle(self.pairs)
        
        logger.info(
            f"VerificationPairDataset: {len(self.pairs)} pairs "
            f"({num_positive_pairs} positive, {num_negative_pairs} negative)"
        )
    
    def _generate_positive_pairs(self, num_pairs: int) -> None:
        """Generate positive pairs (same identity)."""
        count = 0
        attempts = 0
        max_attempts = num_pairs * 10
        
        labels = list(self.base_dataset.identity_indices.keys())
        
        while count < num_pairs and attempts < max_attempts:
            attempts += 1
            
            # Pick random identity with at least 2 samples
            label = random.choice(labels)
            indices = self.base_dataset.identity_indices[label]
            
            if len(indices) < 2:
                continue
            
            idx1, idx2 = random.sample(indices, 2)
            self.pairs.append((idx1, idx2, 1))  # 1 = same identity
            count += 1
    
    def _generate_negative_pairs(self, num_pairs: int) -> None:
        """Generate negative pairs (different identity)."""
        labels = list(self.base_dataset.identity_indices.keys())
        
        for _ in range(num_pairs):
            # Pick two different identities
            label1, label2 = random.sample(labels, 2)
            
            idx1 = random.choice(self.base_dataset.identity_indices[label1])
            idx2 = random.choice(self.base_dataset.identity_indices[label2])
            
            self.pairs.append((idx1, idx2, 0))  # 0 = different identity
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        idx1, idx2, is_same = self.pairs[idx]
        
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        
        return img1, img2, is_same


class GrayscaleDataset(Dataset):
    """
    Dataset wrapper that converts images to grayscale.
    
    Used for iris and fingerprint modalities.
    
    Args:
        base_dataset: Underlying dataset
        output_channels: 1 for grayscale, 3 for RGB (repeated)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        output_channels: int = 1
    ):
        self.base_dataset = base_dataset
        self.output_channels = output_channels
        
        # Grayscale transform
        self.to_gray = transforms.Grayscale(num_output_channels=output_channels)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        
        # Convert to grayscale if needed
        if image.shape[0] == 3:
            # If tensor, convert channels
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            if self.output_channels == 1:
                image = gray.unsqueeze(0)
            else:
                image = gray.unsqueeze(0).repeat(3, 1, 1)
        
        return image, label


def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
    modality: str = 'face'
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        input_size: Target image size (H, W)
        modality: 'face', 'iris', 'fingerprint', 'periocular'
        
    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    if modality == 'face':
        base_transforms.extend([
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    elif modality == 'periocular':
        base_transforms.extend([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    elif modality in ['iris', 'fingerprint']:
        # Less augmentation for texture-based modalities
        base_transforms.extend([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        ])
    
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406] if modality in ['face', 'periocular'] else [0.5],
            std=[0.229, 0.224, 0.225] if modality in ['face', 'periocular'] else [0.5]
        )
    ])
    
    return transforms.Compose(base_transforms)


def get_val_transforms(
    input_size: Tuple[int, int] = (224, 224),
    modality: str = 'face'
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        input_size: Target image size
        modality: 'face', 'iris', 'fingerprint', 'periocular'
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406] if modality in ['face', 'periocular'] else [0.5],
            std=[0.229, 0.224, 0.225] if modality in ['face', 'periocular'] else [0.5]
        )
    ])


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: Source dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        train_ratio: Ratio of data for training
        seed: Random seed for split
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split indices
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    
    split = int(n * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
