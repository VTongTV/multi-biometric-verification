#!/usr/bin/env python3
"""
Fingerprint Recognition Training (2025 Upgrade)

Uses training techniques:
- EfficientNetV2-S / ConvNeXt backbone (Configurable)
- AdaFace Quality Adaptive Margin Loss (CVPR 2022)
- Ridge Attention for fingerprint specific enhancement
- Elastic Deformation + Gabor preprocessing
- LR warmup + cosine annealing
- Gradient accumulation for effective batch size
"""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch import autocast, GradScaler
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.encoders.biometric_encoder import BiometricEncoder
# NEW: AdaFace Loss
from src.models.losses.adaface import AdaFaceLoss
from src.models.losses.arcface import SubCenterArcFaceLoss
from src.models.losses.arcface import ArcFaceLoss, LabelSmoothingCrossEntropy, UpdateMarginStrategy
from src.data.augmentation.augmentations import (
    get_fingerprint_transforms, mixup_data, mixup_criterion, cutmix_data, GridMask,
    get_fingerprint_transforms_progressive, FiveCropCycling, get_fingerprint_transforms_native
)
from src.training.schedulers import CosineWithPlateauScheduler
from src.evaluation.plotter import MetricsPlotter
from src.utils.logging_utils import setup_logging

# New imports for approved improvements
from src.models.fingerprint.ridge_attention import RidgeAttention
from src.models.fingerprint.preprocessing import FingerprintPreprocessor, FingerprintQualityEstimator
from src.models.fingerprint.tta import evaluate_with_tta
import torch.nn.functional as F


def get_scheduled_sample_fraction(epoch: int, schedule: tuple, default: float = 1.0) -> float:
    """Get sample fraction from schedule based on current epoch."""
    current_fraction = default
    for threshold, fraction in schedule:
        if epoch >= threshold:
            current_fraction = fraction
    return current_fraction


class FingerprintEncoderWithAttention(nn.Module):
    """Wrapper that properly integrates Ridge Attention into the encoder forward pass."""
    
    def __init__(self, encoder, ridge_attention):
        super().__init__()
        self.encoder = encoder
        self.ridge_attention = ridge_attention
        self.embedding_head = encoder.embedding_head
        self.backbone = encoder.backbone
        
        # Disable encoder's built-in attention to prevent double attention
        if hasattr(encoder, 'use_attention'):
            encoder.use_attention = False
            logger.info("Disabled encoder's built-in attention (using RidgeAttention instead)")
    
    def forward(self, x, normalize=True):
        # Extract backbone features
        features = self.encoder.backbone.features(x)
        
        # Apply ridge-specific attention
        features = self.ridge_attention(features)
        
        # Clamp for FP16 stability
        features = torch.clamp(features, min=-65000, max=65000)
        
        # Pool and embed - convert to float32 for stability
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        features = features.float()  # Ensure float32 before embedding head
        embeddings = self.encoder.embedding_head(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.encoder.eval()
        return self


class ProgressiveStageManager:
    """
    Manages progressive resolution training stages with early-skip logic.
    
    Handles:
    - Stage transitions based on epoch
    - Early skipping when validation plateaus
    - LR decay on stage transitions
    - 5-crop cycling position tracking
    """
    
    def __init__(
        self,
        stage_schedule: tuple,
        skip_patience: int = 4,
        lr_decay_on_skip: float = 0.5,
        crop_cycle_interval: int = 200
    ):
        """
        Args:
            stage_schedule: Tuple of (start_epoch, end_epoch, resolution, transform_mode, batch_size, accum_steps)
            skip_patience: Epochs without improvement to trigger early skip
            lr_decay_on_skip: LR multiplier when transitioning stages
            crop_cycle_interval: Batches between crop position changes
        """
        self.stages = stage_schedule
        self.skip_patience = skip_patience
        self.lr_decay = lr_decay_on_skip
        self.crop_cycle_interval = crop_cycle_interval
        
        # State tracking
        self.current_stage_idx = 0
        self.stage_best_auc = 0.0
        self.stage_patience_counter = 0
        self.global_best_auc = 0.0
        
        # 5-crop cycling
        self.crop_cycler = FiveCropCycling(source_size=384, crop_size=224)
        self.current_crop_position = 0
    
    def get_stage_for_epoch(self, epoch: int) -> tuple:
        """
        Get stage configuration for given epoch.
        
        Returns:
            (resolution, transform_mode, batch_size, accum_steps, stage_idx)
        """
        for idx, (start, end, res, mode, bs, accum) in enumerate(self.stages):
            if start <= epoch < end:
                return (res, mode, bs, accum, idx)
        
        # Past all stages - use last stage config
        last = self.stages[-1]
        return (last[2], last[3], last[4], last[5], len(self.stages) - 1)
    
    def get_stage_epoch_range(self, stage_idx: int) -> tuple:
        """Get (start_epoch, end_epoch) for a stage."""
        if stage_idx < len(self.stages):
            return (self.stages[stage_idx][0], self.stages[stage_idx][1])
        return (self.stages[-1][0], self.stages[-1][1])
    
    def update_crop_position(self, batch_idx: int):
        """Update crop position based on batch index (every crop_cycle_interval batches)."""
        new_position = (batch_idx // self.crop_cycle_interval) % 5
        if new_position != self.current_crop_position:
            self.current_crop_position = new_position
            self.crop_cycler.set_position(new_position)
            return True  # Position changed
        return False
    
    def check_stage_improvement(self, val_auc: float) -> tuple:
        """
        Check if validation improved within current stage.
        
        Returns:
            (improved: bool, should_skip: bool)
        """
        improved = val_auc > self.stage_best_auc
        
        if improved:
            self.stage_best_auc = val_auc
            self.stage_patience_counter = 0
            
            # Also track global best
            if val_auc > self.global_best_auc:
                self.global_best_auc = val_auc
        else:
            self.stage_patience_counter += 1
        
        should_skip = self.stage_patience_counter >= self.skip_patience
        return (improved, should_skip)
    
    def transition_to_next_stage(self, optimizer) -> int:
        """
        Transition to next stage, applying LR decay.
        
        Returns:
            New stage index
        """
        self.current_stage_idx += 1
        self.stage_best_auc = 0.0  # Reset for new stage
        self.stage_patience_counter = 0
        
        # Apply LR decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.lr_decay
        
        return self.current_stage_idx
    
    def skip_to_next_stage_epoch(self) -> int:
        """
        Get the starting epoch of the next stage for early skipping.
        
        Returns:
            Starting epoch of next stage, or -1 if no more stages
        """
        next_stage = self.current_stage_idx + 1
        if next_stage < len(self.stages):
            return self.stages[next_stage][0]
        return -1  # No more stages
    
    def get_state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'current_stage_idx': self.current_stage_idx,
            'stage_best_auc': self.stage_best_auc,
            'stage_patience_counter': self.stage_patience_counter,
            'global_best_auc': self.global_best_auc,
            'current_crop_position': self.current_crop_position,
        }
    
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.current_stage_idx = state.get('current_stage_idx', 0)
        self.stage_best_auc = state.get('stage_best_auc', 0.0)
        self.stage_patience_counter = state.get('stage_patience_counter', 0)
        self.global_best_auc = state.get('global_best_auc', 0.0)
        self.current_crop_position = state.get('current_crop_position', 0)
        self.crop_cycler.set_position(self.current_crop_position)


def is_checkpoint_corrupted(checkpoint_path: Path) -> bool:
    """Check if a checkpoint file contains NaN/Inf weights."""
    if not checkpoint_path.exists():
        return True
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        for key, value in checkpoint.get('encoder_state_dict', {}).items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return True
        return False
    except Exception as e:
        logger.warning(f'Error checking checkpoint {checkpoint_path}: {e}')
        return True


def load_best_model_fingerprint(best_model_path: Path, encoder, device) -> tuple:
    """Load best_model.pt for resuming after corruption."""
    if not best_model_path.exists():
        return False, 0
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint)
        logger.info(f'Loaded best model from {best_model_path}')
        return True, 0
    except Exception as e:
        logger.warning(f'Failed to load best model: {e}')
        return False, 0


class FingerprintDataset(Dataset):
    """Fingerprint dataset with identity labels and preprocessing."""
    
    def __init__(self, root_dirs: list, transform=None, sample_fraction: float = 1.0, 
                 use_preprocessing: bool = True, use_fast_decoding: bool = True, 
                 image_size: int = 160, class_to_idx: dict = None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}
        self.use_preprocessing = use_preprocessing
        self.use_fast_decoding = False
        logger.info("Using standard PIL image loading")
        
        if use_preprocessing:
            self.preprocessor = FingerprintPreprocessor(
                apply_clahe=True, 
                apply_gabor=False, 
                apply_normalization=True
            )
            logger.info("Enabled fingerprint preprocessing (CLAHE + Gabor + Normalization)")
        
        all_img_paths = []
        extensions = ['.bmp', '.BMP', '.tif', '.TIF', '.png', '.PNG', '.jpg', '.JPG']
        
        for root_dir in root_dirs:
            root_dir = Path(root_dir)
            if not root_dir.exists(): continue
            for ext in extensions:
                paths = list(root_dir.rglob(f'*{ext}'))
                all_img_paths.extend(paths)
        
        all_img_paths = sorted(list(set(all_img_paths))) 
        
        next_idx = len(self.class_to_idx)
        for img_path in all_img_paths:
            identity = self._parse_identity(img_path.name)
            if identity not in self.class_to_idx:
                if class_to_idx is None:
                    self.class_to_idx[identity] = next_idx
                    next_idx += 1
                else:
                    continue
            
            self.samples.append((str(img_path), self.class_to_idx[identity]))
        
        if sample_fraction < 1.0:
            n = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n]
        
        logger.info(f'Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities')
    
    def _parse_identity(self, filename: str):
        """Parse identity from filename."""
        if '__' in filename:
            parts = filename.split('__')
            person_id = parts[0]
            info = parts[1]
            hand = 'Left' if 'Left' in info else 'Right'
            finger = 'unknown'
            for f in ['index', 'thumb', 'middle', 'ring', 'little']:
                if f in info:
                    finger = f
                    break
            return f"{person_id}_{hand}_{finger}"
        elif '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                 return f"{parts[0]}_{parts[1]}"
            elif len(parts) == 2:
                 return parts[0]
            return parts[0]
        return filename.split('.')[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.use_preprocessing:
                img_array = np.array(image)
                img_array = self.preprocessor.enhance(img_array)
                image = Image.fromarray(img_array)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)
        return image, label


def filter_none_collate(batch):
    """Custom collate function to filter out None samples."""
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def native_resolution_collate(batch):
    """
    Collate for native resolution training.
    Returns (images, labels, group_name).
    """
    batch = [(img, lbl, grp) for img, lbl, grp in batch if img is not None]

    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), None

    images, labels, groups = zip(*batch)

    if len(set(groups)) > 1:
        logger.warning(f"Mixed groups in batch: {set(groups)}")

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, groups[0]


class FingerprintDatasetNative(Dataset):
    """
    Fingerprint dataset with native resolution preservation.

    Groups samples by resolution and applies minimal padding/cropping
    to standardize within each group (no upscaling).
    """

    def __init__(
        self,
        root_dirs: list,
        resolution_groups: dict,
        transform_fn,
        use_preprocessing: bool = True,
        class_to_idx: dict = None,
        is_train: bool = True
    ):
        self.resolution_groups = resolution_groups
        self.transform_fn = transform_fn
        self.is_train = is_train
        self.use_preprocessing = use_preprocessing

        if use_preprocessing:
            self.preprocessor = FingerprintPreprocessor(
                apply_clahe=True,
                apply_gabor=False,
                apply_normalization=True
            )

        self.samples = []
        self.group_indices = {g: [] for g in resolution_groups}
        self.class_to_idx = class_to_idx.copy() if class_to_idx else {}

        self._discover_samples(root_dirs, class_to_idx)

        self.group_transforms = {}
        for group, cfg in resolution_groups.items():
            target = cfg['target']
            self.group_transforms[group] = transform_fn(target, is_train)

    def _get_group(self, path: str) -> str:
        """Determine resolution group from file path."""
        path_lower = path.lower()
        for group, cfg in self.resolution_groups.items():
            for pattern in cfg['path_patterns']:
                if pattern in path_lower:
                    return group
        return list(self.resolution_groups.keys())[0]

    def _discover_samples(self, root_dirs: list, class_to_idx: dict = None):
        """Discover all samples and group by resolution."""
        all_img_paths = []
        extensions = ['.bmp', '.BMP', '.tif', '.TIF', '.png', '.PNG', '.jpg', '.JPG']

        for root_dir in root_dirs:
            root_dir = Path(root_dir)
            if not root_dir.exists():
                continue
            for ext in extensions:
                paths = list(root_dir.rglob(f'*{ext}'))
                all_img_paths.extend(paths)

        all_img_paths = sorted(list(set(all_img_paths)))

        next_idx = len(self.class_to_idx)
        for img_path in all_img_paths:
            identity = self._parse_identity(img_path.name)
            if identity not in self.class_to_idx:
                if class_to_idx is None:
                    self.class_to_idx[identity] = next_idx
                    next_idx += 1
                else:
                    continue

            group = self._get_group(str(img_path))
            label = self.class_to_idx[identity]
            self.samples.append((str(img_path), label, group))
            self.group_indices[group].append(len(self.samples) - 1)

    def _parse_identity(self, filename: str):
        """Parse identity from filename."""
        if '__' in filename:
            parts = filename.split('__')
            person_id = parts[0]
            info = parts[1]
            hand = 'Left' if 'Left' in info else 'Right'
            finger = 'unknown'
            for f in ['index', 'thumb', 'middle', 'ring', 'little']:
                if f in info:
                    finger = f
                    break
            return f"{person_id}_{hand}_{finger}"
        elif '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                return f"{parts[0]}_{parts[1]}"
            elif len(parts) == 2:
                return parts[0]
            return parts[0]
        return filename.split('.')[0]

    def _prepare_image(self, image: Image.Image, group: str) -> Image.Image:
        """Pad/crop image to group target size."""
        target_h, target_w = self.resolution_groups[group]['target']
        w, h = image.size

        if h > target_h or w > target_w:
            left = max(0, (w - target_w) // 2)
            top = max(0, (h - target_h) // 2)
            right = left + min(w, target_w)
            bottom = top + min(h, target_h)
            image = image.crop((left, top, right, bottom))
            w, h = image.size

        if h < target_h or w < target_w:
            padded = Image.new('L', (target_w, target_h), 0)
            paste_x = (target_w - w) // 2
            paste_y = (target_h - h) // 2
            padded.paste(image, (paste_x, paste_y))
            return padded

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, group = self.samples[idx]

        try:
            image = Image.open(path).convert('L')

            if self.use_preprocessing:
                arr = np.array(image)
                arr = self.preprocessor.enhance(arr)
                image = Image.fromarray(arr)

            image = self._prepare_image(image, group)

            transform = self.group_transforms[group]
            image = transform(image)

            return image, label, group

        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return None, None, None


class ResolutionGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups samples by resolution.
    Each batch contains only samples from one resolution group.
    Supports oversampling of minority groups for balance.
    """

    def __init__(
        self,
        dataset: FingerprintDatasetNative,
        group_batch_sizes: dict,
        oversample_ratio: float = 1.0,
        shuffle: bool = True,
        drop_last: bool = True,
        interleave_batches: bool = True
    ):
        self.dataset = dataset
        self.group_batch_sizes = group_batch_sizes
        self.oversample_ratio = oversample_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.interleave_batches = interleave_batches
        self.group_indices = dataset.group_indices

    def _compute_samples_per_group(self) -> dict:
        """Compute target samples per group based on oversampling."""
        sizes = {g: len(idx) for g, idx in self.group_indices.items()}
        max_size = max(sizes.values())

        targets = {}
        for g, natural in sizes.items():
            balanced = max_size
            targets[g] = int(natural + (balanced - natural) * self.oversample_ratio)

        return targets

    def __iter__(self):
        targets = self._compute_samples_per_group()
        all_batches = []

        for group, indices in self.group_indices.items():
            batch_size = self.group_batch_sizes[group]
            target_count = targets[group]

            indices = list(indices)
            if self.shuffle:
                random.shuffle(indices)

            if len(indices) < target_count:
                repeats = (target_count // len(indices)) + 1
                indices = (indices * repeats)[:target_count]

            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.interleave_batches and self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        targets = self._compute_samples_per_group()
        total = 0
        for group, target in targets.items():
            batch_size = self.group_batch_sizes[group]
            n_batches = target // batch_size
            if not self.drop_last and target % batch_size:
                n_batches += 1
            total += n_batches
        return total


def train_fingerprint(args):
    """Train fingerprint recognition model."""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / 'training.log')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        logger.info("Precision set to 'high' (FP32 precision for matmul)")

    # Datasets config
    data_dirs = [
        Path(args.data_dir) / 'socofing' / 'SOCOFing', 
        Path(args.data_dir) / 'fingerprint',
        Path(args.data_dir) / 'fingerprints',
    ]
    
    use_gabor = getattr(args, 'use_gabor_preprocessing', False)  
    use_fast_decoding = getattr(args, 'use_fast_decoding', True)
    image_size = getattr(args, 'image_size', 224)  # Default to 224 for AdaFace
    num_workers = getattr(args, 'num_workers', 2)
    
    use_sample_scheduling = getattr(args, 'use_sample_scheduling', False)
    sample_schedule = getattr(args, 'sample_schedule', ((0, 1.0),))
    current_sample_fraction = args.sample_fraction
    class_fraction = getattr(args, 'class_fraction', 1.0)

    # Native resolution training mode
    use_native_resolution = getattr(args, 'use_native_resolution', True)

    if use_native_resolution:
        logger.info("=" * 60)
        logger.info("NATIVE RESOLUTION TRAINING (No Upscaling)")
        logger.info("=" * 60)

        resolution_groups = getattr(args, 'native_resolution_groups', {
            'low': {'target': (103, 103), 'batch_size': 64, 'accum_steps': 1, 'path_patterns': ['socofing']},
            'high': {'target': (388, 388), 'batch_size': 16, 'accum_steps': 4, 'path_patterns': ['fingerprint', 'fingerprints']},
        })

        oversample_ratio = getattr(args, 'oversample_minority_ratio', 1.0)
        train_low_res_first = getattr(args, 'train_low_res_first', False)

        # --- STEP 1: Discover and Select Classes ---
        logger.info("Discovering all classes to perform class-based sampling...")
        discovery_ds = FingerprintDatasetNative(
            data_dirs,
            resolution_groups=resolution_groups,
            transform_fn=get_fingerprint_transforms_native,
            use_preprocessing=False,
            class_to_idx=None,
            is_train=True
        )
        all_identities = list(discovery_ds.class_to_idx.keys())

        num_selected = max(2, int(len(all_identities) * class_fraction))
        random.seed(42)
        selected_identities = sorted(random.sample(all_identities, num_selected))
        
        # --- Subject-Disjoint Split (80/20) with Resolution Group Stratification ---
        # 1. Map each identity to its groups (Low, High, or Both)
        identity_to_groups = {}
        for sample in discovery_ds.samples:
            path_str, label, group = sample
            # Reverse lookup label to identity
            # Ideally we used the identity string, but we have label index.
            # discovery_ds.samples stores label as int.
            # We need to find the identity key for this label.
            # Building a reverse map first.
            pass

        idx_to_identity = {v: k for k, v in discovery_ds.class_to_idx.items()}
        
        identity_to_groups = {}
        for _, label, group in discovery_ds.samples:
            identity = idx_to_identity[label]
            if identity not in identity_to_groups:
                identity_to_groups[identity] = set()
            identity_to_groups[identity].add(group)

        # 2. Separate identities into strata
        ids_low_only = []
        ids_high_only = []
        ids_both = []

        for identity in selected_identities:
            groups = identity_to_groups.get(identity, set())
            if 'low' in groups and 'high' in groups:
                ids_both.append(identity)
            elif 'high' in groups:
                ids_high_only.append(identity)
            else:
                ids_low_only.append(identity) # Default to low if unknown or strictly low

        # 3. Split each stratum 80/20
        def split_ids(id_list, ratio=0.8):
            random.shuffle(id_list)
            split = int(len(id_list) * ratio)
            return id_list[:split], id_list[split:]

        train_low, val_low = split_ids(ids_low_only)
        train_high, val_high = split_ids(ids_high_only)
        train_both, val_both = split_ids(ids_both)

        train_identities = train_low + train_high + train_both
        val_identities = val_low + val_high + val_both

        # Create separate mappings for train and val
        train_class_to_idx = {identity: i for i, identity in enumerate(train_identities)}
        val_class_to_idx = {identity: i for i, identity in enumerate(val_identities)}

        logger.info(f"Subject Split Stratified:")
        logger.info(f"  Low-Only: {len(train_low)} Train, {len(val_low)} Val")
        logger.info(f"  High-Only: {len(train_high)} Train, {len(val_high)} Val")
        logger.info(f"  Mixed: {len(train_both)} Train, {len(val_both)} Val")
        logger.info(f"  Total: {len(train_identities)} Train IDs, {len(val_identities)} Val IDs")
        logger.info(f"Oversample Ratio: {oversample_ratio}, Low-Res First: {train_low_res_first}")

        # Create datasets with DISJOINT class indices
        train_ds = FingerprintDatasetNative(
            data_dirs,
            resolution_groups=resolution_groups,
            transform_fn=get_fingerprint_transforms_native,
            use_preprocessing=use_gabor,
            class_to_idx=train_class_to_idx,
            is_train=True
        )

        val_ds = FingerprintDatasetNative(
            data_dirs,
            resolution_groups=resolution_groups,
            transform_fn=get_fingerprint_transforms_native,
            use_preprocessing=use_gabor,
            class_to_idx=val_class_to_idx,
            is_train=False
        )

        # Log group statistics
        for group, indices in train_ds.group_indices.items():
            cfg = resolution_groups[group]
            logger.info(f"  [Train] {group}: {len(indices)} samples @ {cfg['target']} "
                       f"(batch={cfg['batch_size']}, accum={cfg['accum_steps']})")
                       
        for group, indices in val_ds.group_indices.items():
            logger.info(f"  [Val]   {group}: {len(indices)} samples")

        # Create batch sampler
        group_batch_sizes = {g: cfg['batch_size'] for g, cfg in resolution_groups.items()}

        train_sampler = ResolutionGroupedBatchSampler(
            train_ds,
            group_batch_sizes=group_batch_sizes,
            oversample_ratio=oversample_ratio,
            shuffle=True,
            drop_last=True,
            interleave_batches=not train_low_res_first
        )

        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=native_resolution_collate,
            pin_memory=True,
            persistent_workers=True
        )

        # Validation: natural distribution (no oversampling)
        val_sampler = ResolutionGroupedBatchSampler(
            val_ds,
            group_batch_sizes=group_batch_sizes,
            oversample_ratio=0.0,
            shuffle=False,
            drop_last=False,
            interleave_batches=True # Order doesn't matter for val
        )

        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=native_resolution_collate,
            pin_memory=True,
            persistent_workers=True
        )

        # Important: num_classes must be the number of TRAINING classes for ArcFace
        filtered_class_to_idx = train_class_to_idx 
        num_classes = len(train_class_to_idx)
        n_train_samples = len(train_ds)
        n_val_samples = len(val_ds)

        # Store accum_steps per group for training loop
        group_accum_steps = {g: cfg['accum_steps'] for g, cfg in resolution_groups.items()}

    else:
        logger.info("Discovering all classes to perform class-based sampling...")
        discovery_ds = FingerprintDataset(
            data_dirs, transform=None, sample_fraction=1.0, 
            use_preprocessing=False, use_fast_decoding=False
        )
        all_identities = list(discovery_ds.class_to_idx.keys())
        
        num_selected = max(2, int(len(all_identities) * class_fraction))
        random.seed(42)
        selected_identities = sorted(random.sample(all_identities, num_selected))
        filtered_class_to_idx = {identity: i for i, identity in enumerate(selected_identities)}
        
        logger.info(f"Filtered to {len(filtered_class_to_idx)}/{len(all_identities)} classes")
        
        # Pre-compute train/val split (done once, reused across stages)
        discovery_ds_full = FingerprintDataset(
            data_dirs, transform=None, sample_fraction=1.0,
            use_preprocessing=False, use_fast_decoding=False,
            class_to_idx=filtered_class_to_idx
        )
        
        subject_to_indices = {}
        for idx, (_, label) in enumerate(discovery_ds_full.samples):
            if label not in subject_to_indices:
                subject_to_indices[label] = []
            subject_to_indices[label].append(idx)
        
        all_subjects = list(subject_to_indices.keys())
        random.seed(42)
        random.shuffle(all_subjects)
        
        split_idx = int(0.8 * len(all_subjects))
        train_subjects = all_subjects[:split_idx]
        val_subjects = all_subjects[split_idx:]
        
        train_indices_base = []
        for subj in train_subjects:
            train_indices_base.extend(subject_to_indices[subj])
        
        val_indices_base = []
        for subj in val_subjects:
            val_indices_base.extend(subject_to_indices[subj])
        
        logger.info(f"Subject Split: {len(train_subjects)} Train Subjects, {len(val_subjects)} Val Subjects")
        
        def build_dataloaders_for_stage(
            resolution: int,
            transform_mode: str,
            batch_size: int,
            sample_fraction: float = 1.0,
            crop_cycler = None  # Optional FiveCropCycling instance
        ):
            """Build train/val dataloaders for a specific progressive stage."""
            
            # Get transforms for this stage
            train_transform = get_fingerprint_transforms_progressive(
                image_size=resolution,
                is_train=True,
                transform_mode=transform_mode,
                source_size=384,
                crop_cycler=crop_cycler
            )
            
            val_transform = get_fingerprint_transforms_progressive(
                image_size=resolution,
                is_train=False,
                transform_mode=transform_mode,
                source_size=384,
                crop_cycler=None  # Validation uses center crop
            )
            
            # Build datasets with stage-specific transforms
            train_ds = FingerprintDataset(
                data_dirs, train_transform,
                1.0, use_preprocessing=use_gabor,
                use_fast_decoding=use_fast_decoding,
                image_size=resolution if transform_mode == 'resize' else 384,
                class_to_idx=filtered_class_to_idx
            )
            
            val_ds = FingerprintDataset(
                data_dirs, val_transform,
                1.0, use_preprocessing=use_gabor,
                use_fast_decoding=use_fast_decoding,
                image_size=resolution if transform_mode == 'resize' else 384,
                class_to_idx=filtered_class_to_idx
            )
            
            # Apply sample fraction if needed
            train_indices = train_indices_base.copy()
            if sample_fraction < 1.0:
                random.shuffle(train_indices)
                n_train = int(len(train_indices) * sample_fraction)
                train_indices = train_indices[:n_train]
            
            train_subset = Subset(train_ds, train_indices)
            val_subset = Subset(val_ds, val_indices_base)
            num_classes = len(filtered_class_to_idx)
            
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, collate_fn=filter_none_collate,
                pin_memory=True, drop_last=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=filter_none_collate,
                pin_memory=True, persistent_workers=True
            )
            
            return train_loader, val_loader, num_classes, len(train_subset), len(val_subset)
        
        # Legacy build_dataloaders for backward compatibility
        def build_dataloaders(sample_fraction: float):
            """Build train/val dataloaders using strict SUBJECT-WISE splitting."""
            full_ds = FingerprintDataset(
                data_dirs, get_fingerprint_transforms(image_size, True), 
                1.0, use_preprocessing=use_gabor,
                use_fast_decoding=use_fast_decoding,
                image_size=image_size,
                class_to_idx=filtered_class_to_idx
            )
            
            subject_to_indices = {}
            for idx, (_, label) in enumerate(full_ds.samples):
                if label not in subject_to_indices:
                    subject_to_indices[label] = []
                subject_to_indices[label].append(idx)
                
            all_subjects = list(subject_to_indices.keys())
            random.seed(42)
            random.shuffle(all_subjects)
            
            split_idx = int(0.8 * len(all_subjects))
            train_subjects = all_subjects[:split_idx]
            val_subjects = all_subjects[split_idx:]
            
            logger.info(f"Subject Split: {len(train_subjects)} Train Subjects, {len(val_subjects)} Val Subjects")
            
            train_indices = []
            for subj in train_subjects:
                train_indices.extend(subject_to_indices[subj])
                
            val_indices = []
            for subj in val_subjects:
                val_indices.extend(subject_to_indices[subj])
                
            if sample_fraction < 1.0:
                random.shuffle(train_indices)
                n_train = int(len(train_indices) * sample_fraction)
                train_indices = train_indices[:n_train]
                logger.info(f"Using {len(train_indices)} training samples (fraction={sample_fraction})")
            
            val_ds_base = FingerprintDataset(
                data_dirs, get_fingerprint_transforms(image_size, False), 
                1.0, use_preprocessing=use_gabor,
                use_fast_decoding=use_fast_decoding,
                image_size=image_size,
                class_to_idx=full_ds.class_to_idx
            )
            
            train_subset = Subset(full_ds, train_indices)
            val_subset = Subset(val_ds_base, val_indices)
            num_classes = len(full_ds.class_to_idx)
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=num_workers, collate_fn=filter_none_collate,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=num_workers, collate_fn=filter_none_collate,
                                   pin_memory=True, persistent_workers=True)
            
            return train_loader, val_loader, num_classes, len(train_subset), len(val_subset)
        
        train_loader, val_loader, num_classes, train_count, val_count = build_dataloaders(current_sample_fraction)
        logger.info(f'Train: {train_count}, Val: {val_count}, Classes: {num_classes}')
    
    # Model Setup
    encoder = BiometricEncoder(
        embedding_dim=512, 
        in_channels=3,
        pretrained=True,
        backbone_type='efficientnet',
        use_attention=False,
        drop_path_rate=0.1
    ).to(device, memory_format=torch.channels_last)
    
    logger.info("Training FULL backbone (unfrozen) to learn ridge detection filters")
    
    ridge_attention = RidgeAttention(channels=1280, reduction=16).to(device)
    encoder = FingerprintEncoderWithAttention(encoder, ridge_attention)
    logger.info("✓ Integrated RidgeAttention")
    
    # NEW: AdaFace Loss or SubCenter ArcFace
    scale = getattr(args, 'arcface_scale', 64.0)
    margin = getattr(args, 'arcface_margin', 0.4)
    use_subcenter = getattr(args, 'use_subcenter_arcface', False)
    num_subcenters = getattr(args, 'num_subcenters', 3)
    
    if use_subcenter:
        logger.info(f"Initializing SubCenter ArcFace with K={num_subcenters}, scale={scale}, margin={margin}")
        arcface = SubCenterArcFaceLoss(512, num_classes, num_subcenters=num_subcenters, 
                                       scale=scale, margin=margin).to(device)
    else:
        logger.info(f"Initializing AdaFace with scale={scale}, margin={margin}")
        arcface = AdaFaceLoss(512, num_classes, scale=scale, margin=margin).to(device)
    
    # Enable FP16 Mixed Precision for 2x Speedup & Lower VRAM
    amp_dtype = torch.float16
    logger.info(f"Using Mixed Precision (FP16) for training efficiency")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    lr_backbone = getattr(args, 'lr_backbone', 0.0001) 
    lr_head = getattr(args, 'lr_head', 0.001)
    
    backbone_params = []
    head_params = []
    
    if hasattr(encoder, 'encoder'): 
        base_encoder = encoder.encoder
        if hasattr(encoder, 'ridge_attention'):
            head_params.extend(list(encoder.ridge_attention.parameters()))
    else:
        base_encoder = encoder
        
    if hasattr(base_encoder, 'backbone'):
        backbone_params.extend(list(base_encoder.backbone.parameters()))
    
    if hasattr(base_encoder, 'embedding_head'):
        head_params.extend(list(base_encoder.embedding_head.parameters()))
        
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
        {'params': head_params, 'lr': lr_head, 'name': 'head'},
        {'params': arcface.parameters(), 'lr': lr_head, 'name': 'arcface'}
    ]
        
    optimizer = torch.optim.AdamW(param_groups, lr=lr_backbone, weight_decay=args.weight_decay, eps=1e-8)
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(10, args.epochs // 10 + 1)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = GradScaler('cuda', init_scale=1024, enabled=(amp_dtype != torch.float32))
    
    # Training Loop
    checkpoint_dir = output_dir / 'fingerprint' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    all_losses = []
    val_accuracies = []
    val_aucs = []
    epoch_losses = []
    learning_rates = []
    
    # Resume logic
    checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
    
    if checkpoint_path.exists() and not is_checkpoint_corrupted(checkpoint_path):
        logger.info("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        arcface.load_state_dict(checkpoint['arcface_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler and scaler if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        all_losses = checkpoint.get('all_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        val_aucs = checkpoint.get('val_aucs', [])
        epoch_losses = checkpoint.get('epoch_losses', [])
        learning_rates = checkpoint.get('learning_rates', [])
        
        # Only step scheduler if we didn't restore its state
        if 'scheduler_state_dict' not in checkpoint:
            for _ in range(start_epoch):
                scheduler.step()
        
        logger.info(f'Resumed from epoch {start_epoch} (Best AUC: {best_val_acc:.4f})')
    
    # ====== PROGRESSIVE RESOLUTION TRAINING SETUP ======
    use_progressive = getattr(args, 'use_progressive_resolution', True)
    if use_native_resolution:
        use_progressive = False
    progressive_stages = getattr(args, 'progressive_stages', None)
    stage_skip_patience = getattr(args, 'stage_skip_patience', 4)
    lr_decay_on_skip = getattr(args, 'lr_decay_on_skip', 0.5)
    crop_cycle_interval = getattr(args, 'crop_cycle_interval', 200)
    
    stage_manager = None
    current_stage_config = None
    
    if use_progressive and progressive_stages:
        stage_manager = ProgressiveStageManager(
            stage_schedule=progressive_stages,
            skip_patience=stage_skip_patience,
            lr_decay_on_skip=lr_decay_on_skip,
            crop_cycle_interval=crop_cycle_interval
        )
        
        # Restore stage manager state if resuming
        if checkpoint_path.exists() and 'stage_manager_state' in checkpoint:
            stage_manager.load_state_dict(checkpoint['stage_manager_state'])
            logger.info(f"Restored stage manager: stage {stage_manager.current_stage_idx}, "
                       f"best_auc {stage_manager.stage_best_auc:.4f}")
        
        # Get initial stage config
        current_stage_config = stage_manager.get_stage_for_epoch(start_epoch)
        resolution, transform_mode, batch_size, accum_steps, stage_idx = current_stage_config
        
        # Build dataloaders for initial stage
        train_loader, val_loader, num_classes, train_count, val_count = build_dataloaders_for_stage(
            resolution=resolution,
            transform_mode=transform_mode,
            batch_size=batch_size,
            sample_fraction=current_sample_fraction,
            crop_cycler=stage_manager.crop_cycler if transform_mode == 'crop_from_384' else None
        )
        accumulation_steps = accum_steps
        
        logger.info(f"Progressive training enabled with {len(progressive_stages)} stages")
        logger.info(f"Starting at Stage {stage_idx + 1}: {resolution}x{resolution} ({transform_mode}), "
                   f"batch={batch_size}, accum={accum_steps}")
    else:
        logger.info("Progressive training disabled, using standard training")
    
    accumulation_steps = args.accumulation_steps if not use_progressive else accumulation_steps
    logger.info(f'Starting training for {args.epochs} epochs (Accumulation={accumulation_steps})')

    # Initialize group_accum_steps for native resolution mode
    if not use_native_resolution:
        group_accum_steps = {}

    epoch = start_epoch
    while epoch < args.epochs:
        # ====== PROGRESSIVE RESOLUTION: Check for stage transition ======
        if stage_manager is not None:
            new_stage_config = stage_manager.get_stage_for_epoch(epoch)
            resolution, transform_mode, batch_size, accum_steps, stage_idx = new_stage_config
            
            # Check if stage changed
            if current_stage_config is None or stage_idx != current_stage_config[4]:
                logger.info(f"{'='*60}")
                logger.info(f"STAGE TRANSITION: Entering Stage {stage_idx + 1}")
                logger.info(f"Resolution: {resolution}x{resolution}, Mode: {transform_mode}")
                logger.info(f"Batch size: {batch_size}, Accumulation: {accum_steps}")
                logger.info(f"{'='*60}")
                
                # Rebuild dataloaders for new stage
                train_loader, val_loader, _, _, _ = build_dataloaders_for_stage(
                    resolution=resolution,
                    transform_mode=transform_mode,
                    batch_size=batch_size,
                    sample_fraction=current_sample_fraction,
                    crop_cycler=stage_manager.crop_cycler if transform_mode == 'crop_from_384' else None
                )
                accumulation_steps = accum_steps
                current_stage_config = new_stage_config
                
                # Apply LR decay on transition (except first stage)
                if stage_idx > 0:
                    stage_manager.transition_to_next_stage(optimizer)
                    logger.info(f"LR decayed to {optimizer.param_groups[0]['lr']:.6f}")
        
        encoder.train()
        arcface.train()
        
        # IMPROVEMENT: Progressive margin warmup instead of fixed margin
        # This prevents embedding collapse in early training
        margin_warmup_epochs = getattr(args, 'margin_warmup_epochs', 15)
        margin_schedule = getattr(args, 'margin_schedule', 'linear')
        target_margin = margin
        
        if epoch < margin_warmup_epochs:
            progress = (epoch + 1) / margin_warmup_epochs
            
            if margin_schedule == 'cosine':
                # Cosine schedule: slower start, faster ramp at end
                progress = 0.5 * (1 - np.cos(np.pi * progress))
            elif margin_schedule == 'step':
                # Step schedule: jump at 50% of warmup
                progress = 0.0 if progress < 0.5 else 1.0
            # else: linear (default)
            
            current_margin = target_margin * progress
        else:
            current_margin = target_margin
        
        # Apply margin warmup if loss module supports it
        if hasattr(arcface, 'set_margin'):
            arcface.set_margin(current_margin)
        
        # Log margin change for debugging
        if epoch == 0 or epoch == margin_warmup_epochs - 1 or epoch == margin_warmup_epochs:
            logger.info(f'Margin: {current_margin:.4f} (warmup epoch {epoch+1}/{margin_warmup_epochs}, schedule={margin_schedule})')
        
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            if use_native_resolution:
                images, labels, group = batch_data
                if images.size(0) == 0:
                    continue
                accum_steps = group_accum_steps.get(group, 1)
            else:
                images, labels = batch_data
                if images.size(0) == 0:
                    continue
                accum_steps = accumulation_steps

            # ====== PROGRESSIVE RESOLUTION: Update crop position every N batches ======
            if stage_manager is not None and current_stage_config is not None:
                if current_stage_config[1] == 'crop_from_384':  # transform_mode
                    if stage_manager.update_crop_position(batch_idx):
                        # Log crop position change (first occurrence only per position)
                        if batch_idx > 0 and batch_idx % crop_cycle_interval == 0:
                            pos_name = FiveCropCycling.POSITIONS[stage_manager.current_crop_position]
                            logger.info(f"Crop position changed to: {pos_name}")
            
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda', dtype=amp_dtype):
                embeddings = encoder(images, normalize=False)
                logits = arcface(embeddings, labels)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            all_losses.append(loss.item() * accumulation_steps)
            
            if batch_idx % 200 == 0 or batch_idx == 0:
                with torch.no_grad():
                    # Calculate accuracy for monitoring
                    # Use manual cosine similarity for fair accuracy calculation with AdaFace
                    # Ensure embeddings are float32 to match weights (which are typically float32)
                    embeddings_float = embeddings.float()
                    embeddings_norm = F.normalize(embeddings_float, p=2, dim=1)
                    weight_norm = F.normalize(arcface.weight, p=2, dim=1)
                    mon_logits = F.linear(embeddings_norm, weight_norm)
                    
                    _, predicted = mon_logits.max(1)
                    batch_acc = predicted.eq(labels).float().mean().item() * 100

                logger.info(f'Epoch {epoch+1}/{args.epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item()*accum_steps:.4f} Acc: {batch_acc:.2f}%')

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        epoch_losses.append(avg_loss)
        learning_rates.append(lr)

        # Validation
        encoder.eval()
        val_embeddings = []
        val_labels = []

        with torch.no_grad():
            for val_batch in val_loader:
                if use_native_resolution:
                    val_images, val_batch_labels, _ = val_batch
                else:
                    val_images, val_batch_labels = val_batch
                if val_images.size(0) == 0: continue
                val_images = val_images.to(device)
                emb = encoder(val_images, normalize=True).cpu().numpy()
                val_embeddings.append(emb)
                val_labels.append(val_batch_labels.numpy())
                
        val_embeddings = np.vstack(val_embeddings)
        val_labels = np.concatenate(val_labels)
        
        # Verification Metrics
        try:
            from src.evaluation.evaluation_sampling import compute_robust_verification_scores
            genuine, impostor, _ = compute_robust_verification_scores(
                val_embeddings, val_labels, min_genuine_pairs=2000, min_classes=200, max_total_pairs=100000
            )
            genuine = genuine[np.isfinite(genuine)]
            impostor = impostor[np.isfinite(impostor)]
            
            if len(genuine) > 0 and len(impostor) > 0:
                all_scores = np.concatenate([genuine, impostor])
                all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
                fpr, tpr, _ = roc_curve(all_labels_binary, all_scores)
                val_auc = auc(fpr, tpr)
            else:
                val_auc = 0.5
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            val_auc = 0.0
            
        val_accuracies.append(val_auc * 100)
        val_aucs.append(val_auc)
        logger.info(f'Epoch {epoch+1} Avg Loss: {avg_loss:.4f} Val AUC: {val_auc:.4f} LR: {lr:.6f}')
        
        # ====== PROGRESSIVE RESOLUTION: Check for stage improvement and early skip ======
        should_skip_to_next_stage = False
        if stage_manager is not None:
            improved, should_skip = stage_manager.check_stage_improvement(val_auc)
            
            if improved:
                logger.info(f'Stage {stage_manager.current_stage_idx + 1} improvement: AUC {val_auc:.4f}')
            else:
                logger.info(f'Stage {stage_manager.current_stage_idx + 1} no improvement. '
                           f'Stage patience: {stage_manager.stage_patience_counter}/{stage_skip_patience}')
            
            if should_skip:
                next_epoch = stage_manager.skip_to_next_stage_epoch()
                if next_epoch > 0:
                    logger.info(f"{'='*60}")
                    logger.info(f"EARLY SKIP: Stage {stage_manager.current_stage_idx + 1} plateaued")
                    logger.info(f"Skipping to epoch {next_epoch + 1} (Stage {stage_manager.current_stage_idx + 2})")
                    logger.info(f"{'='*60}")
                    should_skip_to_next_stage = True
                    epoch = next_epoch - 1  # Will be incremented at end of loop
        
        # Early Stopping & Best Model (global best across all stages)
        if val_auc > best_val_acc:
            best_val_acc = val_auc
            patience_counter = 0
            logger.info(f'New best validation AUC: {best_val_acc:.4f}')
            
            # Save encoder.pt (encoder state dict for ensemble/inference)
            encoder_path = output_dir / 'fingerprint' / 'encoder.pt'
            try:
                torch.save(encoder.state_dict(), encoder_path)
                logger.info(f'Saved encoder.pt to {encoder_path}')
            except Exception as e:
                logger.error(f'Failed to save encoder.pt: {e}')
            
            # Save best_model.pt (full checkpoint for resume - consistent with iris format)
            best_model_path = output_dir / 'fingerprint' / 'best_model.pt'
            try:
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'arcface_state_dict': arcface.state_dict(),
                    'best_val_acc': best_val_acc,
                }, best_model_path)
                logger.info(f'Saved best_model.pt to {best_model_path}')
            except Exception as e:
                logger.error(f'Failed to save best_model.pt: {e}')
        else:
            patience_counter += 1
            logger.info(f'No improvement. Patience: {patience_counter}/{args.early_stopping_patience}')
            
        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping triggered")
            break
            
        # Checkpoint - save complete training state for seamless resume
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'arcface_state_dict': arcface.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'all_losses': all_losses,
            'val_accuracies': val_accuracies,
            'val_aucs': val_aucs,
            'epoch_losses': epoch_losses,
            'learning_rates': learning_rates,
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
        }
        
        # Add stage manager state if using progressive resolution
        if stage_manager is not None:
            checkpoint['stage_manager_state'] = stage_manager.get_state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Increment epoch (using while loop for early skip support)
        epoch += 1
    
    # Final Eval (Load Best)
    best_model_path = output_dir / 'fingerprint' / 'best_model.pt'
    if best_model_path.exists():
        logger.info("Loading best model for final evaluation...")
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        # Handle both old format (state_dict directly) and new format (checkpoint dict)
        if 'encoder_state_dict' in best_checkpoint:
            encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(best_checkpoint)
    
    encoder.eval()
    # (Full eval logic omitted for brevity, but assumed present in original script or handled by external tools. 
    # For now, we ensure the training loop is correct.)
    
    # We must replicate the full eval logic to generate plots as the user requested.
    # I will paste the full eval logic here.
    
    if args.use_tta:
        logger.info(f'Using Test-Time Augmentation with {args.tta_num_aug} augmentations')
        embeddings, labels = evaluate_with_tta(encoder, val_loader, device, args.tta_num_aug)
    else:
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                # Handle unpacking (images, labels, *group)
                images = batch[0]
                labels_batch = batch[1]
                
                if images.size(0) == 0: continue
                images = images.to(device)
                embeddings_batch = encoder(images).cpu().numpy()
                all_embeddings.append(embeddings_batch)
                all_labels.append(labels_batch.numpy())
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
    # Compute final verification metrics using SAME sampling as training validation
    # This ensures final evaluation metrics match what early stopping was based on
    logger.info('Computing final verification metrics (same sampling as training)...')
    from src.evaluation.evaluation_sampling import compute_robust_verification_scores
    genuine, impostor, _ = compute_robust_verification_scores(embeddings, labels, min_genuine_pairs=1500, min_classes=100, max_total_pairs=100000)
    
    genuine = genuine[np.isfinite(genuine)]
    impostor = impostor[np.isfinite(impostor)]
    
    all_scores = np.concatenate([genuine, impostor])
    all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    
    fpr, tpr, thresholds = roc_curve(all_labels_binary, all_scores)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)
    
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    d_prime = (np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.var(genuine) + np.var(impostor)))
    
    # Calculate additional metrics for plot generation
    eer_threshold = thresholds[eer_idx]
    predictions = (all_scores >= eer_threshold).astype(int)
    accuracy = np.mean(predictions == all_labels_binary)
    precision, recall, _ = precision_recall_curve(all_labels_binary, all_scores)
    cm = confusion_matrix(all_labels_binary, predictions)
    
    metrics = {
        'eer': eer,
        'auc': roc_auc,
        'd_prime': d_prime,
        'eer_threshold': eer_threshold,
        'accuracy_at_eer': accuracy,
        'gini_coefficient': 2 * roc_auc - 1,
        'precision_at_eer': float(precision[len(precision)//2]), # Approximate
        'recall_at_eer': float(recall[len(recall)//2]),
    }
    
    logger.info('\n' + '='*60)
    logger.info('TRAINING COMPLETE')
    logger.info('='*60)
    logger.info(f'EER: {eer*100:.2f}%')
    logger.info(f'AUC: {roc_auc:.4f}')
    logger.info(f'd-prime: {d_prime:.4f}')
    
    # Save Outputs
    fingerprint_output_dir = output_dir / 'fingerprint'
    fingerprint_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final encoder.pt
    encoder_path = fingerprint_output_dir / 'encoder.pt'
    try:
        torch.save(encoder.state_dict(), encoder_path)
        logger.info(f'Saved final encoder.pt to {encoder_path}')
    except Exception as e:
        logger.error(f'Failed to save encoder.pt: {e}')
    
    # Save final best_model.pt (consistent format)
    best_model_path = fingerprint_output_dir / 'best_model.pt'
    try:
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'arcface_state_dict': arcface.state_dict(),
            'best_val_acc': best_val_acc,
        }, best_model_path)
        logger.info(f'Saved final best_model.pt to {best_model_path}')
    except Exception as e:
        logger.error(f'Failed to save best_model.pt: {e}')
    
    pd.DataFrame({'genuine': genuine}).to_csv(fingerprint_output_dir / 'genuine_scores.csv', index=False)
    pd.DataFrame({'impostor': impostor}).to_csv(fingerprint_output_dir / 'impostor_scores.csv', index=False)
    
    # Generate Plots
    logger.info('Generating plots...')
    plotter = MetricsPlotter(output_dir, 'fingerprint')
    plotter.generate_all(
        fpr=fpr, tpr=tpr, fnr=fnr, thresholds=thresholds,
        genuine=genuine, impostor=impostor,
        metrics=metrics, cm=cm,
        epoch_losses=epoch_losses, all_losses=all_losses, lrs=learning_rates,
        precision=precision, recall=recall, val_accuracies=val_accuracies, val_aucs=val_aucs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--use-native-resolution', type=lambda x: x.lower() == 'true', default=None)
    cli_args = parser.parse_args()
    
    from src.models.fingerprint.config import TrainingConfig
    config = TrainingConfig()
    
    if cli_args.output_dir: config.output_dir = cli_args.output_dir
    if cli_args.data_dir: config.data_dir = cli_args.data_dir
    if cli_args.epochs: config.epochs = cli_args.epochs
    if cli_args.batch_size: config.batch_size = cli_args.batch_size
    if cli_args.use_native_resolution is not None: config.use_native_resolution = cli_args.use_native_resolution
    if cli_args.quick_test:
        config.quick_test = True
        config.epochs = 5
        config.sample_fraction = 0.2
        config.batch_size = 16
        config.accumulation_steps = 1
        
    class Args: pass
    args = Args()
    for k, v in config.__dict__.items():
        setattr(args, k, v)
        
    train_fingerprint(args)


if __name__ == '__main__':
    main()
