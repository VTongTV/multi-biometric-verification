"""
Iris Dataset Implementation.
Handles loading and parsing of multiple iris dataset formats.
"""

import re
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger

from .normalization import robust_normalize


class IrisDataset(Dataset):
    """
    Dataset for iris images supporting multiple formats.
    
    Supported formats:
    - UBIRIS: OperatorA_C{id}_S{session}_I{img}.tiff
    - SBVPI: {id}{L/R}_{direction}_{number}.png
    - CASIA-Thousand: S5{subj}{L/R}{num}.jpg
    - CASIA-Interval: S1{subj}{L/R}{num}.jpg
    - CASIA-Syn: S6{subj}S{num}.jpg
    - CASIA-Lamp: S2{subj}{L/R}{num}.jpg
    - MMU: files in {subject}/{left|right}/ folders
    """
    
    EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    SOURCE_NAMES = ['UBIRIS', 'SBVPI', 'CASIA_T', 'CASIA_I', 'CASIA_S', 'CASIA_L', 'MMU']
    
    # Class-level counters for preprocessing stats
    _cache_hits = 0
    _cache_misses = 0
    _log_interval = 100  # Log every N samples
    
    def __init__(self, data_dirs: List, transform=None, sample_fraction: float = 1.0, is_preprocessed: bool = False, allowed_datasets: set = None):
        self.transform = transform
        self.is_preprocessed = is_preprocessed
        self.allowed_datasets = allowed_datasets
        
        # Reset counters on new dataset
        IrisDataset._cache_hits = 0
        IrisDataset._cache_misses = 0
        self.samples = []
        self.class_to_idx = {}
        
        skipped = 0
        source_counts = {name: 0 for name in self.SOURCE_NAMES}
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.warning(f'Directory not found: {data_path}')
                continue
            
            for ext in self.EXTENSIONS:
                for img_path in data_path.rglob(f'*{ext}'):
                    identity, source = self._parse_identity(img_path)
                    
                    # Domain Filtering
                    if self.allowed_datasets is not None:
                        if source not in self.allowed_datasets:
                            continue
                            
                    if identity:
                        if identity not in self.class_to_idx:
                            self.class_to_idx[identity] = len(self.class_to_idx)
                        self.samples.append((str(img_path), self.class_to_idx[identity]))
                        if source in source_counts:
                            source_counts[source] += 1
                    else:
                        skipped += 1

        
        # Sample fraction
        if sample_fraction < 1.0 and len(self.samples) > 0:
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            self.samples = random.sample(self.samples, n_samples)
        
        logger.info(f'Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities')
        for source, count in source_counts.items():
            if count > 0:
                logger.info(f'  - {source}: {count} samples')
        if skipped > 0:
            logger.warning(f'  - Skipped {skipped} files with unrecognized format')
            
        # Explicit Summary for User Confidence
        logger.info("=" * 40)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 40)
        total_loaded = 0
        for source, count in source_counts.items():
            if count > 0:
                logger.info(f"  > {source}: {count} samples")
                total_loaded += count
        logger.info("-" * 40)
        logger.info(f"  TOTAL: {total_loaded} samples")
        logger.info("=" * 40)
    
    def sample_from_all_sources(self, n_per_source: int = 10):
        """
        Sample evenly from all dataset sources for quick testing.
        Samples N classes per source (with all their samples), ensuring enough data
        for stratified splits.
        
        Args:
            n_per_source: Number of CLASSES (not samples) per source dataset.
        """
        if len(self.samples) == 0:
            return
            
        # Group samples by source (parse from identity key)
        source_classes = {name: {} for name in self.SOURCE_NAMES}  # source -> {class_idx: [samples]}
        
        # Reverse lookup: class_idx -> source
        identity_to_source = {}
        for identity in self.class_to_idx.keys():
            for source_name in self.SOURCE_NAMES:
                if identity.startswith(source_name):
                    identity_to_source[self.class_to_idx[identity]] = source_name
                    break
        
        # Group samples by class within each source
        for path, label in self.samples:
            source = identity_to_source.get(label, 'OTHER')
            if source in source_classes:
                if label not in source_classes[source]:
                    source_classes[source][label] = []
                source_classes[source][label].append((path, label))
        
        # Sample N classes from each source (keep all samples per class)
        import random
        sampled = []
        for source_name, class_dict in source_classes.items():
            if len(class_dict) > 0:
                # Filter classes with >= 3 samples for stratification
                valid_classes = [c for c, samples in class_dict.items() if len(samples) >= 3]
                if len(valid_classes) == 0:
                    # Fallback: take any classes
                    valid_classes = list(class_dict.keys())
                
                n_take = min(n_per_source, len(valid_classes))
                selected_classes = random.sample(valid_classes, n_take)
                
                class_samples = 0
                for cls in selected_classes:
                    sampled.extend(class_dict[cls])
                    class_samples += len(class_dict[cls])
                    
                logger.info(f'  - {source_name}: {n_take} classes, {class_samples} samples from {len(class_dict)} classes')
        
        if len(sampled) == 0:
            logger.warning('No samples after source sampling, keeping original')
            return
            
        self.samples = sampled
        
        # Rebuild class_to_idx with only sampled classes
        used_labels = set(label for _, label in self.samples)
        old_to_new = {old: new for new, old in enumerate(sorted(used_labels))}
        self.samples = [(p, old_to_new[l]) for p, l in self.samples]
        
        new_class_to_idx = {}
        for identity, old_idx in self.class_to_idx.items():
            if old_idx in used_labels:
                new_class_to_idx[identity] = old_to_new[old_idx]
        self.class_to_idx = new_class_to_idx
        
        logger.info(f'Quick test sampling: {len(self.samples)} total samples, {len(self.class_to_idx)} classes')
    
    def _parse_identity(self, img_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Extract identity from path or filename."""
        filename = img_path.stem
        parent_folder = img_path.parent.name.upper()
        grandparent = img_path.parent.parent.name
        
        # UBIRIS: OperatorA_C100_S1_I1
        # UBIRIS: OperatorA_C100_S1_I1 (original) or C100_S1_I1 (processed)
        # Matches _C100_ or starts with C100_
        ubiris_match = re.search(r'(?:^|_)C(\d+)_', filename)
        if ubiris_match:
            return (f'UBIRIS_C{ubiris_match.group(1)}', 'UBIRIS')
        
        # SBVPI: 10L_l_1, 25R_s_3
        sbvpi_match = re.match(r'^(\d+)([LR])_', filename)
        if sbvpi_match:
            return (f'SBVPI_{sbvpi_match.group(1)}_{sbvpi_match.group(2)}', 'SBVPI')
        
        # CASIA-Thousand: S5000L00.jpg
        casia_t_match = re.match(r'^S5(\d{3})([LR])\d+$', filename)
        if casia_t_match:
            return (f'CASIA_T_{casia_t_match.group(1)}_{casia_t_match.group(2)}', 'CASIA_T')
        
        # CASIA-Interval: S1003R01.jpg
        casia_i_match = re.match(r'^S1(\d{3})([LR])\d+$', filename)
        if casia_i_match:
            return (f'CASIA_I_{casia_i_match.group(1)}_{casia_i_match.group(2)}', 'CASIA_I')
        
        # CASIA-Syn: S6000S01.jpg
        casia_s_match = re.match(r'^S6(\d{3})S\d+$', filename)
        if casia_s_match:
            return (f'CASIA_S_{casia_s_match.group(1)}', 'CASIA_S')
        
        # CASIA-Lamp: S2001L01.jpg
        casia_l_match = re.match(r'^S2(\d{3})([LR])\d+$', filename)
        if casia_l_match:
            return (f'CASIA_L_{casia_l_match.group(1)}_{casia_l_match.group(2)}', 'CASIA_L')
        
        # MMU: files in {subject}/{left|right}/ folders
        # Note: Original files are .bmp but preprocessed files are .png
        if parent_folder in ('LEFT', 'RIGHT', 'L', 'R'):
            eye_side = 'L' if parent_folder in ('LEFT', 'L') else 'R'
            return (f'MMU_{grandparent}_{eye_side}', 'MMU')
        
        return (None, None)
    
    def _validate_dataset(self):
        """Validate class distribution and log potential issues."""
        if len(self.samples) == 0:
            logger.error('No samples loaded! Check data paths.')
            return
        
        class_counts = Counter(label for _, label in self.samples)
        min_samples = min(class_counts.values())
        max_samples = max(class_counts.values())
        avg_samples = sum(class_counts.values()) / len(class_counts)
        
        logger.info(f'Class distribution: {len(class_counts)} classes, '
                   f'min={min_samples}, max={max_samples}, avg={avg_samples:.1f}')
        
        if len(class_counts) < 50:
            logger.warning(f'Low class count ({len(class_counts)}). Verify data parsing.')
        if min_samples < 3:
            few_sample_classes = sum(1 for c in class_counts.values() if c < 3)
            logger.warning(f'{few_sample_classes} classes have <3 samples')
    
    def filter_rare_classes(self, min_samples: int = 3):
        """
        Remove classes with fewer than min_samples.
        This enables stratified splits and improves training stability.
        """
        if len(self.samples) == 0:
            return
        
        # Count samples per class
        class_counts = Counter(label for _, label in self.samples)
        valid_classes = {c for c, count in class_counts.items() if count >= min_samples}
        removed_classes = len(class_counts) - len(valid_classes)
        
        if removed_classes == 0:
            logger.info('No rare classes to filter')
            return
        
        # Filter samples
        original_count = len(self.samples)
        self.samples = [(p, l) for p, l in self.samples if l in valid_classes]
        
        # Remap class indices to be contiguous
        old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
        self.samples = [(p, old_to_new[l]) for p, l in self.samples]
        
        # Update class_to_idx
        new_class_to_idx = {}
        for identity, old_idx in self.class_to_idx.items():
            if old_idx in valid_classes:
                new_class_to_idx[identity] = old_to_new[old_idx]
        self.class_to_idx = new_class_to_idx
        
        removed_samples = original_count - len(self.samples)
        logger.info(f'Filtered {removed_classes} rare classes ({removed_samples} samples removed)')
        logger.info(f'Remaining: {len(self.samples)} samples, {len(self.class_to_idx)} classes')
    
    def get_class_weights(self):
        """Get inverse frequency weights for balanced sampling."""
        class_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = []
        for _, label in self.samples:
            weights.append(total / (len(class_counts) * class_counts[label]))
        return weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load preprocessed image (RGB)
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (224, 224), 0)
        
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    """Wrapper dataset that applies transforms to a subset of samples."""
    
    def __init__(self, samples_list, transform=None, is_preprocessed=False):
        self.samples = samples_list
        self.transform = transform
        self.is_preprocessed = is_preprocessed
        
        # Determine if we're dealing with (image_tensor, label) or (path, label)
        # Fingerprint Native dataset might pass tensors directly or paths
        # But for Iris, we refactored to pass (path, label) tuples.
        if len(samples_list) > 0 and isinstance(samples_list[0][0], torch.Tensor):
            self.data_mode = 'tensors'
        else:
            self.data_mode = 'paths'
    
    def __getitem__(self, idx):
        if self.data_mode == 'tensors':
            img, label = self.samples[idx]
            # Transform already applied or not needed? 
            # Usually subset wraps raw data. If tensors, they might be in memory.
            # Assuming if they are tensors, we might still want to apply transforms 
            # (e.g. normalization) if not done.
            # But standard flow is paths.
            return img, label
            
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (224, 224), 0)

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)
