"""
Face Dataset Implementation with Multi-Task Attribute Support.

Handles loading and parsing of LFW and CelebA datasets.
Supports CelebA's 40 binary attributes for multi-task learning.
"""

import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger


class FaceDataset(Dataset):
    """
    Multi-source face dataset that combines LFW and CelebA.
    
    Supports multi-task learning with CelebA attributes.
    
    Handles:
    - LFW: /lfw-deepfunneled/Person_Name/img_xxxx.jpg format
    - CelebA: Uses identity labels and optional attribute labels
    """
    
    def __init__(self, data_dir: Path, transform=None, min_samples: int = 2,
                 use_attributes: bool = False, attribute_names: Optional[List[str]] = None,
                 exclude_sources: Optional[List[str]] = None):
        """
        Args:
            data_dir: Root data directory
            transform: Image transforms
            min_samples: Minimum samples per identity (for metric learning)
            use_attributes: Whether to load CelebA attributes
            attribute_names: List of attribute names to use (subset of 40)
            exclude_sources: List of dataset sources to exclude (e.g., ['lfw', 'celeba'])
        """
        self.transform = transform
        self.use_attributes = use_attributes
        self.attribute_names = attribute_names or []
        self.exclude_sources = [s.lower() for s in (exclude_sources or [])]
        self.samples = []  # List of (img_path, identity_label, attribute_tensor or None)
        self.class_to_idx = {}
        self.celeba_attributes: Dict[str, torch.Tensor] = {}  # img_name -> attr tensor
        
        data_dir = Path(data_dir)
        
        # Load CelebA attributes if needed
        if self.use_attributes and self.attribute_names:
            self._load_celeba_attributes(data_dir)
        
        # Try LFW-deepfunneled first
        if 'lfw' not in self.exclude_sources:
            lfw_dir = data_dir / 'lfw-deepfunneled' / 'lfw-deepfunneled'
            if not lfw_dir.exists():
                lfw_dir = data_dir / 'lfw-deepfunneled'
            if not lfw_dir.exists():
                lfw_dir = data_dir / 'lfw'
            
            if lfw_dir.exists():
                self._load_lfw(lfw_dir, min_samples)
        else:
            logger.info("Excluding LFW from training dataset (Validation Protocol Safe)")
        
        # Always try to load CelebA if available
        if 'celeba' not in self.exclude_sources:
            celeba_dir = data_dir / 'celeba' / 'img_align_celeba'
            if celeba_dir.exists():
                self._load_celeba(data_dir / 'celeba', min_samples)
        else:
            logger.info("Excluding CelebA from training dataset")
                
        # DigiFace-1M disabled for now (too large, slows training)
        # Uncomment to include synthetic faces:
        # digiface_dirs = list(data_dir.glob('DigiFace-1M*'))
        # for d_path in digiface_dirs:
        #     if d_path.is_dir():
        #         self._load_digiface(d_path, min_samples)
        
        logger.info(f'Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities')
        if self.use_attributes:
            logger.info(f'Attribute learning enabled: {len(self.attribute_names)} attributes')
    
    # Source names for quick test sampling
    SOURCE_NAMES = ['lfw', 'celeba', 'digiface']
    
    def sample_from_all_sources(self, n_per_source: int = 10):
        """
        Sample evenly from all dataset sources for quick testing.
        Samples N classes per source (with all their samples).
        
        Args:
            n_per_source: Number of CLASSES (not samples) per source dataset.
        """
        if len(self.samples) == 0:
            return
            
        # Group samples by source (parse from identity key prefix)
        source_classes = {name: {} for name in self.SOURCE_NAMES}
        
        # Map class_idx to source
        idx_to_source = {}
        for identity, idx in self.class_to_idx.items():
            identity_lower = identity.lower()
            if identity_lower.startswith('celeba'):
                idx_to_source[idx] = 'celeba'
            elif identity_lower.startswith('digiface'):
                idx_to_source[idx] = 'digiface'
            else:
                # LFW uses person names directly (no prefix)
                idx_to_source[idx] = 'lfw'
        
        # Group samples by class within each source
        for sample in self.samples:
            img_path, label = sample[0], sample[1]
            attrs = sample[2] if len(sample) > 2 else None
            source = idx_to_source.get(label, 'lfw')
            if source in source_classes:
                if label not in source_classes[source]:
                    source_classes[source][label] = []
                source_classes[source][label].append((img_path, label, attrs))
        
        # Sample N classes from each source (keep all samples per class)
        sampled = []
        for source_name, class_dict in source_classes.items():
            if len(class_dict) > 0:
                # Filter classes with >= 2 samples for stratification
                valid_classes = [c for c, samples in class_dict.items() if len(samples) >= 2]
                if len(valid_classes) == 0:
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
        used_labels = set(s[1] for s in self.samples)
        old_to_new = {old: new for new, old in enumerate(sorted(used_labels))}
        self.samples = [(s[0], old_to_new[s[1]], s[2] if len(s) > 2 else None) for s in self.samples]
        
        new_class_to_idx = {}
        for identity, old_idx in self.class_to_idx.items():
            if old_idx in used_labels:
                new_class_to_idx[identity] = old_to_new[old_idx]
        self.class_to_idx = new_class_to_idx
        
        logger.info(f'Quick test sampling: {len(self.samples)} total samples, {len(self.class_to_idx)} classes')
    
    def _load_celeba_attributes(self, data_dir: Path):
        """Load CelebA attribute labels from CSV."""
        attr_file = data_dir / 'celeba' / 'list_attr_celeba.csv'
        if not attr_file.exists():
            logger.warning(f'CelebA attributes file not found: {attr_file}')
            self.use_attributes = False
            return
        
        # Parse the CSV file
        with open(attr_file, 'r') as f:
            header = f.readline().strip().split(',')
            
            # Find indices of requested attributes
            attr_indices = []
            for attr_name in self.attribute_names:
                if attr_name in header:
                    attr_indices.append(header.index(attr_name))
                else:
                    logger.warning(f'Attribute "{attr_name}" not found in CelebA')
            
            if not attr_indices:
                logger.warning('No valid attributes found, disabling attribute learning')
                self.use_attributes = False
                return
            
            # Parse attribute values (-1, 1) -> (0, 1)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > max(attr_indices):
                    img_name = parts[0]
                    attr_values = [(int(parts[idx]) + 1) // 2 for idx in attr_indices]
                    self.celeba_attributes[img_name] = torch.tensor(attr_values, dtype=torch.float32)
        
        logger.info(f'Loaded attributes for {len(self.celeba_attributes)} CelebA images')
    
    def _load_lfw(self, lfw_dir: Path, min_samples: int):
        """Load LFW dataset - folder per identity format."""
        identity_dirs = [d for d in lfw_dir.iterdir() if d.is_dir()]
        
        for identity_dir in identity_dirs:
            identity_name = identity_dir.name
            images = list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png'))
            
            if len(images) >= min_samples:
                if identity_name not in self.class_to_idx:
                    self.class_to_idx[identity_name] = len(self.class_to_idx)
                
                for img_path in images:
                    # LFW doesn't have attributes, so pass None
                    self.samples.append((str(img_path), self.class_to_idx[identity_name], None))
    
    def _load_celeba(self, celeba_dir: Path, min_samples: int):
        """Load CelebA dataset using identity file if available."""
        # Handle nested directory structure: celeba/img_align_celeba/img_align_celeba/
        img_dir = celeba_dir / 'img_align_celeba'
        if (img_dir / 'img_align_celeba').exists():
            img_dir = img_dir / 'img_align_celeba'
        identity_file = celeba_dir / 'identity_CelebA.txt'
        
        if identity_file.exists():
            # Use identity labels
            identity_map = {}
            with open(identity_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_name, identity = parts
                        if identity not in identity_map:
                            identity_map[identity] = []
                        identity_map[identity].append(img_name)
            
            # Filter and add
            base_idx = len(self.class_to_idx)
            for identity, images in identity_map.items():
                if len(images) >= min_samples:
                    identity_key = f"celeba_{identity}"
                    self.class_to_idx[identity_key] = base_idx + len([k for k in self.class_to_idx if k.startswith('celeba_')])
                    for img_name in images[:20]:  # Limit per identity
                        img_path = img_dir / img_name
                        if img_path.exists():
                            # Get attributes if available
                            attrs = self.celeba_attributes.get(img_name, None)
                            self.samples.append((str(img_path), self.class_to_idx[identity_key], attrs))
        else:
            # Sample random images and treat each as unique (for feature learning)
            images = list(img_dir.glob('*.jpg'))[:5000]
            for i, img_path in enumerate(images):
                identity_key = f"celeba_{i}"
                self.class_to_idx[identity_key] = len(self.class_to_idx)
                attrs = self.celeba_attributes.get(img_path.name, None)
                self.samples.append((str(img_path), self.class_to_idx[identity_key], attrs))
    
    def _load_digiface(self, digiface_dir: Path, min_samples: int):
        """
        Load DigiFace-1M dataset - folder per identity format.
        Structure: DigiFace-1M_1/IdentityID/ImageID.png
        """
        logger.info(f"Loading DigiFace from {digiface_dir.name}...")
        identity_dirs = [d for d in digiface_dir.iterdir() if d.is_dir()]
        
        count = 0
        base_idx = len(self.class_to_idx)
        
        for identity_dir in identity_dirs:
            identity_id = identity_dir.name
            
            # DigiFace uses numeric folder names
            # We prefix to ensure uniqueness across datasets
            identity_key = f"digiface_{identity_id}"
            
            images = list(identity_dir.glob('*.png')) + list(identity_dir.glob('*.jpg'))
            
            if len(images) >= min_samples:
                # Add new identity if not exists
                if identity_key not in self.class_to_idx:
                    self.class_to_idx[identity_key] = base_idx + count
                    count += 1
                
                label_idx = self.class_to_idx[identity_key]
                
                for img_path in images:
                    # DigiFace doesn't have CelebA attributes
                    self.samples.append((str(img_path), label_idx, None))
                    
        logger.info(f"Added {count} identities from {digiface_dir.name}")
                    
    def filter_rare_classes(self, min_samples: int = 2):
        """
        Remove classes with fewer than min_samples.
        This enables stratified splits and improves training stability.
        """
        if len(self.samples) == 0:
            return
        
        # Count samples per class (element 1 is label)
        class_counts = Counter(s[1] for s in self.samples)
        valid_classes = {c for c, count in class_counts.items() if count >= min_samples}
        removed_classes = len(class_counts) - len(valid_classes)
        
        if removed_classes == 0:
            logger.info('No rare classes to filter')
            return
        
        # Filter samples
        original_count = len(self.samples)
        self.samples = [s for s in self.samples if s[1] in valid_classes]
        
        # Remap class indices to be contiguous
        old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
        self.samples = [(s[0], old_to_new[s[1]], s[2]) for s in self.samples]
        
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
        class_counts = Counter(s[1] for s in self.samples)
        total = len(self.samples)
        weights = []
        for s in self.samples:
            weights.append(total / (len(class_counts) * class_counts[s[1]]))
        return weights
    
    @property
    def num_attributes(self) -> int:
        """Return the number of attributes being used."""
        return len(self.attribute_names) if self.use_attributes else 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, attrs = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Return (image, label, attributes) tuple
        if self.use_attributes:
            if attrs is None:
                # No attributes available for this sample, return zeros
                attrs = torch.zeros(len(self.attribute_names), dtype=torch.float32)
            return image, label, attrs
        else:
            return image, label


class TransformSubset(Dataset):
    """Wrapper dataset that applies transforms to a subset of samples."""
    
    def __init__(self, samples_list, transform=None, use_attributes: bool = False,
                 num_attributes: int = 0):
        self.samples = samples_list
        self.transform = transform
        self.use_attributes = use_attributes
        self.num_attributes = num_attributes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if len(sample) == 3:
            img_path, label, attrs = sample
        else:
            img_path, label = sample
            attrs = None
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            img = self.transform(img)
        
        if self.use_attributes:
            if attrs is None:
                attrs = torch.zeros(self.num_attributes, dtype=torch.float32)
            return img, label, attrs
        else:
            return img, label

