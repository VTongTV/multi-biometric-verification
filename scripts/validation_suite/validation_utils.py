
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, Subset
from PIL import Image
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.fingerprint.ridge_attention import RidgeAttention
from loguru import logger

def create_identity_split(dataset, val_ratio=0.2, seed=42):
    """
    Split a dataset by IDENTITY, not by image.
    Ensures that validation set contains people NEVER seen in training set.
    """
    if not hasattr(dataset, 'samples'):
        raise ValueError("Dataset must have 'samples' attribute (list of (path, label))")
    
    # Group indices by label
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    unique_labels = sorted(list(label_to_indices.keys()))
    
    # Shuffle identities
    rng = random.Random(seed)
    rng.shuffle(unique_labels)
    
    # Split identities
    split_idx = int(len(unique_labels) * (1 - val_ratio))
    train_labels = set(unique_labels[:split_idx])
    val_labels = set(unique_labels[split_idx:])
    
    # Collect indices
    train_indices = []
    val_indices = []
    
    for label in train_labels:
        train_indices.extend(label_to_indices[label])
        
    for label in val_labels:
        val_indices.extend(label_to_indices[label])
        
    logger.info(f"Identity Split: {len(train_labels)} Train IDs, {len(val_labels)} Val IDs")
    logger.info(f"Sample Split: {len(train_indices)} Train images, {len(val_indices)} Val images")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

class FingerprintEncoderWithAttention(torch.nn.Module):
    """Wrapper for loading fingerprint models with attention."""
    def __init__(self, encoder, ridge_attention):
        super().__init__()
        self.encoder = encoder
        self.ridge_attention = ridge_attention
        self.embedding_head = encoder.embedding_head
        self.backbone = encoder.backbone
        if hasattr(encoder, 'use_attention'):
            encoder.use_attention = False
    
    def forward(self, x, normalize=True):
        features = self.encoder.backbone.features(x)
        features = self.ridge_attention(features)
        features = torch.clamp(features, min=-65000, max=65000)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        embeddings = self.encoder.embedding_head(features)
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

def load_model(modality, checkpoint_path, device):
    """Load the appropriate model architecture and weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading {modality} model from {checkpoint_path}")
    
    # Default configs
    embedding_dim = 512
    
    if modality == 'fingerprint':
        # Fingerprint uses the special wrapper
        base_encoder = BiometricEncoder(
            embedding_dim=512,
            in_channels=3, # Grayscale duplicated
            pretrained=False,
            backbone_type='efficientnet',
            use_attention=False
        )
        ridge_attention = RidgeAttention(channels=1280, reduction=16)
        model = FingerprintEncoderWithAttention(base_encoder, ridge_attention)
    
    elif modality == 'face':
        # Check for ONNX weights availability (for correct architecture init)
        onnx_path = PROJECT_ROOT / "weights/R50_Glint360K.onnx"
        if onnx_path.exists():
             model = BiometricEncoder(
                backbone_type='iresnet50',
                pretrained_weights_path=str(onnx_path),
                embedding_dim=512,
                pretrained=True
            )
        else:
            model = BiometricEncoder(
                backbone_type='efficientnet',
                embedding_dim=512,
                pretrained=False
            )
            
    elif modality == 'iris':
        # Iris usually uses EfficientNet
        model = BiometricEncoder(
            backbone_type='efficientnet',
            embedding_dim=512,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    model = model.to(device)
    
    # Load weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Weights loaded successfully")
    except Exception as e:
        logger.warning(f"Standard load failed: {e}. Trying wrapper unwrapping...")
        try:
            # Sometimes 'model_state_dict' key is used
            wrapper = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in wrapper:
                model.load_state_dict(wrapper['model_state_dict'])
                logger.info("Loaded from 'model_state_dict' key")
            else:
                raise e
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            raise e2
            
    return model.eval()

class FilenamePatternDataset(Dataset):
    """Dataset for images where identity is encoded in filename pattern."""
    
    def __init__(self, root_dir: Path, pattern_type: str = 'socofing', 
                 transform=None, sample_fraction: float = 1.0, grayscale: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []
        
        # Collect all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        all_images = []
        for ext in extensions:
            all_images.extend(self.root_dir.rglob(f'*{ext}'))
            all_images.extend(self.root_dir.rglob(f'*{ext.upper()}'))
        
        # Parse identities
        self.class_to_idx = {}
        class_idx = 0
        
        for img_path in all_images:
            identity = self._parse_identity(img_path.name, pattern_type)
            if identity is None:
                continue
            
            if identity not in self.class_to_idx:
                self.class_to_idx[identity] = class_idx
                class_idx += 1
            
            self.samples.append((str(img_path), self.class_to_idx[identity]))
        
        # Sample fraction
        if sample_fraction < 1.0:
            rng = random.Random(42)
            rng.shuffle(self.samples)
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            self.samples = self.samples[:n_samples]
            
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities ({pattern_type})")

    def _parse_identity(self, filename: str, pattern_type: str):
        try:
            if pattern_type == 'socofing':
                # 1__M_Left_index_finger.BMP
                parts = filename.split('__')
                if len(parts) >= 1: return parts[0]
            elif pattern_type == 'fvc':
                # 101_1.tif
                name = filename.split('.')[0]
                parts = name.split('_')
                if len(parts) >= 1: return parts[0]
            elif pattern_type == 'sbvpi':
                # Format: 10L_l_1.png -> ID is '10' (first number part)
                # Split by 'L' or 'R' or '_'
                # Usually: 10L_...
                import re
                match = re.match(r'(\d+)[LR]', filename)
                if match:
                    return match.group(1)
                else:
                    # Fallback simple split
                    return filename.split('_')[0].replace('L','').replace('R','')
            elif pattern_type == 'ubiris':
                # Format: C100_S1_I1.tiff -> ID is '100'
                parts = filename.split('_')
                if len(parts) > 0 and parts[0].startswith('C'):
                    return parts[0][1:] # Remove 'C'
                return parts[0]
            elif pattern_type == 'celeba':
                # 000001.jpg -> No identity in filename unless we have identity file.
                # CelebA is usually used for attributes, not identity, unless mapped.
                # We will skip CelebA for identity verification unless we have the mapping file.
                pass
        except:
            pass
        return None

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            if self.grayscale:
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)
