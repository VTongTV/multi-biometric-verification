import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend")) # Add backend source to path

from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.fingerprint.ridge_attention import RidgeAttention
from src.models.fingerprint.preprocessing import FingerprintPreprocessor
from src.evaluation.metrics import MetricsExporter
from src.data.augmentation.augmentations import get_fingerprint_transforms
from src.utils.logging_utils import setup_logging

# ============================================================================
# MODEL DEFINITION (Must match training exactly)
# ============================================================================

class FingerprintEncoderWithAttention(nn.Module):
    """
    Wrapper that properly integrates Ridge Attention into the encoder forward pass.
    Must match the class definition in train.py exactly for state_dict loading.
    """
    
    def __init__(self, encoder, ridge_attention):
        super().__init__()
        self.encoder = encoder
        self.ridge_attention = ridge_attention
        # Expose embedding_head 
        self.embedding_head = encoder.embedding_head
        self.backbone = encoder.backbone
        
        if hasattr(encoder, 'use_attention'):
            encoder.use_attention = False
    
    def forward(self, x, normalize=True):
        # Extract backbone features (without encoder's attention)
        features = self.encoder.backbone.features(x)
        
        # Apply ridge-specific attention (replaces encoder's attention)
        features = self.ridge_attention(features)
        
        # Clamp for FP16 stability (good practice, though we likely use FP32 inference)
        features = torch.clamp(features, min=-65000, max=65000)
        
        # Pool and embed
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        embeddings = self.encoder.embedding_head(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# ============================================================================
# DATASET
# ============================================================================

class ExternalFingerprintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Initialize preprocessor (same as training)
        self.preprocessor = FingerprintPreprocessor(
            apply_clahe=True, 
            apply_gabor=False, 
            apply_normalization=True
        )
        
        # Find all images
        extensions = ['.tif', '.TIF', '.bmp', '.BMP', '.jpg', '.JPG', '.png', '.PNG']
        all_files = []
        for ext in extensions:
            all_files.extend(list(self.root_dir.rglob(f'*{ext}')))
            
        all_files = sorted(list(set(all_files)))
        
        next_idx = 0
        for img_path in all_files:
            # Parse Identity: 101_1.tif -> 101
            # FVC convention: ID_Sample.tif
            filename = img_path.name
            
            # Robust parsing
            if '_' in filename:
                parts = filename.split('_')
                # 101_1.tif -> "101"
                identity = parts[0]
            else:
                # Fallback
                identity = filename.split('.')[0]
            
            if identity not in self.class_to_idx:
                self.class_to_idx[identity] = next_idx
                next_idx += 1
                
            self.samples.append((str(img_path), self.class_to_idx[identity]))
            
        logger.info(f"Found {len(self.samples)} images from {len(self.class_to_idx)} identities in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            
            # Apply Preprocessing
            img_array = np.array(image)
            img_array = self.preprocessor.enhance(img_array)
            image = Image.fromarray(img_array)
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    # Configuration
    DATA_DIR = PROJECT_ROOT / "archive" / "fingerprints"
    if not DATA_DIR.exists():
        DATA_DIR = PROJECT_ROOT / "data" / "raw" / "socofing" / "SOCOFing" / "Real"

    MODEL_PATH = PROJECT_ROOT / "outputs" / "sota" / "fingerprint" / "encoder.pt"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation_external"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    transform = get_fingerprint_transforms(image_size=160, is_training=False)
    dataset = ExternalFingerprintDataset(DATA_DIR, transform=transform)
    
    if len(dataset) == 0:
        logger.error("No images found in dataset directory!")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 2. Initialize Model Architecture
    logger.info("Initializing model architecture...")
    
    # Base Encoder
    base_encoder = BiometricEncoder(
        embedding_dim=512, 
        in_channels=3,  # Grayscale duplicated to 3 channels
        pretrained=False, # No need to download weights, we load state_dict
        backbone_type='efficientnet',
        use_attention=False,
        drop_path_rate=0.0
    )
    
    # Attention Module
    # EfficientNetV2-S has 1280 channels at the final layer
    ridge_attention = RidgeAttention(channels=1280, reduction=16)
    
    # Wrap
    model = FingerprintEncoderWithAttention(base_encoder, ridge_attention).to(device)
    
    # 3. Load Weights
    logger.info(f"Loading weights from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return
        
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return

    model.eval()

    # 4. Extract Embeddings
    logger.info("Extracting embeddings...")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if images.numel() == 0: continue
            
            images = images.to(device)
            # Forward pass
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Extracted {len(embeddings)} embeddings.")

    # 5. Generate Metrics & Plots
    logger.info("Computing metrics and generating plots...")
    exporter = MetricsExporter(OUTPUT_DIR, "fingerprint_external_val")
    
    # Compute everything
    metrics = exporter.compute_all_metrics(embeddings, labels)
    
    # Log results
    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS ON EXTERNAL DATASET")
    logger.info("=" * 50)
    logger.info(f"EER: {metrics.get('eer', 0)*100:.2f}%")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    logger.info(f"Accuracy @ EER: {metrics.get('accuracy_at_eer', 0)*100:.2f}%")
    logger.info(f"d-prime: {metrics.get('d_prime', 0):.4f}")
    
    # Generate Plots
    outputs = exporter.generate_all_outputs()
    logger.info(f"Generated {len(outputs['plots'])} plots in {OUTPUT_DIR}")

if __name__ == "__main__":
    setup_logging(PROJECT_ROOT / "outputs" / "validation_external" / "validation.log")
    main()
