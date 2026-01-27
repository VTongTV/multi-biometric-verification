#!/usr/bin/env python
"""
Validation script for AT&T Face Dataset.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoders.biometric_encoder import BiometricEncoder
from src.evaluation.metrics import MetricsExporter

class ATTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # AT&T structure: s1/1.pgm ... s40/10.pgm
        # Find all identity folders
        identity_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('s')])
        
        for idx, identity_dir in enumerate(identity_dirs):
            identity = identity_dir.name # e.g., s1
            self.class_to_idx[identity] = idx
            
            # Find images
            images = sorted(list(identity_dir.glob('*.pgm')))
            for img_path in images:
                self.samples.append((str(img_path), idx))
                
        logger.info(f"Found {len(self.samples)} images from {len(self.class_to_idx)} identities in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # PGM is grayscale, convert to RGB for model
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return None, None

def main():
    # Config
    DATA_DIR = PROJECT_ROOT / "atnt"
    MODEL_PATH = PROJECT_ROOT / "outputs/face/face/encoder.pt"
    OUTPUT_DIR = PROJECT_ROOT / "outputs/validation_atnt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Dataset
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ATTDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 2. Model
    # Try 512 dim first
    embedding_dim = 512
    logger.info(f"Initializing model with embedding_dim={embedding_dim}")
    
    # Check for ONNX weights
    onnx_path = PROJECT_ROOT / "weights/R50_Glint360K.onnx"
    pretrained = onnx_path.exists()
    
    if pretrained:
        logger.info(f"Found ONNX weights at {onnx_path}, using IResNet50 backbone")
        model = BiometricEncoder(
            backbone_type='iresnet50',
            pretrained_weights_path=str(onnx_path),
            embedding_dim=embedding_dim,
            pretrained=True
        ).to(device)
    else:
        logger.warning("ONNX weights not found, defaulting to EfficientNet (might fail load)")
        model = BiometricEncoder(
            backbone_type='efficientnet',
            embedding_dim=embedding_dim,
            pretrained=False
        ).to(device)
    
    # Load weights
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        # Fallback for older checkpoints or different saves
        try:
             # Sometimes saved as 'state_dict' or 'model_state_dict' inside a dict
             wrapper = torch.load(MODEL_PATH, map_location=device)
             if 'model_state_dict' in wrapper:
                 model.load_state_dict(wrapper['model_state_dict'])
                 logger.info("Loaded from wrapper dict")
             else:
                 raise e
        except Exception as e2:
             logger.error(f"Retry failed: {e2}")
             return

    model.eval()
    
    # 3. Extract Embeddings
    all_embeddings = []
    all_labels = []
    
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for images, labels in dataloader:
            if images is None: continue
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Extracted {embeddings.shape[0]} embeddings")
    
    # 4. Metrics
    exporter = MetricsExporter(OUTPUT_DIR, "atnt_face")
    metrics = exporter.compute_all_metrics(embeddings, labels)
    
    logger.info("="*50)
    logger.info("AT&T DATASET RESULTS")
    logger.info("="*50)
    logger.info(f"EER: {metrics.get('eer', 0)*100:.2f}%")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    logger.info(f"Accuracy @ EER: {metrics.get('accuracy_at_eer', 0)*100:.2f}%")
    logger.info("="*50)
    
    # Save outputs
    exporter.generate_all_outputs()
    logger.info(f"Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
