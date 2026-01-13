#!/usr/bin/env python3
"""
Single Modality Training with Comprehensive Metrics

Trains one encoder at a time with full 30+ metrics and visualizations.
Run one modality, verify outputs, then proceed to next.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.logging_utils import setup_logging
from src.evaluation.metrics import MetricsExporter
from src.training.losses import AdaFace


# ============================================================================
# DATASET CLASSES
# ============================================================================

class IdentityFolderDataset(Dataset):
    """Dataset for identity-based verification from folder structure."""
    
    def __init__(self, root_dir: Path, transform=None, sample_fraction: float = 1.0):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        identity_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for idx, identity_dir in enumerate(sorted(identity_dirs)):
            self.class_to_idx[identity_dir.name] = idx
            for img_path in identity_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    self.samples.append((str(img_path), idx))
        
        if sample_fraction < 1.0:
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]
        
        logger.info(f"IdentityFolderDataset: {len(self.samples)} samples, {len(self.class_to_idx)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        return image, label


class FilenamePatternDataset(Dataset):
    """Dataset where identity is encoded in filename."""
    
    def __init__(self, root_dir: Path, pattern_type: str = 'socofing', 
                 transform=None, sample_fraction: float = 1.0, grayscale: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []
        self.class_to_idx = {}
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.BMP', '.TIF']
        all_images = []
        for ext in extensions:
            all_images.extend(self.root_dir.rglob(f'*{ext}'))
        
        class_idx = 0
        for img_path in all_images:
            identity = self._parse_identity(img_path.name, pattern_type)
            if identity is None:
                continue
            
            if identity not in self.class_to_idx:
                self.class_to_idx[identity] = class_idx
                class_idx += 1
            
            self.samples.append((str(img_path), self.class_to_idx[identity]))
        
        if sample_fraction < 1.0:
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]
        
        logger.info(f"FilenamePatternDataset ({pattern_type}): {len(self.samples)} samples, {len(self.class_to_idx)} identities")
    
    def _parse_identity(self, filename: str, pattern_type: str):
        try:
            if pattern_type == 'socofing':
                parts = filename.split('__')
                return parts[0] if parts else None
            elif pattern_type == 'fvc':
                name = filename.split('.')[0]
                parts = name.split('_')
                return parts[0] if parts else None
            elif pattern_type == 'sbvpi':
                parts = filename.replace('-', '_').split('_')
                return parts[0] if parts else None
            elif pattern_type == 'ubiris':
                parts = filename.split('_')
                return parts[0] if parts else None
        except:
            pass
        return filename.split('.')[0].split('_')[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            if self.grayscale:
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')
        except Exception:
            if self.grayscale:
                image = Image.new('L', (224, 224), color=128)
            else:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# ENCODER
# ============================================================================

class BiometricEncoder(nn.Module):
    """Encoder for biometric verification."""
    
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.embedding = nn.Linear(512, embedding_dim)
    
    def forward(self, x, return_norm=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        norm = torch.norm(x, 2, 1, True)
        x = F.normalize(x, p=2, dim=1)
        
        if return_norm:
            return x, norm
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms(modality: str, image_size: int = 112):
    if modality == 'face':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif modality == 'iris':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif modality == 'fingerprint':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


def load_modality_datasets(data_dir: Path, modality: str, sample_fraction: float = 0.1):
    """Load all datasets for a specific modality."""
    datasets = []
    transform = get_transforms(modality)
    data_dir = Path(data_dir)
    
    if modality == 'face':
        # LFW
        lfw_path = data_dir / 'lfw-deepfunneled' / 'lfw-deepfunneled'
        if lfw_path.exists():
            ds = IdentityFolderDataset(lfw_path, transform, sample_fraction)
            if len(ds) > 0:
                datasets.append(('LFW', ds))
    
    elif modality == 'iris':
        # SBVPI
        sbvpi_path = data_dir / 'sbvpi' / 'SBVPI_Sclera'
        if sbvpi_path.exists():
            ds = FilenamePatternDataset(sbvpi_path, 'sbvpi', transform, sample_fraction)
            if len(ds) > 0:
                datasets.append(('SBVPI', ds))
        
        # UBIRIS
        ubiris_path = data_dir / 'ubiris' / 'ubiris_seg'
        if ubiris_path.exists():
            ds = FilenamePatternDataset(ubiris_path, 'ubiris', transform, sample_fraction)
            if len(ds) > 0:
                datasets.append(('UBIRIS', ds))
    
    elif modality == 'fingerprint':
        # SOCOFing
        socofing_path = data_dir / 'socofing' / 'SOCOFing'
        if socofing_path.exists():
            fp_transform = get_transforms('fingerprint')
            ds = FilenamePatternDataset(socofing_path, 'socofing', fp_transform, sample_fraction, grayscale=True)
            if len(ds) > 0:
                datasets.append(('SOCOFing', ds))
        
        # FVC
        fvc_path = data_dir / 'fingerprint'
        if fvc_path.exists():
            fp_transform = get_transforms('fingerprint')
            ds = FilenamePatternDataset(fvc_path, 'fvc', fp_transform, sample_fraction, grayscale=True)
            if len(ds) > 0:
                datasets.append(('FVC', ds))
    
    return datasets


# ============================================================================
# TRAINING
# ============================================================================

def train_single_modality(
    modality: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    sample_fraction: float = 0.1,
    device: torch.device = None
):
    """Train a single modality encoder with comprehensive metrics."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comprehensive metrics exporter
    metrics_exporter = MetricsExporter(output_dir, modality)
    
    logger.info("=" * 70)
    logger.info(f"TRAINING {modality.upper()} ENCODER")
    logger.info("=" * 70)
    
    # Load datasets
    datasets = load_modality_datasets(data_dir, modality, sample_fraction)
    
    if not datasets:
        logger.error(f"No datasets found for {modality}")
        return None
    
    logger.info(f"Loaded datasets: {[name for name, _ in datasets]}")
    
    # Combine and split
    combined = ConcatDataset([ds for _, ds in datasets])
    
    n = len(combined)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    
    train_ds = Subset(combined, indices[:split])
    val_ds = Subset(combined, indices[split:])
    
    # Get class_to_idx from all datasets
    class_to_idx = {}
    for _, ds in datasets:
        if hasattr(ds, 'class_to_idx'):
            class_to_idx.update(ds.class_to_idx)
    train_ds.class_to_idx = class_to_idx
    val_ds.class_to_idx = class_to_idx
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    logger.info(f"Identities: {len(class_to_idx)}")
    
    # Create encoder
    in_channels = 1 if modality == 'fingerprint' else 3
    encoder = BiometricEncoder(in_channels=in_channels, embedding_dim=256).to(device)
    
    # Classification head for training (AdaFace or Linear)
    num_classes = max(len(class_to_idx), 100)
    
    # User choice: Use AdaFace for Fingerprint, Linear for others (or mostly Linear if lazy)
    # But for this task we force AdaFace implementation for robustness.
    use_adaface = True #(modality == 'fingerprint')
    
    if use_adaface:
        logger.info("Using AdaFace Loss Head (Quality-Adaptive Margin)")
        classifier = AdaFace(embedding_size=256, class_num=num_classes).to(device)
    else:
        classifier = nn.Linear(256, num_classes).to(device)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    encoder.train()
    classifier.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device) % num_classes
            
            optimizer.zero_grad()
            
            with autocast():
                # We need norms for AdaFace
                if use_adaface:
                    embeddings, norms = encoder(images, return_norm=True)
                    logits = classifier(embeddings, norms, labels)
                else:
                    embeddings = encoder(images)
                    logits = classifier(embeddings)
                
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log to metrics exporter
            metrics_exporter.log_training_batch(
                epoch=epoch,
                batch=batch_idx,
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr']
            )
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
        
        scheduler.step()
    
    # Evaluation
    logger.info("Extracting embeddings for evaluation...")
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # TEST-TIME AUGMENTATION (TTA) - Minimal
            # Original + Horizontal Flip
            emb_original = encoder(images) # returns x only if return_norm=False (default)
            emb_flip = encoder(torch.flip(images, dims=[3]))
            
            # Fusion
            embeddings = emb_original + emb_flip
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Computing comprehensive metrics on {len(embeddings)} samples...")
    
    # Compute all metrics
    metrics = metrics_exporter.compute_all_metrics(embeddings, labels)
    
    if metrics:
        logger.info(f"\n{'='*50}")
        logger.info(f"RESULTS FOR {modality.upper()}")
        logger.info(f"{'='*50}")
        logger.info(f"EER: {metrics.get('eer', 0)*100:.2f}%")
        logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
        logger.info(f"d-prime: {metrics.get('d_prime', 0):.4f}")
        logger.info(f"Accuracy at EER: {metrics.get('accuracy_at_eer', 0)*100:.2f}%")
        logger.info(f"Total metrics computed: {len(metrics)}")
    
    # Generate all outputs (plots + CSVs)
    logger.info("Generating comprehensive outputs (30+ metrics, plots, CSVs)...")
    outputs = metrics_exporter.generate_all_outputs()
    
    # Save model
    model_path = output_dir / modality / f'{modality}_model.pt'
    torch.save(encoder.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"OUTPUT SUMMARY FOR {modality.upper()}")
    logger.info(f"{'='*50}")
    logger.info(f"Metrics: {len(outputs['metrics'])} scalar values")
    logger.info(f"Plots: {len(outputs['plots'])} graphs")
    logger.info(f"CSVs: {len(outputs['csvs'])} data files")
    logger.info(f"Output directory: {output_dir / modality}")
    
    return {
        'modality': modality,
        'datasets': [name for name, _ in datasets],
        'metrics': outputs['metrics'],
        'n_plots': len(outputs['plots']),
        'n_csvs': len(outputs['csvs']),
        'output_dir': str(output_dir / modality)
    }


def main():
    parser = argparse.ArgumentParser(description='Single Modality Training with Comprehensive Metrics')
    parser.add_argument('--modality', type=str, required=True, choices=['face', 'iris', 'fingerprint'],
                       help='Which modality to train')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/comprehensive', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--sample-fraction', type=float, default=0.1, help='Fraction of data to use')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir / 'training.log')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    result = train_single_modality(
        modality=args.modality,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_fraction=args.sample_fraction,
        device=device
    )
    
    if result:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Modality: {result['modality']}")
        logger.info(f"Datasets used: {result['datasets']}")
        logger.info(f"Total outputs: {len(result['metrics'])} metrics + {result['n_plots']} plots + {result['n_csvs']} CSVs")
        logger.info(f"= {len(result['metrics']) + result['n_plots'] + result['n_csvs']} total items")


if __name__ == '__main__':
    main()
