#!/usr/bin/env python3
"""
Multi-Dataset Training Pipeline Verification Script

Trains all biometric encoders using ALL available datasets with 10% sampling
to verify the complete training pipeline generates all metrics and graphs.

Datasets used:
- Face: CelebA + LFW-deepfunneled
- Iris: SBVPI + UBIRIS.v2
- Fingerprint: SOCOFing + FVC (CrossMatch/UareU)
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
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging

# ============================================================================
# GENERIC IDENTITY DATASET
# ============================================================================

class IdentityFolderDataset(Dataset):
    """Dataset for identity-based verification from folder structure."""
    
    def __init__(self, root_dir: Path, transform=None, sample_fraction: float = 1.0):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Scan for identity folders
        identity_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for idx, identity_dir in enumerate(sorted(identity_dirs)):
            self.class_to_idx[identity_dir.name] = idx
            for img_path in identity_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    self.samples.append((str(img_path), idx))
        
        # Sample fraction
        if sample_fraction < 1.0:
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return a dummy sample
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FilenamePatternDataset(Dataset):
    """Dataset for images where identity is encoded in filename pattern."""
    
    def __init__(self, root_dir: Path, pattern_type: str = 'socofing', 
                 transform=None, sample_fraction: float = 1.0, grayscale: bool = False):
        """
        pattern_type:
            'socofing': filename like '1__M_Left_index_finger.BMP' -> subject ID is first number
            'fvc': filename like '012_1_3.tif' -> subject ID is XXX, finger is Y
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []
        self.class_to_idx = {}
        
        # Collect all image files recursively
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        all_images = []
        for ext in extensions:
            all_images.extend(self.root_dir.rglob(f'*{ext}'))
            all_images.extend(self.root_dir.rglob(f'*{ext.upper()}'))
        
        # Parse identities
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
            n_samples = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities ({pattern_type})")
    
    def _parse_identity(self, filename: str, pattern_type: str):
        """Extract identity from filename based on pattern."""
        try:
            if pattern_type == 'socofing':
                # Format: 1__M_Left_index_finger.BMP
                parts = filename.split('__')
                if len(parts) >= 1:
                    return parts[0]
            elif pattern_type == 'fvc':
                # Format: 012_1_3.tif (subject_finger_sample)
                name = filename.split('.')[0]
                parts = name.split('_')
                if len(parts) >= 1:
                    return parts[0]
            elif pattern_type == 'sbvpi':
                # Format varies - try to extract subject ID
                parts = filename.replace('-', '_').split('_')
                if len(parts) >= 1:
                    return parts[0]
            elif pattern_type == 'ubiris':
                # Format: C1_S1_I1.tiff or similar
                parts = filename.split('_')
                if len(parts) >= 1:
                    return parts[0]
        except:
            pass
        return filename.split('.')[0].split('_')[0]  # Fallback
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            if self.grayscale:
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            if self.grayscale:
                image = Image.new('L', (224, 224), color=128)
            else:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# SIMPLE ENCODER 
# ============================================================================

class SimpleEncoder(nn.Module):
    """Lightweight encoder for pipeline verification."""
    
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
            nn.AdaptiveAvgPool2d(1)
        )
        self.embedding = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ============================================================================
# METRICS & VISUALIZATION
# ============================================================================

class MetricsExporter:
    """Export all metrics to CSV and generate graphs."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = []
        self.eval_results = {}
    
    def log_training_batch(self, epoch: int, batch: int, loss: float, lr: float):
        self.training_history.append({
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        })
    
    def compute_verification_metrics(self, embeddings: np.ndarray, labels: np.ndarray):
        """Compute genuine and impostor scores for verification."""
        n = len(embeddings)
        genuine_scores = []
        impostor_scores = []
        
        # Limit comparisons for speed
        max_comparisons = 5000
        indices = list(range(n))
        random.shuffle(indices)
        indices = indices[:min(n, 200)]
        
        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j])
                
                if labels[i] == labels[j]:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)
                
                if len(genuine_scores) + len(impostor_scores) >= max_comparisons:
                    break
            if len(genuine_scores) + len(impostor_scores) >= max_comparisons:
                break
        
        return np.array(genuine_scores), np.array(impostor_scores)
    
    def compute_eer(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray):
        """Compute Equal Error Rate."""
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        fnr = 1 - tpr
        
        # Find EER point
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer, fpr, tpr, thresholds
    
    def compute_all_metrics(self, embeddings: np.ndarray, labels: np.ndarray):
        """Compute all biometric metrics."""
        genuine, impostor = self.compute_verification_metrics(embeddings, labels)
        
        if len(genuine) == 0 or len(impostor) == 0:
            logger.warning("Insufficient data for metrics computation")
            return {}
        
        eer, fpr, tpr, thresholds = self.compute_eer(genuine, impostor)
        roc_auc = auc(fpr, tpr)
        
        # FAR/FRR at various thresholds
        far_at_frr_01 = self._far_at_frr(fpr, tpr, 0.01)
        far_at_frr_001 = self._far_at_frr(fpr, tpr, 0.001)
        
        metrics = {
            'eer': float(eer),
            'auc': float(roc_auc),
            'far_at_frr_1': float(far_at_frr_01),
            'far_at_frr_01': float(far_at_frr_001),
            'genuine_mean': float(np.mean(genuine)),
            'genuine_std': float(np.std(genuine)),
            'impostor_mean': float(np.mean(impostor)),
            'impostor_std': float(np.std(impostor)),
            'n_genuine': len(genuine),
            'n_impostor': len(impostor)
        }
        
        self.eval_results = {
            'metrics': metrics,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'genuine_scores': genuine.tolist(),
            'impostor_scores': impostor.tolist()
        }
        
        return metrics
    
    def _far_at_frr(self, fpr, tpr, target_frr):
        """Get FAR at specific FRR."""
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - target_frr))
        return fpr[idx]
    
    def save_training_csv(self, encoder_name: str):
        """Save training history to CSV."""
        df = pd.DataFrame(self.training_history)
        csv_path = self.output_dir / f'{encoder_name}_training_history.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved training history to {csv_path}")
        return csv_path
    
    def save_metrics_csv(self, encoder_name: str, metrics: dict):
        """Save metrics to CSV."""
        df = pd.DataFrame([metrics])
        csv_path = self.output_dir / f'{encoder_name}_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
        return csv_path
    
    def save_scores_csv(self, encoder_name: str):
        """Save genuine/impostor scores to CSV."""
        if 'genuine_scores' not in self.eval_results:
            return None
        
        genuine = self.eval_results['genuine_scores']
        impostor = self.eval_results['impostor_scores']
        
        # Create DataFrame
        data = []
        for s in genuine:
            data.append({'score': s, 'type': 'genuine'})
        for s in impostor:
            data.append({'score': s, 'type': 'impostor'})
        
        df = pd.DataFrame(data)
        csv_path = self.output_dir / f'{encoder_name}_verification_scores.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved verification scores to {csv_path}")
        return csv_path
    
    def plot_roc_curve(self, encoder_name: str):
        """Generate ROC curve plot."""
        if 'fpr' not in self.eval_results:
            return None
        
        fpr = self.eval_results['fpr']
        tpr = self.eval_results['tpr']
        roc_auc = self.eval_results['metrics']['auc']
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {encoder_name}', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / f'{encoder_name}_roc_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curve to {plot_path}")
        return plot_path
    
    def plot_det_curve(self, encoder_name: str):
        """Generate DET curve plot."""
        if 'fpr' not in self.eval_results:
            return None
        
        fpr = np.array(self.eval_results['fpr'])
        fnr = 1 - np.array(self.eval_results['tpr'])
        
        # Filter valid points
        valid = (fpr > 0) & (fnr > 0)
        fpr = fpr[valid]
        fnr = fnr[valid]
        
        if len(fpr) < 2:
            return None
        
        plt.figure(figsize=(8, 8))
        plt.loglog(fpr * 100, fnr * 100, 'b-', linewidth=2)
        plt.xlabel('False Accept Rate (%)', fontsize=12)
        plt.ylabel('False Reject Rate (%)', fontsize=12)
        plt.title(f'DET Curve - {encoder_name}', fontsize=14)
        plt.grid(True, alpha=0.3, which='both')
        
        plot_path = self.output_dir / f'{encoder_name}_det_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved DET curve to {plot_path}")
        return plot_path
    
    def plot_score_distribution(self, encoder_name: str):
        """Generate score distribution plot."""
        if 'genuine_scores' not in self.eval_results:
            return None
        
        genuine = self.eval_results['genuine_scores']
        impostor = self.eval_results['impostor_scores']
        
        plt.figure(figsize=(10, 6))
        plt.hist(impostor, bins=50, alpha=0.6, color='red', label='Impostor', density=True)
        plt.hist(genuine, bins=50, alpha=0.6, color='green', label='Genuine', density=True)
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Score Distribution - {encoder_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / f'{encoder_name}_score_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved score distribution to {plot_path}")
        return plot_path
    
    def plot_training_curves(self, encoder_name: str):
        """Generate training curves plot."""
        if not self.training_history:
            return None
        
        df = pd.DataFrame(self.training_history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        epoch_loss = df.groupby('epoch')['loss'].mean()
        ax1.plot(epoch_loss.index, epoch_loss.values, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        epoch_lr = df.groupby('epoch')['learning_rate'].first()
        ax2.plot(epoch_lr.index, epoch_lr.values, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'{encoder_name}_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {plot_path}")
        return plot_path
    
    def generate_all_outputs(self, encoder_name: str, metrics: dict):
        """Generate all CSV files and plots."""
        outputs = {
            'training_csv': str(self.save_training_csv(encoder_name)),
            'metrics_csv': str(self.save_metrics_csv(encoder_name, metrics)),
            'scores_csv': str(self.save_scores_csv(encoder_name)),
            'roc_curve': str(self.plot_roc_curve(encoder_name)),
            'det_curve': str(self.plot_det_curve(encoder_name)),
            'score_distribution': str(self.plot_score_distribution(encoder_name)),
            'training_curves': str(self.plot_training_curves(encoder_name))
        }
        
        # Save summary JSON
        summary = {
            'encoder': encoder_name,
            'metrics': metrics,
            'outputs': outputs,
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / f'{encoder_name}_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return outputs


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder_name: str,
    num_epochs: int,
    output_dir: Path,
    device: torch.device
):
    """Train encoder and generate all metrics."""
    
    exporter = MetricsExporter(output_dir / encoder_name)
    
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Classification head for training
    num_classes = len(train_loader.dataset.class_to_idx) if hasattr(train_loader.dataset, 'class_to_idx') else 1000
    if hasattr(train_loader.dataset, 'datasets'):
        # ConcatDataset
        num_classes = 0
        for ds in train_loader.dataset.datasets:
            if hasattr(ds, 'class_to_idx'):
                num_classes = max(num_classes, len(ds.class_to_idx))
    
    num_classes = max(num_classes, 100)  # Minimum classes
    classifier = nn.Linear(encoder.embedding.out_features, num_classes).to(device)
    
    encoder.train()
    classifier.train()
    
    logger.info(f"Training {encoder_name} for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Clamp labels to valid range
            labels = labels % num_classes
            
            optimizer.zero_grad()
            
            with autocast():
                embeddings = encoder(images)
                logits = classifier(embeddings)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log every batch
            exporter.log_training_batch(
                epoch=epoch,
                batch=batch_idx,
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr']
            )
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {avg_loss:.4f}")
        
        scheduler.step()
    
    # Evaluation
    logger.info("Extracting embeddings for evaluation...")
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            embeddings = encoder(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Computing metrics on {len(embeddings)} samples...")
    metrics = exporter.compute_all_metrics(embeddings, labels)
    
    if metrics:
        logger.info(f"Metrics for {encoder_name}:")
        logger.info(f"  EER: {metrics['eer']*100:.2f}%")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
    
    # Generate all outputs
    outputs = exporter.generate_all_outputs(encoder_name, metrics)
    
    # Save model
    model_path = output_dir / encoder_name / f'{encoder_name}_model.pt'
    torch.save(encoder.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    return metrics, outputs


# ============================================================================
# DATASET LOADING
# ============================================================================

def get_transforms(modality: str, image_size: int = 112):
    """Get transforms for each modality."""
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
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])


def load_all_datasets(data_dir: Path, sample_fraction: float = 0.1):
    """Load all available datasets."""
    datasets = {'face': [], 'iris': [], 'fingerprint': []}
    
    data_dir = Path(data_dir)
    
    # FACE DATASETS
    # LFW
    lfw_path = data_dir / 'lfw-deepfunneled' / 'lfw-deepfunneled'
    if lfw_path.exists():
        logger.info(f"Loading LFW from {lfw_path}")
        ds = IdentityFolderDataset(lfw_path, get_transforms('face'), sample_fraction)
        if len(ds) > 0:
            datasets['face'].append(('LFW', ds))
    
    # CelebA (use partially extracted)
    celeba_path = data_dir / 'celeba' / 'img_align_celeba'
    if celeba_path.exists():
        logger.info(f"Loading CelebA from {celeba_path}")
        # CelebA doesn't have identity folders, use filename pattern
        ds = FilenamePatternDataset(celeba_path, 'fvc', get_transforms('face'), sample_fraction)
        if len(ds) > 0:
            datasets['face'].append(('CelebA', ds))
    
    # IRIS DATASETS  
    # SBVPI
    sbvpi_path = data_dir / 'sbvpi' / 'SBVPI_Sclera'
    if sbvpi_path.exists():
        logger.info(f"Loading SBVPI from {sbvpi_path}")
        ds = FilenamePatternDataset(sbvpi_path, 'sbvpi', get_transforms('iris'), sample_fraction)
        if len(ds) > 0:
            datasets['iris'].append(('SBVPI', ds))
    
    # UBIRIS
    ubiris_path = data_dir / 'ubiris' / 'ubiris_seg'
    if ubiris_path.exists():
        logger.info(f"Loading UBIRIS from {ubiris_path}")
        ds = FilenamePatternDataset(ubiris_path, 'ubiris', get_transforms('iris'), sample_fraction)
        if len(ds) > 0:
            datasets['iris'].append(('UBIRIS', ds))
    
    # FINGERPRINT DATASETS
    # SOCOFing
    socofing_path = data_dir / 'socofing' / 'SOCOFing'
    if socofing_path.exists():
        logger.info(f"Loading SOCOFing from {socofing_path}")
        ds = FilenamePatternDataset(socofing_path, 'socofing', get_transforms('fingerprint'), sample_fraction, grayscale=True)
        if len(ds) > 0:
            datasets['fingerprint'].append(('SOCOFing', ds))
    
    # FVC + CrossMatch + UareU
    fvc_path = data_dir / 'fingerprint'
    if fvc_path.exists():
        logger.info(f"Loading FVC/CrossMatch from {fvc_path}")
        ds = FilenamePatternDataset(fvc_path, 'fvc', get_transforms('fingerprint'), sample_fraction, grayscale=True)
        if len(ds) > 0:
            datasets['fingerprint'].append(('FVC', ds))
    
    return datasets


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset Training Pipeline Verification')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Root data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/pipeline_test', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per encoder')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--sample-fraction', type=float, default=0.1, help='Fraction of data to use')
    parser.add_argument('--modalities', type=str, default='face,iris,fingerprint', help='Which modalities to train')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir / 'training.log')
    
    logger.info("=" * 60)
    logger.info("MULTI-DATASET TRAINING PIPELINE VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sample fraction: {args.sample_fraction * 100}%")
    logger.info(f"Modalities: {args.modalities}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load all datasets
    datasets = load_all_datasets(Path(args.data_dir), args.sample_fraction)
    
    # Print dataset summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    
    all_results = {}
    modalities = args.modalities.split(',')
    
    for modality in modalities:
        modality = modality.strip()
        if modality not in datasets or not datasets[modality]:
            logger.warning(f"No datasets found for {modality}")
            continue
        
        logger.info(f"\n{modality.upper()} datasets:")
        total_samples = 0
        for name, ds in datasets[modality]:
            logger.info(f"  {name}: {len(ds)} samples")
            total_samples += len(ds)
        logger.info(f"  TOTAL: {total_samples} samples")
        
        # Combine datasets
        combined = ConcatDataset([ds for _, ds in datasets[modality]])
        
        # Split into train/val
        n = len(combined)
        indices = list(range(n))
        random.shuffle(indices)
        split = int(0.8 * n)
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        train_ds = Subset(combined, train_indices)
        val_ds = Subset(combined, val_indices)
        
        # Attach class_to_idx for compatibility
        class_to_idx = {}
        for _, ds in datasets[modality]:
            if hasattr(ds, 'class_to_idx'):
                class_to_idx.update(ds.class_to_idx)
        train_ds.class_to_idx = class_to_idx
        val_ds.class_to_idx = class_to_idx
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # Create encoder
        in_channels = 1 if modality == 'fingerprint' else 3
        encoder = SimpleEncoder(in_channels=in_channels, embedding_dim=256).to(device)
        
        # Train
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING {modality.upper()} ENCODER")
        logger.info(f"{'='*60}")
        
        metrics, outputs = train_encoder(
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            encoder_name=modality,
            num_epochs=args.epochs,
            output_dir=output_dir,
            device=device
        )
        
        all_results[modality] = {
            'metrics': metrics,
            'outputs': outputs,
            'datasets_used': [name for name, _ in datasets[modality]]
        }
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    
    summary_data = []
    for modality, result in all_results.items():
        logger.info(f"\n{modality.upper()}:")
        logger.info(f"  Datasets: {', '.join(result['datasets_used'])}")
        if result['metrics']:
            logger.info(f"  EER: {result['metrics']['eer']*100:.2f}%")
            logger.info(f"  AUC: {result['metrics']['auc']:.4f}")
        
        summary_data.append({
            'modality': modality,
            'datasets': ', '.join(result['datasets_used']),
            **result['metrics']
        })
    
    # Save master summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'all_modalities_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Save full results JSON
    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
