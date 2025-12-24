#!/usr/bin/env python
"""
Comprehensive training script with full metrics and visualization.

Outputs:
- All training metrics to CSV (loss, accuracy, lr per epoch)
- All evaluation metrics (EER, FAR, FRR, AUC, etc.)
- Plots: ROC, DET, FAR/FRR, score distribution, training curves
- Embeddings for analysis

Usage:
    python scripts/train_full.py --data-dir data/raw/lfw
    python scripts/train_full.py --encoder face --epochs 30
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from loguru import logger

from models.encoders import create_encoder
from models.losses import ArcFaceLoss, create_loss_function
from evaluation.metrics import compute_all_metrics, compute_eer
from evaluation.visualization import MetricsVisualizer
from utils.hardware import detect_hardware_config, clear_gpu_memory
from utils.logging_utils import setup_logging


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_synthetic_dataset(
    num_identities: int = 100,
    samples_per_identity: int = 10,
    image_size: tuple = (224, 224),
    channels: int = 3
):
    """
    Create synthetic dataset for testing when real data unavailable.
    
    Each identity gets slightly different noise pattern.
    """
    from torch.utils.data import Dataset
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_identities, samples_per_identity, image_size, channels):
            self.num_identities = num_identities
            self.samples_per_identity = samples_per_identity
            self.image_size = image_size
            self.channels = channels
            self.total_samples = num_identities * samples_per_identity
            
            # Create identity-specific patterns
            self.identity_patterns = {}
            for i in range(num_identities):
                # Each identity has a unique base pattern
                pattern = torch.randn(channels, image_size[0], image_size[1]) * 0.3
                self.identity_patterns[i] = pattern
        
        def __len__(self):
            return self.total_samples
        
        def __getitem__(self, idx):
            label = idx // self.samples_per_identity
            
            # Get identity pattern and add noise
            base = self.identity_patterns[label]
            noise = torch.randn_like(base) * 0.1
            image = base + noise
            
            # Normalize
            image = (image - image.mean()) / (image.std() + 1e-8)
            
            return image, label
        
        @property
        def num_classes(self):
            return self.num_identities
    
    return SyntheticDataset(num_identities, samples_per_identity, image_size, channels)


def try_load_lfw_dataset(data_dir: Path, transform):
    """
    Try to load LFW dataset, fall back to synthetic if not available.
    """
    from data.datasets import LFWDataset, IdentityFolderDataset
    
    # Check for LFW
    lfw_paths = [
        data_dir / 'lfw',
        data_dir / 'lfw_funneled',
        data_dir / 'raw' / 'lfw',
        data_dir
    ]
    
    for path in lfw_paths:
        if path.exists():
            # Check if it has identity folders
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) > 100:
                logger.info(f"Found LFW dataset at {path}")
                try:
                    train_ds = LFWDataset(path, transform=transform, split='train')
                    val_ds = LFWDataset(path, transform=transform, split='val')
                    return train_ds, val_ds, train_ds.num_classes
                except Exception as e:
                    logger.warning(f"Failed to load LFW: {e}")
    
    # Fall back to synthetic
    logger.warning("LFW not found, using synthetic dataset for demonstration")
    synthetic = create_synthetic_dataset(
        num_identities=500,
        samples_per_identity=8,
        image_size=(224, 224),
        channels=3
    )
    
    # Split
    train_size = int(0.8 * len(synthetic))
    val_size = len(synthetic) - train_size
    train_ds, val_ds = random_split(
        synthetic, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_ds, val_ds, synthetic.num_classes


class ComprehensiveTrainer:
    """
    Training with comprehensive metrics logging and visualization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        experiment_name: str,
        config: dict
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Output directories
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Visualizer
        self.visualizer = MetricsVisualizer(self.output_dir, experiment_name)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'time_per_epoch': []
        }
        
        self.best_val_acc = 0.0
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        self.criterion.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        per_batch_data = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                embeddings = self.model(images)
                loss, logits = self.criterion(embeddings, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Per-batch logging
            per_batch_data.append({
                'epoch': epoch,
                'batch': batch_idx,
                'loss': loss.item(),
                'accuracy': predicted.eq(labels).float().mean().item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            if batch_idx % 20 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%"
                )
        
        # Save per-batch data
        batch_df = pd.DataFrame(per_batch_data)
        batch_csv = self.visualizer.csv_dir / f"epoch_{epoch}_batches.csv"
        batch_df.to_csv(batch_csv, index=False)
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate and compute metrics."""
        self.model.eval()
        self.criterion.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_embeddings = []
        all_labels = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.use_amp):
                embeddings = self.model(images)
                loss, logits = self.criterion(embeddings, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total,
            'embeddings': embeddings,
            'labels': labels
        }
    
    def compute_verification_scores(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_pairs: int = 3000
    ) -> tuple:
        """Generate positive and negative pair scores."""
        positive_scores = []
        negative_scores = []
        
        n = len(embeddings)
        unique_labels = np.unique(labels)
        
        # Positive pairs (same identity)
        for _ in range(num_pairs):
            label = np.random.choice(unique_labels)
            indices = np.where(labels == label)[0]
            if len(indices) >= 2:
                i, j = np.random.choice(indices, 2, replace=False)
                sim = np.dot(embeddings[i], embeddings[j])
                positive_scores.append(sim)
        
        # Negative pairs (different identity)
        for _ in range(num_pairs):
            i = np.random.randint(n)
            diff_indices = np.where(labels != labels[i])[0]
            if len(diff_indices) > 0:
                j = np.random.choice(diff_indices)
                sim = np.dot(embeddings[i], embeddings[j])
                negative_scores.append(sim)
        
        return np.array(positive_scores), np.array(negative_scores)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        
        path = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoints_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (acc: {self.best_val_acc:.4f})")
    
    def train(self) -> dict:
        """Full training loop with comprehensive logging."""
        logger.info(f"Starting training for {self.config['epochs']} epochs")
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['time_per_epoch'].append(epoch_time)
            
            logger.info(
                f"Epoch {epoch}/{self.config['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Check best
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            # Save checkpoint
            if epoch % self.config.get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Save history so far
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(self.visualizer.csv_dir / 'training_history.csv', index=False)
        
        # Final evaluation with full metrics
        logger.info("Running final evaluation...")
        final_val = self.validate()
        
        # Generate verification scores
        positive_scores, negative_scores = self.compute_verification_scores(
            final_val['embeddings'],
            final_val['labels'],
            num_pairs=min(5000, len(final_val['labels']) * 2)
        )
        
        # Compute all metrics
        all_metrics = compute_all_metrics(positive_scores, negative_scores)
        all_metrics['final_train_loss'] = self.history['train_loss'][-1]
        all_metrics['final_train_acc'] = self.history['train_acc'][-1]
        all_metrics['final_val_loss'] = self.history['val_loss'][-1]
        all_metrics['final_val_acc'] = self.history['val_acc'][-1]
        all_metrics['best_val_acc'] = self.best_val_acc
        all_metrics['total_epochs'] = self.config['epochs']
        
        # Generate all plots
        logger.info("Generating plots...")
        self.visualizer.generate_all_plots(
            positive_scores,
            negative_scores,
            training_history=self.history,
            prefix=self.config.get('encoder', 'model')
        )
        
        # Export metrics summary
        self.visualizer.export_summary_csv(all_metrics, 'final_metrics')
        
        # Save embeddings
        embeddings_path = self.visualizer.csv_dir / 'final_embeddings.csv'
        emb_df = pd.DataFrame(final_val['embeddings'])
        emb_df['label'] = final_val['labels']
        emb_df.to_csv(embeddings_path, index=False)
        
        # Save scores
        scores_path = self.visualizer.csv_dir / 'verification_scores.csv'
        max_len = max(len(positive_scores), len(negative_scores))
        scores_df = pd.DataFrame({
            'positive_scores': np.pad(positive_scores, (0, max_len - len(positive_scores)), constant_values=np.nan),
            'negative_scores': np.pad(negative_scores, (0, max_len - len(negative_scores)), constant_values=np.nan)
        })
        scores_df.to_csv(scores_path, index=False)
        
        # Save final summary
        summary = {
            'experiment': self.config.get('experiment_name', 'unknown'),
            'encoder': self.config.get('encoder', 'face'),
            'epochs': self.config['epochs'],
            'best_val_acc': self.best_val_acc,
            'eer': all_metrics['eer'],
            'auc': all_metrics['auc'],
            'final_metrics': all_metrics,
            'output_dir': str(self.output_dir)
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training complete! Results saved to {self.output_dir}")
        logger.info(f"Best Val Acc: {self.best_val_acc:.4f}")
        logger.info(f"EER: {all_metrics['eer']:.4f}")
        logger.info(f"AUC: {all_metrics['auc']:.4f}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Comprehensive biometric training')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--encoder', type=str, default='face', 
                       choices=['face', 'periocular', 'iris', 'fingerprint'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--experiment', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment or f"{args.encoder}_{timestamp}"
    
    # Setup logging
    setup_logging(
        log_dir=args.output_dir / 'logs',
        experiment_name=experiment_name
    )
    
    logger.info("=" * 60)
    logger.info(f"COMPREHENSIVE TRAINING: {args.encoder.upper()}")
    logger.info("=" * 60)
    
    # Get transform
    from data.datasets import get_train_transforms, get_val_transforms
    
    input_sizes = {
        'face': (224, 224),
        'periocular': (112, 112),
        'iris': (224, 224),
        'fingerprint': (224, 224)
    }
    input_size = input_sizes[args.encoder]
    
    train_transform = get_train_transforms(input_size, args.encoder)
    val_transform = get_val_transforms(input_size, args.encoder)
    
    # Load dataset
    train_ds, val_ds, num_classes = try_load_lfw_dataset(args.data_dir, train_transform)
    
    logger.info(f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {num_classes} classes")
    
    # Create dataloaders (num_workers=0 for Windows compatibility with synthetic dataset)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = create_encoder(args.encoder, embedding_dim=512, pretrained=True)
    logger.info(f"Created {args.encoder} encoder")
    
    # Create loss
    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=512,
        scale=30.0,
        margin=0.5
    )
    
    # Config
    config = {
        'encoder': args.encoder,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'use_amp': args.use_amp,
        'seed': args.seed,
        'num_classes': num_classes,
        'experiment_name': experiment_name,
        'save_every': 5
    }
    
    # Train
    trainer = ComprehensiveTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        config=config
    )
    
    results = trainer.train()
    
    # Cleanup
    clear_gpu_memory()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Results: {args.output_dir / experiment_name}")
    logger.info("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
