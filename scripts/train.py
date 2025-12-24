#!/usr/bin/env python
"""
Training script for biometric encoders.

Usage:
    python scripts/train.py --encoder face --data-dir data/raw/lfw
    python scripts/train.py --encoder iris --data-dir data/raw/casia_iris_v4
    python scripts/train.py --config configs/custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from loguru import logger

from models.encoders import create_encoder
from models.losses import ArcFaceLoss, create_loss_function
from training import BiometricTrainer, TrainerConfig
from data.datasets.biometric_datasets import (
    LFWDataset,
    IdentityFolderDataset,
    get_train_transforms,
    get_val_transforms,
    create_dataloaders
)
from utils.hardware import detect_hardware_config, clear_gpu_memory
from utils.logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train biometric encoder models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--encoder', '-e',
        type=str,
        choices=['face', 'periocular', 'iris', 'fingerprint'],
        default='face',
        help='Encoder type to train'
    )
    
    # Data
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['lfw', 'folder'],
        default='folder',
        help='Dataset type'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    
    # Loss
    parser.add_argument(
        '--loss',
        type=str,
        choices=['arcface', 'cosface', 'combined'],
        default='arcface',
        help='Loss function'
    )
    parser.add_argument('--margin', type=float, default=0.5, help='Angular margin')
    parser.add_argument('--scale', type=float, default=30.0, help='Feature scale')
    
    # Model
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    
    # Optimization
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--grad-accum', type=int, default=2, help='Gradient accumulation steps')
    
    # Scheduler
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['cosine', 'step', 'plateau'],
        default='cosine',
        help='LR scheduler'
    )
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'), help='Checkpoint directory')
    parser.add_argument('--resume', type=Path, default=None, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log-dir', type=Path, default=Path('logs'), help='Log directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--config', type=Path, default=None, help='Config file (overrides CLI args)')
    
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_input_size(encoder_type: str) -> tuple:
    """Get input size for encoder type."""
    sizes = {
        'face': (224, 224),
        'periocular': (112, 112),
        'iris': (224, 224),
        'fingerprint': (224, 224)
    }
    return sizes[encoder_type]


def main():
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.get('training', {}).items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    # Set experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.encoder}_{args.loss}_e{args.epochs}"
    
    # Setup logging
    setup_logging(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    logger.info("=" * 60)
    logger.info(f"Training {args.encoder.upper()} Encoder")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Detect hardware
    hw_config = detect_hardware_config(target_batch_size=args.batch_size)
    
    # Get input size for modality
    input_size = get_input_size(args.encoder)
    
    # Create transforms
    train_transform = get_train_transforms(input_size, args.encoder)
    val_transform = get_val_transforms(input_size, args.encoder)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    
    if args.dataset_type == 'lfw':
        train_dataset = LFWDataset(
            args.data_dir,
            transform=train_transform,
            split='train'
        )
        val_dataset = LFWDataset(
            args.data_dir,
            transform=val_transform,
            split='val'
        )
    else:
        # Generic folder dataset
        full_dataset = IdentityFolderDataset(
            args.data_dir,
            transform=train_transform,
            min_samples=2
        )
        
        # Split into train/val
        from torch.utils.data import random_split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        # Update val transform (hacky but works)
        val_dataset.dataset = IdentityFolderDataset(
            args.data_dir,
            transform=val_transform,
            min_samples=2
        )
    
    # Get number of classes
    if hasattr(train_dataset, 'num_classes'):
        num_classes = train_dataset.num_classes
    else:
        # For subset, get from parent
        num_classes = train_dataset.dataset.num_classes
    
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val, {num_classes} classes")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating {args.encoder} encoder...")
    encoder = create_encoder(
        args.encoder,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    )
    
    # Create loss
    logger.info(f"Creating {args.loss} loss...")
    loss_fn = create_loss_function(
        args.loss,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        scale=args.scale,
        margin=args.margin
    )
    
    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        scheduler_type=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        use_amp=args.amp,
        checkpoint_dir=str(args.checkpoint_dir / args.experiment_name),
        log_dir=str(args.log_dir / args.experiment_name),
        patience=args.patience,
        seed=args.seed
    )
    
    # Create trainer
    trainer = BiometricTrainer(
        model=encoder,
        criterion=loss_fn,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    results = trainer.train()
    
    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation metric: {results['best_val_metric']:.4f}")
    logger.info(f"Epochs trained: {results['epochs_trained']}")
    logger.info("=" * 60)
    
    # Cleanup
    clear_gpu_memory()
    
    return results


if __name__ == '__main__':
    main()
