"""
Iris Recognition Training - Refactored Entry Point.

Modular training script optimized for 95%+ accuracy with:
- MixUp augmentation for better generalization
- ArcFace with fixed margin 0.3
- RadialAttention + ECA attention modules
- TTA at evaluation
- Support for UBIRIS, SBVPI, CASIA (Thousand/Interval/Syn/Lamp), MMU datasets
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
# Robust autocast import
if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast_mode'):
    from torch.amp.autocast_mode import autocast
else:
    from torch.cuda.amp import autocast
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.losses.arcface import ArcFaceLoss, SubCenterArcFaceLoss
from src.evaluation.plotter import MetricsPlotter
from src.utils.logging_utils import setup_logging

# Use absolute imports for standalone execution
from src.models.iris.config import TrainingConfig, get_data_dirs
from src.models.iris.augmentations import get_iris_transforms, get_iris_transforms_by_strength
from src.models.iris.dataset import IrisDataset, TransformSubset
from src.models.iris.trainer import IrisTrainer
from src.models.iris.domain_config import DOMAINS, get_domain_config_overrides, get_augmentation_strength

# Setup logging
setup_logging()


def train_iris(config: TrainingConfig):
    """Main iris training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info('=' * 60)
    logger.info('IRIS TRAINING - REFACTORED VERSION')
    logger.info('=' * 60)
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    
    if config.allowed_datasets:
        logger.info("=" * 40)
        logger.info(f"ACTIVE DOMAIN RESTRICTION: {config.allowed_datasets}")
        logger.info("Only datasets matching this list will be loaded.")
        logger.info("=" * 40)
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        torch.backends.cudnn.benchmark = True
    
    # Get dataset paths
    data_dirs = get_data_dirs(project_root)
    logger.info('Data directories:')
    for d in data_dirs:
        logger.info(f'  - {d} (exists: {d.exists()})')
    
    # Validation: Check if preprocessed data exists
    processed_root = project_root / 'data' / 'processed_onnx'
    use_preprocessed = processed_root.exists() and len(list(processed_root.glob('**/*.png'))) > 1000
    
    if use_preprocessed:
        logger.info(f"Found preprocessed dataset at {processed_root}!")
        logger.info("Using OFFLINE NORMALIZED data (Fast CPU/GPU training)")
        # For preprocessed, we just point to the root. The structure mirrors raw.
        # But dataset.py expects list of source dirs. 
        # We can just pass the root, and since it contains mirrored 'UBIRIS', 'SBVPI' etc folders, 
        # the dataset regex *should* still find them if we pass the subfolders.
        # Actually, let's just pass the processed_root if the dataset class handles recursion (rglob).
        data_dirs = [processed_root]
    else:
        logger.warning(f"Preprocessed dataset NOT found. Using RAW data + ON-THE-FLY Normalization (Slow!)")
        logger.info("Run `python -m src.models.iris.preprocess` to fix speed.")
        # data_dirs already set via get_data_dirs
    
    # Create full dataset
    full_dataset = IrisDataset(
        data_dirs, 
        transform=None, 
        sample_fraction=config.sample_fraction, 
        is_preprocessed=use_preprocessed,
        allowed_datasets=config.allowed_datasets  # NEW: Domain filtering
    )
    
    if len(full_dataset) == 0:
        logger.error('No data loaded! Exiting.')
        return
    
    # Quick test mode: sample from all sources
    if config.quick_test:
        logger.info('=' * 40)
        logger.info('QUICK TEST MODE ENABLED')
        logger.info('=' * 40)
        logger.info(f'Sampling {config.quick_test_samples_per_source} samples per source...')
        full_dataset.sample_from_all_sources(n_per_source=config.quick_test_samples_per_source)
        # Override epochs for quick test
        config.epochs = config.quick_test_epochs
        config.num_workers = 0  # Simpler debugging
        logger.info(f'Quick test epochs: {config.epochs}')
    
    # Filter rare classes to enable stratified splits
    logger.info(f'Filtering classes with <{config.min_samples_per_class} samples...')
    full_dataset.filter_rare_classes(min_samples=config.min_samples_per_class)
    
    num_classes = len(full_dataset.class_to_idx)
    logger.info(f'After filtering - Samples: {len(full_dataset)}, Classes: {num_classes}')
    
    # Stratified Subject-Disjoint Split (80/20)
    # We must split by IDENTITY, not by sample, to ensure Open-Set evaluation.
    logger.info('Creating Subject-Disjoint train/val split (80/20)...')
    
    # Get all identity indices
    all_identities = list(full_dataset.class_to_idx.values())
    
    # Random split of identities
    import random
    random.seed(42)
    random.shuffle(all_identities)
    
    split_idx = int(0.8 * len(all_identities))
    train_identities = set(all_identities[:split_idx])
    val_identities = set(all_identities[split_idx:])
    
    logger.info(f'Split: {len(train_identities)} Train Identities, {len(val_identities)} Val Identities')
    
    # Filter samples based on identity split
    train_samples = [s for s in full_dataset.samples if s[1] in train_identities]
    val_samples = [s for s in full_dataset.samples if s[1] in val_identities]
    
    # Remap labels for ArcFace (must be 0..N-1)
    # Create mapping: Old_ID -> New_Train_ID
    train_label_map = {old_id: new_id for new_id, old_id in enumerate(sorted(train_identities))}
    
    # Remap training samples
    new_train_samples = []
    for s in train_samples:
        new_label = train_label_map[s[1]]
        # IrisDataset samples are (path, label)
        new_train_samples.append((s[0], new_label))
    train_samples = new_train_samples
    
    # Update num_classes
    num_classes = len(train_identities)
    logger.info(f'Re-mapped training labels to 0..{num_classes-1}')
    
    logger.info(f'Train Samples: {len(train_samples)}, Val Samples: {len(val_samples)}')
    
    # Create datasets with transforms
    train_transform = get_iris_transforms(config.image_size, True)
    val_transform = get_iris_transforms(config.image_size, False)
    
    train_dataset = TransformSubset(train_samples, train_transform, is_preprocessed=use_preprocessed)
    val_dataset = TransformSubset(val_samples, val_transform, is_preprocessed=use_preprocessed)
    
    # Compute class weights for balanced loss
    from collections import Counter
    train_labels = [s[1] for s in train_samples]
    class_counts = Counter(train_labels)
    num_classes_actual = len(class_counts)
    total_samples = len(train_labels)
    
    # Inverse frequency weighting
    class_weights = torch.zeros(num_classes)
    for cls_idx, count in class_counts.items():
        class_weights[cls_idx] = total_samples / (num_classes_actual * count)
    class_weights = class_weights / class_weights.mean()
    
    logger.info(f'Class weights computed: min={class_weights.min():.2f}, max={class_weights.max():.2f}')
    
    # Data loaders
    logger.info(f'Using {config.num_workers} workers for data loading')
    logger.info(f'Using class-weighted loss (no sampler overhead)')
    
    # Windows optimization: use persistent workers to avoid spawn overhead
    use_persistent = config.num_workers > 0
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True,
        'persistent_workers': use_persistent,
        'prefetch_factor': 2 if config.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )
    
    
    # Create model
    encoder = BiometricEncoder(
        embedding_dim=config.embedding_dim,
        in_channels=3,
        pretrained=True,
        dropout=config.dropout
    ).to(device, memory_format=torch.channels_last) # OPTIMIZATION: Channels Last
    
    # IMPROVEMENT: Support SubCenter ArcFace for L/R eye variation
    use_subcenter = getattr(config, 'use_subcenter_arcface', False)
    num_subcenters = getattr(config, 'num_subcenters', 3)
    
    if use_subcenter:
        logger.info(f'Using SubCenter ArcFace with K={num_subcenters} subcenters per class')
        arcface = SubCenterArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=config.arcface_scale,
            margin=config.arcface_margin
        ).to(device)
    else:
        arcface = ArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            scale=config.arcface_scale,
            margin=config.arcface_margin
        ).to(device)
    
    # RTX 3050 optimizations (Windows-compatible)
    if device.type == 'cuda':
        # TF32 for faster matrix ops on Ampere GPUs (RTX 30 series)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info('TF32 precision enabled for faster computation')
        # Note: channels_last and torch.compile disabled - not faster on RTX 3050/Windows
    
    # Log configuration
    logger.info('Hyperparameters:')
    logger.info(f'  - Epochs: {config.epochs}')
    logger.info(f'  - Batch size: {config.batch_size}')
    logger.info(f'  - Effective batch size: {config.effective_batch_size}')
    logger.info(f'  - Base LR: {config.lr}')
    logger.info(f'  - MixUp: {config.use_mixup} (alpha={config.mixup_alpha})')
    logger.info(f'  - ArcFace: scale={config.arcface_scale}, margin={config.arcface_margin}')
    logger.info(f'  - Dropout: {config.dropout}')
    
    # Create trainer with class weights for balanced loss
    trainer = IrisTrainer(config, encoder, arcface, device, class_weights=class_weights)
    
    # Setup checkpoint paths
    checkpoint_dir = output_dir / 'iris' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Smart resume: automatically checks for corruption and falls back to best_model if needed
    resume_result = trainer.smart_resume(checkpoint_dir, output_dir)
    logger.info(f'Resume strategy used: {resume_result}')
    
    # Train
    best_acc = trainer.fit(train_loader, val_loader, output_dir, checkpoint_dir)
    
    # Load best model before evaluation
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        logger.info(f'Loading best model from {best_model_path} for final evaluation...')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if 'arcface_state_dict' in checkpoint:
            arcface.load_state_dict(checkpoint['arcface_state_dict'])
        logger.info('Best model loaded successfully.')
    
    # Generate evaluation plots
    logger.info('Generating evaluation plots...')
    generate_evaluation(
        encoder, arcface, val_loader, device, output_dir,
        trainer.train_losses, trainer.val_losses, trainer.val_accuracies, trainer.val_aucs,
        config, best_acc
    )
    
    logger.info('\n' + '=' * 60)
    logger.info('TRAINING COMPLETE')
    logger.info('=' * 60)
    logger.info(f'Best Accuracy: {best_acc:.2f}%')


def generate_evaluation(encoder, arcface, val_loader, device, output_dir,
                       train_losses, val_losses, val_accuracies, val_aucs, config, best_acc):
    """Generate comprehensive evaluation plots and metrics."""
    
     # Extract embeddings
    logger.info('Extracting embeddings for evaluation...')
    encoder.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # OPTIMIZATION: Channels Last
            images = images.to(memory_format=torch.channels_last)
            
            # OPTIMIZATION: BF16
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            with autocast('cuda', dtype=amp_dtype):
                # TTA: Average original + horizontal flip
                emb_orig = encoder(images)
                emb_flip = encoder(torch.flip(images, [3]))
                embeddings = F.normalize(emb_orig + emb_flip, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    
    # Compute similarity scores using SAME sampling as training validation
    # This ensures final evaluation metrics match what early stopping was based on
    logger.info('Computing verification metrics with class-aware sampling (same as training)...')
    from src.evaluation.evaluation_sampling import compute_robust_verification_scores
    
    genuine, impostor, sample_stats = compute_robust_verification_scores(
        all_embeddings, all_labels, min_genuine_pairs=500, min_classes=100, max_total_pairs=250000
    )
    
    # Use same data for plotting
    impostor_for_plotting = impostor
    
    if len(genuine) < 5 or len(impostor) < 5:
        logger.warning(f'Insufficient verification pairs: {len(genuine)} genuine, {len(impostor)} impostor')
        save_minimal_results(output_dir, best_acc, train_losses)
        return
    
    # Compute metrics
    all_scores = np.concatenate([genuine, impostor])
    all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    
    # Filter out NaN values (can occur if model weights are corrupted)
    valid_mask = np.isfinite(all_scores)
    if not valid_mask.all():
        nan_count = (~valid_mask).sum()
        logger.warning(f"Filtering {nan_count} NaN similarity scores from evaluation")
        all_scores = all_scores[valid_mask]
        all_labels_binary = all_labels_binary[valid_mask]
        
        if len(all_scores) < 10:
            logger.error("Too few valid scores remaining after NaN filtering. Cannot compute metrics.")
            save_minimal_results(output_dir, best_acc, train_losses)
            return
    
    fpr, tpr, thresholds = roc_curve(all_labels_binary, all_scores)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)
    
    eer_diff = np.abs(fpr - fnr)
    eer_idx = np.nanargmin(eer_diff)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
    
    d_prime = (np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.var(genuine) + np.var(impostor) + 1e-8))
    
    predictions = (all_scores >= eer_threshold).astype(int)
    accuracy = np.mean(predictions == all_labels_binary)
    precision, recall, _ = precision_recall_curve(all_labels_binary, all_scores)
    
    metrics = {
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'auc': float(roc_auc),
        'd_prime': float(d_prime),
        'accuracy_at_eer': float(accuracy),
        'best_classification_accuracy': float(best_acc),
        'genuine_mean': float(np.mean(genuine)),
        'impostor_mean': float(np.mean(impostor)),
        'n_genuine': len(genuine),
        'n_impostor': len(impostor),
    }
    
    # Generate plots
    logger.info('Generating plots...')
    plotter = MetricsPlotter(output_dir, 'iris')
    cm = confusion_matrix(all_labels_binary, predictions)
    
    # Handle case where training was skipped (empty losses)
    if not train_losses:
        logger.warning("No training losses to plot (training skipped). Using dummy data for plots.")
        train_losses = [0.0]
        learning_rates = [config.lr]
    else:
        learning_rates = [config.lr * (epoch + 1) / config.warmup_epochs if epoch < config.warmup_epochs else
                          config.lr * 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / 
                          (config.epochs - config.warmup_epochs)))
                          for epoch in range(len(train_losses))]
    
    plotter.generate_all(
        fpr=fpr, tpr=tpr, fnr=fnr, thresholds=thresholds,
        genuine=genuine, impostor=impostor_for_plotting,
        metrics=metrics, cm=cm,
        epoch_losses=train_losses, all_losses=train_losses, lrs=learning_rates,
        precision=precision, recall=recall,
        val_accuracies=val_accuracies,
        val_aucs=val_aucs
    )
    
    # Save outputs
    iris_output_dir = output_dir / 'iris'
    iris_output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame([metrics]).to_csv(iris_output_dir / 'metrics.csv', index=False)
    
    # Ensure all history arrays have the same length
    min_len = min(len(train_losses), len(val_losses), len(val_accuracies)) if train_losses and val_losses and val_accuracies else 0
    if min_len > 0:
        pd.DataFrame({
            'epoch': range(min_len),
            'loss': train_losses[:min_len],
            'val_loss': val_losses[:min_len],
            'val_acc': val_accuracies[:min_len]
        }).to_csv(iris_output_dir / 'training_history.csv', index=False)
    else:
        # No training history to save (evaluation-only mode)
        logger.warning("No training history to save (training was skipped).")
    
    # Save summary
    import json
    summary = {
        'model': 'SOTA Iris (EfficientNetV2-S + ArcFace)',
        'epochs': len(train_losses),
        'batch_size': config.batch_size,
        'lr': config.lr,
        'best_accuracy': best_acc,
        'eer': float(eer * 100),
        'auc': float(roc_auc),
        'd_prime': float(d_prime),
        'mixup_enabled': config.use_mixup,
        'timestamp': datetime.now().isoformat(),
    }
    with open(iris_output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f'EER: {eer*100:.2f}%')
    logger.info(f'AUC: {roc_auc:.4f}')
    logger.info(f'd-prime: {d_prime:.4f}')


def save_minimal_results(output_dir, best_acc, train_losses):
    """Save minimal results when verification data is insufficient."""
    import json
    iris_output_dir = output_dir / 'iris'
    iris_output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model': 'SOTA Iris (EfficientNetV2-S + ArcFace)',
        'epochs': len(train_losses),
        'best_accuracy': best_acc,
        'warning': 'Insufficient data for verification metrics',
        'timestamp': datetime.now().isoformat(),
    }
    with open(iris_output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Iris Recognition Training')
    parser.add_argument('--output-dir', type=str, default='outputs/iris')
    # Default to None to allow config.py defaults to take precedence
    parser.add_argument('--epochs', type=int, default=None)  
    parser.add_argument('--batch-size', type=int, default=None)  
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--sample-fraction', type=float, default=None)
    parser.add_argument('--no-mixup', action='store_true', help='Disable MixUp augmentation')
    parser.add_argument('--subcenter', action='store_true',
                        help='Use SubCenter ArcFace for L/R eye variation handling')
    parser.add_argument('--quick-test', action='store_true', 
                        help='Quick test mode: sample from all sources, run few epochs, verify graphs generate')
    parser.add_argument('--domain', type=str, default=None, choices=list(DOMAINS.keys()),
                        help='Train on specific domain (nir_casia or visible)')
    args = parser.parse_args()
    
    # Only pass arguments that are provided (not None)
    config_kwargs = {'output_dir': args.output_dir}
    if args.epochs is not None: config_kwargs['epochs'] = args.epochs
    if args.batch_size is not None: config_kwargs['batch_size'] = args.batch_size
    if args.lr is not None: config_kwargs['lr'] = args.lr
    if args.sample_fraction is not None: config_kwargs['sample_fraction'] = args.sample_fraction
    if args.no_mixup: config_kwargs['use_mixup'] = False
    if args.quick_test: config_kwargs['quick_test'] = True
    if args.subcenter: config_kwargs['use_subcenter_arcface'] = True
    
    # Handle domain-specific training
    if args.domain:
        domain_info = DOMAINS[args.domain]
        config_kwargs['output_dir'] = f'outputs/iris_domain_{args.domain}'
        config_kwargs['allowed_datasets'] = set(domain_info['datasets'])
        
        # IMPROVEMENT: Apply domain-specific config overrides
        domain_overrides = get_domain_config_overrides(args.domain)
        for key, value in domain_overrides.items():
            # Only apply if not explicitly set via CLI
            if key not in config_kwargs:
                config_kwargs[key] = value
                logger.info(f"  Domain override: {key} = {value}")
        
        logger.info(f"Domain training: {domain_info['name']}")
        logger.info(f"Datasets: {domain_info['datasets']}")
    
    config = TrainingConfig(**config_kwargs)
    
    # Log final configuration
    logger.info("=" * 50)
    logger.info("FINAL TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"  epochs: {config.epochs}")
    logger.info(f"  lr: {config.lr}")
    logger.info(f"  backbone_lr_mult: {config.backbone_lr_mult}")
    logger.info(f"  arcface_margin: {config.arcface_margin}")
    logger.info(f"  margin_warmup_epochs: {config.margin_warmup_epochs}")
    logger.info(f"  early_stop_patience: {config.early_stop_patience}")
    logger.info(f"  use_subcenter_arcface: {config.use_subcenter_arcface}")
    logger.info("=" * 50)
    
    train_iris(config)


if __name__ == '__main__':
    main()
