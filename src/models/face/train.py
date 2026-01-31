"""
Face Recognition Training - Refactored Entry Point.

Modular training script optimized for accuracy with:
- MixUp augmentation for better generalization
- ArcFace with scale=64, margin=0.3
- TTA at evaluation
- Support for LFW and CelebA datasets
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

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

# Import modular components
from src.models.face.config import TrainingConfig, get_data_dirs
from src.models.face.augmentations import get_face_transforms
from src.models.face.dataset import FaceDataset, TransformSubset
from src.models.face.trainer import FaceTrainer
from src.models.face.adaface import AdaFaceLoss

# Setup logging
setup_logging()


from src.evaluation.lfw_evaluator import LFWEvaluator
from src.evaluation.atnt_evaluator import ATNTEvaluator


def train_face(config: TrainingConfig):
    """Main face training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info('=' * 60)
    logger.info('FACE TRAINING - REFACTORED VERSION')
    logger.info('=' * 60)
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        torch.backends.cudnn.benchmark = True
    
    # Load dataset with optional attribute support
    data_dir = Path('data/raw')
    full_dataset = FaceDataset(
        data_dir, transform=None, min_samples=config.min_samples_per_class,
        use_attributes=config.use_attributes,
        attribute_names=config.attribute_names if config.use_attributes else None,
        exclude_sources=config.exclude_sources
    )
    
    if len(full_dataset) == 0:
        logger.error('No face data found! Ensure CelebA dataset is in data/raw/ and LFW is configured correctly.')
        return
    
    # Quick test mode: sample from all sources
    if config.quick_test:
        logger.info('=' * 40)
        logger.info('QUICK TEST MODE ENABLED')
        logger.info('=' * 40)
        logger.info(f'Sampling {config.quick_test_samples_per_source} classes per source...')
        full_dataset.sample_from_all_sources(n_per_source=config.quick_test_samples_per_source)
        # Override epochs for quick test
        config.epochs = config.quick_test_epochs
        config.num_workers = 0  # Simpler debugging
        logger.info(f'Quick test epochs: {config.epochs}')
    
    # Filter rare classes
    logger.info(f'Filtering classes with <{config.min_samples_per_class} samples...')
    full_dataset.filter_rare_classes(min_samples=config.min_samples_per_class)
    
    num_classes = len(full_dataset.class_to_idx)
    logger.info(f'After filtering - Samples: {len(full_dataset)}, Classes: {num_classes}')
    
    # Stratified Subject-Disjoint Split
    # We must split by IDENTITY, not by sample, to ensure Open-Set evaluation.
    logger.info('Creating Subject-Disjoint train/val split (80/20)...')
    
    # Get all identity indices
    all_identities = list(full_dataset.class_to_idx.values())
    
    # Stratify by source (LFW vs CelebA) if possible, otherwise random
    # Since dataset class abstracts source, we can just split identities randomly
    # but ensuring we have enough samples for val.
    
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
    # We only need to remap TRAIN labels. Val labels just need to be consistent for metric learning.
    
    # Create mapping: Old_ID -> New_Train_ID
    train_label_map = {old_id: new_id for new_id, old_id in enumerate(sorted(train_identities))}
    
    # Remap training samples
    # Structure: (path, label, attrs) or (path, label)
    new_train_samples = []
    for s in train_samples:
        new_label = train_label_map[s[1]]
        if len(s) == 3:
            new_train_samples.append((s[0], new_label, s[2]))
        else:
            new_train_samples.append((s[0], new_label))
    train_samples = new_train_samples
    
    # Update num_classes for ArcFace to match training set size
    num_classes = len(train_identities)
    logger.info(f'Re-mapped training labels to 0..{num_classes-1}')
    
    logger.info(f'Train Samples: {len(train_samples)}, Val Samples: {len(val_samples)}')
    
    # Create datasets with transforms
    train_transform = get_face_transforms(config.image_size, True, config.use_randaugment)
    val_transform = get_face_transforms(config.image_size, False, False)
    
    # Create datasets with attribute support
    use_attrs = full_dataset.use_attributes
    num_attrs = full_dataset.num_attributes
    train_dataset = TransformSubset(train_samples, train_transform, use_attributes=use_attrs, num_attributes=num_attrs)
    val_dataset = TransformSubset(val_samples, val_transform, use_attributes=use_attrs, num_attributes=num_attrs)
    
    # Compute class weights for balanced loss
    train_labels = [s[1] for s in train_samples]
    class_counts = Counter(train_labels)
    num_classes_actual = len(class_counts)
    total_samples = len(train_labels)
    
    class_weights = torch.zeros(num_classes)
    for cls_idx, count in class_counts.items():
        class_weights[cls_idx] = total_samples / (num_classes_actual * count)
    class_weights = class_weights / class_weights.mean()
    
    logger.info(f'Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}')
    
    # Data loaders
    logger.info(f'Using {config.num_workers} workers for data loading')
    use_persistent = config.num_workers > 0
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True,
        'persistent_workers': use_persistent,
        'prefetch_factor': 2 if config.num_workers > 0 else None,
    }
    
    # Balanced class sampling (inverse frequency weighted)
    if getattr(config, 'use_balanced_sampling', False):
        from torch.utils.data import WeightedRandomSampler
        # Compute sample weights based on class frequency
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_kwargs)
        logger.info('Using balanced class sampling (WeightedRandomSampler)')
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    # Initialize Evaluator (LFW or AT&T)
    evaluator = None
    evaluator_name = "LFW"
    
    # Check for AT&T override
    if getattr(config, 'atnt_path', None):
        atnt_path = Path(config.atnt_path)
        if atnt_path.exists():
            logger.info(f"Initializing AT&T Evaluator with {atnt_path}")
            evaluator = ATNTEvaluator(atnt_path, config.image_size, config.batch_size)
            evaluator_name = "AT&T"
        else:
            logger.warning(f"AT&T path {atnt_path} does not exist.")
            
    # Fallback to LFW if no AT&T or AT&T failed (and LFW exists)
    if evaluator is None:
        lfw_pairs_path = data_dir / 'pairs.csv'
        if lfw_pairs_path.exists():
            logger.info(f"Initializing LFW Evaluator with {lfw_pairs_path}")
            evaluator = LFWEvaluator(data_dir, lfw_pairs_path, config.image_size, config.batch_size)
        else:
            logger.warning(f"LFW pairs.csv not found at {lfw_pairs_path}. Skipping verification benchmark.")
    
    # Create model
    encoder = BiometricEncoder(
        embedding_dim=config.embedding_dim,
        in_channels=3,
        pretrained=config.use_pretrained_face,
        dropout=config.dropout,
        backbone_type=config.backbone_type,
        pretrained_weights_path=getattr(config, 'pretrained_weights_path', '')
    ).to(device, memory_format=torch.channels_last) # OPTIMIZATION: Channels Last
    
    # ====== NOVEL CONTRIBUTION 1: AdaFace vs ArcFace vs SubCenter ======
    use_subcenter = getattr(config, 'use_subcenter_arcface', False)
    num_subcenters = getattr(config, 'num_subcenters', 3)
    
    if config.use_adaface:
        logger.info(f'Using AdaFace Loss (scale={config.adaface_scale}, margin={config.adaface_margin}, h={config.adaface_h})')
        loss_module = AdaFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            scale=config.adaface_scale,
            margin=config.adaface_margin,
            h=config.adaface_h,
            t_alpha=config.adaface_t_alpha
        ).to(device)
    elif use_subcenter:
        logger.info(f'Using SubCenter ArcFace Loss (K={num_subcenters}, scale={config.arcface_scale}, margin={config.arcface_margin})')
        loss_module = SubCenterArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=config.arcface_scale,
            margin=config.arcface_margin
        ).to(device)
    else:
        logger.info(f'Using ArcFace Loss (scale={config.arcface_scale}, margin={config.arcface_margin})')
        loss_module = ArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            scale=config.arcface_scale,
            margin=config.arcface_margin
        ).to(device)
    
    # GPU optimizations
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info('TF32 precision enabled')
    
    # Log configuration
    logger.info('Hyperparameters:')
    logger.info(f'  - Epochs: {config.epochs}')
    logger.info(f'  - Batch size: {config.batch_size}')
    logger.info(f'  - Effective batch size: {config.effective_batch_size}')
    logger.info(f'  - Base LR: {config.lr}')
    logger.info(f'  - MixUp: {config.use_mixup} (alpha={config.mixup_alpha})')
    if config.use_adaface:
        logger.info(f'  - AdaFace: scale={config.adaface_scale}, margin={config.adaface_margin}, h={config.adaface_h}')
    else:
        logger.info(f'  - ArcFace: scale={config.arcface_scale}, margin={config.arcface_margin}')
    
    # ====== NOVEL CONTRIBUTION 2: Multi-Task Attribute Learning ======
    if full_dataset.use_attributes:
        logger.info(f'  - Attributes: {full_dataset.num_attributes} ({config.attribute_loss_weight} weight)')
    
    # Create trainer with novel contributions support
    trainer = FaceTrainer(
        config, encoder, loss_module, device,
        class_weights=class_weights,
        num_attributes=full_dataset.num_attributes if full_dataset.use_attributes else 0
    )
    
    # Setup checkpoint paths
    checkpoint_dir = output_dir / 'face' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if config.evaluate_only:
        logger.info('=' * 40)
        logger.info('EVALUATION ONLY MODE')
        logger.info('=' * 40)
        
        # Identify checkpoint to load
        model_path = None
        if config.checkpoint_path:
            model_path = Path(config.checkpoint_path)
        else:
            # Try standard locations
            candidates = [
                output_dir / 'best_model.pt',
                Path('outputs/face/best_model.pt'),
                checkpoint_dir / 'latest_checkpoint.pt'
            ]
            for p in candidates:
                if p.exists():
                    model_path = p
                    break
        
        if model_path and model_path.exists():
            logger.info(f'Loading model from {model_path}...')
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if 'arcface_state_dict' in checkpoint:
                loss_module.load_state_dict(checkpoint['arcface_state_dict'])
            best_acc = checkpoint.get('best_acc', 0.0)
            logger.info(f'Model loaded. Best recorded accuracy: {best_acc:.2f}%')
        else:
            logger.error(f'No checkpoint found at {model_path} or default locations!')
            return

        # Dummy history for plotting to prevent crashing
        trainer.train_losses = [0.0]
        trainer.val_losses = [0.0]
        trainer.val_accuracies = [0.0]
        trainer.val_aucs = [0.0]
        
    else:
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
                loss_module.load_state_dict(checkpoint['arcface_state_dict'])
            logger.info('Best model loaded successfully.')
    
    # Generate evaluation plots
    logger.info('Generating evaluation plots...')
    generate_evaluation(
        encoder, loss_module, val_loader, device, output_dir,
        trainer.train_losses, trainer.val_losses, trainer.val_accuracies, trainer.val_aucs,
        config, best_acc
    )
    
    # Run Benchmark if available
    if evaluator is not None:
        logger.info('=' * 60)
        logger.info(f'RUNNING {evaluator_name} BENCHMARK')
        logger.info('=' * 60)
        
        # Evaluate returns different things based on evaluator
        if evaluator_name == "LFW":
            acc, std = evaluator.evaluate(encoder, device)
            result_str = f'LFW Accuracy: {acc:.2f}% (+/- {std:.2f}%)'
        else:
            # AT&T returns (acc, auc)
            acc, auc_score = evaluator.evaluate(encoder, device)
            result_str = f'AT&T Accuracy: {acc:.2f}%, AUC: {auc_score:.4f}'
            
        logger.info(result_str)
        
        # Save results
        with open(output_dir / 'face' / 'benchmark_results.txt', 'w') as f:
            f.write(result_str + '\n')
    
    logger.info('\n' + '=' * 60)
    logger.info('TRAINING COMPLETE')
    logger.info('=' * 60)
    logger.info(f'Best Accuracy: {best_acc:.2f}%')


def generate_evaluation(encoder, arcface, val_loader, device, output_dir,
                       train_losses, val_losses, val_accuracies, val_aucs, config, best_acc):
    """Generate comprehensive evaluation plots and metrics."""
    
    # Extract embeddings with TTA
    logger.info('Extracting embeddings for evaluation...')
    encoder.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]  # Ignore attributes if present
            images = images.to(device)
            
            # OPTIMIZATION: Channels Last & BF16
            images = images.to(memory_format=torch.channels_last)
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            with autocast('cuda', dtype=amp_dtype):
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
        all_embeddings, all_labels, min_genuine_pairs=1500, min_classes=100, max_total_pairs=100000
    )
    
    if len(genuine) < 5 or len(impostor) < 5:
        logger.warning(f'Insufficient verification pairs: {len(genuine)} genuine, {len(impostor)} impostor')
        save_minimal_results(output_dir, best_acc, train_losses, config, encoder, arcface)
        return
    
    # Compute metrics
    all_scores = np.concatenate([genuine, impostor])
    all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    
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
        'gini_coefficient': float(2 * roc_auc - 1),
        'precision_at_eer': float(precision[len(precision)//2]),
        'recall_at_eer': float(recall[len(recall)//2]),
    }
    metrics['f1_at_eer'] = float(2 * metrics['precision_at_eer'] * metrics['recall_at_eer'] / 
                                  (metrics['precision_at_eer'] + metrics['recall_at_eer'] + 1e-8))
    
    # Generate all 30 plots
    logger.info('Generating 30 plots...')
    plotter = MetricsPlotter(output_dir, 'face')
    cm = confusion_matrix(all_labels_binary, predictions)
    
    learning_rates = [config.lr * (epoch + 1) / config.warmup_epochs if epoch < config.warmup_epochs else
                      config.lr * 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / 
                      (config.epochs - config.warmup_epochs)))
                      for epoch in range(len(train_losses))]
    
    plotter.generate_all(
        fpr=fpr, tpr=tpr, fnr=fnr, thresholds=thresholds,
        genuine=genuine, impostor=impostor,
        metrics=metrics, cm=cm,
        epoch_losses=train_losses, all_losses=train_losses, lrs=learning_rates,
        precision=precision, recall=recall,
        val_accuracies=val_accuracies,
        val_aucs=val_aucs
    )
    
    # Save outputs
    face_output_dir = output_dir / 'face'
    face_output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame([metrics]).to_csv(face_output_dir / 'metrics.csv', index=False)
    pd.DataFrame({
        'epoch': range(len(train_losses)),
        'loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }).to_csv(face_output_dir / 'training_history.csv', index=False)
    
    # Save summary
    import json
    loss_name = "AdaFace" if config.use_adaface else ("SubCenter ArcFace" if getattr(config, 'use_subcenter_arcface', False) else "ArcFace")
    summary = {
        'model': f'SOTA Face ({config.backbone_type} + {loss_name})',
        'epochs': len(train_losses),
        'batch_size': config.batch_size,
        'lr': config.lr,
        'best_accuracy': best_acc,
        'eer': float(eer * 100),
        'auc': float(roc_auc),
        'd_prime': float(d_prime),
        'mixup_enabled': config.use_mixup,
        'plots_generated': plotter.plot_count,
        'timestamp': datetime.now().isoformat(),
    }
    with open(face_output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save encoder with proper error handling and logging
    encoder_path = face_output_dir / 'encoder.pt'
    best_model_path = output_dir / 'best_model.pt'
    
    try:
        # Save encoder state dict
        torch.save(encoder.state_dict(), encoder_path)
        logger.info(f'Saved encoder to {encoder_path}')
        
        # Also save as best_model.pt for consistency with resume logic
        full_checkpoint = {
            'epoch': len(train_losses),
            'encoder_state_dict': encoder.state_dict(),
            'loss_module_state_dict': arcface.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(full_checkpoint, best_model_path)
        logger.info(f'Saved best_model.pt to {best_model_path}')
        
    except Exception as e:
        logger.error(f'Failed to save model: {e}')
        raise
    
    logger.info(f'EER: {eer*100:.2f}%')
    logger.info(f'AUC: {roc_auc:.4f}')
    logger.info(f'd-prime: {d_prime:.4f}')
    logger.info(f'Plots generated: {plotter.plot_count}')


def save_minimal_results(output_dir, best_acc, train_losses, config=None, encoder=None, arcface=None):
    """Save minimal results when verification data is insufficient."""
    import json
    face_output_dir = output_dir / 'face'
    face_output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = 'SOTA Face (EfficientNetV2-S + ArcFace)'
    if config:
        loss_name = "AdaFace" if config.use_adaface else ("SubCenter ArcFace" if getattr(config, 'use_subcenter_arcface', False) else "ArcFace")
        model_name = f'SOTA Face ({config.backbone_type} + {loss_name})'
    
    summary = {
        'model': model_name,
        'epochs': len(train_losses),
        'best_accuracy': best_acc,
        'warning': 'Insufficient data for verification metrics',
        'timestamp': datetime.now().isoformat(),
    }
    with open(face_output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save encoder even with insufficient verification data
    if encoder is not None:
        encoder_path = face_output_dir / 'encoder.pt'
        try:
            torch.save(encoder.state_dict(), encoder_path)
            logger.info(f'Saved encoder to {encoder_path}')
            
            # Also save as best_model.pt
            if arcface is not None:
                full_checkpoint = {
                    'epoch': len(train_losses),
                    'encoder_state_dict': encoder.state_dict(),
                    'loss_module_state_dict': arcface.state_dict(),
                    'best_acc': best_acc,
                }
                best_model_path = output_dir / 'best_model.pt'
                torch.save(full_checkpoint, best_model_path)
                logger.info(f'Saved best_model.pt to {best_model_path}')
        except Exception as e:
            logger.error(f'Failed to save model: {e}')


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Training')
    parser.add_argument('--output-dir', type=str, default='outputs/face')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--sample-fraction', type=float, default=None)
    parser.add_argument('--no-mixup', action='store_true', help='Disable MixUp augmentation')
    parser.add_argument('--atnt-path', type=str, default=None, help='Path to AT&T dataset for validation')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training and run evaluation only')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint for evaluation')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode: sample from all sources, run few epochs, verify graphs generate')
    args = parser.parse_args()
    
    config_kwargs = {'output_dir': args.output_dir}
    if args.epochs is not None: config_kwargs['epochs'] = args.epochs
    if args.batch_size is not None: config_kwargs['batch_size'] = args.batch_size
    if args.lr is not None: config_kwargs['lr'] = args.lr
    if args.sample_fraction is not None: config_kwargs['sample_fraction'] = args.sample_fraction
    if args.no_mixup: config_kwargs['use_mixup'] = False
    if args.quick_test: config_kwargs['quick_test'] = True
    if args.atnt_path: config_kwargs['atnt_path'] = args.atnt_path
    if args.evaluate_only: config_kwargs['evaluate_only'] = True
    if args.checkpoint_path: config_kwargs['checkpoint_path'] = args.checkpoint_path
    
    config = TrainingConfig(**config_kwargs)
    
    train_face(config)


if __name__ == '__main__':
    main()
