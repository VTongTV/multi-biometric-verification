"""
Standalone script to evaluate the best saved model and generate all graphs.
This addresses the bug where final evaluation was run on the latest model instead of best.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.amp import autocast
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.losses.arcface import ArcFaceLoss, SubCenterArcFaceLoss
from src.evaluation.plotter import MetricsPlotter
from src.utils.logging_utils import setup_logging

setup_logging()


def evaluate_iris_visible():
    """Evaluate the best iris visible model and generate all graphs."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    output_dir = Path("outputs/iris_domain_visible")
    best_model_path = output_dir / "best_model.pt"
    
    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}")
        return
    
    logger.info(f"Loading best model from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Best model from epoch: {checkpoint['epoch']}")
    if 'best_val_auc' in checkpoint:
        logger.info(f"Best validation AUC: {checkpoint['best_val_auc']:.4f}")
    
    # Get model config from checkpoint or use defaults
    num_classes = checkpoint.get('num_classes', 370)
    embedding_dim = checkpoint.get('embedding_dim', 512)
    
    logger.info(f"Model config: {num_classes} classes, {embedding_dim}D embeddings")
    
    # Create encoder
    encoder = BiometricEncoder(
        embedding_dim=embedding_dim,
        dropout=0.6,
        use_attention=True
    ).to(device)
    encoder.to(memory_format=torch.channels_last)
    
    # Load encoder weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    logger.info("Encoder loaded successfully.")
    
    # Load dataset
    from src.models.iris.config import TrainingConfig, get_data_dirs
    from src.models.iris.augmentations import get_iris_transforms_by_strength
    from src.models.iris.dataset import IrisDataset, TransformSubset
    from src.models.iris.domain_config import get_domain_config_overrides
    
    config = TrainingConfig()
    domain_overrides = get_domain_config_overrides('visible')
    for key, value in domain_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Allowed datasets for visible domain
    allowed_datasets = {'UBIRIS', 'SBVPI'}
    
    # Check for preprocessed data
    processed_dir = Path("data/processed_onnx")
    if processed_dir.exists():
        logger.info("Using preprocessed dataset")
        full_dataset = IrisDataset(
            data_dirs=[processed_dir],
            transform=None,
            use_preprocessed=True,
            allowed_datasets=allowed_datasets
        )
    else:
        logger.info("Using raw dataset")
        data_dirs = get_data_dirs()
        full_dataset = IrisDataset(
            data_dirs=data_dirs,
            transform=None,
            allowed_datasets=allowed_datasets
        )
    
    logger.info(f"Dataset: {len(full_dataset)} samples, {full_dataset.num_classes} classes")
    
    # Create same train/val split as training
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create validation transforms
    val_transforms = get_iris_transforms_by_strength('very_heavy', train=False)
    val_subset = TransformSubset(full_dataset, val_indices, val_transforms)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Validation set: {len(val_subset)} samples")
    
    # Extract embeddings with TTA
    logger.info("Extracting embeddings with TTA...")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
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
    
    logger.info(f"Extracted {len(all_embeddings)} embeddings")
    
    # Compute verification metrics
    logger.info("Computing verification metrics...")
    from src.evaluation.evaluation_sampling import compute_robust_verification_scores
    
    genuine, impostor, sample_stats = compute_robust_verification_scores(
        all_embeddings, all_labels, min_genuine_pairs=500, min_classes=100, max_total_pairs=250000
    )
    
    logger.info(f"Generated {len(genuine)} genuine pairs, {len(impostor)} impostor pairs")
    
    # Compute metrics
    all_scores = np.concatenate([genuine, impostor])
    all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    
    # Filter NaN
    valid_mask = np.isfinite(all_scores)
    all_scores = all_scores[valid_mask]
    all_labels_binary = all_labels_binary[valid_mask]
    
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
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (Best Model)")
    logger.info("=" * 60)
    logger.info(f"EER: {eer * 100:.2f}%")
    logger.info(f"AUC: {roc_auc:.4f}")
    logger.info(f"d-prime: {d_prime:.4f}")
    logger.info(f"Accuracy at EER: {accuracy * 100:.2f}%")
    logger.info("=" * 60)
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Subsample impostors for plotting
    impostor_for_plotting = impostor
    if len(impostor) > 50000:
        np.random.seed(42)
        indices = np.random.choice(len(impostor), size=50000, replace=False)
        impostor_for_plotting = impostor[indices]
    
    metrics = {
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'auc': float(roc_auc),
        'd_prime': float(d_prime),
        'accuracy_at_eer': float(accuracy),
        'genuine_mean': float(np.mean(genuine)),
        'genuine_std': float(np.std(genuine)),
        'impostor_mean': float(np.mean(impostor)),
        'impostor_std': float(np.std(impostor)),
    }
    
    # Save metrics
    import json
    metrics_path = output_dir / 'best_model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate all plots
    plotter = MetricsPlotter(output_dir, 'iris')
    
    # Try to load training history from checkpoint or log
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    val_aucs = checkpoint.get('val_aucs', [])
    
    plotter.generate_all(
        genuine_scores=genuine,
        impostor_scores=impostor_for_plotting,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        eer=eer,
        eer_threshold=eer_threshold,
        train_losses=train_losses if train_losses else None,
        val_losses=val_losses if val_losses else None,
        val_accuracies=val_accuracies if val_accuracies else None,
        val_aucs=val_aucs if val_aucs else None,
        embeddings=all_embeddings,
        labels=all_labels,
        d_prime=d_prime,
        roc_auc=roc_auc
    )
    
    logger.info(f"All plots saved to {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    evaluate_iris_visible()
