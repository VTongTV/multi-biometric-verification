"""
Exact Evaluation Script for Iris Models.

Replicates the EXACT evaluation settings used during training to get
the precise AUC value that was recorded as "best".

Key settings matched from trainer.py validate():
- NO TTA (Test-Time Augmentation) - same as training epochs
- compute_robust_verification_scores with min_genuine_pairs=500, max_total_pairs=150000
- Same embedding extraction pipeline

Usage:
    python -m src.models.iris.exact_eval --output-dir outputs/iris_domain_visible
    python -m src.models.iris.exact_eval --output-dir outputs/iris_domain_nir_casia --domain nir_casia
"""
import sys
from pathlib import Path
from loguru import logger
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Project root setup
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.iris.dataset import IrisDataset, TransformSubset
from src.models.iris.config import TrainingConfig, get_data_dirs
from src.models.iris.domain_config import get_domain_config_overrides, DOMAINS
from src.models.iris.augmentations import get_iris_transforms
from src.evaluation.evaluation_sampling import compute_robust_verification_scores
from src.models.encoders.biometric_encoder import BiometricEncoder


def extract_embeddings_exact(encoder, dataloader, device, use_tta=False):
    """
    Extract embeddings using EXACT same pipeline as trainer.py validate().

    Args:
        encoder: The encoder model
        dataloader: Validation dataloader
        device: torch device
        use_tta: Whether to use TTA (default: False to match training)
    """
    encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            images = images.to(memory_format=torch.channels_last)

            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if use_tta:
                    # TTA: Horizontal flip (for final evaluation only)
                    emb_orig = encoder(images)
                    emb_flip = encoder(torch.flip(images, [3]))
                    embeddings = F.normalize(emb_orig + emb_flip, p=2, dim=1)
                else:
                    # NO TTA - same as training epochs
                    embeddings = encoder(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    return all_embeddings, all_labels


def compute_auc_exact(embeddings, labels):
    """
    Compute AUC using EXACT same settings as trainer.py validate().

    Uses compute_robust_verification_scores with:
    - min_genuine_pairs=500
    - max_total_pairs=150000
    """
    try:
        genuine, impostor, _ = compute_robust_verification_scores(
            embeddings, labels,
            min_genuine_pairs=500,
            max_total_pairs=150000
        )

        genuine = genuine[np.isfinite(genuine)]
        impostor = impostor[np.isfinite(impostor)]

        if len(genuine) > 0 and len(impostor) > 0:
            all_scores = np.concatenate([genuine, impostor])
            all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
            fpr, tpr, _ = roc_curve(all_labels_binary, all_scores)
            val_auc = auc(fpr, tpr)

            stats = {
                'n_genuine': len(genuine),
                'n_impostor': len(impostor),
                'auc': val_auc
            }
            return val_auc, stats
        else:
            return 0.5, {'n_genuine': 0, 'n_impostor': 0, 'auc': 0.5}

    except Exception as e:
        logger.warning(f"AUC calculation failed: {e}")
        return 0.5, {'n_genuine': 0, 'n_impostor': 0, 'auc': 0.5}


def run_exact_evaluation(output_dir: str, domain: str = 'visible', use_tta: bool = False):
    """
    Run exact evaluation matching training settings.

    Args:
        output_dir: Path to output directory containing best_model.pt
        domain: 'nir_casia' or 'visible'
        use_tta: Whether to use TTA (default False to match training)
    """
    output_path = Path(output_dir)
    best_model_path = output_path / 'best_model.pt'

    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}")
        return None

    logger.info(f"{'=' * 60}")
    logger.info("EXACT EVALUATION - Matching Training Settings")
    logger.info(f"{'=' * 60}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Use TTA: {use_tta}")
    logger.info(f"Model: {best_model_path}")
    logger.info(f"{'=' * 60}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load domain config
    domain_info = DOMAINS.get(domain, DOMAINS['visible'])
    config_kwargs = {
        'output_dir': output_dir,
        'allowed_datasets': set(domain_info['datasets']),
    }
    domain_overrides = get_domain_config_overrides(domain)
    for key, value in domain_overrides.items():
        config_kwargs[key] = value

    config = TrainingConfig(**config_kwargs)

    # Create encoder
    encoder = BiometricEncoder(
        embedding_dim=config.embedding_dim,
        in_channels=3,
        pretrained=False,  # Load from checkpoint, no pretrained needed
        dropout=0.0  # No dropout for evaluation
    ).to(device)

    # Load best model
    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    logger.info(f"Best model loaded (epoch: {checkpoint.get('epoch', 'unknown')})")

    # Setup dataset (same as training)
    logger.info("Setting up validation dataset...")
    data_dirs = get_data_dirs(project_root)
    processed_root = project_root / 'data' / 'processed_onnx'
    use_preprocessed = processed_root.exists() and len(list(processed_root.glob('**/*.png'))) > 1000

    if use_preprocessed:
        data_dirs = [processed_root]
        logger.info(f"Using preprocessed data from {processed_root}")

    full_dataset = IrisDataset(
        data_dirs,
        transform=None,
        sample_fraction=config.sample_fraction,
        is_preprocessed=use_preprocessed,
        allowed_datasets=config.allowed_datasets
    )

    # Filter rare classes
    full_dataset.filter_rare_classes(min_samples=config.min_samples_per_class)

    # Replicate stratified split
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
    except ValueError:
        logger.warning("Stratified split failed, falling back to simple split")
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

    logger.info(f"Validation samples: {len(val_idx)}, Classes: {len(set([full_dataset.samples[i][1] for i in val_idx]))}")

    # Create validation dataset with transforms
    val_transform = get_iris_transforms(config.image_size, False)
    val_samples = [full_dataset.samples[i] for i in val_idx]
    val_dataset = TransformSubset(val_samples, val_transform, is_preprocessed=use_preprocessed)

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # Extract embeddings (EXACT same as training - NO TTA by default)
    logger.info(f"Extracting embeddings (TTA={use_tta})...")
    embeddings, labels = extract_embeddings_exact(encoder, val_loader, device, use_tta=use_tta)
    logger.info(f"Extracted {len(embeddings)} embeddings from {len(set(labels))} classes")

    # Compute AUC (EXACT same as training)
    logger.info("Computing AUC with training settings...")
    val_auc, stats = compute_auc_exact(embeddings, labels)

    # Results
    logger.info(f"{'=' * 60}")
    logger.info("RESULTS (Exact Training Settings)")
    logger.info(f"{'=' * 60}")
    logger.info(f"Genuine pairs: {stats['n_genuine']}")
    logger.info(f"Impostor pairs: {stats['n_impostor']}")
    logger.info(f"Val AUC: {val_auc:.6f}")
    logger.info(f"{'=' * 60}")

    return val_auc


def main():
    parser = argparse.ArgumentParser(
        description='Exact evaluation matching training settings'
    )
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory containing best_model.pt')
    parser.add_argument('--domain', type=str, default='visible',
                        choices=['nir_casia', 'visible'],
                        help='Domain for dataset loading')
    parser.add_argument('--use-tta', action='store_true',
                        help='Use TTA (horizontal flip) - for comparison only')

    args = parser.parse_args()

    setup_logging()

    # Run exact evaluation
    auc_value = run_exact_evaluation(args.output_dir, args.domain, args.use_tta)

    if auc_value is not None:
        logger.info(f"\nFinal AUC: {auc_value:.6f}")
        logger.info("This should match the 'Best AUC' recorded during training.")
    else:
        logger.error("Evaluation failed")


def setup_logging():
    """Setup logging configuration."""
    from src.utils.logging_utils import setup_logging
    setup_logging()


if __name__ == '__main__':
    main()
