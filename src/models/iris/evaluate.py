import argparse
import sys
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.logging_utils import setup_logging
from src.models.iris.config import TrainingConfig, get_data_dirs
from src.models.iris.dataset import IrisDataset, TransformSubset
from src.models.iris.augmentations import get_iris_transforms
from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.losses.arcface import ArcFaceLoss, SubCenterArcFaceLoss
from src.models.iris.domain_config import DOMAINS, get_domain_config_overrides
from src.models.iris.train import generate_evaluation

setup_logging()

def main():
    parser = argparse.ArgumentParser(description='Iris Recognition Evaluation')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory containing best_model.pt')
    parser.add_argument('--domain', type=str, required=True, choices=list(DOMAINS.keys()), help='Domain used for training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    best_model_path = output_dir / 'best_model.pt'
    
    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}")
        return

    # Load configuration based on domain (replicate training config)
    logger.info(f"Setting up evaluation for domain: {args.domain}")
    
    # Base config
    config_kwargs = {'output_dir': args.output_dir}
    
    # Domain overrides
    domain_info = DOMAINS[args.domain]
    config_kwargs['allowed_datasets'] = set(domain_info['datasets'])
    domain_overrides = get_domain_config_overrides(args.domain)
    for key, value in domain_overrides.items():
        config_kwargs[key] = value
    
    config = TrainingConfig(**config_kwargs)
    
    # Override batch size if provided
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers

    # --- Data Loading (Replicate split) ---
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
    
    if len(full_dataset) == 0:
        logger.error("No data found!")
        return

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
        logger.warning("Stratified split failed, falling back to simple split (might mismatch training if fallback happened there)")
        import random
        # Re-seed random for deterministic fallback if possible, though train.py didn't set seed for fallback
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]
        
    logger.info(f"Validation set size: {len(val_idx)}")
    
    val_samples = [full_dataset.samples[i] for i in val_idx]
    val_transform = get_iris_transforms(config.image_size, False)
    val_dataset = TransformSubset(val_samples, val_transform, is_preprocessed=use_preprocessed)
    
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # --- Model Setup ---
    num_classes = len(full_dataset.class_to_idx)
    logger.info(f"Number of classes: {num_classes}")
    
    encoder = BiometricEncoder(
        embedding_dim=config.embedding_dim,
        in_channels=3,
        pretrained=False, # Weights will be loaded
        dropout=config.dropout
    ).to(device, memory_format=torch.channels_last)
    
    # ArcFace Setup (needed for generate_evaluation signature, though not used for inference usually)
    use_subcenter = getattr(config, 'use_subcenter_arcface', False)
    num_subcenters = getattr(config, 'num_subcenters', 3)
    
    if use_subcenter:
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
        
    # --- Load Checkpoint ---
    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    if 'arcface_state_dict' in checkpoint:
        arcface.load_state_dict(checkpoint['arcface_state_dict'])
    
    best_acc = checkpoint.get('best_acc', 0.0) # Or read from log if not in checkpoint
    
    # --- Evaluation ---
    # We pass empty lists for history since we only care about final metrics/plots based on current model
    # generate_evaluation handles empty history gracefully (plots might be empty/dummy for history)
    
    generate_evaluation(
        encoder, arcface, val_loader, device, output_dir,
        train_losses=[], val_losses=[], val_accuracies=[], val_aucs=[],
        config=config, best_acc=best_acc
    )
    logger.info("Evaluation complete.")

if __name__ == '__main__':
    main()
