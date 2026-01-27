
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation_utils import create_identity_split, load_model, collate_fn
from src.data.datasets.biometric_datasets import IdentityFolderDataset
from src.evaluation.metrics import MetricsExporter
from loguru import logger

def main():
    MODALITY = 'face'
    DATA_DIR = PROJECT_ROOT / "data/raw/lfw-deepfunneled/lfw-deepfunneled"
    MODEL_PATH = PROJECT_ROOT / "outputs/face/face/encoder.pt"
    OUTPUT_DIR = PROJECT_ROOT / "outputs/validation_suite/face"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Validating {MODALITY} on {device}")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if not DATA_DIR.exists():
        logger.error(f"Data not found: {DATA_DIR}")
        return

    full_dataset = IdentityFolderDataset(DATA_DIR, transform=transform)
    
    # 2. Split by Identity (The Fix)
    train_subset, val_subset = create_identity_split(full_dataset, val_ratio=0.3)
    
    # We only care about Val for this test
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 3. Load Model
    model = load_model(MODALITY, MODEL_PATH, device)
    
    # 4. Extract
    embeddings = []
    labels = []
    
    logger.info("Extracting embeddings from Validation Set (Unseen Identities)...")
    with torch.no_grad():
        for imgs, lbls in val_loader:
            if imgs.numel() == 0: continue
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())
            
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # 5. Metrics
    exporter = MetricsExporter(OUTPUT_DIR, f"{MODALITY}_split_val")
    metrics = exporter.compute_all_metrics(embeddings, labels)
    exporter.generate_all_outputs()
    
    logger.info("="*50)
    logger.info(f"{MODALITY.upper()} SPLIT VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"EER: {metrics.get('eer', 0)*100:.2f}%")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
