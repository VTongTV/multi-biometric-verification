
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation_utils import create_identity_split, load_model, collate_fn, FilenamePatternDataset
from src.evaluation.metrics import MetricsExporter
from loguru import logger

def main():
    MODALITY = 'fingerprint'
    DATA_DIR = PROJECT_ROOT / "data/raw/socofing/SOCOFing"
    MODEL_PATH = PROJECT_ROOT / "outputs/fingerprint/fingerprint/encoder.pt"
    OUTPUT_DIR = PROJECT_ROOT / "outputs/validation_suite/fingerprint"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Validating {MODALITY} on {device}")
    
    if not DATA_DIR.exists():
        logger.error(f"Data not found: {DATA_DIR}")
        # Try alternate path
        DATA_DIR = PROJECT_ROOT / "data/raw/fingerprint"
        if not DATA_DIR.exists():
            logger.error("No fingerprint data found.")
            return
        logger.info(f"Using alternate path: {DATA_DIR}")
        PATTERN = 'fvc'
    else:
        PATTERN = 'socofing'
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((160, 160)), # EfficientNet standard for fingerprint in this repo
        transforms.Grayscale(num_output_channels=3), # Model expects 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = FilenamePatternDataset(DATA_DIR, pattern_type=PATTERN, transform=transform, grayscale=True)
    
    # 2. Split by Identity
    _, val_subset = create_identity_split(full_dataset, val_ratio=0.3)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 3. Load Model
    model = load_model(MODALITY, MODEL_PATH, device)
    
    # 4. Extract
    embeddings = []
    labels = []
    
    logger.info("Extracting embeddings...")
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
