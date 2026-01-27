
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation_utils import create_identity_split, load_model, collate_fn, FilenamePatternDataset
from src.evaluation.metrics import MetricsExporter
from loguru import logger

def main():
    MODALITY = 'iris'
    # Paths verified via ls
    SBVPI_DIR = PROJECT_ROOT / "data/raw/sbvpi/SBVPI_Sclera/Training/Images"
    UBIRIS_DIR = PROJECT_ROOT / "data/raw/ubiris/ubiris2_1/CLASSES_400_300_Part1"
    
    # Model path
    MODEL_PATH = PROJECT_ROOT / "outputs/iris_domain_nir_casia/iris/encoder.pt"
    # Fallback to visible if NIR not found
    if not MODEL_PATH.exists():
        MODEL_PATH = PROJECT_ROOT / "outputs/iris_domain_visible/iris/encoder.pt"
        
    OUTPUT_DIR = PROJECT_ROOT / "outputs/validation_suite/iris"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Validating {MODALITY} on {device}")
    
    # 1. Load Data
    # Iris usually 224x224 in this repo (or rectangular, but resizing to 224x224 is standard for CNNs)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    datasets = []
    
    if SBVPI_DIR.exists():
        logger.info(f"Loading SBVPI from {SBVPI_DIR}")
        ds1 = FilenamePatternDataset(SBVPI_DIR, pattern_type='sbvpi', transform=transform)
        if len(ds1) > 0: datasets.append(ds1)
        
    if UBIRIS_DIR.exists():
        logger.info(f"Loading UBIRIS from {UBIRIS_DIR}")
        ds2 = FilenamePatternDataset(UBIRIS_DIR, pattern_type='ubiris', transform=transform)
        if len(ds2) > 0: datasets.append(ds2)
        
    if not datasets:
        logger.error("No iris datasets found.")
        return
        
    full_dataset = ConcatDataset(datasets)
    # We need to manually combine samples list for split logic
    full_dataset.samples = []
    for ds in datasets:
        full_dataset.samples.extend(ds.samples)
    
    # 2. Split by Identity
    _, val_subset = create_identity_split(full_dataset, val_ratio=0.3)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 3. Load Model
    # Note: Iris model config might vary (NIR vs Visible). 
    # If trained on CASIA (NIR), it might perform poorly on UBIRIS (Visible).
    # But we validate on what we have.
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
