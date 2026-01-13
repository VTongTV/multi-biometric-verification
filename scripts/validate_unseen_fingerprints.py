
import sys
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / 'webapp' / 'backend'
# sys.path.insert(0, str(BACKEND_DIR))

try:
    from src.models.encoders.biometric_encoder import BiometricEncoder
    from src.evaluation.metrics import MetricsExporter
    from loguru import logger
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

class UnseenFingerprintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Support DB1_B, DB2_B, etc.
        # Structure: root/DB1_B/101_1.tif
        
        # Recursive search for .tif files
        image_files = list(self.root_dir.rglob('*.tif')) + list(self.root_dir.rglob('*.bmp')) + list(self.root_dir.rglob('*.jpg'))
        
        if not image_files:
            print(f"No images found in {self.root_dir}")
        
        for img_path in image_files:
            # Parse identity from filename
            # Format usually: ID_Impression.tif (e.g., 101_1.tif)
            filename = img_path.name
            try:
                # Assuming format ID_Impression.ext
                # Adjust splitting logic based on common FVC formats
                if '_' in filename:
                    identity = filename.split('_')[0]
                else:
                    # Fallback or specific handling
                    identity = filename.split('.')[0]
                
                # We need numeric labels for metric computation usually, but string IDs work for some logic
                # Let's map string IDs to integers
                self.samples.append((str(img_path), identity))
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        # Map string identities to integers
        self.unique_identities = sorted(list(set(s[1] for s in self.samples)))
        self.class_to_idx = {name: i for i, name in enumerate(self.unique_identities)}
        
        print(f"Found {len(self.samples)} images from {len(self.unique_identities)} identities.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, identity = self.samples[idx]
        label = self.class_to_idx[identity]
        
        try:
            image = Image.open(img_path).convert('L') # Fingerprints are grayscale
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            return torch.zeros((1, 224, 224)), label, img_path

def main():
    parser = argparse.ArgumentParser(description='Validate Fingerprint Model on Unseen Data')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--model-path', type=str, default=None, help='Path to .pt model file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger.add(output_dir / "validation.log")
    logger.info(f"Starting validation on {args.data_dir}")

    # 2. Dataset & Dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = UnseenFingerprintDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        logger.error("Dataset is empty.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. Load Model
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Default path
        model_path = BACKEND_DIR / 'outputs' / 'sota' / 'fingerprint' / 'encoder.pt'
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return
        
    logger.info(f"Loading model from {model_path}")
    
    # Initialize model (Matching training config)
    # BiometricEncoder defaults: embedding_dim=512, but we need to check if training used 256 or 512
    # The log `train_Fingerprint_20260111_093804.log` mentions "Initializing ArcFace with scale=32.0..." but doesn't explicitly state embedding dim in the visible snippet.
    # However, `biometric_encoder.py` defaults to 512.
    # If load_state_dict fails, we might need to retry with 256.
    
    try:
        encoder = BiometricEncoder(embedding_dim=512, in_channels=1, backbone_type='efficientnet').to(device)
        state_dict = torch.load(model_path, map_location=device)
        encoder.load_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"Failed to load with dim=512: {e}")
        logger.info("Retrying with dim=256...")
        try:
            encoder = BiometricEncoder(embedding_dim=256, in_channels=1, backbone_type='efficientnet').to(device)
            state_dict = torch.load(model_path, map_location=device)
            encoder.load_state_dict(state_dict)
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            return

    encoder.eval()

    # 4. Extract Embeddings
    all_embeddings = []
    all_labels = []
    
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader):
            images = images.to(device)
            
            # Simple TTA: Original + Flip
            emb_orig = encoder(images)
            emb_flip = encoder(torch.flip(images, [3]))
            embeddings = torch.nn.functional.normalize(emb_orig + emb_flip, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    # 5. Compute Metrics & Plots
    logger.info(f"Computing metrics for {len(all_embeddings)} samples...")
    
    exporter = MetricsExporter(output_dir, "Unseen_Fingerprint_Validation")
    
    # Compute all metrics
    metrics = exporter.compute_all_metrics(all_embeddings, all_labels)
    
    # Generate outputs
    logger.info("Generating plots and CSVs...")
    outputs = exporter.generate_all_outputs()
    
    logger.info("="*50)
    logger.info("VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"EER: {metrics.get('eer', 0)*100:.2f}%")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    logger.info(f"d-prime: {metrics.get('d_prime', 0):.4f}")
    logger.info(f"Accuracy at EER: {metrics.get('accuracy_at_eer', 0)*100:.2f}%")
    logger.info(f"Plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
