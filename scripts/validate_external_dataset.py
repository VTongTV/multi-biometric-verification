
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

# Setup paths
# scripts/validate_external_dataset.py -> <project_root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add webapp/backend to path to find src modules if needed, though most seem to be under PROJECT_ROOT/src based on imports
# The training script used sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent)) which points to PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from project
try:
    from src.models.encoders.biometric_encoder import BiometricEncoder
    from src.models.fingerprint.ridge_attention import RidgeAttention
    from src.evaluation.metrics import MetricsExporter
    from loguru import logger
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# ============================================================================
# MODEL DEFINITION (Copied from train.py)
# ============================================================================

class FingerprintEncoderWithAttention(nn.Module):
    """
    Wrapper that properly integrates Ridge Attention into the encoder forward pass.
    """
    
    def __init__(self, encoder, ridge_attention):
        super().__init__()
        self.encoder = encoder
        self.ridge_attention = ridge_attention
        # Expose embedding_head for optimizer param groups
        self.embedding_head = encoder.embedding_head
        self.backbone = encoder.backbone
        
        # CRITICAL: Disable encoder's built-in attention to prevent double attention
        if hasattr(encoder, 'use_attention'):
            encoder.use_attention = False
    
    def forward(self, x, normalize=True):
        # Extract backbone features (without encoder's attention)
        features = self.encoder.backbone.features(x)
        
        # Apply ridge-specific attention (replaces encoder's attention)
        features = self.ridge_attention(features)
        
        # Clamp for FP16 stability before pooling (keeping consistent with training)
        features = torch.clamp(features, min=-65000, max=65000)
        
        # Pool and embed
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        embeddings = self.encoder.embedding_head(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# ============================================================================
# DATASET HANDLING
# ============================================================================

class GrayscaleTo3Channel:
    """Convert 1-channel grayscale to 3-channel by duplication."""
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

def get_validation_transform(image_size=160):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        GrayscaleTo3Channel(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def parse_fvc_identity(filename):
    """
    Parse FVC filename format: 101_1.tif -> Identity 101
    """
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return filename

def load_external_dataset(data_dir, transform):
    samples = []
    labels = []
    paths = []
    
    data_dir = Path(data_dir)
    print(f"Scanning {data_dir}...")
    
    # Extensions to look for
    exts = ['*.tif', '*.TIF', '*.bmp', '*.BMP', '*.jpg', '*.png']
    all_files = []
    for ext in exts:
        all_files.extend(list(data_dir.rglob(ext)))
    
    all_files = sorted(list(set(all_files)))
    print(f"Found {len(all_files)} images.")
    
    # Map identities to integers 0..N
    identity_map = {}
    next_idx = 0
    
    valid_images = []
    
    for file_path in all_files:
        try:
            # Parse identity
            identity_str = parse_fvc_identity(file_path.name)
            
            if identity_str not in identity_map:
                identity_map[identity_str] = next_idx
                next_idx += 1
            
            label = identity_map[identity_str]
            
            # Load and transform
            img = Image.open(file_path).convert('L')
            if transform:
                img_tensor = transform(img)
                samples.append(img_tensor.numpy())
                labels.append(label)
                paths.append(str(file_path))
                
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            
    return np.array(samples), np.array(labels), paths

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model_path = PROJECT_ROOT / 'outputs' / 'sota' / 'fingerprint' / 'encoder.pt'
    if not model_path.exists():
        # Try checking webapp backend path
        model_path = PROJECT_ROOT / 'webapp' / 'backend' / 'outputs' / 'sota' / 'fingerprint' / 'encoder.pt'
        
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    
    # Initialize base encoder
    base_encoder = BiometricEncoder(
        embedding_dim=512, 
        in_channels=3, 
        pretrained=False, # No need to download pretrained weights, we load from checkpoint
        backbone_type='efficientnet',
        use_attention=False,
        drop_path_rate=0.0
    )
    
    # Initialize attention
    ridge_attention = RidgeAttention(channels=1280, reduction=16)
    
    # Wrap
    model = FingerprintEncoderWithAttention(base_encoder, ridge_attention).to(device)
    
    # Load weights
    state_dict = None
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("[SUCCESS] Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Try loading into just the base encoder if wrapper fails (fallback)
        if state_dict is not None:
            try:
                base_encoder.load_state_dict(state_dict, strict=False)
                model = base_encoder.to(device)
                print("! Loaded weights into base encoder (wrapper mismatch?)")
            except:
                print("FATAL: Could not load model.")
                return
        else:
            print("FATAL: Could not load model file.")
            return

    model.eval()
    
    # 2. Load Data
    data_path = PROJECT_ROOT / 'archive' / 'fingerprints'
    if not data_path.exists():
        print(f"Data path not found: {data_path}")
        return
        
    transform = get_validation_transform(image_size=160)
    
    print("Loading dataset into memory...")
    images, labels, paths = load_external_dataset(data_path, transform)
    
    if len(images) == 0:
        print("No images found.")
        return
        
    print(f"Loaded {len(images)} images from {len(np.unique(labels))} identities.")
    
    # 3. Extract Embeddings
    print("Extracting embeddings...")
    batch_size = 32
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.tensor(images[i:i+batch_size]).to(device)
            emb = model(batch, normalize=True)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.vstack(embeddings)
    
    # 4. Generate Metrics
    output_dir = PROJECT_ROOT / 'outputs' / 'validation_external'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating metrics in {output_dir}...")
    exporter = MetricsExporter(output_dir, 'fingerprint_external')
    
    # Compute all metrics
    metrics = exporter.compute_all_metrics(embeddings, labels)
    
    # Generate all outputs
    exporter.generate_all_outputs()
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS (EXTERNAL DATASET)")
    print("="*50)
    print(f"EER: {metrics.get('eer', 0)*100:.2f}%")
    print(f"AUC: {metrics.get('auc', 0):.4f}")
    print(f"Accuracy @ EER: {metrics.get('accuracy_at_eer', 0)*100:.2f}%")
    print(f"d-prime: {metrics.get('d_prime', 0):.4f}")
    print("-" * 50)
    print(f"Detailed results saved to {output_dir / 'fingerprint_external'}")

if __name__ == '__main__':
    main()
