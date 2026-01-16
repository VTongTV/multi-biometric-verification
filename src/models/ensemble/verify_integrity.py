
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble.ensemble_model import BiometricEnsemble, FingerprintEncoderWithAttention
from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.fingerprint.ridge_attention import RidgeAttention
from src.evaluation.lfw_evaluator import LFWEvaluator

def load_single_model(path, modality):
    """Load a single modality model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['encoder_state_dict'] if 'encoder_state_dict' in checkpoint else checkpoint
    
    if modality == 'face':
        model = BiometricEncoder(embedding_dim=512, backbone_type='iresnet50', pretrained=False)
        # Handle iresnet structure if needed, or if it was efficientnet
        # The ensemble uses iresnet50 if onnx provided, else efficientnet.
        # Let's check the path or assume efficientnet if trained via our script
        # Training script uses EfficientNetV2-S by default usually unless specified.
        # But wait, Face training script default is EfficientNet.
        model = BiometricEncoder(embedding_dim=512, backbone_type='iresnet50', pretrained=False)

    elif 'iris' in modality:
        model = BiometricEncoder(embedding_dim=512, backbone_type='efficientnet', use_attention=True, pretrained=False)
    elif modality == 'fingerprint':
        # Reconstruct the wrapper structure
        base = BiometricEncoder(embedding_dim=512, in_channels=3, backbone_type='efficientnet', use_attention=False, pretrained=False)
        attn = RidgeAttention(channels=1280, reduction=16)
        model = FingerprintEncoderWithAttention(base, attn)
    
    # Clean keys
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('encoder.', '') 
        clean_state_dict[k] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model

class EnsembleWrapper(nn.Module):
    """Wraps ensemble to look like a single model for a specific modality."""
    def __init__(self, ensemble, modality):
        super().__init__()
        self.ensemble = ensemble
        self.modality = modality
        
    def forward(self, x):
        inputs = {self.modality: x}
        out = self.ensemble(inputs)
        return out['embeddings'][self.modality]

def verify_integrity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running verification on {device}...")
    
    # 1. Load Ensemble
    ensemble_path = project_root / 'outputs' / 'ensemble' / 'ensemble_model.pt'
    logger.info(f"Loading Ensemble from {ensemble_path}")
    
    # Initialize with face weights to match the saved model structure
    face_weights_path = str(project_root / 'outputs' / 'face' / 'best_model.pt')
    ensemble = BiometricEnsemble(
        face_weights_path=face_weights_path,
        device=device
    )
    
    # Load the actual weights
    state_dict = torch.load(ensemble_path, map_location='cpu')
    ensemble.load_state_dict(state_dict)
    ensemble.to(device)
    ensemble.eval()
    
    # 2. Define Single Models to Check
    checks = [
        {
            'name': 'Face',
            'modality': 'face',
            'path': project_root / 'outputs' / 'face' / 'best_model.pt',
            'input_shape': (1, 3, 112, 112)
        },
        {
            'name': 'Fingerprint',
            'modality': 'fingerprint',
            'path': project_root / 'outputs' / 'fingerprint' / 'fingerprint' / 'best_model.pt',
            'input_shape': (1, 3, 160, 160) 
        },
        {
            'name': 'Iris NIR',
            'modality': 'iris_nir',
            'path': project_root / 'outputs' / 'iris_domain_nir_casia' / 'best_model.pt',
            'input_shape': (1, 3, 224, 224) 
        },
        {
            'name': 'Iris Visible',
            'modality': 'iris_vis',
            'path': project_root / 'outputs' / 'iris_domain_visible' / 'best_model.pt',
            'input_shape': (1, 3, 224, 224)
        }
    ]
    
    logger.info("="*60)
    logger.info("TEST 1: EMBEDDING CONSISTENCY CHECK")
    logger.info("Verifying that Ensemble output matches Single Model output exactly.")
    logger.info("="*60)
    
    for check in checks:
        name = check['name']
        modality = check['modality']
        path = check['path']
        
        if not path.exists():
            logger.warning(f"Skipping {name}: Model file not found at {path}")
            continue
            
        logger.info(f"Checking {name}...")
        
        # Load single model
        try:
            single_model = load_single_model(path, modality).to(device)
        except Exception as e:
            logger.error(f"Failed to load single model for {name}: {e}")
            continue
            
        # Create random input
        x = torch.randn(check['input_shape']).to(device)
        
        # Get outputs
        with torch.no_grad():
            # Single Model Output
            y_single = single_model(x)
            
            # Ensemble call
            inputs = {modality: x}
            out = ensemble(inputs)
            y_ensemble = out['embeddings'][modality]
            
            # Normalize single output if needed to match ensemble
            # (BiometricEncoder returns normalized usually, but we check)
            y_single = F.normalize(y_single, p=2, dim=1)
            # Ensemble already normalizes in forward()
            
            # Calculate difference
            diff = torch.abs(y_single - y_ensemble).max().item()
            
            if diff < 1e-4: # Allow small FP error
                logger.info(f"  [PASS] {name} Embeddings Match! (Max Diff: {diff:.8f})")
            else:
                logger.error(f"  [FAIL] {name} Embeddings Mismatch! (Max Diff: {diff:.8f})")
                
    logger.info("\n" + "="*60)
    logger.info("TEST 2: LFW EVALUATION (FACE ONLY)")
    logger.info("Running standard LFW protocol using Ensemble Face Branch.")
    logger.info("="*60)
    
    # Setup LFW Evaluator
    data_dir = project_root / 'data' / 'raw'
    lfw_pairs = data_dir / 'pairs.csv'
    
    if lfw_pairs.exists():
        evaluator = LFWEvaluator(data_dir, lfw_pairs, image_size=112, batch_size=32)
        
        # Wrap ensemble
        face_wrapper = EnsembleWrapper(ensemble, 'face')
        
        mean, std = evaluator.evaluate(face_wrapper, device)
        logger.info(f"Ensemble Face Accuracy on LFW: {mean:.2f}% (+/- {std:.2f}%)")
    else:
        logger.warning(f"LFW pairs.csv not found at {lfw_pairs}. Skipping LFW test.")

if __name__ == "__main__":
    verify_integrity()
