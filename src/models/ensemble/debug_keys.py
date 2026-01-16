
import torch
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.encoders.biometric_encoder import BiometricEncoder

def debug_keys():
    # 1. Load Checkpoint
    checkpoint_path = project_root / 'outputs' / 'face' / 'best_model.pt'
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
        logger.info("Found 'encoder_state_dict' in checkpoint")
    else:
        state_dict = checkpoint
        logger.info("Using raw checkpoint as state_dict")
        
    ckpt_keys = list(state_dict.keys())
    logger.info(f"Checkpoint Keys ({len(ckpt_keys)} total):")
    for k in ckpt_keys[:10]:
        logger.info(f"  {k}")
        
    # 2. Instantiate Model
    logger.info("Instantiating BiometricEncoder(backbone_type='efficientnet')...")
    model = BiometricEncoder(embedding_dim=512, backbone_type='efficientnet', pretrained=False)
    model_keys = list(model.state_dict().keys())
    
    logger.info(f"Model Keys ({len(model_keys)} total):")
    for k in model_keys[:10]:
        logger.info(f"  {k}")
        
    # 3. Check Overlap
    # Simulate the cleaning logic from ensemble_model.py
    clean_ckpt_keys = [k.replace('module.', '').replace('encoder.', '') for k in ckpt_keys]
    
    intersection = set(clean_ckpt_keys).intersection(set(model_keys))
    logger.info(f"Intersection: {len(intersection)} keys match")
    
    if len(intersection) == 0:
        logger.error("ZERO KEYS MATCH! Architecture mismatch confirmed.")
    elif len(intersection) < len(model_keys) * 0.5:
        logger.warning(f"Low match rate ({len(intersection)}/{len(model_keys)}).")
        
    # Specific check for backbone
    logger.info("Checking backbone prefix...")
    ckpt_backbone = [k for k in clean_ckpt_keys if 'backbone' in k]
    model_backbone = [k for k in model_keys if 'backbone' in k]
    
    if ckpt_backbone:
        logger.info(f"Checkpoint backbone sample: {ckpt_backbone[0]}")
    if model_backbone:
        logger.info(f"Model backbone sample: {model_backbone[0]}")

if __name__ == "__main__":
    debug_keys()
