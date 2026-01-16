
import torch
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble.ensemble_model import BiometricEnsemble


def find_model_path(candidates: list, model_name: str) -> Path:
    """Find the first existing model path from candidates list."""
    for p in candidates:
        if p.exists():
            logger.info(f"  Found {model_name}: {p}")
            return p
    logger.warning(f"  {model_name}: NOT FOUND (searched {len(candidates)} locations)")
    return None


def build_ensemble():
    """Build and save the Multi-Modal Biometric Ensemble.
    
    Searches for trained models in expected output directories and
    builds an ensemble with all available modalities.
    """
    output_dir = project_root / 'outputs' / 'ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BUILDING MULTI-MODAL BIOMETRIC ENSEMBLE")
    logger.info("=" * 60)
    
    # 1. Face - Prioritize Fine-Tuned Model over Generic ONNX
    # The locally trained model matches the normalization/augmentation pipeline
    face_candidates = [
        project_root / 'outputs' / 'face' / 'best_model.pt',
        project_root / 'outputs' / 'face' / 'encoder.pt',
    ]
    face_path = find_model_path(face_candidates, "Face")
    
    # Fallback to ONNX backbone if local training not found
    if not face_path:
        onnx_path = project_root / 'weights' / 'R50_Glint360K.onnx'
        if onnx_path.exists():
            logger.info(f"  Found Face ONNX (Fallback): {onnx_path}")
            face_path = onnx_path
        else:
            logger.warning(f"  Face ONNX NOT FOUND: {onnx_path}")
    
    # 2. Fingerprint - use specified location
    finger_candidates = [
        project_root / 'outputs' / 'fingerprint' / 'fingerprint' / 'best_model.pt',
        project_root / 'outputs' / 'fingerprint' / 'fingerprint' / 'encoder.pt',
        project_root / 'outputs' / 'fingerprint' / 'encoder.pt',
        project_root / 'outputs' / 'fingerprint' / 'best_model.pt',
    ]
    finger_path = find_model_path(finger_candidates, "Fingerprint")
    
    # 3. Iris NIR (CASIA) - use specified location
    nir_candidates = [
        project_root / 'outputs' / 'iris_domain_nir_casia' / 'best_model.pt',
        project_root / 'outputs' / 'iris_domain_nir_casia' / 'encoder.pt',
        project_root / 'outputs' / 'iris_domain_nir_casia' / 'iris' / 'checkpoints' / 'best_model.pt',
    ]
    iris_nir_path = find_model_path(nir_candidates, "Iris NIR")
    
    # 4. Iris Visible (UBIRIS/SBVPI) - use specified location
    vis_candidates = [
        project_root / 'outputs' / 'iris_domain_visible' / 'best_model.pt',
        project_root / 'outputs' / 'iris_domain_visible' / 'encoder.pt',
        project_root / 'outputs' / 'iris_domain_visible' / 'iris' / 'checkpoints' / 'best_model.pt',
    ]
    iris_vis_path = find_model_path(vis_candidates, "Iris Visible")
    
    # Count available modalities
    available_count = sum([
        face_path is not None,
        finger_path is not None,
        iris_nir_path is not None,
        iris_vis_path is not None,
    ])
    
    logger.info(f"\nAvailable modalities: {available_count}/4")
    
    if available_count == 0:
        logger.error("No trained models found! Cannot build ensemble.")
        logger.error("Please train at least one modality first.")
        return None
    
    # Instantiate ensemble
    logger.info("\nBuilding ensemble model...")
    model = BiometricEnsemble(
        face_weights_path=str(face_path) if face_path else None,
        iris_nir_weights_path=str(iris_nir_path) if iris_nir_path else None,
        iris_vis_weights_path=str(iris_vis_path) if iris_vis_path else None,
        fingerprint_weights_path=str(finger_path) if finger_path else None,
        device='cpu'
    )
    
    # Save ensemble
    save_path = output_dir / 'ensemble_model.pt'
    try:
        torch.save(model.state_dict(), save_path)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ENSEMBLE SAVED: {save_path}")
        logger.info(f"Size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"{'=' * 60}")
    except Exception as e:
        logger.error(f"Failed to save ensemble: {e}")
        return None
    
    # Summary
    logger.info("\nModel Composition:")
    logger.info(f"  Face:         {'LOADED' if face_path else 'NOT AVAILABLE'}")
    logger.info(f"  Fingerprint:  {'LOADED' if finger_path else 'NOT AVAILABLE'}")
    logger.info(f"  Iris NIR:     {'LOADED' if iris_nir_path else 'NOT AVAILABLE'}")
    logger.info(f"  Iris Visible: {'LOADED' if iris_vis_path else 'NOT AVAILABLE'}")
    
    return model


if __name__ == "__main__":
    build_ensemble()
