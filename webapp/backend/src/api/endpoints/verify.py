
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from typing import Optional, Annotated
import torch

from src.api.schemas import BiometricResponse, ErrorResponse
from src.api.dependencies import get_model, get_gallery_cache, load_image_tensor, face_transform, iris_transform, finger_transform
from models.ensemble.ensemble_model import BiometricEnsemble

router = APIRouter()

@router.post("/verify", response_model=BiometricResponse)
async def verify(
    identity: Annotated[str, Form()],
    protection_key: Optional[str] = Form(None),
    face_file: Optional[UploadFile] = File(None),
    iris_nir_file: Optional[UploadFile] = File(None),
    iris_vis_file: Optional[UploadFile] = File(None),
    fingerprint_file: Optional[UploadFile] = File(None),
    model: BiometricEnsemble = Depends(get_model),
    gallery_cache: dict = Depends(get_gallery_cache)
):
    """
    Verify 1:1 match against a specific identity.
    """
    # 1. Get Gallery Data
    if identity in gallery_cache:
        gallery_out = gallery_cache[identity]
    else:
        raise HTTPException(status_code=404, detail="Identity not found in gallery")

    # 2. Process Probe Data
    probe_inputs = {}

    if face_file:
        probe_inputs['face'] = load_image_tensor(face_file.file, face_transform)
    if iris_nir_file:
        # Apply Worldcoin segmentation for iris (models trained on segmented ROIs)
        probe_inputs['iris_nir'] = load_image_tensor(iris_nir_file.file, iris_transform, apply_segmentation=True)
    if iris_vis_file:
        # Apply Worldcoin segmentation for iris (models trained on segmented ROIs)
        probe_inputs['iris_vis'] = load_image_tensor(iris_vis_file.file, iris_transform, apply_segmentation=True)
    if fingerprint_file:
        probe_inputs['fingerprint'] = load_image_tensor(fingerprint_file.file, finger_transform, preprocessing_type='fingerprint')

    # Remove None entries
    probe_inputs = {k: v for k, v in probe_inputs.items() if v is not None}

    if not probe_inputs:
        raise HTTPException(status_code=400, detail="No valid biometric data provided")

    # 3. Verification
    with torch.no_grad():
        probe_out = model(probe_inputs)
        
        # Apply Template Protection if key provided
        if protection_key:
            # Protect Probe
            probe_out = model.protect_embeddings(probe_out, protection_key)
            # Protect Gallery (on-the-fly from raw cache)
            gallery_out = model.protect_embeddings(gallery_out, protection_key)
            
        result = model.compute_score(probe_out, gallery_out)

    score = result['fused_score'].item()
    details = {k: v.item() for k, v in result['individual_scores'].items()}

    # DEBUG: Log details for troubleshooting
    from loguru import logger
    logger.info(f"Verification details: {details}")

    # Decision
    decision = "AUTHORIZED" if score > 0.6 else "REJECTED"

    return BiometricResponse(
        decision=decision,
        score=score,
        details=details
    )

