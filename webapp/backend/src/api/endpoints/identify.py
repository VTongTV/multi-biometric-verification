
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, List
import torch

from src.api.schemas import IdentificationResponse
from src.api.dependencies import get_model, get_gallery_cache, load_image_tensor, face_transform, iris_transform, finger_transform
from models.ensemble.ensemble_model import BiometricEnsemble

router = APIRouter()

@router.post("/identify", response_model=IdentificationResponse)
async def identify(
    face_file: Optional[UploadFile] = File(None),
    iris_nir_file: Optional[UploadFile] = File(None),
    iris_vis_file: Optional[UploadFile] = File(None),
    fingerprint_file: Optional[UploadFile] = File(None),
    active_ids: Optional[str] = Form(None),
    model: BiometricEnsemble = Depends(get_model),
    gallery_cache: dict = Depends(get_gallery_cache)
):
    """
    Identify 1:N match against the entire gallery or a subset.
    """
    if not gallery_cache:
        raise HTTPException(status_code=500, detail="Gallery is empty or not loaded")

    # Parse active_ids if present
    valid_ids = None
    if active_ids:
        valid_ids = set(active_ids.split(','))

    # 1. Process Probe Data
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

    probe_inputs = {k: v for k, v in probe_inputs.items() if v is not None}

    if not probe_inputs:
        raise HTTPException(status_code=400, detail="No valid biometric data provided")

    # 2. Identification
    best_score = -1.0
    best_id = "Unknown"
    best_details = {}

    with torch.no_grad():
        probe_out = model(probe_inputs)

        for identity_id, gallery_out in gallery_cache.items():
            # Skip if subset is defined and ID is not in it
            if valid_ids and identity_id not in valid_ids:
                continue

            result = model.compute_score(probe_out, gallery_out)
            score = result['fused_score'].item()

            if score > best_score:
                best_score = score
                best_id = identity_id
                best_details = {k: v.item() for k, v in result['individual_scores'].items()}

    decision = best_id if best_score > 0.6 else "Unknown"

    return IdentificationResponse(
        match_id=decision,
        score=best_score,
        details=best_details
    )
