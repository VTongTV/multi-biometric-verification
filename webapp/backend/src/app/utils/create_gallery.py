import os
import shutil
import random
from pathlib import Path
from loguru import logger
from PIL import Image

# webapp/backend/src/app/utils/create_gallery.py -> project root is parents[5]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_ROOT = PROJECT_ROOT / "data/raw"
GALLERY_DIR = PROJECT_ROOT / "webapp/backend/src/app/static/gallery"

def create_gallery(num_identities=20):
    """
    Creates a synthetic gallery of N identities.
    Each identity is a folder containing:
    - face.jpg
    - fingerprint.bmp
    - iris_nir.jpg
    - iris_vis.jpg
    """
    logger.info(f"Creating synthetic gallery for {num_identities} identities...")
    
    if GALLERY_DIR.exists():
        shutil.rmtree(GALLERY_DIR)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect Sources
    # Face: CelebA
    face_src = DATA_ROOT / "celeba" / "img_align_celeba" / "img_align_celeba"
    
    # Fingerprint: SOCOFing
    finger_src = DATA_ROOT / "socofing" / "SOCOFing" / "Real"
    
    # Iris NIR: CASIA-Iris-Interval
    nir_src = DATA_ROOT / "CASIA-Iris-Interval"
    
    # Iris Vis: UBIRIS
    vis_src = DATA_ROOT / "ubiris"
    
    # Get file lists
    # Face
    faces = sorted(list(face_src.glob("*.jpg")))[:num_identities]
    
    # Fingerprint
    fingers = sorted(list(finger_src.glob("*.BMP")))[:num_identities]
    
    # Iris NIR
    nirs = []
    for f in nir_src.glob("**/*.jpg"):
        nirs.append(f)
        if len(nirs) >= num_identities: break
        
    # Iris Vis (TIFFs)
    viss = []
    for f in vis_src.glob("**/*.tiff"):
        viss.append(f)
        if len(viss) >= num_identities: break
        
    # Check counts
    logger.info(f"Found: {len(faces)} faces, {len(fingers)} fingers, {len(nirs)} NIRs, {len(viss)} VISs")
    
    if len(faces) == 0: logger.error(f"No faces found at {face_src}")
    if len(fingers) == 0: logger.error(f"No fingers found at {finger_src}")
    if len(viss) == 0: logger.error(f"No VIS images found at {vis_src}")
    
    # Create Identities
    for i in range(num_identities):
        identity_id = f"user_{i+1:02d}"
        id_dir = GALLERY_DIR / identity_id
        id_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy Face
        if i < len(faces):
            shutil.copy(faces[i], id_dir / "face.jpg")
            
        # Copy Fingerprint
        if i < len(fingers):
            shutil.copy(fingers[i], id_dir / "fingerprint.bmp")
            
        # Copy NIR
        if i < len(nirs):
            shutil.copy(nirs[i], id_dir / "iris_nir.jpg")
            
        # CMS VIS (Convert TIFF to JPG)
        if i < len(viss):
            try:
                with Image.open(viss[i]) as img:
                    img.convert('RGB').save(id_dir / "iris_vis.jpg")
            except Exception as e:
                logger.error(f"Failed to convert {viss[i]}: {e}")
            
        logger.info(f"Created {identity_id}")
        
    logger.info("Gallery creation complete.")

if __name__ == "__main__":
    create_gallery()
