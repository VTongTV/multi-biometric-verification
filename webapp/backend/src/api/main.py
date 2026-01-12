from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn
import random
import hashlib
from PIL import Image
import io

from src.api.core.config import API_V1_STR, PROJECT_NAME, GALLERY_DIR, DATA_DIR
from src.api.endpoints import verify, identify
from src.api.dependencies import get_model, get_gallery_cache

# Cache for raw files to support random sampling from full dataset
RAW_FILES_CACHE = {}
# Set of hashes for all files currently in the gallery (identities)
GALLERY_HASHES = set()

def get_file_hash(path):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_gallery_hashes():
    """Index all files in the gallery to prevent random sampling overlaps."""
    global GALLERY_HASHES
    GALLERY_HASHES.clear()
    
    if not GALLERY_DIR.exists():
        return
        
    logger.info("Indexing gallery hashes to exclude enrolled data...")
    count = 0
    for user_dir in GALLERY_DIR.iterdir():
        if user_dir.is_dir():
            for file_path in user_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_hash = get_file_hash(file_path)
                        GALLERY_HASHES.add(file_hash)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to hash {file_path}: {e}")
                        
    logger.info(f"Indexed {count} gallery files (unique hashes: {len(GALLERY_HASHES)})")

def preload_raw_datasets():
    """Background task to index raw datasets for random sampling."""
    try:
        logger.info("Indexing raw datasets for random sampling...")
        
        # Base Raw Dir
        raw_dir = DATA_DIR / "raw"
        
        # Face (CelebA)
        # Note: Adjust path if structure is slightly different (e.g. redundant nesting)
        face_src = raw_dir / "celeba/img_align_celeba/img_align_celeba"
        if not face_src.exists(): # Try alternative path
             face_src = raw_dir / "celeba/img_align_celeba"
        
        if face_src.exists():
            RAW_FILES_CACHE['face'] = list(face_src.glob("*.jpg"))
        else:
            logger.warning(f"Face source not found at {face_src}")
            
        # Fingerprint (SOCOFing)
        finger_src = raw_dir / "socofing/SOCOFing/Real"
        if finger_src.exists():
            RAW_FILES_CACHE['fingerprint'] = list(finger_src.glob("*.BMP"))
        else:
            logger.warning(f"Fingerprint source not found at {finger_src}")
            
        # Iris NIR (CASIA)
        nir_src = raw_dir / "CASIA-Iris-Interval"
        if nir_src.exists():
            RAW_FILES_CACHE['irisNir'] = list(nir_src.glob("**/*.jpg"))
        else:
            logger.warning(f"Iris NIR source not found at {nir_src}")
            
        # Iris Vis (UBIRIS)
        vis_src = raw_dir / "ubiris"
        if vis_src.exists():
            RAW_FILES_CACHE['irisVis'] = list(vis_src.glob("**/*.tiff"))
        else:
            logger.warning(f"Iris Vis source not found at {vis_src}")
            
        logger.info(f"Indexed raw datasets: { {k: len(v) for k,v in RAW_FILES_CACHE.items()} }")
    except Exception as e:
        logger.error(f"Failed to index raw datasets: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and cache on startup
    logger.info("Startup: Loading models...")
    get_model()
    get_gallery_cache()
    load_gallery_hashes()
    preload_raw_datasets()
    yield
    # Cleanup (if needed)

app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS
origins = [
    "*", # Allow all origins for dev flexibility
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(verify.router, prefix=API_V1_STR, tags=["Verification"])
app.include_router(identify.router, prefix=API_V1_STR, tags=["Identification"])

# New Endpoint: List Identities
@app.get(f"{API_V1_STR}/identities")
def list_identities():
    if not GALLERY_DIR.exists():
        return []
    # List subdirectories in gallery dir
    identities = [d.name for d in GALLERY_DIR.iterdir() if d.is_dir()]
    return sorted(identities)

@app.get(f"{API_V1_STR}/random-sample")
def get_random_sample(modality: str):
    """
    Returns a random image from the full raw dataset for the specified modality.
    Ensures the returned image is NOT currently assigned to any enrolled user.
    """
    files = RAW_FILES_CACHE.get(modality)
    if not files:
        # Fallback if cache empty (e.g. dev mode restart or path issue)
        # Try to reload or just fail
        return Response(status_code=404, content="No raw data available for this modality")
    
    # Retry loop to find unassigned file
    max_retries = 20
    for _ in range(max_retries):
        file_path = random.choice(files)
        
        # Check if hash exists in gallery
        try:
            f_hash = get_file_hash(file_path)
            if f_hash in GALLERY_HASHES:
                # Collision with enrolled user, skip
                continue
        except Exception:
            # If read fails, skip
            continue
            
        # If we get here, it's unique (or hashing failed but we try anyway)
        
        # Handle TIFF conversion for browsers
        if file_path.suffix.lower() in ['.tiff', '.tif']:
            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    buf.seek(0)
                    return Response(content=buf.getvalue(), media_type="image/jpeg")
            except Exception as e:
                logger.error(f"Error converting TIFF {file_path}: {e}")
                return Response(status_code=500, content="Image conversion failed")
                
        return FileResponse(file_path)
    
    return Response(status_code=500, content="Failed to find unassigned random sample")

# Mount Static/Gallery for visualization
# Ensure we mount it to /static/gallery to match the legacy frontend expectation
if GALLERY_DIR.exists():
    app.mount("/static/gallery", StaticFiles(directory=str(GALLERY_DIR)), name="gallery")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
