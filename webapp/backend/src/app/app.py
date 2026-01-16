
import os
import torch
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from loguru import logger
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble.ensemble_model import BiometricEnsemble

app = Flask(__name__)
model = None
device = 'cpu'

# --- Configuration ---
GALLERY_DIR = project_root / "src/app/static/gallery"
MODEL_PATH = project_root / "outputs/ensemble/ensemble_model.pt"

# --- Segmentation Utils ---
_iris_segmenter = None

def get_iris_segmenter():
    """Lazy load Worldcoin iris segmentation model."""
    global _iris_segmenter
    if _iris_segmenter is None:
        try:
            from src.models.iris.segmentation.worldcoin_wrapper import WorldcoinIrisSegmenter
            _iris_segmenter = WorldcoinIrisSegmenter()
            logger.info("Worldcoin iris segmentation model loaded successfully")
        except ImportError as e:
            logger.warning(f"onnxruntime not available, segmentation disabled: {e}")
            _iris_segmenter = None
        except Exception as e:
            logger.error(f"Failed to load Worldcoin segmenter: {e}")
            _iris_segmenter = None
    return _iris_segmenter

def _segment_iris_image(image_source, target_size=(224, 224)):
    """Segment iris image using Worldcoin ONNX model."""
    segmenter = get_iris_segmenter()
    if segmenter is None:
        return None

    try:
        if isinstance(image_source, (str, Path)):
            img = Image.open(str(image_source)).convert('RGB')
            img_array = np.array(img)
        else:
            if isinstance(image_source, Image.Image):
                img_array = np.array(image_source.convert('RGB'))
            else:
                img_array = image_source

        roi, quality = segmenter.extract_iris_roi_mask(img_array, target_size=target_size)
        if roi is not None and quality >= 0.3:
            return Image.fromarray(roi)
        else:
            logger.warning(f"Iris segmentation failed or low quality (q={quality})")
            return None
    except Exception as e:
        logger.error(f"Iris segmentation error: {e}")
        return None

# --- Preprocessing (Aligned with training) ---
# Face transforms - Match training (ImageNet normalization)
face_transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Iris transforms - Match training (224x224, ImageNet normalization)
iris_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fingerprint transforms - Match training (224x224, ImageNet normalization)
finger_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(source, transform, apply_segmentation=False):
    """Load image with optional iris segmentation."""
    if isinstance(source, (str, Path)):
        if not os.path.exists(source):
            return None
        try:
            img = Image.open(source).convert('RGB')
            if apply_segmentation:
                segmented = _segment_iris_image(img, target_size=(224, 224))
                if segmented is not None:
                    img = segmented
            return transform(img).unsqueeze(0).to(device)
        except Exception as e:
            logger.error(f"Error loading {source}: {e}")
            return None
    else:
        try:
            img = Image.open(source).convert('RGB')
            if apply_segmentation:
                segmented = _segment_iris_image(img, target_size=(224, 224))
                if segmented is not None:
                    img = segmented
            return transform(img).unsqueeze(0).to(device)
        except Exception as e:
            logger.error(f"Error loading uploaded file: {e}")
            return None

def load_model():
    global model
    try:
        face_path = project_root / 'weights' / 'R50_Glint360K.onnx'

        model = BiometricEnsemble(
            face_weights_path=str(face_path),
            device=device
        )

        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

# Global Cache
GALLERY_CACHE = {}

def load_gallery_cache():
    """Pre-compute embeddings for all identities in gallery."""
    global GALLERY_CACHE
    if not model:
        return

    logger.info("Caching gallery embeddings...")
    identities = sorted([d.name for d in GALLERY_DIR.iterdir() if d.is_dir()])

    for identity in identities:
        gallery_path = GALLERY_DIR / identity
        # Apply segmentation for iris images (required - models trained on segmented ROIs)
        inputs = {
            'face': load_image(gallery_path / 'face.jpg', face_transform),
            'iris_nir': load_image(gallery_path / 'iris_nir.jpg', iris_transform, apply_segmentation=True),
            'iris_vis': load_image(gallery_path / 'iris_vis.jpg', iris_transform, apply_segmentation=True),
            'fingerprint': load_image(gallery_path / 'fingerprint.bmp', finger_transform)
        }
        inputs = {k: v for k, v in inputs.items() if v is not None}
        if inputs:
            with torch.no_grad():
                GALLERY_CACHE[identity] = model(inputs)

    logger.info(f"Cached {len(GALLERY_CACHE)} identities.")

# Initialize model and cache
load_model()
load_gallery_cache()

@app.route('/')
def index():
    identities = sorted([d.name for d in GALLERY_DIR.iterdir() if d.is_dir()])
    return render_template('index.html', identities=identities)

@app.route('/verify', methods=['POST'])
def verify():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    claimed_id = request.form.get('identity')
    logger.info(f"Received verify request. Form: {request.form}, Files: {request.files.keys()}")
    logger.info(f"Claimed ID: '{claimed_id}'")

    if claimed_id in GALLERY_CACHE:
        gallery_out = GALLERY_CACHE[claimed_id]
    else:
        gallery_path = GALLERY_DIR / str(claimed_id)
        if not gallery_path.exists():
            return jsonify({'error': 'Identity not found in gallery'}), 404

        # Apply segmentation for iris images
        gallery_inputs = {
            'face': load_image(gallery_path / 'face.jpg', face_transform),
            'iris_nir': load_image(gallery_path / 'iris_nir.jpg', iris_transform, apply_segmentation=True),
            'iris_vis': load_image(gallery_path / 'iris_vis.jpg', iris_transform, apply_segmentation=True),
            'fingerprint': load_image(gallery_path / 'fingerprint.bmp', finger_transform)
        }
        gallery_inputs = {k: v for k, v in gallery_inputs.items() if v is not None}
        if not gallery_inputs:
            return jsonify({'error': 'No valid gallery data'}), 400
        with torch.no_grad():
            gallery_out = model(gallery_inputs)

    # Load Probe Data
    probe_inputs = {}
    if 'face_file' in request.files:
        probe_inputs['face'] = load_image(request.files['face_file'], face_transform)
    if 'iris_nir_file' in request.files:
        # Apply segmentation for iris
        probe_inputs['iris_nir'] = load_image(request.files['iris_nir_file'], iris_transform, apply_segmentation=True)
    if 'iris_vis_file' in request.files:
        # Apply segmentation for iris
        probe_inputs['iris_vis'] = load_image(request.files['iris_vis_file'], iris_transform, apply_segmentation=True)
    if 'fingerprint_file' in request.files:
        probe_inputs['fingerprint'] = load_image(request.files['fingerprint_file'], finger_transform)

    probe_inputs = {k: v for k, v in probe_inputs.items() if v is not None}

    if not probe_inputs:
        return jsonify({'error': 'No valid probe data provided'}), 400

    # Run Verification
    with torch.no_grad():
        probe_out = model(probe_inputs)
        result = model.compute_score(probe_out, gallery_out)

    score = result['fused_score'].item()
    details = {k: v.item() for k, v in result['individual_scores'].items()}

    decision = "AUTHORIZED" if score > 0.6 else "REJECTED"

    return jsonify({
        'decision': decision,
        'score': score,
        'details': details
    })

@app.route('/identify', methods=['POST'])
def identify():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    if not GALLERY_CACHE:
        load_gallery_cache()

    # Load Probe Data
    probe_inputs = {}
    if 'face_file' in request.files:
        probe_inputs['face'] = load_image(request.files['face_file'], face_transform)
    if 'iris_nir_file' in request.files:
        # Apply segmentation for iris
        probe_inputs['iris_nir'] = load_image(request.files['iris_nir_file'], iris_transform, apply_segmentation=True)
    if 'iris_vis_file' in request.files:
        # Apply segmentation for iris
        probe_inputs['iris_vis'] = load_image(request.files['iris_vis_file'], iris_transform, apply_segmentation=True)
    if 'fingerprint_file' in request.files:
        probe_inputs['fingerprint'] = load_image(request.files['fingerprint_file'], finger_transform)

    if not probe_inputs:
        return jsonify({'error': 'No valid biometric data provided'}), 400

    best_score = -1.0
    best_id = "Unknown"
    best_details = {}

    with torch.no_grad():
        probe_out = model(probe_inputs)

        for identity_id, gallery_out in GALLERY_CACHE.items():
            result = model.compute_score(probe_out, gallery_out)
            score = result['fused_score'].item()

            if score > best_score:
                best_score = score
                best_id = identity_id
                best_details = {k: v.item() for k, v in result['individual_scores'].items()}

    decision = best_id if best_score > 0.6 else "Unknown"

    return jsonify({
        'match_id': decision,
        'score': best_score,
        'details': best_details
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
