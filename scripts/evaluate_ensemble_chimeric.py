"""
Evaluate Biometric Ensemble using Chimeric (Virtual) Subjects.

Since datasets are disjoint (different people), we create 'Virtual Identities'
by randomly pairing Face ID X + Iris ID Y + Fingerprint ID Z.

Protocol:
1. Load validation sets for Face, Iris, and Fingerprint.
2. Extract embeddings for all validation samples.
3. Group embeddings by class (Identity).
4. Create N virtual subjects.
   - Assign (Face_ID, Iris_ID, Finger_ID) to each Virtual_ID.
5. Generate Pairs:
   - Genuine: Sample 2 instances from Face_ID, 2 from Iris_ID, 2 from Finger_ID.
   - Impostor: Sample instances from different Virtual_IDs.
6. Compute Ensemble Scores and Metrics (AUC, EER).
"""

import torch
import numpy as np
import random
import cv2
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset
from PIL import Image

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble.ensemble_model import BiometricEnsemble
from src.models.face.config import TrainingConfig as FaceConfig
from src.models.face.dataset import FaceDataset
from src.models.face.augmentations import get_face_transforms
from src.models.iris.config import TrainingConfig as IrisConfig
from src.models.iris.dataset import IrisDataset
from src.models.iris.augmentations import get_iris_transforms
from src.models.fingerprint.config import TrainingConfig as FingerConfig
from src.models.fingerprint.preprocessing import FingerprintPreprocessor

# --- Define FingerprintDataset locally since it's not exposed in a module ---
class FingerprintDataset(Dataset):
    """Fingerprint dataset with identity labels and preprocessing."""
    
    def __init__(self, root_dirs: list, transform=None, sample_fraction: float = 1.0, 
                 use_preprocessing: bool = True, use_fast_decoding: bool = True, 
                 image_size: int = 160, class_to_idx: dict = None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}
        self.use_preprocessing = use_preprocessing
        
        if use_preprocessing:
            self.preprocessor = FingerprintPreprocessor(
                apply_clahe=True, 
                apply_gabor=False, 
                apply_normalization=True
            )
        
        all_img_paths = []
        extensions = ['.bmp', '.BMP', '.tif', '.TIF', '.png', '.PNG', '.jpg', '.JPG']
        
        for root_dir in root_dirs:
            root_dir = Path(root_dir)
            if not root_dir.exists(): continue
            for ext in extensions:
                paths = list(root_dir.rglob(f'*{ext}'))
                all_img_paths.extend(paths)
        
        all_img_paths = sorted(list(set(all_img_paths))) 
        
        next_idx = len(self.class_to_idx)
        for img_path in all_img_paths:
            identity = self._parse_identity(img_path.name)
            if identity not in self.class_to_idx:
                if class_to_idx is None:
                    self.class_to_idx[identity] = next_idx
                    next_idx += 1
                else:
                    continue
            
            self.samples.append((str(img_path), self.class_to_idx[identity]))
        
        if sample_fraction < 1.0 and len(self.samples) > 0:
            n = max(1, int(len(self.samples) * sample_fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n]
        
        logger.info(f'Loaded {len(self.samples)} samples from {len(self.class_to_idx)} identities')
    
    def get_class_counts(self):
        from collections import Counter
        return Counter(label for _, label in self.samples)

    def _parse_identity(self, filename: str):
        """Parse identity from filename."""
        if '__' in filename:
            parts = filename.split('__')
            person_id = parts[0]
            info = parts[1]
            hand = 'Left' if 'Left' in info else 'Right'
            finger = 'unknown'
            for f in ['index', 'thumb', 'middle', 'ring', 'little']:
                if f in info:
                    finger = f
                    break
            return f"{person_id}_{hand}_{finger}"
        elif '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                 return f"{parts[0]}_{parts[1]}"
            elif len(parts) == 2:
                 return parts[0]
            return parts[0]
        return filename.split('.')[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.use_preprocessing:
                img_array = np.array(image)
                img_array = self.preprocessor.enhance(img_array)
                image = Image.fromarray(img_array)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data_and_extract(modality, data_dirs, config, transform, encoder, device, max_subjects=50):
    """Load data, extract embeddings, and group by identity."""
    logger.info(f"Processing {modality}...")
    
    # 1. Load Dataset
    if modality == 'face':
        if Path(data_dirs[0]).name == 'atnt':
             dataset = FaceDataset(Path(data_dirs[0]), transform=transform, min_samples=2)
        else:
             dataset = FaceDataset(data_dirs[0], transform=transform, min_samples=2)
    elif modality == 'iris':
        # IrisDataset takes a list of strings/paths
        dataset = IrisDataset(data_dirs, transform=transform)
        dataset.filter_rare_classes(min_samples=2)
    elif modality == 'fingerprint':
        dataset = FingerprintDataset(data_dirs, transform=transform)
        # Manually filter rare classes
        counts = dataset.get_class_counts()
        valid = {c for c, cnt in counts.items() if cnt >= 2}
        dataset.samples = [s for s in dataset.samples if s[1] in valid]
        
    if len(dataset) == 0:
        logger.warning(f"No data found for {modality}")
        return {}

    # Quick sampling for speed
    class_counts = Counter(label for _, label in dataset.samples) if hasattr(dataset, 'samples') else {}
    if not class_counts and hasattr(dataset, 'get_class_counts'):
        class_counts = dataset.get_class_counts()
        
    valid_classes = [c for c, count in class_counts.items() if count >= 2]
    if len(valid_classes) > max_subjects:
        valid_classes = random.sample(valid_classes, max_subjects)
    
    logger.info(f"Selected {len(valid_classes)} subjects for {modality}")
    
    # Extract Embeddings
    encoder.eval()
    embeddings_by_id = defaultdict(list)
    
    indices = [i for i, (p, l) in enumerate(dataset.samples) if l in valid_classes]
    
    # Limit total samples to prevent OOM/Timeouts
    if len(indices) > 500:
        indices = random.sample(indices, 500)
        
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Custom collate to handle None from FingerprintDataset errors
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch: return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    loader = torch.utils.data.DataLoader(subset, batch_size=32, num_workers=0, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {modality}"):
            if batch[0] is None: continue
            images, labels = batch
            images = images.to(device)
            
            # Use encoder directly
            if modality == 'fingerprint':
                emb = encoder(images, normalize=True)
            elif modality == 'face':
                emb = encoder(images)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            else: # iris
                emb = encoder(images)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                
            for e, l in zip(emb, labels):
                embeddings_by_id[l.item()].append(e.cpu())
                
    return embeddings_by_id

from collections import Counter

def evaluate_ensemble():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Ensemble
    logger.info("Loading Ensemble...")
    ensemble_path = Path("outputs/ensemble/ensemble_model.pt")
    if not ensemble_path.exists():
        logger.error("Ensemble model not found. Run build_ensemble.py first.")
        return

    model = BiometricEnsemble(device=device)
    # Allow missing keys (e.g. if we only trained some modalities)
    state = torch.load(ensemble_path)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Extract Features
    
    # --- FACE ---
    face_data = ["data/raw/atnt"] 
    if not Path(face_data[0]).exists():
        logger.warning("AT&T not found, using LFW/CelebA paths")
        face_data = ["data/raw/lfw-deepfunneled/lfw-deepfunneled"]
        
    face_embeddings = load_data_and_extract(
        'face', face_data, FaceConfig(), 
        get_face_transforms(112, False), 
        model.face_encoder, device
    )
    
    # --- IRIS ---
    iris_data = ["data/raw/casia/NIR"]
    if not Path(iris_data[0]).exists():
        iris_data = ["data/raw/casia-iris-thousand"]
        
    iris_embeddings = load_data_and_extract(
        'iris', iris_data, IrisConfig(),
        get_iris_transforms(224, False),
        model.iris_nir_encoder, device
    )
    
    # --- FINGERPRINT ---
    finger_data = ["data/raw/socofing/SOCOFing/Real"]
    finger_embeddings = load_data_and_extract(
        'fingerprint', finger_data, FingerConfig(),
        None, # FingerprintDataset handles transforms if needed, but we passed None. 
              # Actually FingerprintDataset expects a transform to tensor.
              # Let's import the transform function.
        model.fingerprint_encoder, device
    )
    
    # Check if we need to add ToTensor for fingerprint
    # FingerprintDataset returns PIL image if transform is None.
    # We need a basic transform.
    from torchvision import transforms
    finger_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Re-load fingerprint with transform
    finger_embeddings = load_data_and_extract(
        'fingerprint', finger_data, FingerConfig(),
        finger_transform,
        model.fingerprint_encoder, device
    )
    
    # 3. Create Chimeric Subjects
    n_subjects = min(len(face_embeddings), len(iris_embeddings), len(finger_embeddings))
    logger.info(f"Overlapping subjects count: {n_subjects}")
    
    if n_subjects < 2:
        logger.error("Insufficient data overlap.")
        return

    logger.info(f"Creating {n_subjects} Chimeric Identities...")
    
    face_ids = list(face_embeddings.keys())[:n_subjects]
    iris_ids = list(iris_embeddings.keys())[:n_subjects]
    finger_ids = list(finger_embeddings.keys())[:n_subjects]
    
    virtual_subjects = []
    for i in range(n_subjects):
        virtual_subjects.append({
            'face': face_embeddings[face_ids[i]],
            'iris': iris_embeddings[iris_ids[i]],
            'fingerprint': finger_embeddings[finger_ids[i]]
        })
        
    # 4. Generate Scores
    genuine_scores = []
    impostor_scores = []
    
    # Genuine
    logger.info("Computing Genuine Scores...")
    for subj in virtual_subjects:
        if len(subj['face']) < 2 or len(subj['iris']) < 2 or len(subj['fingerprint']) < 2:
            continue
            
        probe = {
            'embeddings': {
                'face': subj['face'][0].unsqueeze(0).to(device),
                'iris_nir': subj['iris'][0].unsqueeze(0).to(device),
                'fingerprint': subj['fingerprint'][0].unsqueeze(0).to(device)
            },
            'qualities': {'face': torch.tensor([1.0], device=device), 'iris_nir': torch.tensor([1.0], device=device), 'fingerprint': torch.tensor([1.0], device=device)}
        }
        gallery = {
            'embeddings': {
                'face': subj['face'][1].unsqueeze(0).to(device),
                'iris_nir': subj['iris'][1].unsqueeze(0).to(device),
                'fingerprint': subj['fingerprint'][1].unsqueeze(0).to(device)
            },
            'qualities': {'face': torch.tensor([1.0], device=device), 'iris_nir': torch.tensor([1.0], device=device), 'fingerprint': torch.tensor([1.0], device=device)}
        }
        
        score_out = model.compute_score(probe, gallery)
        genuine_scores.append(score_out['fused_score'].item())

    # Impostor
    logger.info("Computing Impostor Scores...")
    import itertools
    pairs = list(itertools.combinations(range(len(virtual_subjects)), 2))
    if len(pairs) > 5000:
        pairs = random.sample(pairs, 5000)
        
    for i, j in pairs:
        subj_i = virtual_subjects[i]
        subj_j = virtual_subjects[j]
        
        if len(subj_i['face']) < 1 or len(subj_j['face']) < 1: continue
        
        probe = {
            'embeddings': {
                'face': subj_i['face'][0].unsqueeze(0).to(device),
                'iris_nir': subj_i['iris'][0].unsqueeze(0).to(device),
                'fingerprint': subj_i['fingerprint'][0].unsqueeze(0).to(device)
            },
            'qualities': {'face': torch.tensor([1.0], device=device), 'iris_nir': torch.tensor([1.0], device=device), 'fingerprint': torch.tensor([1.0], device=device)}
        }
        gallery = {
            'embeddings': {
                'face': subj_j['face'][0].unsqueeze(0).to(device),
                'iris_nir': subj_j['iris'][0].unsqueeze(0).to(device),
                'fingerprint': subj_j['fingerprint'][0].unsqueeze(0).to(device)
            },
            'qualities': {'face': torch.tensor([1.0], device=device), 'iris_nir': torch.tensor([1.0], device=device), 'fingerprint': torch.tensor([1.0], device=device)}
        }
        
        score_out = model.compute_score(probe, gallery)
        impostor_scores.append(score_out['fused_score'].item())
        
    # 5. Metrics
    if not genuine_scores or not impostor_scores:
        logger.error("No scores computed.")
        return

    y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_scores = genuine_scores + impostor_scores
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    logger.info("=" * 40)
    logger.info("ENSEMBLE RESULTS (Chimeric Protocol)")
    logger.info("=" * 40)
    logger.info(f"Genuine Pairs: {len(genuine_scores)}")
    logger.info(f"Impostor Pairs: {len(impostor_scores)}")
    logger.info(f"AUC: {roc_auc:.5f}")
    
    return roc_auc

if __name__ == "__main__":
    evaluate_ensemble()
