"""
AT&T (ORL) Face Dataset Evaluator.
Implements all-pairs verification protocol for the AT&T dataset.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple
import itertools
import random

class ATNTImageDataset(Dataset):
    """Simple dataset to load AT&T images."""
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # AT&T images are PGM (grayscale), convert to RGB for model compatibility
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, str(path)

class ATNTEvaluator:
    """
    Evaluator for AT&T (ORL) dataset.
    Performs verification evaluation using all possible genuine pairs
    and a large subset of impostor pairs.
    """
    
    def __init__(self, data_dir: Path, image_size: int = 112, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Standard test transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.image_paths, self.labels = self._find_images()
        
    def _find_images(self) -> Tuple[List[Path], List[int]]:
        """Find all PGM images and parse labels from folder names (s1..s40)."""
        image_paths = []
        labels = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"AT&T dataset not found at {self.data_dir}")
            
        # AT&T structure: s1/1.pgm ... s40/10.pgm
        for subject_dir in self.data_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('s'):
                try:
                    label = int(subject_dir.name[1:])
                    for img_path in subject_dir.glob('*.pgm'):
                        image_paths.append(img_path)
                        labels.append(label)
                except ValueError:
                    continue
                    
        return image_paths, labels

    @torch.inference_mode()
    def evaluate(self, model, device) -> Tuple[float, float]:
        """
        Run evaluation.
        Returns: (accuracy, auc) - Accuracy is calculated at best threshold.
        """
        if len(self.image_paths) == 0:
            logger.warning("No AT&T images found!")
            return 0.0, 0.0

        logger.info(f"Found {len(self.image_paths)} images from {len(set(self.labels))} subjects.")

        # Create dataloader
        dataset = ATNTImageDataset(self.image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)
        
        # Extract embeddings
        logger.info("Extracting embeddings...")
        model.eval()
        embeddings = []
        
        for images, _ in tqdm(dataloader, desc="AT&T Embeddings"):
            images = images.to(device)
            # Mixed precision inference if available
            with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
                emb = model(images)
                emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
            
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(self.labels)
        
        # Generate pairs and scores
        logger.info("Computing pair scores...")
        genuine_scores = []
        impostor_scores = []
        
        # Efficient pair generation using matrix multiplication
        # Since dataset is small (400 images), 400x400 matrix is tiny.
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Extract scores
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j]
                if labels[i] == labels[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
                    
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        logger.info(f"Pairs: {len(genuine_scores)} genuine, {len(impostor_scores)} impostor")
        
        # Calculate Metrics
        # 1. Best Accuracy
        best_acc, best_thresh = self._calculate_best_accuracy(genuine_scores, impostor_scores)
        
        # 2. AUC
        from sklearn.metrics import roc_curve, auc
        y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        logger.info(f"AT&T Results - Acc: {best_acc*100:.2f}%, AUC: {roc_auc:.4f}")
        return best_acc * 100, roc_auc

    def _calculate_best_accuracy(self, genuine_scores, impostor_scores):
        """Find optimal threshold and accuracy."""
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        thresholds = np.arange(min_score, max_score, 0.01)
        best_acc = 0.0
        best_thresh = 0.0
        
        for thresh in thresholds:
            tp = np.sum(genuine_scores >= thresh)
            tn = np.sum(impostor_scores < thresh)
            acc = (tp + tn) / (len(genuine_scores) + len(impostor_scores))
            
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                
        return best_acc, best_thresh
