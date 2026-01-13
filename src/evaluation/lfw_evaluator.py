"""
LFW Verification Evaluator.
Implements standard 10-fold cross-validation protocol for face verification.
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
from typing import List, Tuple, Dict
import pandas as pd

class LFWImageDataset(Dataset):
    """Simple dataset to load unique LFW images for embedding extraction."""
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, str(path)

class LFWEvaluator:
    """
    Evaluator for LFW verification protocol.
    Parses pairs.csv, computes embeddings, and calculates 10-fold CV accuracy.
    """
    
    def __init__(self, data_dir: Path, pairs_path: Path, image_size: int = 224, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.pairs_path = Path(pairs_path)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Standard test transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Locate LFW directory
        self.lfw_dir = self._find_lfw_dir()
        
    def _find_lfw_dir(self) -> Path:
        """Find the actual LFW image directory."""
        candidates = [
            self.data_dir / 'lfw-deepfunneled' / 'lfw-deepfunneled',
            self.data_dir / 'lfw-deepfunneled',
            self.data_dir / 'lfw',
            self.data_dir / 'raw' / 'lfw-deepfunneled' / 'lfw-deepfunneled',
            self.data_dir / 'raw' / 'lfw-deepfunneled'
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        raise FileNotFoundError(f"Could not find LFW directory in {self.data_dir}")

    def _parse_pairs(self) -> Tuple[List[Tuple[str, str, int]], List[str]]:
        """
        Parse pairs.csv to get pair paths and unique image paths.
        Format in CSV:
        Matched: name,imagenum1,imagenum2,
        Mismatched: name1,imagenum1,name2,imagenum2
        """
        pairs = []
        unique_paths = set()
        
        # Read CSV simply to handle potentially malformed lines if any
        # But pandas is robust. Note the trailing comma in matched lines based on user input
        
        # We'll read line by line to handle the variable number of columns safely
        with open(self.pairs_path, 'r') as f:
            lines = f.readlines()[1:] # Skip header
            
        for line in lines:
            parts = line.strip().split(',')
            parts = [p for p in parts if p] # Remove empty strings from trailing commas
            
            if len(parts) == 3: # Matched pair: name, num1, num2
                name = parts[0]
                n1, n2 = int(parts[1]), int(parts[2])
                path1 = self._get_image_path(name, n1)
                path2 = self._get_image_path(name, n2)
                label = 1
                
            elif len(parts) == 4: # Mismatched: name1, num1, name2, num2
                name1, n1 = parts[0], int(parts[1])
                name2, n2 = parts[2], int(parts[3])
                path1 = self._get_image_path(name1, n1)
                path2 = self._get_image_path(name2, n2)
                label = 0
            else:
                continue # Skip invalid lines
                
            if path1.exists() and path2.exists():
                pairs.append((str(path1), str(path2), label))
                unique_paths.add(str(path1))
                unique_paths.add(str(path2))
            else:
                # logger.warning(f"Missing image for pair: {line.strip()}")
                pass
                
        return pairs, list(unique_paths)

    def _get_image_path(self, name: str, num: int) -> Path:
        # LFW deep funneled format: Name/Name_0001.jpg
        # Construct filename with zero padding to 4 digits
        filename = f"{name}_{num:04d}.jpg"
        return self.lfw_dir / name / filename

    @torch.inference_mode()
    def evaluate(self, model, device) -> Tuple[float, float]:
        """
        Run full evaluation.
        Returns: (mean_accuracy, std_accuracy)
        """
        logger.info("Parsing LFW pairs...")
        pairs, unique_paths = self._parse_pairs()
        logger.info(f"Found {len(pairs)} validation pairs and {len(unique_paths)} unique images")
        
        if len(pairs) == 0:
            logger.warning("No valid pairs found! Check dataset path.")
            return 0.0, 0.0

        # Create dataloader for unique images
        dataset = LFWImageDataset([Path(p) for p in unique_paths], transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        
        # Extract embeddings
        logger.info("Extracting embeddings...")
        model.eval()
        path_to_embedding = {}
        
        for images, paths in tqdm(dataloader, desc="LFW Embeddings"):
            images = images.to(device)
            # Support both BiometricEncoder (returns normalized) and others
            # Assuming model.forward returns normalized embeddings directly
            emb = model(images) 
            emb = F.normalize(emb, p=2, dim=1) # Ensure normalization
            emb = emb.cpu().numpy()
            
            for path, e in zip(paths, emb):
                path_to_embedding[path] = e
        
        # Calculate scores
        logger.info("Calculating similarity scores...")
        scores = []
        labels = []
        for p1, p2, label in pairs:
            if p1 in path_to_embedding and p2 in path_to_embedding:
                e1 = path_to_embedding[p1]
                e2 = path_to_embedding[p2]
                score = np.dot(e1, e2) # Cosine similarity since normalized
                scores.append(score)
                labels.append(label)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # 10-fold cross-validation
        accs = self._k_fold_evaluation(scores, labels, k=10)
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        
        logger.info(f"LFW Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
        return mean_acc, std_acc

    def _k_fold_evaluation(self, scores, labels, k=10):
        # Calculate optimal threshold accuracy for each fold
        n = len(scores)
        fold_size = n // k
        accs = []
        
        indices = np.arange(n)
        
        for i in range(k):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            # Find best threshold on training set
            best_thresh = self._find_best_threshold(scores[train_indices], labels[train_indices])
            
            # Evaluate on test set
            acc = self._calculate_accuracy(scores[test_indices], labels[test_indices], best_thresh)
            accs.append(acc)
            
        return accs

    def _find_best_threshold(self, scores, labels):
        thresholds = np.arange(-1.0, 1.0, 0.005)
        best_acc = 0
        best_thresh = 0
        
        for thresh in thresholds:
            acc = self._calculate_accuracy(scores, labels, thresh)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        return best_thresh

    def _calculate_accuracy(self, scores, labels, threshold):
        predictions = (scores > threshold).astype(int)
        correct = (predictions == labels).sum()
        return correct / len(labels)
