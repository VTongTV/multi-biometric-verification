"""
Inference pipeline for biometric verification.

Provides a unified interface for:
- Loading trained models
- Preprocessing input images
- Extracting embeddings
- Computing similarity and verification
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoders import create_encoder, BaseEncoder
from models.fusion import EyeFusionModule, TriModalFusionNet
from data.preprocessing.preprocessors import (
    FacePreprocessor,
    PeriocularPreprocessor,
    IrisPreprocessor,
    FingerprintPreprocessor,
    create_preprocessor
)


class BiometricPredictor:
    """
    Unified predictor for biometric verification.
    
    Handles model loading, preprocessing, embedding extraction,
    and similarity computation for all modalities.
    
    Args:
        device: Device to run inference on ('cuda' or 'cpu')
        default_threshold: Default similarity threshold for verification
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        default_threshold: float = 0.5
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.default_threshold = default_threshold
        
        # Model containers
        self.encoders: Dict[str, nn.Module] = {}
        self.preprocessors: Dict[str, object] = {}
        self.fusion_modules: Dict[str, nn.Module] = {}
        
        # Thresholds per modality
        self.thresholds: Dict[str, float] = {
            'face': 0.5,
            'periocular': 0.45,
            'iris': 0.4,
            'fingerprint': 0.45,
            'fused': 0.55
        }
        
        logger.info(f"BiometricPredictor initialized on {self.device}")
    
    def load_encoder(
        self,
        modality: str,
        checkpoint_path: Optional[Path] = None,
        pretrained: bool = True
    ) -> None:
        """
        Load an encoder for a modality.
        
        Args:
            modality: 'face', 'periocular', 'iris', 'fingerprint'
            checkpoint_path: Path to trained checkpoint (optional)
            pretrained: Use ImageNet pretrained weights if no checkpoint
        """
        encoder = create_encoder(modality, pretrained=pretrained)
        
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    encoder.load_state_dict(checkpoint['model_state_dict'])
                else:
                    encoder.load_state_dict(checkpoint)
                
                logger.info(f"Loaded {modality} encoder from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        encoder.to(self.device)
        encoder.eval()
        
        self.encoders[modality] = encoder
        self.preprocessors[modality] = create_preprocessor(modality)
        
        logger.info(f"{modality.capitalize()} encoder loaded")
    
    def load_fusion_module(
        self,
        fusion_type: str,
        checkpoint_path: Optional[Path] = None
    ) -> None:
        """
        Load a fusion module.
        
        Args:
            fusion_type: 'eye' or 'trimodal'
            checkpoint_path: Path to trained checkpoint
        """
        if fusion_type == 'eye':
            module = EyeFusionModule()
        elif fusion_type == 'trimodal':
            module = TriModalFusionNet()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                module.load_state_dict(checkpoint['model_state_dict'])
            else:
                module.load_state_dict(checkpoint)
            logger.info(f"Loaded {fusion_type} fusion from {checkpoint_path}")
        
        module.to(self.device)
        module.eval()
        
        self.fusion_modules[fusion_type] = module
    
    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        modality: str
    ) -> Optional[torch.Tensor]:
        """
        Preprocess an image for a modality.
        
        Args:
            image: Input image (numpy, PIL, or path)
            modality: Target modality
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Get preprocessor
        if modality not in self.preprocessors:
            self.preprocessors[modality] = create_preprocessor(modality)
        
        preprocessor = self.preprocessors[modality]
        
        # Preprocess
        processed = preprocessor(image)
        if processed is None:
            return None
        
        # Convert to tensor
        tensor = preprocessor.to_tensor(processed)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def extract_embedding(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        modality: str
    ) -> Optional[np.ndarray]:
        """
        Extract embedding for an image.
        
        Args:
            image: Input image
            modality: 'face', 'periocular', 'iris', 'fingerprint'
            
        Returns:
            512-dim embedding as numpy array
        """
        if modality not in self.encoders:
            logger.error(f"Encoder for {modality} not loaded")
            return None
        
        # Preprocess
        tensor = self.preprocess(image, modality)
        if tensor is None:
            logger.warning(f"Preprocessing failed for {modality}")
            return None
        
        # Extract embedding
        encoder = self.encoders[modality]
        embedding = encoder(tensor)
        
        return embedding.cpu().numpy()[0]
    
    @torch.no_grad()
    def extract_fused_embedding(
        self,
        face_image: Optional[Union[np.ndarray, Image.Image, Path]] = None,
        periocular_image: Optional[Union[np.ndarray, Image.Image, Path]] = None,
        iris_image: Optional[Union[np.ndarray, Image.Image, Path]] = None,
        fingerprint_image: Optional[Union[np.ndarray, Image.Image, Path]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract fused embedding from multiple modalities.
        
        Args:
            face_image: Face image (optional)
            periocular_image: Periocular image (optional)
            iris_image: Iris image (optional)
            fingerprint_image: Fingerprint image (optional)
            
        Returns:
            Fused 512-dim embedding
        """
        embeddings = {}
        
        # Extract available embeddings
        if face_image is not None and 'face' in self.encoders:
            emb = self.extract_embedding(face_image, 'face')
            if emb is not None:
                embeddings['face'] = torch.tensor(emb).unsqueeze(0).to(self.device)
        
        if periocular_image is not None and 'periocular' in self.encoders:
            emb = self.extract_embedding(periocular_image, 'periocular')
            if emb is not None:
                embeddings['periocular'] = torch.tensor(emb).unsqueeze(0).to(self.device)
        
        if iris_image is not None and 'iris' in self.encoders:
            emb = self.extract_embedding(iris_image, 'iris')
            if emb is not None:
                embeddings['iris'] = torch.tensor(emb).unsqueeze(0).to(self.device)
        
        if fingerprint_image is not None and 'fingerprint' in self.encoders:
            emb = self.extract_embedding(fingerprint_image, 'fingerprint')
            if emb is not None:
                embeddings['fingerprint'] = torch.tensor(emb).unsqueeze(0).to(self.device)
        
        if not embeddings:
            logger.warning("No embeddings extracted")
            return None
        
        # If only one modality, return that embedding
        if len(embeddings) == 1:
            return list(embeddings.values())[0].cpu().numpy()[0]
        
        # Fuse embeddings
        if 'trimodal' in self.fusion_modules:
            fusion = self.fusion_modules['trimodal']
            
            # Get eye embedding if we have both periocular and iris
            eye_emb = None
            if 'eye' in self.fusion_modules and 'periocular' in embeddings and 'iris' in embeddings:
                eye_fusion = self.fusion_modules['eye']
                eye_emb = eye_fusion(embeddings['periocular'], embeddings['iris'])
            elif 'periocular' in embeddings:
                eye_emb = embeddings['periocular']
            elif 'iris' in embeddings:
                eye_emb = embeddings['iris']
            
            fused = fusion(
                face_emb=embeddings.get('face'),
                eye_emb=eye_emb,
                fingerprint_emb=embeddings.get('fingerprint')
            )
            
            return fused.cpu().numpy()[0]
        else:
            # Simple average fusion
            all_embs = torch.cat(list(embeddings.values()), dim=0)
            fused = all_embs.mean(dim=0, keepdim=True)
            fused = F.normalize(fused, p=2, dim=1)
            return fused.cpu().numpy()[0]
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score in [-1, 1]
        """
        # Normalize
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def verify(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        modality: str = 'fused',
        threshold: Optional[float] = None
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify if two embeddings belong to the same identity.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            modality: Modality for threshold selection
            threshold: Custom threshold (uses modality default if None)
            
        Returns:
            Dict with 'verified' (bool), 'similarity' (float), 'threshold' (float)
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        
        if threshold is None:
            threshold = self.thresholds.get(modality, self.default_threshold)
        
        verified = similarity >= threshold
        
        return {
            'verified': bool(verified),
            'similarity': similarity,
            'threshold': threshold
        }
    
    def verify_images(
        self,
        image1: Union[np.ndarray, Image.Image, Path],
        image2: Union[np.ndarray, Image.Image, Path],
        modality: str
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify if two images belong to the same identity.
        
        Args:
            image1: First image
            image2: Second image
            modality: Image modality
            
        Returns:
            Verification result
        """
        emb1 = self.extract_embedding(image1, modality)
        emb2 = self.extract_embedding(image2, modality)
        
        if emb1 is None or emb2 is None:
            return {
                'verified': False,
                'similarity': 0.0,
                'threshold': self.thresholds.get(modality, self.default_threshold),
                'error': 'Embedding extraction failed'
            }
        
        return self.verify(emb1, emb2, modality)
    
    def set_threshold(self, modality: str, threshold: float) -> None:
        """Set verification threshold for a modality."""
        self.thresholds[modality] = threshold
        logger.info(f"Threshold for {modality} set to {threshold}")


def load_predictor(
    checkpoint_dir: Optional[Path] = None,
    modalities: List[str] = ['face'],
    device: Optional[str] = None
) -> BiometricPredictor:
    """
    Convenience function to load a predictor with models.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        modalities: List of modalities to load
        device: Device to use
        
    Returns:
        Configured BiometricPredictor
    """
    predictor = BiometricPredictor(device=device)
    
    for modality in modalities:
        checkpoint_path = None
        if checkpoint_dir:
            # Look for best model checkpoint
            potential_paths = [
                checkpoint_dir / modality / 'best_model.pt',
                checkpoint_dir / f'{modality}_best.pt',
                checkpoint_dir / f'{modality}.pt'
            ]
            for path in potential_paths:
                if path.exists():
                    checkpoint_path = path
                    break
        
        predictor.load_encoder(
            modality,
            checkpoint_path=checkpoint_path,
            pretrained=checkpoint_path is None
        )
    
    return predictor
