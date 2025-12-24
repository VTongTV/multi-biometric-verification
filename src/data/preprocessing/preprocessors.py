"""
Image preprocessing pipelines for each biometric modality.

Handles:
- Face: detection, alignment, cropping
- Periocular: eye region extraction from face
- Iris: segmentation and normalization
- Fingerprint: enhancement and ridge normalization
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger


class BasePreprocessor:
    """Base class for biometric image preprocessing."""
    
    def __init__(
        self,
        output_size: Tuple[int, int],
        output_channels: int = 3
    ):
        self.output_size = output_size
        self.output_channels = output_channels
    
    def __call__(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Process an image. Override in subclasses."""
        raise NotImplementedError
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized torch tensor."""
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        # HWC -> CHW
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Normalize to [0, 1]
        tensor = tensor / 255.0
        
        # ImageNet normalization for RGB
        if self.output_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
        else:
            # Simple normalization for grayscale
            tensor = (tensor - 0.5) / 0.5
        
        return tensor


class FacePreprocessor(BasePreprocessor):
    """
    Face image preprocessor.
    
    Uses OpenCV's Haar cascade for face detection and applies alignment.
    For production, consider using facenet-pytorch or MTCNN.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        min_face_size: int = 50,
        margin: float = 0.2
    ):
        super().__init__(output_size, output_channels=3)
        
        self.min_face_size = min_face_size
        self.margin = margin
        
        # Load Haar cascade
        cv2_data_path = cv2.data.haarcascades
        cascade_path = str(Path(cv2_data_path) / "haarcascade_frontalface_default.xml")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning("Failed to load face cascade. Using fallback detection.")
            self._use_fallback = True
        else:
            self._use_fallback = False
            logger.debug("Face cascade loaded successfully")
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image.
        
        Returns:
            Tuple of (x, y, w, h) or None if no face detected
        """
        if self._use_fallback or image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])
    
    def __call__(
        self,
        image: np.ndarray,
        detect: bool = True
    ) -> Optional[np.ndarray]:
        """
        Preprocess face image.
        
        Args:
            image: Input BGR image
            detect: Whether to detect and crop face
            
        Returns:
            Preprocessed face image or None if detection fails
        """
        if image is None:
            return None
        
        if detect:
            face_box = self.detect_face(image)
            
            if face_box is None:
                # Fallback: center crop
                h, w = image.shape[:2]
                size = min(h, w)
                start_x = (w - size) // 2
                start_y = (h - size) // 2
                image = image[start_y:start_y+size, start_x:start_x+size]
            else:
                x, y, w, h = face_box
                
                # Add margin
                margin_w = int(w * self.margin)
                margin_h = int(h * self.margin)
                
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h)
                x2 = min(image.shape[1], x + w + margin_w)
                y2 = min(image.shape[0], y + h + margin_h)
                
                image = image[y1:y2, x1:x2]
        
        # Resize
        image = cv2.resize(image, self.output_size)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image


class PeriocularPreprocessor(BasePreprocessor):
    """
    Periocular (eye region) image preprocessor.
    
    Extracts the periocular region from a face image or
    directly processes periocular crops.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        eye: str = 'both'  # 'left', 'right', or 'both'
    ):
        super().__init__(output_size, output_channels=3)
        
        self.eye = eye
        
        # Load eye cascade
        cv2_data_path = cv2.data.haarcascades
        cascade_path = str(Path(cv2_data_path) / "haarcascade_eye.xml")
        
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        if self.eye_cascade.empty():
            logger.warning("Failed to load eye cascade")
            self._use_fallback = True
        else:
            self._use_fallback = False
    
    def detect_eyes(
        self,
        image: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], ...]]:
        """Detect eyes in image."""
        if self._use_fallback:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        if len(eyes) < 2:
            return None
        
        # Sort by x coordinate (left to right)
        eyes = sorted(eyes, key=lambda e: e[0])
        return tuple([tuple(eyes[0]), tuple(eyes[1])])
    
    def __call__(
        self,
        image: np.ndarray,
        detect: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract periocular region.
        
        Args:
            image: Input face image
            detect: Whether to detect eye regions
            
        Returns:
            Periocular region image
        """
        if image is None:
            return None
        
        if detect:
            eyes = self.detect_eyes(image)
            
            if eyes is not None:
                # Extract region containing both eyes
                x1 = min(e[0] for e in eyes)
                y1 = min(e[1] for e in eyes)
                x2 = max(e[0] + e[2] for e in eyes)
                y2 = max(e[1] + e[3] for e in eyes)
                
                # Expand region
                h, w = image.shape[:2]
                pad_x = int((x2 - x1) * 0.3)
                pad_y = int((y2 - y1) * 0.5)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                image = image[y1:y2, x1:x2]
            else:
                # Fallback: top 40% of face
                h, w = image.shape[:2]
                image = image[:int(h * 0.4), :]
        
        # Resize
        image = cv2.resize(image, self.output_size)
        
        return image


class IrisPreprocessor(BasePreprocessor):
    """
    Visible-light iris image preprocessor.
    
    Handles RGB iris images from webcams (more challenging than NIR).
    Applies grayscale conversion and contrast enhancement.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        apply_clahe: bool = True
    ):
        super().__init__(output_size, output_channels=1)
        
        self.apply_clahe = apply_clahe
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def __call__(
        self,
        image: np.ndarray,
        enhance: bool = True
    ) -> Optional[np.ndarray]:
        """
        Preprocess iris image.
        
        Args:
            image: Input eye region image
            enhance: Whether to apply contrast enhancement
            
        Returns:
            Preprocessed grayscale iris image
        """
        if image is None:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        if enhance and self.apply_clahe:
            gray = self.clahe.apply(gray)
        
        # Resize
        gray = cv2.resize(gray, self.output_size)
        
        return gray


class FingerprintPreprocessor(BasePreprocessor):
    """
    Fingerprint image preprocessor.
    
    Applies ridge enhancement, normalization, and optional
    Gabor filtering for improved ridge clarity.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        enhance_ridges: bool = True,
        apply_gabor: bool = False
    ):
        super().__init__(output_size, output_channels=1)
        
        self.enhance_ridges = enhance_ridges
        self.apply_gabor = apply_gabor
        
        # CLAHE for contrast
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to standard mean and variance."""
        mean = np.mean(image)
        std = np.std(image)
        
        if std < 1e-6:
            return image
        
        normalized = (image - mean) / std
        
        # Scale to 0-255
        normalized = ((normalized - normalized.min()) / 
                     (normalized.max() - normalized.min() + 1e-6) * 255)
        
        return normalized.astype(np.uint8)
    
    def enhance_ridge_pattern(self, image: np.ndarray) -> np.ndarray:
        """Enhance ridge patterns using morphological operations."""
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        
        # Morphological closing to connect broken ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Light Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def __call__(
        self,
        image: np.ndarray,
        enhance: bool = True
    ) -> Optional[np.ndarray]:
        """
        Preprocess fingerprint image.
        
        Args:
            image: Input fingerprint image
            enhance: Whether to apply ridge enhancement
            
        Returns:
            Preprocessed grayscale fingerprint image
        """
        if image is None:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize
        gray = self.normalize_image(gray)
        
        # Enhance ridges
        if enhance and self.enhance_ridges:
            gray = self.enhance_ridge_pattern(gray)
        
        # Resize
        gray = cv2.resize(gray, self.output_size)
        
        return gray


def create_preprocessor(
    modality: str,
    output_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> BasePreprocessor:
    """
    Factory function to create preprocessors.
    
    Args:
        modality: One of 'face', 'periocular', 'iris', 'fingerprint'
        output_size: Optional custom output size
        **kwargs: Additional preprocessor-specific arguments
        
    Returns:
        Preprocessor instance
    """
    modality = modality.lower()
    
    if modality == 'face':
        size = output_size or (224, 224)
        return FacePreprocessor(
            output_size=size,
            min_face_size=kwargs.get('min_face_size', 50),
            margin=kwargs.get('margin', 0.2)
        )
    elif modality == 'periocular':
        size = output_size or (112, 112)
        return PeriocularPreprocessor(
            output_size=size,
            eye=kwargs.get('eye', 'both')
        )
    elif modality == 'iris':
        size = output_size or (224, 224)
        return IrisPreprocessor(
            output_size=size,
            apply_clahe=kwargs.get('apply_clahe', True)
        )
    elif modality == 'fingerprint':
        size = output_size or (224, 224)
        return FingerprintPreprocessor(
            output_size=size,
            enhance_ridges=kwargs.get('enhance_ridges', True),
            apply_gabor=kwargs.get('apply_gabor', False)
        )
    else:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Choose from: face, periocular, iris, fingerprint"
        )
