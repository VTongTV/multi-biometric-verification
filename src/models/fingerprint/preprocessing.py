"""
Fingerprint Preprocessing Pipeline.

Implements preprocessing techniques:
- Gabor filter enhancement for ridge detection
- CLAHE for contrast enhancement
- Adaptive histogram equalization
- Ridge orientation estimation
- Quality assessment
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from PIL import Image
import math


class GaborFilterBank:
    """
    Gabor filter bank for fingerprint ridge enhancement.
    
    Creates filters at multiple orientations and frequencies to enhance
    ridge patterns regardless of local orientation.
    """
    
    def __init__(
        self, 
        orientations: int = 8,
        frequencies: Tuple[float, ...] = (0.1, 0.15, 0.2),
        kernel_size: int = 21
    ):
        """
        Initialize Gabor filter bank.
        
        Args:
            orientations: Number of orientations (typically 8 or 16)
            frequencies: Ridge frequencies to detect (in cycles/pixel)
            kernel_size: Size of Gabor kernel (must be odd)
        """
        self.orientations = orientations
        self.frequencies = frequencies
        self.kernel_size = kernel_size
        
        # Pre-compute filter bank
        self.filters = self._create_filter_bank()
        
    def _create_filter_bank(self) -> list:
        """Create bank of Gabor filters at different orientations and frequencies."""
        filters = []
        
        for freq in self.frequencies:
            for i in range(self.orientations):
                theta = i * np.pi / self.orientations
                kernel = self._create_gabor_kernel(freq, theta)
                filters.append(kernel)
        
        return filters
    
    def _create_gabor_kernel(self, frequency: float, theta: float) -> np.ndarray:
        """
        Create single Gabor filter kernel.
        
        Args:
            frequency: Ridge frequency
            theta: Orientation angle
            
        Returns:
            Gabor kernel (kernel_size x kernel_size)
        """
        sigma = 4.0  # Standard deviation of Gaussian envelope
        lambda_val = 1.0 / frequency  # Wavelength
        gamma = 0.5  # Aspect ratio
        psi = 0  # Phase offset
        
        # Create mesh grid
        k = self.kernel_size
        x = np.linspace(-(k // 2), k // 2, k)
        y = np.linspace(-(k // 2), k // 2, k)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor function
        gaussian = np.exp(-(X_theta**2 + gamma**2 * Y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * X_theta / lambda_val + psi)
        
        kernel = gaussian * sinusoid
        
        # Normalize to zero mean
        kernel = kernel - np.mean(kernel)
        
        return kernel.astype(np.float32)
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filter bank and return maximum response.
        
        Args:
            image: Grayscale fingerprint image (H, W)
            
        Returns:
            Enhanced image (H, W)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Store responses
        responses = []
        
        # Apply each filter
        for kernel in self.filters:
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(response)
        
        # Take maximum response across orientations (non-maximal suppression)
        responses = np.array(responses)
        enhanced = np.max(responses, axis=0)
        
        # Normalize to [0, 255]
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced


class FingerprintPreprocessor:
    """
    Comprehensive fingerprint preprocessing pipeline.
    
    Applies multiple enhancement techniques in optimal order:
    1. Normalization (mean=128, variance normalization)
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Gabor filtering (ridge enhancement)
    4. Optional: Binarization and thinning
    """
    
    def __init__(
        self,
        apply_clahe: bool = True,
        apply_gabor: bool = True,
        apply_normalization: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        gabor_orientations: int = 8,
        gabor_frequencies: Tuple[float, ...] = (0.1, 0.15, 0.2),
        target_mean: float = 128.0,
        target_variance: float = 100.0
    ):
        """
        Initialize preprocessor.
        
        Args:
            apply_clahe: Enable CLAHE
            apply_gabor: Enable Gabor filtering
            apply_normalization: Enable intensity normalization
            clahe_clip_limit: CLAHE clip limit (2.0 is standard)
            clahe_tile_size: CLAHE grid size (8x8 is standard)
            gabor_orientations: Number of Gabor filter orientations
            gabor_frequencies: Ridge frequencies to enhance
            target_mean: Target mean intensity
            target_variance: Target variance
        """
        self.apply_clahe = apply_clahe
        self.apply_gabor = apply_gabor
        self.apply_normalization = apply_normalization
        
        # Store CLAHE parameters (not the object itself for pickle compatibility)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        # Gabor filter bank
        if apply_gabor:
            self.gabor_bank = GaborFilterBank(
                orientations=gabor_orientations,
                frequencies=gabor_frequencies
            )
        else:
            self.gabor_bank = None
        
        self.target_mean = target_mean
        self.target_variance = target_variance
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to fixed mean and variance.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Normalized image
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Calculate statistics
        mean = np.mean(img_float)
        std = np.std(img_float)
        
        # Normalize
        if std > 1e-6:  # Avoid division by zero
            normalized = (img_float - mean) / std
            normalized = normalized * np.sqrt(self.target_variance) + self.target_mean
        else:
            normalized = img_float
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 255)
        
        return normalized.astype(np.uint8)
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input fingerprint image (can be RGB or grayscale)
            
        Returns:
            Enhanced grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                image = image[:, :, 0]  # Take first channel
        
        # Step 1: Intensity normalization
        if self.apply_normalization:
            image = self.normalize_intensity(image)
        
        # Step 2: CLAHE (contrast enhancement)
        # Create CLAHE on-the-fly for pickle compatibility
        if self.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
            image = clahe.apply(image)
        
        # Step 3: Gabor filtering (ridge enhancement)
        if self.apply_gabor and self.gabor_bank is not None:
            image = self.gabor_bank.apply(image)
        
        return image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Alias for enhance()."""
        return self.enhance(image)


class FingerprintQualityEstimator:
    """
    Estimate fingerprint image quality.
    
    Uses multiple quality metrics:
    - Local contrast
    - Ridge clarity
    - Edge strength
    """
    
    @staticmethod
    def estimate_quality(image: np.ndarray) -> float:
        """
        Estimate quality score [0, 1].
        
        Args:
            image: Grayscale fingerprint image
            
        Returns:
            Quality score (higher is better)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Metric 1: Local contrast (standard deviation in blocks)
        block_size = 16
        h, w = image.shape
        contrasts = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                contrasts.append(np.std(block))
        
        mean_contrast = np.mean(contrasts) if contrasts else 0
        contrast_score = min(mean_contrast / 50.0, 1.0)  # Normalize
        
        # Metric 2: Edge strength
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
        edge_score = min(np.mean(np.abs(edges)) / 50.0, 1.0)
        
        # Metric 3: Focus (Laplacian variance)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        focus_score = min(np.var(laplacian) / 500.0, 1.0)
        
        # Combine metrics (weighted average)
        quality = 0.4 * contrast_score + 0.3 * edge_score + 0.3 * focus_score
        
        return float(quality)


def preprocess_dataset_offline(
    input_dir: str,
    output_dir: str,
    preprocessor: Optional[FingerprintPreprocessor] = None,
    extensions: Tuple[str, ...] = ('.bmp', '.tif', '.png', '.jpg')
):
    """
    Preprocess entire dataset offline and save to disk.
    
    This allows preprocessing once and reusing enhanced images,
    significantly speeding up training.
    
    Args:
        input_dir: Input directory with raw fingerprints
        output_dir: Output directory for enhanced fingerprints
        preprocessor: Preprocessor instance (creates default if None)
        extensions: File extensions to process
    """
    from pathlib import Path
    import os
    
    if preprocessor is None:
        preprocessor = FingerprintPreprocessor()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to preprocess")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Enhance
        enhanced = preprocessor.enhance(img)
        
        # Save with same relative path
        rel_path = img_path.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(out_file), enhanced)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(image_files)} images")
    
    print(f"Preprocessing complete! Enhanced images saved to {output_dir}")
