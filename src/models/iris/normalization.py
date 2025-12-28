"""
Daugman's Rubber Sheet Normalization for Iris Recognition.

This module provides functions to:
1. Segment the iris and pupil using Hough Transforms.
2. Unroll the annular iris region into a rectangular polar image (Daugman Normalization).
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from loguru import logger

def segment_iris(image_bgr: np.ndarray):
    """
    Find pupil and iris circles using Hough Transform.
    
    Args:
        image_bgr: Input image in BGR format (H, W, 3).
        
    Returns:
        tuple: (inner_circle, outer_circle) or (None, None) if failed.
               Both are (x, y, r).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing to improve circle detection (contrast limits)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Blur to reduce noise
    blurred = cv2.medianBlur(enhanced, 5)
    
    h, w = gray.shape
    min_dist = h // 8
    
    # 1. Detect Pupil (Inner Circle)
    # Pupils are generally dark and smaller
    pupil_circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=min_dist,
        param1=50, # Gradient threshold (high)
        param2=30, # Accumulator threshold (lower = more sensitive)
        minRadius=h // 20, 
        maxRadius=h // 5
    )
    
    pupil = None
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        # Take the strongest circle (first one)
        pupil = pupil_circles[0, 0]
    # No fallback needed - try iris detection next

    if pupil is None:
        return None, None

    px, py, pr = pupil
    
    # 2. Detect Iris (Outer Circle) - Multi-Pass Approach
    # Try progressively looser parameters to maximize detection rate
    
    iris = None
    
    # Parameter sets: (param1, param2, min_radius_multiplier, max_radius_multiplier)
    # param1: Canny edge threshold, param2: accumulator threshold (lower = more sensitive)
    param_sets = [
        (50, 30, 1.5, 3.5),   # Pass 1: Standard parameters
        (40, 20, 1.3, 4.0),   # Pass 2: More sensitive, wider radius range
        (30, 15, 1.2, 4.5),   # Pass 3: Even more sensitive
    ]
    
    for pass_idx, (p1, p2, min_mult, max_mult) in enumerate(param_sets):
        iris_circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=min_dist, 
            param1=p1, 
            param2=p2, 
            minRadius=int(pr * min_mult), 
            maxRadius=min(int(pr * max_mult), int(h * 0.75))
        )
        
        if iris_circles is not None:
            iris_circles = np.uint16(np.around(iris_circles))
            
            # Find circle closest to pupil center
            best_dist = float('inf')
            best_candidate = None
            for c in iris_circles[0, :]:
                ix, iy, ir = map(float, c)
                px_f, py_f, pr_f = map(float, pupil)
                
                dist = np.sqrt((ix - px_f)**2 + (iy - py_f)**2)
                # Center offset shouldn't be huge (iris and pupil are concentric)
                if dist < pr_f * 0.8 and dist < best_dist:
                    best_dist = dist
                    best_candidate = np.array([ix, iy, ir], dtype=np.uint16)
            
            if best_candidate is not None:
                iris = best_candidate
                if pass_idx > 0:
                    logger.debug(f'Iris found on pass {pass_idx + 1} with params p1={p1}, p2={p2}')
                break
                
    if iris is None:
        # All passes failed - use adaptive geometric heuristic
        # Based on physiological ratios and image characteristics
        
        # Base ratio depends on pupil size (smaller pupils often have larger relative iris)
        if pr < 20:
            # Very small pupil - likely zoomed out or small eye
            ratio = 3.2
        elif pr < 35:
            # Medium pupil - standard ratio
            ratio = 2.8
        elif pr < 50:
            # Large pupil - iris is proportionally smaller
            ratio = 2.5
        else:
            # Very large pupil (dilated or close-up) - conservative ratio
            ratio = 2.3
        
        # Adjust for image size (larger images tend to have more visible sclera)
        if h > 400:
            ratio += 0.2  # Slightly larger iris estimate for high-res images
        elif h < 200:
            ratio -= 0.2  # Smaller iris estimate for low-res images
        
        iris_radius = int(pr * ratio)
        
        # Sanity check: iris shouldn't exceed image bounds
        max_radius = min(px, py, w - px, h - py) - 5
        iris_radius = min(iris_radius, max(int(max_radius), int(pr * 1.5)))
        
        logger.warning(f'Iris detection failed after {len(param_sets)} passes (pupil r={pr}), using adaptive heuristic (ratio={ratio:.1f})')
        iris = (px, py, iris_radius)
        
    return pupil, iris

def daugman_normalization(image: np.ndarray, inner_circle, outer_circle, height=64, width=512):
    """
    Unroll iris into a rectangular block.
    
    Args:
        image: Source image (H, W, C) or (H, W).
        inner_circle: (x, y, r) of pupil.
        outer_circle: (x, y, r) of iris boundary.
        height: Output height (radial resolution).
        width: Output width (angular resolution).
        
    Returns:
        np.ndarray: Normalized image (height, width, C).
    """
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
        
    # Create polar grid
    # theta: 0 to 2*pi (width steps)
    theta = np.linspace(0, 2 * np.pi, width)
    
    # r: 0 to 1 (height steps) -> mapped between inner and outer radius
    # We ignore the pupil center offset for simplicity in this standard implementation
    # (Rubber sheet model usually assumes linear interpolation between r_in and r_out at each theta)
    
    norm_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    xp, yp, rp = inner_circle
    xi, yi, ri = outer_circle
    
    # For optimization, we compute map_x and map_y for cv2.remap
    
    # Grid of (r, theta) coordinates
    # Rows are distinct radius levels (0..1), Cols are distinct theta levels (0..2pi)
    r_grid = np.linspace(0, 1, height).reshape(-1, 1) # (H, 1)
    theta_grid = np.linspace(0, 2 * np.pi, width).reshape(1, -1) # (1, W)
    
    # Calculate boundary coordinates at each theta
    # Pupil boundary (xp, yp, rp)
    xl = xp + rp * np.cos(theta_grid)
    yl = yp + rp * np.sin(theta_grid)
    
    # Iris boundary (xi, yi, ri)
    xr = xi + ri * np.cos(theta_grid)
    yr = yi + ri * np.sin(theta_grid)
    
    # Linearly interpolate radius based on r_grid
    # x = xl + (xr - xl) * r_grid
    map_x = xl + (xr - xl) * r_grid
    map_y = yl + (yr - yl) * r_grid
    
    # Convert to float32 for remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    
    # Check bounds (optional, generic remap handles BORDER_CONSTANT)
    
    norm_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return norm_image

def daugman_normalization_torch(image_tensor: torch.Tensor, inner_circle, outer_circle, height=64, width=512):
    """
    GPU-accelerated Daugman normalization using grid_sample.
    
    Args:
        image_tensor: (C, H, W) tensor on GPU.
        inner_circle: (x, y, r).
        outer_circle: (x, y, r).
        height: Output height.
        width: Output width.
        
    Returns:
        torch.Tensor: Normalized image (C, height, width).
    """
    device = image_tensor.device
    C, H, W = image_tensor.shape
    
    # Create grid
    # y is radius (0 to 1), x is theta (0 to 2pi)
    # Normalized coordinates for grid_sample are [-1, 1]
    
    # Theta: 0..2pi
    theta = torch.linspace(0, 2 * np.pi, width, device=device).unsqueeze(0).repeat(height, 1) # (H, W)
    
    # Radius: 0..1
    r_grid = torch.linspace(0, 1, height, device=device).unsqueeze(1).repeat(1, width) # (H, W)
    
    xp, yp, rp = inner_circle
    xi, yi, ri = outer_circle
    
    # Calculate sampling coordinates
    # Pupil boundary
    xl = xp + rp * torch.cos(theta)
    yl = yp + rp * torch.sin(theta)
    
    # Iris boundary
    xr = xi + ri * torch.cos(theta)
    yr = yi + ri * torch.sin(theta)
    
    # Interpolate
    map_x = xl + (xr - xl) * r_grid
    map_y = yl + (yr - yl) * r_grid
    
    # Normalize to [-1, 1] for grid_sample
    # formula: 2 * x / (W - 1) - 1
    grid_x = 2.0 * map_x / (W - 1) - 1.0
    grid_y = 2.0 * map_y / (H - 1) - 1.0
    
    # Stack -> (H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # Sample
    # image_tensor needs to be (N, C, H, W) -> (1, C, H, W)
    out = torch.nn.functional.grid_sample(
        image_tensor.unsqueeze(0), 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return out.squeeze(0) # (C, H, W)


def robust_normalize(image_path_or_array, target_shape=(512, 64), device='cpu'):
    """
    Wrapper that handles loading, segmentation, and normalization with fallback.
    
    Args:
        image_path_or_array: Path str or numpy array (BGR).
        target_shape: (Width, Height).
        device: 'cpu' or 'cuda' (or torch.device).
    
    Returns:
        np.ndarray: Processed image (H, W, C) RGB format.
    """
    width, height = target_shape
    
    if isinstance(image_path_or_array, (str, Path)):
        # Load image
        try:
            image = cv2.imread(str(image_path_or_array))
            if image is None:
                raise ValueError(f"Could not load {image_path_or_array}")
        except Exception:
            # Return blank if IO failure
            return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        image = image_path_or_array
        
    # Attempt Segmentation
    try:
        pupil, iris = segment_iris(image)
        
        if pupil is None or iris is None:
            raise ValueError("Segmentation failed")
            
        # Normalization
        if str(device) != 'cpu' and torch.cuda.is_available():
            # GPU Path
            # Convert to Tensor (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device)
            norm_tensor = daugman_normalization_torch(img_tensor, pupil, iris, height=height, width=width)
            
            # Back to CPU numpy for compatibility with rest of pipeline
            norm_img = norm_tensor.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # Daugman torch already returns (C, H, W) -> permute back to (H, W, C)
        else:
            # CPU Path
            norm_img = daugman_normalization(image, pupil, iris, height=height, width=width)
        
        # Check if result is empty/black (segmentation was way off)
        if norm_img.mean() < 5: 
             # Too dark, likely sampled outside image or pupil detected in black corner
             raise ValueError("Normalized image is empty")
             
        # Convert BGR to RGB for PyTorch (if originally BGR)
        # Note: cv2.imread is BGR. dataset is RGB. 
        # CAREFUL here. If input was numpy, what format is it?
        # segment_iris expects BGR. 
        # Let's assume input is BGR as per docstring.
        norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
        return norm_img
        
    except ValueError:
        # Fallback: Center crop and resize
        h, w = image.shape[:2]
        center_crop_size = min(h, w)
        cy, cx = h // 2, w // 2
        half = center_crop_size // 2
        crop = image[cy-half:cy+half, cx-half:cx+half]
        
        fallback_img = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        fallback_img = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
        return fallback_img

