"""
Test-Time Augmentation (TTA) for Fingerprint Recognition.

Applies multiple augmentations during inference and averages predictions
for improved accuracy and robustness.
"""

import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F


def tta_predict(model, images, num_augmentations=5, device='cuda'):
    """
    Apply Test-Time Augmentation for improved predictions.
    
    Args:
        model: Trained encoder model
        images: Batch of images (B, C, H, W)
        num_augmentations: Number of augmented predictions to average
        device: Device to run on
        
    Returns:
        Averaged embeddings (B, embedding_dim)
    """
    model.eval()
    
    # List to store predictions
    all_embeddings = []
    
    with torch.no_grad():
        # Original prediction
        embeddings = model(images, normalize=True)
        all_embeddings.append(embeddings)
        
        # Augmented predictions
        for _ in range(num_augmentations - 1):
            # Apply random augmentations
            aug_images = apply_tta_augmentations(images)
            aug_embeddings = model(aug_images, normalize=True)
            all_embeddings.append(aug_embeddings)
    
    # Average all predictions
    avg_embeddings = torch.stack(all_embeddings).mean(dim=0)
    
    # Re-normalize
    avg_embeddings = F.normalize(avg_embeddings, p=2, dim=1)
    
    return avg_embeddings


def apply_tta_augmentations(images):
    """
    Apply random augmentations for TTA.
    
    Uses lightweight augmentations that preserve fingerprint identity:
    - Small rotations
    - Minor horizontal/vertical shifts
    - Slight scaling
    
    Args:
        images: Batch of images (B, C, H, W)
        
    Returns:
        Augmented images (B, C, H, W)
    """
    import torchvision.transforms.functional as TF
    import random
    
    batch_size = images.shape[0]
    augmented = []
    
    for i in range(batch_size):
        img = images[i]
        
        # Random rotation (-5 to +5 degrees)
        angle = random.uniform(-5, 5)
        img = TF.rotate(img, angle)
        
        # Random affine (translation)
        translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        # Convert to pixel values
        h, w = img.shape[-2:]
        translate_px = (int(translate[0] * w), int(translate[1] * h))
        img = TF.affine(img, angle=0, translate=translate_px, scale=1.0, shear=0)
        
        # Random scale (0.95 to 1.05)
        scale = random.uniform(0.95, 1.05)
        img = TF.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)
        
        augmented.append(img)
    
    return torch.stack(augmented)


def evaluate_with_tta(model, dataloader, device='cuda', num_augmentations=5):
    """
    Evaluate model with Test-Time Augmentation.
    
    Args:
        model: Trained encoder
        dataloader: Validation/test DataLoader
        device: Device to run on
        num_augmentations: Number of TTA augmentations
        
    Returns:
        all_embeddings: Numpy array of embeddings (N, embedding_dim)
        all_labels: Numpy array of labels (N,)
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (handle varying lengths: images, labels, *extra)
            images = batch[0]
            labels = batch[1]
            
            if images.size(0) == 0:
                continue
            images = images.to(device)
            
            # Apply TTA
            embeddings = tta_predict(model, images, num_augmentations, device)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels
