"""
Evaluation Sampling for Biometric Verification.

Provides class-aware stratified sampling to guarantee sufficient genuine pairs
for meaningful ROC curves and verification metrics. 

The naive approach of sampling N random embeddings fails when datasets have
many classes with few samples each (e.g., 3000 classes with ~2.5 samples/class).
This module ensures minimum genuine pairs by prioritizing multi-sample classes.
"""

import numpy as np
from collections import defaultdict
from loguru import logger


def compute_robust_verification_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_genuine_pairs: int = 2000,
    min_classes: int = 200,
    max_impostor_ratio: int = 10,
    max_total_pairs: int = 200000
) -> tuple:
    """
    Compute verification scores with class-aware sampling.
    
    Guarantees sufficient genuine pairs by:
    1. Grouping embeddings by class
    2. Prioritizing classes with 2+ samples (required for genuine pairs)
    3. Dynamically scaling sample size until min_genuine_pairs is achievable
    
    Args:
        embeddings: (N, D) normalized embedding vectors
        labels: (N,) class labels for each embedding
        min_genuine_pairs: Minimum genuine pairs to generate (default 2000 for stable AUC)
        min_classes: Minimum number of classes to include for robust AUC (default 200)
        max_impostor_ratio: Max ratio of impostor:genuine pairs (default 10)
        max_total_pairs: Cap on total pairs to prevent memory issues
        
    Returns:
        (genuine_scores, impostor_scores, stats_dict)
    """
    # Group embeddings by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
    
    # Filter to classes with 2+ samples (needed for genuine pairs)
    multi_sample_classes = {
        cls: indices for cls, indices in class_to_indices.items() 
        if len(indices) >= 2
    }
    
    if not multi_sample_classes:
        logger.warning("No classes with 2+ samples found! Using random sampling fallback.")
        return _fallback_random_sampling(embeddings, labels)
    
    # Calculate available genuine pairs per class: n*(n-1)/2
    class_pair_counts = {
        cls: len(indices) * (len(indices) - 1) // 2 
        for cls, indices in multi_sample_classes.items()
    }
    total_available_genuine = sum(class_pair_counts.values())
    
    logger.info(f"Classes with 2+ samples: {len(multi_sample_classes)}/{len(class_to_indices)}")
    logger.info(f"Total available genuine pairs: {total_available_genuine}")
    
    # Strategy: Select classes greedily until we have enough genuine pairs
    # Sort classes by sample count descending (more samples = more pairs)
    sorted_classes = sorted(
        multi_sample_classes.keys(),
        key=lambda c: len(multi_sample_classes[c]),
        reverse=True
    )
    
    selected_indices = []
    selected_genuine_pairs = 0
    
    for cls in sorted_classes:
        indices = multi_sample_classes[cls]
        n = len(indices)
        pairs_from_class = n * (n - 1) // 2
        
        selected_indices.extend(indices)
        selected_genuine_pairs += pairs_from_class
        
        # Stop when we have enough potential genuine pairs AND enough class diversity
        # We need at least 2 classes for impostor pairs, but 5+ is better for robust stats
        current_classes_count = 0 
        # Optimize: Avoid full set construction inside loop if possible, 
        # but since we append indices, we need to look up labels.
        # However, we are iterating through 'sorted_classes', which are unique classes.
        # So we can just count iterations.
        
        # Calculate how many classes we have processed so far
        # We are at the end of the loop body, so we have processed 'cls'
        classes_processed = sorted_classes.index(cls) + 1
        
        if selected_genuine_pairs >= min_genuine_pairs and classes_processed >= min_classes:
             break
    
    # If still not enough, we've selected all multi-sample classes
    if selected_genuine_pairs < min_genuine_pairs:
        logger.warning(
            f"Only {selected_genuine_pairs} genuine pairs available "
            f"(requested {min_genuine_pairs}). Proceeding with available data."
        )
    
    # Also add some samples from single-sample classes for impostor diversity
    single_sample_classes = [
        indices[0] for cls, indices in class_to_indices.items() 
        if len(indices) == 1
    ]
    
    # Add up to 100 single-sample embeddings for impostor diversity
    if single_sample_classes:
        n_single_to_add = min(100, len(single_sample_classes))
        np.random.shuffle(single_sample_classes)
        selected_indices.extend(single_sample_classes[:n_single_to_add])
    
    # Remove duplicates and convert to array
    selected_indices = list(set(selected_indices))
    selected_embeddings = embeddings[selected_indices]
    selected_labels = labels[selected_indices]
    
    logger.info(f"Selected {len(selected_indices)} embeddings from {len(set(selected_labels))} classes")
    
    # Group selected embeddings by class for efficient pair generation
    selected_class_to_indices = defaultdict(list)
    for idx, label in enumerate(selected_labels):
        selected_class_to_indices[label].append(idx)
    
    # STEP 1: Generate ALL genuine pairs first (critical for unbiased AUC)
    genuine_scores = []
    for cls_indices in selected_class_to_indices.values():
        n = len(cls_indices)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(selected_embeddings[cls_indices[i]], selected_embeddings[cls_indices[j]])
                genuine_scores.append(sim)
    
    logger.info(f"Generated {len(genuine_scores)} genuine pairs from {len(selected_class_to_indices)} classes")
    
    # STEP 2: Sample impostor pairs (cross-class) up to target ratio
    # Instead of exhaustive generation, sample random cross-class pairs
    target_impostors = min(len(genuine_scores) * max_impostor_ratio, max_total_pairs - len(genuine_scores))
    
    # Create list of all classes for sampling
    class_list = list(selected_class_to_indices.keys())
    impostor_scores = []
    
    # Efficient impostor sampling: pick random pairs from different classes
    attempts = 0
    max_attempts = target_impostors * 3  # Allow some failed attempts
    
    while len(impostor_scores) < target_impostors and attempts < max_attempts:
        # Pick two random classes
        cls1, cls2 = np.random.choice(len(class_list), size=2, replace=False)
        cls1, cls2 = class_list[cls1], class_list[cls2]
        
        # Pick one random embedding from each class
        idx1 = np.random.choice(selected_class_to_indices[cls1])
        idx2 = np.random.choice(selected_class_to_indices[cls2])
        
        sim = np.dot(selected_embeddings[idx1], selected_embeddings[idx2])
        impostor_scores.append(sim)
        attempts += 1
    
    logger.info(f"Sampled {len(impostor_scores)} impostor pairs (target: {target_impostors})")
    
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    
    # Filter NaN/Inf values
    genuine = genuine[np.isfinite(genuine)]
    impostor = impostor[np.isfinite(impostor)]
    
    stats = {
        'n_genuine': len(genuine),
        'n_impostor': len(impostor),
        'n_selected_embeddings': len(selected_indices),
        'n_multi_sample_classes': len(multi_sample_classes),
        'n_total_classes': len(class_to_indices),
        'ratio': len(impostor) / max(len(genuine), 1)
    }
    
    logger.info(f"Generated {stats['n_genuine']} genuine, {stats['n_impostor']} impostor pairs (ratio 1:{stats['ratio']:.1f})")
    
    return genuine, impostor, stats

def _fallback_random_sampling(embeddings: np.ndarray, labels: np.ndarray) -> tuple:
    """Fallback to random sampling when no multi-sample classes exist."""
    n_samples = min(500, len(embeddings))
    sample_indices = np.random.permutation(len(embeddings))[:n_samples]
    
    genuine_scores = []
    impostor_scores = []
    
    for i in range(len(sample_indices)):
        for j in range(i + 1, len(sample_indices)):
            sim = np.dot(embeddings[sample_indices[i]], embeddings[sample_indices[j]])
            if labels[sample_indices[i]] == labels[sample_indices[j]]:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)
    
    genuine = np.array(genuine_scores) if genuine_scores else np.array([0.5])
    impostor = np.array(impostor_scores) if impostor_scores else np.array([0.5])
    
    stats = {
        'n_genuine': len(genuine),
        'n_impostor': len(impostor),
        'n_selected_embeddings': n_samples,
        'fallback': True
    }
    
    return genuine, impostor, stats


def compute_full_verification_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_total_pairs: int = 500000
) -> tuple:
    """
    Compute verification scores on ALL available pairs (no sampling).
    
    This provides the true AUC on the complete validation set without
    any sampling variance. Use this for final evaluation on best model.
    
    Args:
        embeddings: (N, D) normalized embedding vectors
        labels: (N,) class labels for each embedding
        max_total_pairs: Cap on total pairs to prevent memory issues
        
    Returns:
        (genuine_scores, impostor_scores, stats_dict)
    """
    n_embeddings = len(embeddings)
    n_classes = len(np.unique(labels))
    
    logger.info(f"Computing full verification on {n_embeddings} embeddings from {n_classes} classes")
    
    genuine_scores = []
    impostor_scores = []
    total_computed = 0
    
    for i in range(n_embeddings):
        for j in range(i + 1, n_embeddings):
            sim = np.dot(embeddings[i], embeddings[j])
            
            if labels[i] == labels[j]:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)
            
            total_computed += 1
            
            if total_computed >= max_total_pairs:
                logger.warning(f"Reached max_total_pairs limit: {max_total_pairs}")
                break
        
        if total_computed >= max_total_pairs:
            break
    
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    
    genuine = genuine[np.isfinite(genuine)]
    impostor = impostor[np.isfinite(impostor)]
    
    stats = {
        'n_genuine': len(genuine),
        'n_impostor': len(impostor),
        'n_embeddings': n_embeddings,
        'n_classes': n_classes,
        'total_pairs_computed': total_computed,
        'ratio': len(impostor) / max(len(genuine), 1)
    }
    
    logger.info(f"Full evaluation: {stats['n_genuine']} genuine, {stats['n_impostor']} impostor pairs")
    
    return genuine, impostor, stats
