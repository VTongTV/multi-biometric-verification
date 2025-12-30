"""
Incremental Iris Preprocessing Script using Worldcoin ONNX Model.

This script:
1. Scans all defined RAW datasets.
2. Checks if a segmented version already exists in `data/processed_onnx`.
3. If MISSING, runs ONNX segmentation (GPU/CPU) to generate it.
4. Skips already processed files to save time.

Usage:
    python -m src.models.iris.preprocess
    python -m src.models.iris.preprocess --domain nir_casia
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Add project root (src/models/iris -> project root)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.models.iris.config import get_data_dirs
from src.models.iris.segmentation.worldcoin_wrapper import WorldcoinIrisSegmenter
from src.models.iris.domain_config import get_domain_datasets, DOMAINS

def preprocess_incremental(domain_name=None):
    # Setup paths
    raw_dirs = get_data_dirs(project_root)
    output_base = project_root / 'data' / 'processed_onnx'
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Identify allowed datasets if domain is specified
    allowed_datasets = None
    if domain_name:
        if domain_name in DOMAINS:
            allowed_datasets = set(get_domain_datasets(domain_name))
            logger.info(f"Filtering for domain '{domain_name}': {allowed_datasets}")
        else:
             logger.warning(f"Unknown domain {domain_name}, processing ALL.")
    
    # 1. Identify files to process
    queue = []
    
    logger.info("Scanning datasets for missing processed files...")
    
    for data_dir in raw_dirs:
        data_dir_upper = str(data_dir).upper()
        if 'UBIRIS' in data_dir_upper: dataset_name = 'UBIRIS'
        elif 'SBVPI' in data_dir_upper: dataset_name = 'SBVPI'
        elif 'CASIA-IRIS-THOUSAND' in data_dir_upper: dataset_name = 'CASIA_T'
        elif 'CASIA-IRIS-INTERVAL' in data_dir_upper: dataset_name = 'CASIA_I'
        elif 'CASIA-IRIS-LAMP' in data_dir_upper: dataset_name = 'CASIA_L'
        elif 'MMU' in data_dir_upper: dataset_name = 'MMU'
        else: dataset_name = data_dir.name
        
        # FILTER: Skip if not in allowed list
        if allowed_datasets and dataset_name not in allowed_datasets:
            continue
            
        save_dir = output_base / dataset_name
        
        # Find raw images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(data_dir.rglob(ext)))
            
        for img_path in images:
            # Replicate relative structure
            rel_path = img_path.relative_to(data_dir)
            target_path = save_dir / rel_path
            target_path = target_path.with_suffix('.png') # Always png
            
            if not target_path.exists():
                queue.append((img_path, target_path))
    
    if len(queue) == 0:
        logger.info("All images are already processed! Nothing to do.")
        return

    logger.info(f"Found {len(queue)} new/missing images to process.")
    
    # 2. Initialize Segmenter (Lazy load only if needed)
    logger.info("Initializing Worldcoin ONNX Segmenter (GPU)...")
    try:
        segmenter = WorldcoinIrisSegmenter()
    except Exception as e:
        logger.error(f"Failed to initialize segmenter: {e}")
        return

    # 3. Process Queue
    success_count = 0
    fail_count = 0
    
    # Process in loop
    for img_path, target_path in tqdm(queue, desc="Segmenting"):
        try:
            # Ensure dir exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = cv2.imread(str(img_path))
            if img is None:
                fail_count += 1
                continue
                
            # Segment with retry
            roi, quality = segmenter.extract_iris_roi_mask(img, target_size=(224, 224))
            
            # Retry once if failed
            if roi is None or quality < 0.3:
                roi, quality = segmenter.extract_iris_roi_mask(img, target_size=(224, 224))
            
            # Reject if still low quality
            if roi is None or quality < 0.3:
                fail_count += 1
                continue
            
            # Save
            cv2.imwrite(str(target_path), roi)
            success_count += 1
            
        except Exception as e:
            # logger.warning(f"Error {img_path.name}: {e}")
            fail_count += 1

    logger.info("="*40)
    logger.info(f"Incremental Preprocessing Complete.")
    logger.info(f"New Files Created: {success_count}")
    logger.info(f"Failures: {fail_count}")
    logger.info("="*40)

if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, help='Process only specific domain datasets')
    args = parser.parse_args()
    
    preprocess_incremental(args.domain)
