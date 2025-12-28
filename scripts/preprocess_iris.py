
"""
Offline Preprocessing Script for Iris Dataset.
Perform Daugman normalization once and save to disk.
"""

import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.iris.config import TrainingConfig, get_data_dirs
from src.models.iris.normalization import robust_normalize
from src.models.iris.dataset import IrisDataset
from loguru import logger

def process_single_image(args):
    """
    Process a single image path.
    Args:
        args: tuple(img_path, output_root, relative_path)
    Returns:
        success (bool)
    """
    img_path, output_root, relative_path = args
    
    output_path = output_root / relative_path
    
    # Skip if already exists (resume capability)
    if output_path.exists():
        return True
        
    try:
        # Normalize (this does segmentation + unrolling)
        # robust_normalize returns RGB, cv2 expects BGR
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        norm_img_rgb = robust_normalize(str(img_path), target_shape=(512, 64), device=device)
        norm_img_bgr = cv2.cvtColor(norm_img_rgb, cv2.COLOR_RGB2BGR)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        cv2.imwrite(str(output_path), norm_img_bgr)
        return True
    except Exception as e:
        # logger.warning(f"Failed to process {img_path}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Offline Iris Preprocessing')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: process only 20 samples per dataset source')
    parser.add_argument('--force', action='store_true',
                        help='Force re-processing of existing files')
    args = parser.parse_args()
    
    logger.info("Starting Offline Iris Preprocessing...")
    if args.quick:
        logger.info("QUICK MODE: Processing limited samples for testing")
    if args.force:
        logger.info("FORCE MODE: Re-processing existing files")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dirs = get_data_dirs(project_root)
    
    # We need a common root to determine relative paths
    # This is tricky because data_dirs are scattered.
    # We will assume all are under 'data/raw' or similar.
    # Actually, IrisDataset scans them. We can just use the absolute path vs project root?
    
    # Let's verify where the data comes from
    # We'll replicate the structure relative to 'data/raw' if possible.
    raw_root = project_root / 'data' / 'raw'
    output_root = project_root / 'data' / 'processed_iris_512x64'
    
    if output_root.exists():
        logger.warning(f"Output directory {output_root} already exists. Merging/Skipping...")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        
    # Gather all files first
    logger.info("Scanning for files...")
    all_files = []
    
    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    
    # Track files by source for quick mode
    source_files = {}
    
    for d in data_dirs:
        if not d.exists():
            continue
        # Determine source name from path
        source_name = d.name
        if source_name not in source_files:
            source_files[source_name] = []
            
        for ext in extensions:
            for p in d.rglob(f'*{ext}'):
                # Determine relative path for mirroring
                try:
                    rel_path = p.relative_to(raw_root)
                except ValueError:
                    # If not under data/raw, try relative to project root
                    try:
                        rel_path = p.relative_to(project_root)
                    except ValueError:
                         # Flatten if completely outside (unlikely)
                         rel_path = Path(p.parent.name) / p.name
                
                source_files[source_name].append((p, output_root, rel_path))
    
    # Quick mode: sample from each source
    if args.quick:
        import random
        samples_per_source = 20
        for source, files in source_files.items():
            n_take = min(samples_per_source, len(files))
            sampled = random.sample(files, n_take) if len(files) > 0 else []
            all_files.extend(sampled)
            logger.info(f"  - {source}: {n_take} of {len(files)} files")
    else:
        for files in source_files.values():
            all_files.extend(files)
    
    # Filter out existing files unless --force
    if not args.force:
        original_count = len(all_files)
        all_files = [(p, o, r) for p, o, r in all_files if not (o / r).exists()]
        skipped = original_count - len(all_files)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files (use --force to re-process)")
                
    logger.info(f"Processing {len(all_files)} images...")
    
    if len(all_files) == 0:
        logger.info("No files to process!")
        return
    
    # Multiprocessing
    # Limit to 4 cores to keep system responsive
    num_workers = min(4, max(1, cpu_count() - 2))
    logger.info(f"Using {num_workers} CPU workers (limited for system responsiveness).")
    
    success_count = 0
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image, all_files), total=len(all_files)))
    
    success_count = sum(results)
    logger.info(f"Processing Complete. Success: {success_count}/{len(all_files)} ({success_count/len(all_files)*100:.1f}%)")
    logger.info(f"Data saved to: {output_root}")

if __name__ == "__main__":
    main()

