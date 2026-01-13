
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.models.fingerprint.config import TrainingConfig, get_data_dirs
from src.models.fingerprint.train import (
    FingerprintDataset, FingerprintEncoderWithAttention, filter_none_collate,
    get_fingerprint_transforms
)
from src.models.encoders.biometric_encoder import BiometricEncoder
from src.models.losses.arcface import ArcFaceLoss
from src.models.fingerprint.ridge_attention import RidgeAttention

def setup_simple_logging():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def sanity_check():
    setup_simple_logging()
    logger.info("="*60)
    logger.info("STARTING SANITY CHECK: Overfitting 20 Identities")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 1. Config
    config = TrainingConfig()
    config.use_gabor_preprocessing = True
    config.batch_size = 16
    image_size = 160
    
    # 2. Data Loading (Subset)
    data_dirs = get_data_dirs(PROJECT_ROOT)
    logger.info(f"Scanning data directories: {[d.name for d in data_dirs]}")
    
    # Load dataset WITHOUT preprocessing first to scan classes quickly
    scan_ds = FingerprintDataset(
        data_dirs, transform=None, sample_fraction=1.0, 
        use_preprocessing=False, use_fast_decoding=False
    )
    
    all_identities = sorted(list(scan_ds.class_to_idx.keys()))
    logger.info(f"Found {len(all_identities)} total identities.")
    
    if len(all_identities) < 20:
        logger.error("Not enough identities for sanity check!")
        return

    # SELECT 20 IDENTITIES
    target_identities = all_identities[:20]
    logger.info(f"Selected 20 identities for overfitting: {target_identities[0]} ... {target_identities[-1]}")
    
    # Create subset mapping
    subset_class_to_idx = {identity: i for i, identity in enumerate(target_identities)}
    
    # Load ACTUAL dataset (with Gabor)
    logger.info("Loading images with Gabor Preprocessing (this may take 10-20s)...")
    full_ds = FingerprintDataset(
        data_dirs, 
        get_fingerprint_transforms(image_size, True), # Train transforms
        sample_fraction=1.0, 
        use_preprocessing=True, # TEST GABOR
        image_size=image_size,
        class_to_idx=subset_class_to_idx # Filter to 20 classes
    )
    
    logger.info(f"Loaded {len(full_ds)} images for 20 identities.")
    
    # Split Train/Val
    # CRITICAL FIX: Random shuffle is REQUIRED for sanity check to ensure 
    # we have the same classes in Train and Val (Closed-Set Overfitting Test).
    # Without shuffle, we train on Persons 1-16 and validate on Persons 17-20,
    # resulting in 0% accuracy because the classes are disjoint.
    n = len(full_ds)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices) # <--- ADDED SHUFFLE
    
    split = int(0.8 * n)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_loader = DataLoader(
        Subset(full_ds, train_indices), 
        batch_size=16, shuffle=True, collate_fn=filter_none_collate
    )
    val_loader = DataLoader(
        Subset(full_ds, val_indices), 
        batch_size=16, shuffle=False, collate_fn=filter_none_collate
    )
    
    # 3. Model Setup
    logger.info("Initializing Model...")
    encoder = BiometricEncoder(
        embedding_dim=512, in_channels=3, pretrained=True, 
        backbone_type='efficientnet', use_attention=False
    ).to(device)
    
    ridge_attention = RidgeAttention(channels=1280, reduction=16).to(device)
    model = FingerprintEncoderWithAttention(encoder, ridge_attention).to(device)
    
    # ArcFace
    arcface = ArcFaceLoss(512, 20, scale=32.0, margin=0.5).to(device) # 20 classes
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': arcface.parameters()}
    ], lr=0.001, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    epochs = 30 # Increased to ensure convergence
    logger.info("Starting Training (30 Epochs)...")
    
    # Easify ArcFace for sanity check
    arcface.margin = 0.1
    arcface.scale = 16.0
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        arcface.train()
        total_loss = 0
        batches = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            if images.size(0) == 0: continue
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            embeddings = model(images, normalize=False)
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            # Train Acc
            with torch.no_grad():
                 cosine_logits = arcface.get_cosine_logits(embeddings)
                 _, pred = cosine_logits.max(1)
                 correct_train += pred.eq(labels).sum().item()
                 total_train += labels.size(0)
            
        avg_loss = total_loss / max(1, batches)
        train_acc = 100 * correct_train / max(1, total_train)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images.size(0) == 0: continue
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images, normalize=False)
                # Use cosine similarity for prediction
                logits = arcface.get_cosine_logits(embeddings)
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        
        val_acc = 100 * correct / max(1, total)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
    logger.info("="*60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("="*60)
    
    if history['val_acc'][-1] > 85.0:
        logger.info("✅ SUCCESS: Model successfully overfitted small subset (>85% Acc)")
        logger.info("This confirms the pipeline CAN learn identities.")
    elif history['val_acc'][-1] > 70.0:
        logger.info("⚠️ WARNING: Model learning is slow (>70% Acc). Might need LR tuning.")
    else:
        logger.info("❌ FAILURE: Model failed to learn even 20 identities (<70% Acc).")
        logger.info("Check preprocessing or data labels.")

if __name__ == "__main__":
    sanity_check()
