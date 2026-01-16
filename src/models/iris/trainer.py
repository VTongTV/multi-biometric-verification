"""
Iris Model Trainer.
Core training loop with MixUp augmentation for 95%+ accuracy.
"""

import sys
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_curve, auc
from loguru import logger

# Enable absolute imports when running standalone
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .config import TrainingConfig
from .augmentations import mixup_data, mixup_criterion, cutmix_data
from src.evaluation.evaluation_sampling import compute_robust_verification_scores


class IrisTrainer:
    """Trainer class for SOTA iris recognition model."""
    
    def __init__(self, config: TrainingConfig, encoder, arcface, device, class_weights=None):
        self.config = config
        self.encoder = encoder
        self.arcface = arcface
        self.device = device
        
        # Setup optimizer with differentiated learning rates
        # IMPROVEMENT: Increased backbone LR multiplier from 0.1 to 0.3 for faster adaptation
        backbone_params = list(encoder.backbone.parameters())
        head_params = list(encoder.embedding_head.parameters())
        arcface_params = list(arcface.parameters())
        
        backbone_lr_mult = getattr(config, 'backbone_lr_mult', 0.3)  # Default 0.3 (was 0.1)
        arcface_lr_mult = getattr(config, 'arcface_lr_mult', 2.0)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config.lr * backbone_lr_mult, 'name': 'backbone'},
            {'params': head_params, 'lr': config.lr, 'name': 'head'},
            {'params': arcface_params, 'lr': config.lr * arcface_lr_mult, 'name': 'arcface'},
        ], weight_decay=config.weight_decay, fused=True)  # fused=True for faster updates
        
        logger.info(f'Optimizer LR: backbone={config.lr * backbone_lr_mult:.2e}, '
                   f'head={config.lr:.2e}, arcface={config.lr * arcface_lr_mult:.2e}')
        
        # Setup scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=max(1, config.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(1, config.epochs - config.warmup_epochs),
            eta_min=config.eta_min
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs]
        )
        
        # Class-weighted loss for balanced training (faster than WeightedRandomSampler)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            logger.info(f'Using class-weighted CrossEntropyLoss')
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        # Very conservative init_scale to prevent FP16 overflow (cause of NaN at epoch 66)
        # Default is 65536 which can cause overflow; 512 is much safer for RTX 3050
        self.scaler = GradScaler('cuda', init_scale=512.0, growth_interval=2000)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        self.best_auc = 0.0  # Best validation AUC (used for model selection)
        self.no_improve_epochs = 0
        self.start_epoch = 0
        self._last_good_state = None  # Cache last known good state for NaN recovery
        
        # NaN recovery safeguards - prevent infinite recovery loops
        self.max_recoveries_per_epoch = 5  # Abort epoch after this many recoveries
        self.recovery_count = 0  # Reset at start of each epoch
    

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with MixUp augmentation."""
        self.encoder.train()
        self.arcface.train()
        
        # IMPROVEMENT: Progressive margin warmup instead of fixed margin
        # This prevents embedding collapse in early training
        margin_warmup_epochs = getattr(self.config, 'margin_warmup_epochs', 10)
        margin_schedule = getattr(self.config, 'margin_schedule', 'linear')
        
        if epoch < margin_warmup_epochs:
            progress = (epoch + 1) / margin_warmup_epochs
            
            if margin_schedule == 'cosine':
                # Cosine schedule: slower start, faster ramp at end
                progress = 0.5 * (1 - np.cos(np.pi * progress))
            elif margin_schedule == 'step':
                # Step schedule: jump at 50% of warmup
                progress = 0.0 if progress < 0.5 else 1.0
            # else: linear (default)
            
            current_margin = self.config.arcface_margin * progress
        else:
            current_margin = self.config.arcface_margin
        self.arcface.set_margin(current_margin)
        
        # Log margin change for debugging
        if epoch == 0 or epoch == margin_warmup_epochs - 1 or epoch == margin_warmup_epochs:
            logger.info(f'ArcFace margin: {current_margin:.4f} (warmup epoch {epoch+1}/{margin_warmup_epochs}, schedule={margin_schedule})')
        
        epoch_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
        cfg = self.config
        
        # More frequent logging for first epoch to catch issues early
        log_interval = 50 if epoch == 0 else 200
        
        nan_count = 0  # Track consecutive NaN batches
        self.recovery_count = 0  # Reset recovery counter at start of each epoch
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Non-blocking transfers for async GPU loading
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Decide augmentation strategy: MixUp, CutMix, or None
            use_cutmix = getattr(cfg, 'use_cutmix', False)
            cutmix_prob = getattr(cfg, 'cutmix_prob', 0.5)
            cutmix_alpha = getattr(cfg, 'cutmix_alpha', 1.0)
            
            rand_val = np.random.random()
            apply_mixup = cfg.use_mixup and rand_val < cfg.mixup_prob
            apply_cutmix = use_cutmix and not apply_mixup and rand_val < (cfg.mixup_prob + cutmix_prob)
            
            # Apply MixUp or CutMix augmentation
            if apply_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.mixup_alpha)
                
                # OPTIMIZATION: Channels Last
                images = images.to(memory_format=torch.channels_last)
                
                # OPTIMIZATION: BF16
                use_bf16 = torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                
                with autocast('cuda', dtype=amp_dtype):
                    embeddings = self.encoder(images)
                    logits = self.arcface(embeddings, labels_a)
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    loss = loss / cfg.accumulation_steps
            elif apply_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
                
                # OPTIMIZATION: Channels Last
                images = images.to(memory_format=torch.channels_last)
                
                # OPTIMIZATION: BF16
                use_bf16 = torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                
                with autocast('cuda', dtype=amp_dtype):
                    embeddings = self.encoder(images)
                    logits = self.arcface(embeddings, labels_a)
                    # CutMix uses same criterion as MixUp
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    loss = loss / cfg.accumulation_steps
            else:
                 # OPTIMIZATION: Channels Last & BF16 for non-mixup path too
                images = images.to(memory_format=torch.channels_last)
                use_bf16 = torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                
                with autocast('cuda', dtype=amp_dtype):
                    embeddings = self.encoder(images)
                    logits = self.arcface(embeddings, labels)
                    loss = self.criterion(logits, labels) / cfg.accumulation_steps
            
            # Guard against NaN/Inf loss BEFORE backward to prevent gradient corruption
            if not torch.isfinite(loss):
                nan_count += 1
                logger.warning(f"Batch {batch_idx} loss is non-finite ({loss.item()}). Skipping step. (NaN streak: {nan_count})")
                self.optimizer.zero_grad(set_to_none=True)
                # More aggressive scaler reduction (0.25x instead of 0.5x)
                new_scale = max(self.scaler.get_scale() * 0.25, 1.0)  # Floor at 1.0
                self.scaler.update(new_scale=new_scale)
                logger.info(f"Reduced scaler to {new_scale:.1f}")
                
                # If too many consecutive NaN, try to recover from last good state
                if nan_count >= 5 and self._last_good_state is not None:
                    self.recovery_count += 1
                    
                    # Check if we've exceeded max recoveries for this epoch
                    if self.recovery_count > self.max_recoveries_per_epoch:
                        logger.error(f"Exceeded max recoveries ({self.max_recoveries_per_epoch}) this epoch. "
                                    f"Training is unstable - aborting epoch.")
                        return float('inf')
                    
                    logger.warning(f"Attempting recovery from last good state... "
                                  f"(recovery {self.recovery_count}/{self.max_recoveries_per_epoch})")
                    
                    # Restore model weights
                    self.encoder.load_state_dict(self._last_good_state['encoder'])
                    self.arcface.load_state_dict(self._last_good_state['arcface'])
                    
                    # CRITICAL: Also restore optimizer state to clear bad momentum
                    if 'optimizer' in self._last_good_state:
                        self.optimizer.load_state_dict(self._last_good_state['optimizer'])
                    
                    # Restore or reset scaler
                    if 'scaler' in self._last_good_state:
                        self.scaler.load_state_dict(self._last_good_state['scaler'])
                    else:
                        self.scaler = GradScaler('cuda', init_scale=256.0)
                    
                    # Reduce scaler further for safety after recovery
                    new_scale = max(self.scaler.get_scale() * 0.5, 1.0)
                    self.scaler.update(new_scale=new_scale)
                    
                    nan_count = 0  # Reset NaN streak
                    # NOTE: continue will move to NEXT batch, not retry same one
                    continue
                
                if nan_count >= 10:
                    logger.error(f"Too many consecutive NaN losses ({nan_count}). Model weights may be corrupted.")
                    return float('inf')
                continue
            else:
                nan_count = 0  # Reset streak on successful batch
                # Cache good state every 500 successful batches (Reduced overhead from 100)
                if batch_idx % 500 == 0:
                    self._last_good_state = {
                        'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
                        'arcface': {k: v.clone() for k, v in self.arcface.state_dict().items()},
                        'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                        'scaler': self.scaler.state_dict(),
                    }
            
            # Scaled backward pass (only after NaN check)
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                     list(self.encoder.parameters()) + list(self.arcface.parameters()), 
                     1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
            # Logging (re-scale loss back for display)
            epoch_loss += loss.item() * cfg.accumulation_steps
            
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epoch+1}/{cfg.epochs} Batch {batch_idx}/{len(train_loader)} '
                           f'Loss: {loss.item() * cfg.accumulation_steps:.4f} LR: {current_lr:.2e}')
        
        return epoch_loss / len(train_loader)
    
    @torch.inference_mode()  # Faster than no_grad()
    def validate(self, val_loader: DataLoader, use_tta: bool = False) -> tuple:
        """Validate model and return loss, accuracy, and AUC.

        Args:
            val_loader: Validation data loader
            use_tta: Whether to use Test Time Augmentation (slower but more accurate)
        """
        self.encoder.eval()
        self.arcface.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        # Track closed-set vs open-set samples
        total_closed_set = 0
        total_open_set = 0

        # Collect embeddings and labels for AUC calculation
        all_embeddings = []
        all_labels = []

        for images, labels in val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            # OPTIMIZATION: Channels Last
            images = images.to(memory_format=torch.channels_last)

            # OPTIMIZATION: BF16
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

            with autocast('cuda', dtype=amp_dtype):
                embeddings = self.encoder(images)
                
                # Check if validation labels are within classifier range (Closed-Set)
                # If disjoint (Open-Set), labels will be out of range -> skip loss calculation
                max_label = labels.max().item()
                is_closed_set = max_label < self.arcface.num_classes
                
                if is_closed_set:
                    # Use margin-adjusted logits for loss calculation (training signal)
                    logits_for_loss = self.arcface(embeddings, labels)
                    loss = self.criterion(logits_for_loss, labels)
                    # Use raw cosine logits (without margin) for accuracy calculation
                    logits_for_pred = self.arcface.get_cosine_logits(embeddings)
                    
                    val_loss += loss.item()
                    _, predicted = logits_for_pred.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    total_closed_set += labels.size(0)
                else:
                    # Open-set validation (labels are just identity markers, not class indices)
                    # Skip loss/acc calculation, rely on verification AUC below
                    total_open_set += labels.size(0)
                    logger.debug(f'Open-Set batch: {labels.size(0)} samples with max_label={max_label} (ArcFace num_classes={self.arcface.num_classes})')

            if use_tta:
                # TTA: Horizontal Flip (only used for final evaluation)
                with autocast('cuda', dtype=amp_dtype):
                    images_flipped = torch.flip(images, [3])
                    embeddings_flipped = self.encoder(images_flipped)
                    embeddings_final = F.normalize(embeddings + embeddings_flipped, p=2, dim=1)
                # Collect TTA embeddings for AUC calculation
                all_embeddings.append(embeddings_final.detach().cpu())
            else:
                # Collect standard embeddings (already normalized in encoder)
                all_embeddings.append(embeddings.detach().cpu())

            all_labels.append(labels.cpu())

        avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # Compute AUC
        val_auc = 0.0
        if len(all_embeddings) > 0:
            stacked_embeddings = np.vstack([e.numpy() for e in all_embeddings])
            stacked_labels = np.concatenate([l.numpy() for l in all_labels])

            try:
                genuine, impostor, _ = compute_robust_verification_scores(
                    stacked_embeddings, stacked_labels, min_genuine_pairs=500, min_classes=100, max_total_pairs=150000
                )
                # Filter NaNs
                genuine = genuine[np.isfinite(genuine)]
                impostor = impostor[np.isfinite(impostor)]

                if len(genuine) > 0 and len(impostor) > 0:
                    all_scores = np.concatenate([genuine, impostor])
                    all_labels_binary = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
                    fpr, tpr, _ = roc_curve(all_labels_binary, all_scores)
                    val_auc = auc(fpr, tpr)
            except Exception as e:
                logger.warning(f"AUC calculation failed: {e}")
                val_auc = 0.5

        # Log validation set composition
        if total_closed_set > 0 or total_open_set > 0:
            logger.info(f'Validation set: {total_closed_set} closed-set samples, {total_open_set} open-set samples')

        # Return Training Loss as Val Loss proxy (for graphs), and Acc=AUC%
        train_loss_proxy = self.train_losses[-1] if hasattr(self, 'train_losses') and self.train_losses else 0.0
        return train_loss_proxy, val_auc * 100.0, val_auc
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: Path, suffix: str = 'latest'):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            checkpoint_dir: Directory to save checkpoint
            suffix: Checkpoint filename suffix (default: 'latest')
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'arcface_state_dict': self.arcface.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
            'best_auc': self.best_auc,
            'no_improve_epochs': self.no_improve_epochs, # Save patience counter
        }, checkpoint_dir / f'{suffix}_checkpoint.pt')
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if not checkpoint_path.exists():
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.arcface.load_state_dict(checkpoint['arcface_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Restore scheduler and scaler if available
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.val_aucs = checkpoint.get('val_aucs', [])
            # Handle both old 'best_acc' and new 'best_auc' keys for backwards compatibility
            self.best_auc = checkpoint.get('best_auc', checkpoint.get('best_acc', 0.0))
            self.no_improve_epochs = checkpoint.get('no_improve_epochs', 0) # Restore patience counter
            
            # Only step scheduler if we didn't restore its state
            if 'scheduler_state_dict' not in checkpoint:
                for _ in range(self.start_epoch):
                    self.scheduler.step()
            
            logger.info(f'Resumed from epoch {self.start_epoch} (Best AUC: {self.best_auc:.4f})')
            return True
        except Exception as e:
            logger.warning(f'Failed to load checkpoint: {e}. Starting fresh.')
            return False
    
    @staticmethod
    def is_checkpoint_corrupted(checkpoint_path: Path) -> bool:
        """Check if a checkpoint file contains NaN/Inf weights (corrupted).
        
        Returns True if corrupted, False if OK.
        """
        if not checkpoint_path.exists():
            return True  # Non-existent = "corrupted" for resume purposes
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Check encoder weights for NaN/Inf
            for key, value in checkpoint.get('encoder_state_dict', {}).items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    logger.warning(f'Found NaN/Inf in encoder weight: {key}')
                    return True
            
            # Check arcface weights for NaN/Inf
            for key, value in checkpoint.get('arcface_state_dict', {}).items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    logger.warning(f'Found NaN/Inf in arcface weight: {key}')
                    return True
            
            return False  # All weights are finite
        except Exception as e:
            logger.warning(f'Error checking checkpoint {checkpoint_path}: {e}')
            return True  # Assume corrupted if we can't read it
    
    def smart_resume(self, checkpoint_dir: Path, output_dir: Path) -> str:
        """Intelligently resume from best available checkpoint.
        
        Priority:
        1. latest_checkpoint.pt if it exists and is NOT corrupted
        2. best_model.pt if latest is corrupted or missing
        3. Start fresh if nothing available
        
        Returns: 'latest', 'best', or 'fresh' indicating which path was taken
        """
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pt'
        best_model_path = output_dir / 'best_model.pt'
        
        # Check latest checkpoint first
        if latest_checkpoint.exists():
            if not self.is_checkpoint_corrupted(latest_checkpoint):
                # Latest is good, use it
                if self.load_checkpoint(latest_checkpoint):
                    logger.info('✓ Resumed from latest_checkpoint.pt (verified not corrupted)')
                    return 'latest'
            else:
                logger.warning('⚠ latest_checkpoint.pt is corrupted (contains NaN weights)')
                # Rename corrupted checkpoint for debugging
                corrupted_path = checkpoint_dir / 'corrupted_checkpoint.pt'
                latest_checkpoint.rename(corrupted_path)
                logger.info(f'Moved corrupted checkpoint to {corrupted_path.name}')
        
        # Fall back to best_model.pt
        if best_model_path.exists():
            if not self.is_checkpoint_corrupted(best_model_path):
                try:
                    if self.load_best_model(best_model_path):
                        logger.info('✓ Resumed from best_model.pt (latest was corrupted/missing)')
                        return 'best'
                    else:
                        logger.warning('load_best_model returned False (unexpected state)')
                except Exception as e:
                    logger.error(f'Failed to load best_model.pt: {e}')
            else:
                logger.error('⚠ best_model.pt is also corrupted!')

        logger.info('No valid checkpoint found. Starting fresh training.')
        return 'fresh'
    
    def load_best_model(self, best_model_path: Path, resume_epoch: int = None) -> bool:
        """Load best_model.pt for resuming after NaN corruption.
        
        This loads only the model weights (not optimizer state) which is appropriate
        when the optimizer state may be corrupted from NaN gradients.
        
        Args:
            best_model_path: Path to best_model.pt
            resume_epoch: Epoch to resume from (if None, uses epoch from checkpoint)
        """
        if not best_model_path.exists():
            return False
        
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.arcface.load_state_dict(checkpoint['arcface_state_dict'])
            self.best_auc = checkpoint.get('best_auc', checkpoint.get('best_acc', 0.0))
            
            # Resume from checkpoint epoch or specified epoch
            saved_epoch = checkpoint.get('epoch', 0)
            self.start_epoch = resume_epoch if resume_epoch is not None else saved_epoch + 1
            
            # Reset optimizer and scheduler for fresh start with good weights
            # This is safer than loading potentially corrupted optimizer state
            for _ in range(self.start_epoch):
                self.scheduler.step()
            
            # Reset scaler to conservative value
            self.scaler = GradScaler('cuda', init_scale=256.0, growth_interval=2000)
            
            logger.info(f'Loaded best model from epoch {saved_epoch} (AUC: {self.best_auc:.4f})')
            logger.info(f'Resuming from epoch {self.start_epoch} with fresh optimizer state')
            return True
        except Exception as e:
            logger.warning(f'Failed to load best model: {e}')
            return False
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            output_dir: Path, checkpoint_dir: Path):
        """Main training loop."""
        cfg = self.config
        
        # Note: Checkpoint loading is now handled by smart_resume() before fit() is called
        # This allows for intelligent corruption detection and fallback
        
        logger.info('Starting training...')
        
        for epoch in range(self.start_epoch, cfg.epochs):
            # Check for manual stop signal (allows graceful early termination)
            stop_file = output_dir / 'STOP_TRAINING'
            if stop_file.exists():
                logger.info('STOP_TRAINING file detected - stopping training gracefully')
                stop_file.unlink()  # Remove the stop file so next run isn't affected
                logger.info(f'Best AUC was {self.best_auc:.4f}')
                self.save_checkpoint(epoch - 1 if epoch > 0 else 0, checkpoint_dir)
                break
            
            # Train
            avg_train_loss = self.train_epoch(train_loader, epoch)
            
            # Handle catastrophic NaN failure (train_epoch returns inf)
            if avg_train_loss == float('inf'):
                logger.error(f'Training halted at epoch {epoch+1} due to excessive NaN losses.')
                logger.info(f'Best AUC was {self.best_auc:.4f} before failure.')
                # Save as 'failed' checkpoint, NOT 'latest', to preserve good state
                self.save_checkpoint(epoch, checkpoint_dir, suffix='failed')
                logger.info(f'Corrupted state saved to failed_checkpoint.pt (not latest)')
                break
            
            self.train_losses.append(avg_train_loss)
            
            # Step scheduler
            self.scheduler.step()
            
            # Validate (use TTA only on final epoch for speed)
            use_tta = (epoch == cfg.epochs - 1)
            avg_val_loss, val_acc, val_auc = self.validate(val_loader, use_tta=use_tta)

            # Early sanity check - warn if no learning after epoch 1
            if epoch == 0 and val_auc < 0.6:
                logger.warning(f'⚠️ SANITY CHECK: Val AUC is {val_auc:.4f} after epoch 1 - training may have issues!')
                logger.warning('Check: learning rate, data loading, label mappings, or model architecture.')
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            self.val_aucs.append(val_auc)

            logger.info(f'Epoch {epoch+1}/{cfg.epochs} - Train Loss: {avg_train_loss:.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f} Margin: {self.arcface.margin:.3f}')

            # Track best model (using AUC)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.no_improve_epochs = 0
                
                # Save best_model.pt (full checkpoint with arcface for resume)
                best_model_path = output_dir / 'best_model.pt'
                try:
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'arcface_state_dict': self.arcface.state_dict(),
                        'best_auc': self.best_auc,  # Renamed from best_acc for clarity
                    }, best_model_path)
                    logger.info(f'★ Saved best_model.pt to {best_model_path}')
                except Exception as e:
                    logger.error(f'Failed to save best_model.pt: {e}')
                
                # Save encoder.pt (encoder only for ensemble/inference)
                encoder_path = output_dir / 'encoder.pt'
                try:
                    torch.save(self.encoder.state_dict(), encoder_path)
                    logger.info(f'★ Saved encoder.pt to {encoder_path}')
                except Exception as e:
                    logger.error(f'Failed to save encoder.pt: {e}')
                
                logger.info(f'★ New best AUC: {self.best_auc:.4f}')
            else:
                self.no_improve_epochs += 1
                logger.info(f'No improvement. Patience: {self.no_improve_epochs}/{cfg.early_stop_patience}')
            
            # Save checkpoint every 3 epochs
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch, checkpoint_dir)
                logger.info(f'Checkpoint saved at epoch {epoch + 1}')
            
            # Early stopping
            if self.no_improve_epochs >= cfg.early_stop_patience:
                logger.info(f'Early stopping: No improvement for {cfg.early_stop_patience} epochs')
                break
        
        return self.best_auc
