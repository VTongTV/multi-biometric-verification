"""
Face Model Trainer.

Core training loop with:
- AdaFace: Quality-adaptive margin loss (CVPR 2022)
- Multi-task attribute learning with CelebA attributes
- MixUp augmentation for improved accuracy
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

from src.models.face.config import TrainingConfig
from src.models.face.augmentations import mixup_data, mixup_criterion, cutmix_data
from src.models.face.adaface import AdaFaceLoss
from src.evaluation.evaluation_sampling import compute_robust_verification_scores


class AttributeHead(nn.Module):
    """Multi-task attribute prediction head."""
    
    def __init__(self, embedding_dim: int, num_attributes: int, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_attributes)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)


class FaceTrainer:
    """Trainer class for SOTA face recognition model with novel contributions."""
    
    def __init__(self, config: TrainingConfig, encoder, loss_module, device,
                 class_weights=None, num_attributes: int = 0):
        self.config = config
        self.encoder = encoder
        self.loss_module = loss_module  # Can be ArcFace or AdaFace
        self.device = device
        self.use_attributes = config.use_attributes and num_attributes > 0
        self.num_attributes = num_attributes
        
        # Create attribute prediction head if using multi-task learning
        self.attribute_head = None
        if self.use_attributes:
            self.attribute_head = AttributeHead(
                embedding_dim=config.embedding_dim,
                num_attributes=num_attributes
            ).to(device)
            logger.info(f'Attribute head created for {num_attributes} attributes')
        
        # Setup optimizer with differentiated learning rates
        # IMPROVEMENT: Use configurable LR multipliers from config
        backbone_params = list(encoder.backbone.parameters())
        head_params = list(encoder.embedding_head.parameters())
        loss_params = list(loss_module.parameters())
        
        backbone_lr_mult = getattr(config, 'backbone_lr_mult', 0.1)
        head_lr_mult = getattr(config, 'head_lr_mult', 1.0)
        loss_lr_mult = getattr(config, 'loss_lr_mult', 1.0)  # Changed from hardcoded 2.0
        
        param_groups = [
            {'params': backbone_params, 'lr': config.lr * backbone_lr_mult, 'name': 'backbone'},
            {'params': head_params, 'lr': config.lr * head_lr_mult, 'name': 'head'},
            {'params': loss_params, 'lr': config.lr * loss_lr_mult, 'name': 'loss_module'},
        ]
        
        logger.info(f'LR multipliers: backbone={backbone_lr_mult}, head={head_lr_mult}, loss={loss_lr_mult}')
        
        if self.attribute_head is not None:
            param_groups.append({
                'params': self.attribute_head.parameters(),
                'lr': config.lr,
                'name': 'attribute_head'
            })
        
        # Choose optimizer based on config (SGD is standard for face recognition)
        if getattr(config, 'use_sgd', False):
            self.optimizer = optim.SGD(
                param_groups, 
                momentum=getattr(config, 'sgd_momentum', 0.9),
                weight_decay=config.weight_decay,
                nesterov=True
            )
            logger.info(f'Using SGD optimizer (momentum={config.sgd_momentum}, nesterov=True)')
        else:
            self.optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay, fused=True)
            logger.info('Using AdamW optimizer')
        
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
        
        # Class-weighted loss for balanced training
        if class_weights is not None:
            class_weights = class_weights.to(device)
            logger.info('Using class-weighted CrossEntropyLoss')
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Attribute loss (BCE for multi-label classification)
        self.attribute_criterion = nn.BCEWithLogitsLoss()
        
        # Very conservative init_scale to prevent FP16 overflow (cause of NaN at epoch 66 in iris)
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
    
    def _compute_loss(self, embeddings, labels, attributes=None, use_mixup_labels=None):
        """Compute combined loss (identity + optional attributes)."""
        # Identity loss via AdaFace/ArcFace
        if use_mixup_labels is not None:
            labels_a, labels_b, lam = use_mixup_labels
            logits = self.loss_module(embeddings, labels_a)
            identity_loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
        else:
            logits = self.loss_module(embeddings, labels)
            identity_loss = self.criterion(logits, labels)
        
        # Optional attribute loss (multi-task learning)
        attr_loss = torch.tensor(0.0, device=self.device)
        if self.use_attributes and self.attribute_head is not None and attributes is not None:
            attr_logits = self.attribute_head(embeddings)
            attr_loss = self.attribute_criterion(attr_logits, attributes)
        
        # Combined loss
        total_loss = identity_loss + self.config.attribute_loss_weight * attr_loss
        
        return total_loss, identity_loss, attr_loss, logits
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with MixUp augmentation."""
        import time as _time  # Debug timing
        
        self.encoder.train()
        self.loss_module.train()
        if self.attribute_head is not None:
            self.attribute_head.train()
        
        # IMPROVEMENT: Progressive margin warmup instead of fixed margin
        # This prevents embedding collapse in early training
        margin_warmup_epochs = getattr(self.config, 'margin_warmup_epochs', 10)
        margin_schedule = getattr(self.config, 'margin_schedule', 'linear')
        target_margin = self.config.adaface_margin if self.config.use_adaface else self.config.arcface_margin
        
        if epoch < margin_warmup_epochs:
            progress = (epoch + 1) / margin_warmup_epochs
            
            if margin_schedule == 'cosine':
                # Cosine schedule: slower start, faster ramp at end
                progress = 0.5 * (1 - np.cos(np.pi * progress))
            elif margin_schedule == 'step':
                # Step schedule: jump at 50% of warmup
                progress = 0.0 if progress < 0.5 else 1.0
            # else: linear (default)
            
            current_margin = target_margin * progress
        else:
            current_margin = target_margin
        
        if hasattr(self.loss_module, 'set_margin'):
            self.loss_module.set_margin(current_margin)
        
        # Log margin change for debugging
        if epoch == 0 or epoch == margin_warmup_epochs - 1 or epoch == margin_warmup_epochs:
            logger.info(f'Margin: {current_margin:.4f} (warmup epoch {epoch+1}/{margin_warmup_epochs}, schedule={margin_schedule})')
        
        epoch_loss = 0.0
        epoch_identity_loss = 0.0
        epoch_attr_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        cfg = self.config
        
        # Log every 200 batches
        log_interval = 200
        nan_count = 0  # Track consecutive NaN batches
        self.recovery_count = 0  # Reset recovery counter at start of each epoch
        
        # DEBUG: Track timing for crash diagnosis
        epoch_start_time = _time.time()
        batch_times = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start = _time.time()
            
            # DEBUG: Log GPU state every 100 batches for crash diagnosis
            if batch_idx % 100 == 0:
                try:
                    mem_alloc = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    elapsed = _time.time() - epoch_start_time
                    avg_batch_time = sum(batch_times[-100:]) / max(1, len(batch_times[-100:]))
                    logger.debug(f"[GPU DEBUG] Batch {batch_idx} | Elapsed: {elapsed:.1f}s | "
                                f"Mem: {mem_alloc:.0f}/{mem_reserved:.0f}MB | "
                                f"Avg batch time: {avg_batch_time*1000:.1f}ms")
                except Exception as e:
                    logger.warning(f"[GPU DEBUG] Error getting stats: {e}")
            # Handle both (image, label) and (image, label, attributes) formats
            if len(batch) == 3:
                images, labels, attributes = batch
                attributes = attributes.to(self.device, non_blocking=True)
            else:
                images, labels = batch
                attributes = None
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long() # Restore: Move labels to GPU!

            # Apply MixUp or CutMix augmentation
            mixup_labels = None
            use_cutmix = getattr(cfg, 'use_cutmix', False)
            cutmix_prob = getattr(cfg, 'cutmix_prob', 0.3)
            cutmix_alpha = getattr(cfg, 'cutmix_alpha', 1.0)
            
            rand_val = np.random.random()
            apply_mixup = cfg.use_mixup and rand_val < cfg.mixup_prob
            apply_cutmix = use_cutmix and not apply_mixup and rand_val < (cfg.mixup_prob + cutmix_prob)
            
            if apply_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.mixup_alpha)
                mixup_labels = (labels_a, labels_b, lam)
            elif apply_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
                mixup_labels = (labels_a, labels_b, lam)
            
            # OPTIMIZATION: Channels Last for Tensor Cores
            images = images.to(memory_format=torch.channels_last)
            
            # OPTIMIZATION: Use BFloat16 (BF16) if available
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            with autocast('cuda', dtype=amp_dtype):
                embeddings = self.encoder(images)
                loss, id_loss, attr_loss, _ = self._compute_loss(
                    embeddings, labels, attributes, mixup_labels
                )
                loss = loss / cfg.accumulation_steps
            
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
                    self.loss_module.load_state_dict(self._last_good_state['loss_module'])
                    
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
                        'loss_module': {k: v.clone() for k, v in self.loss_module.state_dict().items()},
                        'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                        'scaler': self.scaler.state_dict(),
                    }
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=1.0)
                if self.attribute_head is not None:
                    torch.nn.utils.clip_grad_norm_(self.attribute_head.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * cfg.accumulation_steps
            epoch_identity_loss += id_loss.item()
            epoch_attr_loss += attr_loss.item() if attr_loss.item() > 0 else 0
            
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (f'Epoch {epoch+1}/{cfg.epochs} Batch {batch_idx}/{len(train_loader)} '
                          f'Loss: {loss.item() * cfg.accumulation_steps:.4f} LR: {current_lr:.2e}')
                if self.use_attributes:
                    log_msg += f' Attr Loss: {attr_loss.item():.4f}'
                logger.info(log_msg)
            
            # DEBUG: Track batch time for crash diagnosis
            batch_times.append(_time.time() - batch_start)
        
        return epoch_loss / len(train_loader)
    
    @torch.inference_mode()
    def validate(self, val_loader: DataLoader, use_tta: bool = False) -> tuple:
        """Validate model and return loss, accuracy, and AUC."""
        self.encoder.eval()
        self.loss_module.eval()
        if self.attribute_head is not None:
            self.attribute_head.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        # Collect embeddings and labels for AUC calculation
        all_embeddings = []
        all_labels = []

        debug_logged = False
        for batch in val_loader:
            if len(batch) == 3:
                images, labels, attributes = batch
                attributes = attributes.to(self.device, non_blocking=True)
            else:
                images, labels = batch
                attributes = None

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()  # Ensure long dtype

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
                is_closed_set = max_label < self.loss_module.num_classes
                
                if is_closed_set:
                    loss, _, _, _ = self._compute_loss(embeddings, labels, attributes)
                    val_loss += loss.item()
                    
                    # Use raw cosine logits (without margin) for accuracy calculation
                    logits_for_pred = self.loss_module.get_cosine_logits(embeddings)
                    _, predicted = logits_for_pred.max(1)
                    
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Debug: Log info for first batch to diagnose accuracy
                    if not debug_logged:
                        num_classes = logits_for_pred.size(1)
                        label_min, label_max = labels.min().item(), labels.max().item()
                        pred_min, pred_max = predicted.min().item(), predicted.max().item()
                        batch_correct = predicted.eq(labels).sum().item()
                        logger.info(f'Val debug: num_classes={num_classes}, labels=[{label_min},{label_max}], '
                                    f'pred=[{pred_min},{pred_max}], batch_correct={batch_correct}/{labels.size(0)}')
                        debug_logged = True
                else:
                    # Open-Set validation: Skip loss/acc, rely on AUC
                    pass

            if use_tta:
                # TTA: Horizontal Flip
                with autocast('cuda', dtype=amp_dtype):
                    images_flipped = torch.flip(images, [3])
                    embeddings_flipped = self.encoder(images_flipped)
                    embeddings_final = F.normalize(embeddings + embeddings_flipped, p=2, dim=1)
                    # Collect TTA embeddings
                    all_embeddings.append(embeddings_final.detach().cpu())
            else:
                # Collect standard embeddings
                all_embeddings.append(embeddings.detach().cpu())

            all_labels.append(labels.cpu())

        avg_loss = val_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        # Compute AUC
        val_auc = 0.0
        if len(all_embeddings) > 0:
            stacked_embeddings = np.vstack([e.numpy() for e in all_embeddings])
            stacked_labels = np.concatenate([l.numpy() for l in all_labels])

            try:
                genuine, impostor, _ = compute_robust_verification_scores(
                    stacked_embeddings, stacked_labels, min_genuine_pairs=1500, min_classes=100, max_total_pairs=100000
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
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'loss_module_state_dict': self.loss_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
            'best_auc': self.best_auc,  # Renamed from best_acc for clarity
        }
        if self.attribute_head is not None:
            checkpoint['attribute_head_state_dict'] = self.attribute_head.state_dict()
        torch.save(checkpoint, checkpoint_dir / f'{suffix}_checkpoint.pt')
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if not checkpoint_path.exists():
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            # Handle both old arcface and new loss_module keys
            if 'loss_module_state_dict' in checkpoint:
                self.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
            elif 'arcface_state_dict' in checkpoint:
                self.loss_module.load_state_dict(checkpoint['arcface_state_dict'])
            
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
            
            if self.attribute_head is not None and 'attribute_head_state_dict' in checkpoint:
                self.attribute_head.load_state_dict(checkpoint['attribute_head_state_dict'])
            
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
    def get_checkpoint_epoch(checkpoint_path: Path) -> int:
        """Get epoch number from a checkpoint file safely.
        
        Returns -1 if file doesn't exist or is invalid.
        """
        if not checkpoint_path.exists():
            return -1
            
        try:
            # Load map_location='cpu' to avoid OOM just for checking epoch
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint.get('epoch', -1)
        except Exception as e:
            logger.error(f'Failed to read epoch from checkpoint {checkpoint_path}: {e}')
            return -1

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
            
            # Check loss_module weights for NaN/Inf
            loss_key = 'loss_module_state_dict' if 'loss_module_state_dict' in checkpoint else 'arcface_state_dict'
            for key, value in checkpoint.get(loss_key, {}).items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    logger.warning(f'Found NaN/Inf in loss_module weight: {key}')
                    return True
            
            return False  # All weights are finite
        except Exception as e:
            logger.warning(f'Error checking checkpoint {checkpoint_path}: {e}')
            return True  # Assume corrupted if we can't read it
    
    def smart_resume(self, checkpoint_dir: Path, output_dir: Path) -> str:
        """Intelligently resume from newer of latest or best checkpoint.
        
        Priority:
        1. Checkpoint with highest epoch number (latest vs best)
        2. Must be non-corrupted (no NaNs)
        
        Returns: 'latest', 'best', or 'fresh'
        """
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pt'
        best_model_path = output_dir / 'best_model.pt'
        
        candidates = []
        
        # 1. Evaluate latest_checkpoint.pt
        if latest_checkpoint.exists():
            if not self.is_checkpoint_corrupted(latest_checkpoint):
                epoch = self.get_checkpoint_epoch(latest_checkpoint)
                if epoch >= 0:
                    candidates.append(('latest', epoch, latest_checkpoint))
            else:
                logger.warning('⚠ latest_checkpoint.pt is corrupted. Moving it.')
                try:
                    latest_checkpoint.rename(checkpoint_dir / 'corrupted_checkpoint.pt')
                except Exception as e:
                    logger.error(f'Failed to rename corrupted checkpoint: {e}')
        
        # 2. Evaluate best_model.pt
        if best_model_path.exists():
            if not self.is_checkpoint_corrupted(best_model_path):
                epoch = self.get_checkpoint_epoch(best_model_path)
                if epoch >= 0:
                    candidates.append(('best', epoch, best_model_path))
            else:
                logger.error('⚠ best_model.pt is also corrupted!')

        # 3. Sort by epoch descending to prioritize newer state
        # If epochs are equal, stable sort preserves order (latest preferred if equal? No, actually latest has optimizer state, best might not)
        # best_model.pt usually lacks optimizer state in this codebase.
        # So if epochs are equal, we should prefer 'latest'.
        # Since we appended 'latest' first, sorting by epoch desc will keep 'latest' first if epochs match (stable sort).
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not candidates:
            logger.info('No valid checkpoint found. Starting fresh training.')
            return 'fresh'
            
        # 4. Attempt load
        source, epoch, path = candidates[0]
        
        if source == 'latest':
            if self.load_checkpoint(path):
                logger.info(f'✓ Resumed from latest_checkpoint.pt (Epoch {epoch})')
                return 'latest'
        else:
            # Source is 'best'
            if self.load_best_model(path):
                logger.info(f'✓ Resumed from best_model.pt (Epoch {epoch} is newer/same as latest)')
                return 'best'
        
        logger.warning('Failed to load selected candidate. Starting fresh.')
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
            
            # Handle both old arcface and new loss_module keys
            if 'loss_module_state_dict' in checkpoint:
                self.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
            elif 'arcface_state_dict' in checkpoint:
                self.loss_module.load_state_dict(checkpoint['arcface_state_dict'])
            
            self.best_auc = checkpoint.get('best_auc', checkpoint.get('best_acc', 0.0))
            
            # Resume from checkpoint epoch or specified epoch
            saved_epoch = checkpoint.get('epoch', 0)
            self.start_epoch = resume_epoch if resume_epoch is not None else saved_epoch + 1
            
            # Reset optimizer and scheduler for fresh start with good weights
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
        
        # Log training setup
        loss_type = 'AdaFace' if cfg.use_adaface else 'ArcFace'
        logger.info(f'Starting training with {loss_type} loss...')
        if self.use_attributes:
            logger.info(f'Multi-task learning enabled: {self.num_attributes} attributes')
        
        for epoch in range(self.start_epoch, cfg.epochs):
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
            
            # Validate with TTA for better model selection (enabled on all epochs)
            # TTA adds ~2x inference time but improves accuracy measurement
            use_tta = True  # Always use TTA for accurate validation
            avg_val_loss, val_acc, val_auc = self.validate(val_loader, use_tta=use_tta)

            # Early sanity check
            if epoch == 0 and val_auc < 0.6:
                logger.warning(f'⚠️ SANITY CHECK: Val AUC is {val_auc:.4f} after epoch 1')

            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            self.val_aucs.append(val_auc)

            logger.info(f'Epoch {epoch+1}/{cfg.epochs} - Train Loss: {avg_train_loss:.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')

            # Track best model (using AUC)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.no_improve_epochs = 0
                best_checkpoint = {
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'loss_module_state_dict': self.loss_module.state_dict(),
                    'best_auc': self.best_auc,  # Renamed from best_acc for clarity
                }
                if self.attribute_head is not None:
                    best_checkpoint['attribute_head_state_dict'] = self.attribute_head.state_dict()
                
                # Save with explicit error handling and path logging
                best_model_path = output_dir / 'best_model.pt'
                try:
                    torch.save(best_checkpoint, best_model_path)
                    logger.info(f'★ New best AUC: {self.best_auc:.4f} - Saved to {best_model_path}')
                except Exception as e:
                    logger.error(f'Failed to save best model to {best_model_path}: {e}')
            else:
                self.no_improve_epochs += 1
                logger.info(f'No improvement. Patience: {self.no_improve_epochs}/{cfg.early_stop_patience}')
            
            # Save checkpoint every 3 epochs
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch, checkpoint_dir)
                logger.info(f'Checkpoint saved at epoch {epoch+1}')
            
            # Early stopping
            if self.no_improve_epochs >= cfg.early_stop_patience:
                logger.info(f'Early stopping: No improvement for {cfg.early_stop_patience} epochs')
                break
        
        # Final checkpoint save at end of training
        self.save_checkpoint(epoch, checkpoint_dir)
        logger.info(f'Final checkpoint saved at epoch {epoch+1}')
        
        # Verify best_model.pt exists
        best_model_path = output_dir / 'best_model.pt'
        if best_model_path.exists():
            logger.info(f'Training complete. Best model saved at: {best_model_path}')
        else:
            logger.warning(f'Warning: best_model.pt not found at {best_model_path}')
        
        return self.best_auc

