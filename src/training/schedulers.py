"""
Learning Rate Schedulers for SOTA Training.

Implements warmup + cosine annealing and other advanced schedules.
"""

import math
from torch.optim import Optimizer


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate (default: 1e-6)
        base_lr: Base/peak learning rate (default: 1e-3)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        base_lr: float = 1e-3
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self) -> float:
        """Advance one epoch and return new learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class WarmupStepScheduler:
    """
    Learning rate scheduler with warmup and step decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        step_epochs: List of epochs to decay LR
        gamma: Decay factor (default: 0.1)
        base_lr: Base learning rate
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        step_epochs: list,
        gamma: float = 0.1,
        base_lr: float = 1e-3
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.step_epochs = step_epochs
        self.gamma = gamma
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self) -> float:
        """Advance one epoch and return new learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            lr = self.base_lr
            for step_epoch in self.step_epochs:
                if self.current_epoch >= step_epoch:
                    lr *= self.gamma
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class CosineWithPlateauScheduler:
    """
    Learning Rate Scheduler with Plateau Detection.
    
    Combines warmup + cosine annealing with automatic plateau detection
    that boosts learning rate when loss stagnates.
    
    Features:
    - Linear warmup phase
    - Cosine annealing with restarts on plateau
    - Automatic LR boost when plateau detected
    - Higher base LR (3e-3) for faster initial convergence
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
        base_lr: Base/peak learning rate (default: 3e-3 for robust training)
        patience: Epochs to wait before plateau detection (default: 3)
        plateau_threshold: Minimum loss decrease to not be considered plateau (default: 0.05)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        base_lr: float = 3e-3,  # Higher base LR for robust training
        patience: int = 3,
        plateau_threshold: float = 0.05
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.patience = patience
        self.plateau_threshold = plateau_threshold
        self.current_epoch = 0
        
        # Plateau detection state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.plateau_boosts = 0
        self.max_plateau_boosts = 3  # Max times to boost before giving up
        
        # Restart state for cosine annealing with warm restarts
        self.restart_epoch = 0
        self.current_max_lr = base_lr
    
    def step(self, epoch_loss: float = None) -> float:
        """
        Advance one epoch and return new learning rate.
        
        Args:
            epoch_loss: Average loss for the epoch (for plateau detection)
        """
        self.current_epoch += 1
        
        # Check for plateau if loss provided
        if epoch_loss is not None:
            self._check_plateau(epoch_loss)
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup with higher target
            lr = self.current_max_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing from restart point
            epochs_since_restart = self.current_epoch - max(self.warmup_epochs, self.restart_epoch)
            remaining_epochs = max(1, self.total_epochs - max(self.warmup_epochs, self.restart_epoch))
            progress = epochs_since_restart / remaining_epochs
            lr = self.min_lr + 0.5 * (self.current_max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _check_plateau(self, loss: float):
        """Check if training is in plateau and boost LR if needed."""
        improvement = self.best_loss - loss
        
        if improvement > self.plateau_threshold:
            # Good progress - reset counter
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            # No significant improvement
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                if self.plateau_boosts < self.max_plateau_boosts:
                    # Boost LR - warm restart
                    self._warm_restart()
    
    def _warm_restart(self):
        """Perform warm restart - boost LR back up."""
        self.plateau_boosts += 1
        self.epochs_without_improvement = 0
        self.restart_epoch = self.current_epoch
        
        # Boost factor decreases with each restart
        boost_factor = 1.5 / self.plateau_boosts  # 1.5x, 0.75x, 0.5x
        self.current_max_lr = min(self.base_lr * boost_factor, self.base_lr)
        
        # Apply boosted LR immediately
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_max_lr
        
        import logging
        logging.getLogger(__name__).warning(
            f"Plateau detected! Warm restart #{self.plateau_boosts}: "
            f"Boosting LR to {self.current_max_lr:.6f}"
        )
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_status(self) -> dict:
        """Get scheduler status for logging."""
        return {
            'current_epoch': self.current_epoch,
            'current_lr': self.get_lr(),
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'plateau_boosts': self.plateau_boosts
        }

