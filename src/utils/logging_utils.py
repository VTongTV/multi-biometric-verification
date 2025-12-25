"""
Logging configuration utilities using loguru.

Provides structured logging for training, evaluation, and inference.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> None:
    """
    Configure loguru for the biometric system.
    
    Args:
        log_dir: Directory for log files. If None, logs only to console.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old log files
        console_format: Custom format for console output
        file_format: Custom format for file output
        experiment_name: Name for the experiment (used in log filenames)
    """
    # Remove default handler
    logger.remove()
    
    # Default formats
    if console_format is None:
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    if file_format is None:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handlers if directory specified
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"biometric_{timestamp}.log"
        
        # Main log file (all levels)
        logger.add(
            log_dir / log_filename,
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Separate error log
        error_filename = log_filename.replace(".log", "_errors.log")
        logger.add(
            log_dir / error_filename,
            format=file_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"Logging to: {log_dir / log_filename}")


def get_experiment_logger(experiment_name: str, log_dir: Path) -> None:
    """
    Configure logging for a specific training experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
    """
    setup_logging(
        log_dir=log_dir,
        log_level="DEBUG",  # Capture all during training
        experiment_name=experiment_name,
        rotation="50 MB",
        retention="1 month"
    )


class TrainingLogger:
    """
    Structured logger for training metrics.
    
    Provides consistent formatting for training progress reporting.
    """
    
    def __init__(
        self,
        experiment_name: str,
        total_epochs: int,
        log_every_n_steps: int = 100
    ):
        self.experiment_name = experiment_name
        self.total_epochs = total_epochs
        self.log_every_n_steps = log_every_n_steps
        self.current_epoch = 0
        self.global_step = 0
        
    def log_epoch_start(self, epoch: int) -> None:
        """Log the start of an epoch."""
        self.current_epoch = epoch
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch}/{self.total_epochs} - {self.experiment_name}")
        logger.info(f"{'='*60}")
        
    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """Log epoch summary."""
        msg = f"Epoch {epoch}/{self.total_epochs} complete"
        
        metrics = []
        metrics.append(f"train_loss={train_loss:.4f}")
        
        if val_loss is not None:
            metrics.append(f"val_loss={val_loss:.4f}")
        if train_acc is not None:
            metrics.append(f"train_acc={train_acc:.2%}")
        if val_acc is not None:
            metrics.append(f"val_acc={val_acc:.2%}")
        if lr is not None:
            metrics.append(f"lr={lr:.2e}")
        if duration_seconds is not None:
            metrics.append(f"time={duration_seconds:.1f}s")
            
        logger.info(f"{msg} | {' | '.join(metrics)}")
        
    def log_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        lr: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log training step (respects log_every_n_steps)."""
        self.global_step += 1
        
        if step % self.log_every_n_steps == 0 or step == total_steps - 1:
            progress = f"[{step+1}/{total_steps}]"
            metrics = [f"loss={loss:.4f}"]
            
            if lr is not None:
                metrics.append(f"lr={lr:.2e}")
                
            for key, value in kwargs.items():
                if isinstance(value, float):
                    metrics.append(f"{key}={value:.4f}")
                else:
                    metrics.append(f"{key}={value}")
                    
            logger.info(f"Step {progress} | {' | '.join(metrics)}")
            
    def log_validation(
        self,
        metrics: dict,
        prefix: str = "Validation"
    ) -> None:
        """Log validation metrics."""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if value < 0.01 or value > 100:
                    formatted.append(f"{key}={value:.4e}")
                elif key.endswith("acc") or key.endswith("accuracy"):
                    formatted.append(f"{key}={value:.2%}")
                else:
                    formatted.append(f"{key}={value:.4f}")
            else:
                formatted.append(f"{key}={value}")
                
        logger.info(f"{prefix}: {' | '.join(formatted)}")
        
    def log_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Log checkpoint save."""
        if is_best:
            logger.success(f"New best model saved: {path}")
        else:
            logger.info(f"Checkpoint saved: {path}")
            
    def log_early_stopping(self, patience_counter: int, max_patience: int) -> None:
        """Log early stopping status."""
        logger.warning(
            f"No improvement for {patience_counter}/{max_patience} epochs. "
            "Early stopping may trigger."
        )


class MetricTracker:
    """
    Track and log metrics during training.
    
    Provides running averages and best metric tracking.
    """
    
    def __init__(self, metrics_to_track: list[str]):
        self.metrics_to_track = metrics_to_track
        self.reset()
        
    def reset(self) -> None:
        """Reset all tracked values."""
        self._sums = {m: 0.0 for m in self.metrics_to_track}
        self._counts = {m: 0 for m in self.metrics_to_track}
        self._best = {m: float('-inf') for m in self.metrics_to_track}
        
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self._sums:
                self._sums[key] += value
                self._counts[key] += 1
                
    def get_average(self, metric: str) -> float:
        """Get running average for a metric."""
        if self._counts.get(metric, 0) == 0:
            return 0.0
        return self._sums[metric] / self._counts[metric]
    
    def get_all_averages(self) -> dict:
        """Get all running averages."""
        return {
            m: self.get_average(m) 
            for m in self.metrics_to_track
        }
        
    def check_best(self, metric: str, value: float, higher_is_better: bool = True) -> bool:
        """
        Check if this is the best value for a metric.
        
        Returns True if this is a new best.
        """
        if higher_is_better:
            is_best = value > self._best[metric]
        else:
            is_best = value < self._best.get(metric, float('inf'))
            
        if is_best:
            self._best[metric] = value
            
        return is_best
    
    def reset_epoch(self) -> None:
        """Reset sums and counts for new epoch, keeping best values."""
        for m in self.metrics_to_track:
            self._sums[m] = 0.0
            self._counts[m] = 0
