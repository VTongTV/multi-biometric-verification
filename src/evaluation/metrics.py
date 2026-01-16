"""
Biometric Metrics and Visualization Module

Generates 30+ metrics and visualizations per model including:
- Performance metrics (EER, AUC, accuracy, d-prime, Gini, etc.)
- Operating point metrics (FAR/FRR at multiple thresholds)
- Score statistics (mean, std, quantiles, kurtosis, skewness)
- Graphs (ROC, DET, FAR/FRR curves, distributions, embedding viz)
- CSV exports for all data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.manifold import TSNE
from scipy import stats
from scipy.spatial.distance import cdist
from loguru import logger


class MetricsExporter:
    """
    Generate 30+ metrics and visualizations for biometric verification.
    
    Outputs include:
    - 15+ scalar metrics
    - 8+ visualization plots
    - 10+ CSV data exports
    """
    
    def __init__(self, output_dir: Path, encoder_name: str):
        self.output_dir = Path(output_dir) / encoder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_name = encoder_name
        self.training_history = []
        self.metrics = {}
        self.raw_data = {}
        
    # =========================================================================
    # TRAINING LOGGING
    # =========================================================================
    
    def log_training_batch(self, epoch: int, batch: int, loss: float, lr: float,
                          accuracy: float = None, grad_norm: float = None):
        """Log training batch statistics."""
        entry = {
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        }
        if accuracy is not None:
            entry['accuracy'] = accuracy
        if grad_norm is not None:
            entry['gradient_norm'] = grad_norm
        self.training_history.append(entry)
    
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float = None,
                     train_acc: float = None, val_acc: float = None):
        """Log epoch-level statistics."""
        entry = {
            'epoch': epoch,
            'epoch_end': True,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(entry)
    
    # =========================================================================
    # METRIC COMPUTATION
    # =========================================================================
    
    def compute_verification_scores(self, embeddings: np.ndarray, labels: np.ndarray,
                                    max_comparisons: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute genuine and impostor similarity scores."""
        n = len(embeddings)
        genuine_scores = []
        impostor_scores = []
        
        # Efficient pairwise comparison with limit
        indices = np.random.permutation(n)[:min(n, 500)]
        
        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j])
                
                if labels[i] == labels[j]:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)
                
                if len(genuine_scores) + len(impostor_scores) >= max_comparisons:
                    break
            if len(genuine_scores) + len(impostor_scores) >= max_comparisons:
                break
        
        self.raw_data['genuine_scores'] = np.array(genuine_scores)
        self.raw_data['impostor_scores'] = np.array(impostor_scores)
        
        return self.raw_data['genuine_scores'], self.raw_data['impostor_scores']
    
    def compute_all_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics (30+ values)."""
        
        genuine, impostor = self.compute_verification_scores(embeddings, labels)
        
        if len(genuine) < 5 or len(impostor) < 5:
            logger.warning(f"Insufficient verification pairs: {len(genuine)} genuine, {len(impostor)} impostor")
            return {}
        
        # Combine for ROC computation
        all_scores = np.concatenate([genuine, impostor])
        all_labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        fnr = 1 - tpr
        
        self.raw_data['fpr'] = fpr
        self.raw_data['tpr'] = tpr
        self.raw_data['fnr'] = fnr
        self.raw_data['thresholds'] = thresholds
        
        # =====================================================================
        # 1. CORE BIOMETRIC METRICS
        # =====================================================================
        
        # Equal Error Rate (EER)
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
        
        # Area Under ROC Curve
        roc_auc = auc(fpr, tpr)
        
        # d-prime (decidability index)
        d_prime = (np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.var(genuine) + np.var(impostor)))
        
        # Gini coefficient
        gini = 2 * roc_auc - 1
        
        self.metrics.update({
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'auc': float(roc_auc),
            'd_prime': float(d_prime),
            'gini_coefficient': float(gini),
        })
        
        # =====================================================================
        # 2. OPERATING POINT METRICS (FAR/FRR at various thresholds)
        # =====================================================================
        
        # FAR at specific FRR values
        for target_fnr in [0.001, 0.01, 0.05, 0.1, 0.2]:
            idx = np.nanargmin(np.abs(fnr - target_fnr))
            far = fpr[idx] if idx < len(fpr) else 1.0
            self.metrics[f'far_at_fnr_{target_fnr}'] = float(far)
        
        # FRR at specific FAR values
        for target_fpr in [0.001, 0.01, 0.05, 0.1]:
            idx = np.nanargmin(np.abs(fpr - target_fpr))
            frr = fnr[idx] if idx < len(fnr) else 1.0
            self.metrics[f'frr_at_far_{target_fpr}'] = float(frr)
        
        # Threshold at specific operating points
        for target_far in [0.01, 0.001, 0.0001]:
            idx = np.nanargmin(np.abs(fpr - target_far))
            thresh = thresholds[idx] if idx < len(thresholds) else 0.5
            self.metrics[f'threshold_at_far_{target_far}'] = float(thresh)
        
        # =====================================================================
        # 3. SCORE DISTRIBUTION STATISTICS
        # =====================================================================
        
        # Genuine score statistics
        self.metrics['genuine_mean'] = float(np.mean(genuine))
        self.metrics['genuine_std'] = float(np.std(genuine))
        self.metrics['genuine_min'] = float(np.min(genuine))
        self.metrics['genuine_max'] = float(np.max(genuine))
        self.metrics['genuine_median'] = float(np.median(genuine))
        self.metrics['genuine_q25'] = float(np.percentile(genuine, 25))
        self.metrics['genuine_q75'] = float(np.percentile(genuine, 75))
        self.metrics['genuine_skewness'] = float(stats.skew(genuine))
        self.metrics['genuine_kurtosis'] = float(stats.kurtosis(genuine))
        
        # Impostor score statistics
        self.metrics['impostor_mean'] = float(np.mean(impostor))
        self.metrics['impostor_std'] = float(np.std(impostor))
        self.metrics['impostor_min'] = float(np.min(impostor))
        self.metrics['impostor_max'] = float(np.max(impostor))
        self.metrics['impostor_median'] = float(np.median(impostor))
        self.metrics['impostor_q25'] = float(np.percentile(impostor, 25))
        self.metrics['impostor_q75'] = float(np.percentile(impostor, 75))
        self.metrics['impostor_skewness'] = float(stats.skew(impostor))
        self.metrics['impostor_kurtosis'] = float(stats.kurtosis(impostor))
        
        # Score overlap metrics
        overlap_min = max(np.min(genuine), np.min(impostor))
        overlap_max = min(np.max(genuine), np.max(impostor))
        overlap_ratio = (overlap_max - overlap_min) / (np.max(genuine) - np.min(impostor)) if overlap_max > overlap_min else 0
        self.metrics['score_overlap_ratio'] = float(overlap_ratio)
        
        # =====================================================================
        # 4. SAMPLE COUNTS
        # =====================================================================
        
        self.metrics['n_genuine_pairs'] = int(len(genuine))
        self.metrics['n_impostor_pairs'] = int(len(impostor))
        self.metrics['n_total_pairs'] = int(len(genuine) + len(impostor))
        self.metrics['n_unique_identities'] = int(len(np.unique(labels)))
        self.metrics['n_total_samples'] = int(len(embeddings))
        
        # =====================================================================
        # 5. PRECISION/RECALL AT EER
        # =====================================================================
        
        predictions = (all_scores >= eer_threshold).astype(int)
        self.metrics['accuracy_at_eer'] = float(accuracy_score(all_labels, predictions))
        self.metrics['precision_at_eer'] = float(precision_score(all_labels, predictions, zero_division=0))
        self.metrics['recall_at_eer'] = float(recall_score(all_labels, predictions, zero_division=0))
        self.metrics['f1_at_eer'] = float(f1_score(all_labels, predictions, zero_division=0))
        
        # =====================================================================
        # 6. EMBEDDING QUALITY METRICS
        # =====================================================================
        
        # Intra-class cohesion (average genuine similarity)
        self.metrics['intra_class_cohesion'] = float(np.mean(genuine))
        
        # Inter-class separation (1 - average impostor similarity)
        self.metrics['inter_class_separation'] = float(1 - np.mean(impostor))
        
        # Silhouette-like score
        self.metrics['separability_score'] = float((np.mean(genuine) - np.mean(impostor)) / (np.std(genuine) + np.std(impostor) + 1e-8))
        
        logger.info(f"Computed {len(self.metrics)} metrics")
        return self.metrics
    
    # =========================================================================
    # VISUALIZATION GENERATION
    # =========================================================================
    
    def plot_roc_curve(self) -> Path:
        """Generate ROC curve with annotations."""
        if 'fpr' not in self.raw_data:
            return None
        
        fpr, tpr = self.raw_data['fpr'], self.raw_data['tpr']
        roc_auc = self.metrics.get('auc', 0)
        eer = self.metrics.get('eer', 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        # Mark EER point
        eer_idx = np.nanargmin(np.abs(fpr - self.raw_data['fnr']))
        ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], c='red', s=100, zorder=5, 
                   label=f'EER = {eer*100:.2f}%')
        
        # Mark operating points
        for far in [0.01, 0.001]:
            idx = np.nanargmin(np.abs(fpr - far))
            ax.scatter([fpr[idx]], [tpr[idx]], c='green', s=50, zorder=4)
            ax.annotate(f'FAR={far}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(5,5))
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (1-FRR)', fontsize=12)
        ax.set_title(f'ROC Curve - {self.encoder_name.upper()}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        path = self.output_dir / f'{self.encoder_name}_01_roc_curve.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_det_curve(self) -> Path:
        """Generate DET curve (log-log scale)."""
        if 'fpr' not in self.raw_data:
            return None
        
        fpr = self.raw_data['fpr']
        fnr = self.raw_data['fnr']
        
        # Filter for valid points
        valid = (fpr > 0) & (fnr > 0) & (fpr < 1) & (fnr < 1)
        fpr, fnr = fpr[valid], fnr[valid]
        
        if len(fpr) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.loglog(fpr * 100, fnr * 100, 'b-', linewidth=2)
        ax.loglog([0.01, 100], [0.01, 100], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Accept Rate (%)', fontsize=12)
        ax.set_ylabel('False Reject Rate (%)', fontsize=12)
        ax.set_title(f'DET Curve - {self.encoder_name.upper()}', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0.01, 100])
        ax.set_ylim([0.01, 100])
        
        path = self.output_dir / f'{self.encoder_name}_02_det_curve.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_far_frr_curves(self) -> Path:
        """Generate FAR/FRR vs threshold curves."""
        if 'thresholds' not in self.raw_data:
            return None
        
        thresholds = self.raw_data['thresholds']
        fpr = self.raw_data['fpr']
        fnr = self.raw_data['fnr']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, fpr * 100, 'r-', linewidth=2, label='FAR')
        ax.plot(thresholds, fnr * 100, 'b-', linewidth=2, label='FRR')
        
        # Mark EER
        eer_threshold = self.metrics.get('eer_threshold', 0.5)
        eer = self.metrics.get('eer', 0)
        ax.axvline(eer_threshold, color='green', linestyle='--', label=f'EER Threshold ({eer_threshold:.3f})')
        ax.scatter([eer_threshold], [eer * 100], c='green', s=100, zorder=5)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title(f'FAR/FRR vs Threshold - {self.encoder_name.upper()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = self.output_dir / f'{self.encoder_name}_03_far_frr_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_score_distributions(self) -> Path:
        """Generate score distribution histograms."""
        if 'genuine_scores' not in self.raw_data:
            return None
        
        genuine = self.raw_data['genuine_scores']
        impostor = self.raw_data['impostor_scores']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overlapping histogram
        ax = axes[0]
        ax.hist(impostor, bins=50, alpha=0.6, color='red', label='Impostor', density=True)
        ax.hist(genuine, bins=50, alpha=0.6, color='green', label='Genuine', density=True)
        ax.axvline(self.metrics.get('eer_threshold', 0.5), color='blue', linestyle='--', label='EER Threshold')
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distributions (Overlapping)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Separated distributions
        ax = axes[1]
        ax.boxplot([impostor, genuine], labels=['Impostor', 'Genuine'], bootstrap=100)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title('Score Distributions (Box Plot)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f'{self.encoder_name}_04_score_distributions.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_confusion_matrix(self) -> Path:
        """Generate confusion matrix at EER threshold."""
        if 'genuine_scores' not in self.raw_data:
            return None
        
        genuine = self.raw_data['genuine_scores']
        impostor = self.raw_data['impostor_scores']
        eer_threshold = self.metrics.get('eer_threshold', 0.5)
        
        # Predictions at EER threshold
        all_scores = np.concatenate([genuine, impostor])
        all_labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        predictions = (all_scores >= eer_threshold).astype(int)
        
        cm = confusion_matrix(all_labels, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Reject', 'Accept'], yticklabels=['Impostor', 'Genuine'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix at EER Threshold - {self.encoder_name.upper()}', fontsize=14)
        
        path = self.output_dir / f'{self.encoder_name}_05_confusion_matrix.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_training_curves(self) -> Path:
        """Generate training loss and learning rate curves."""
        if not self.training_history:
            return None
        
        df = pd.DataFrame(self.training_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss per batch
        ax = axes[0, 0]
        ax.plot(df['loss'].values, 'b-', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Batch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title('Training Loss (per batch)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Loss per epoch (averaged)
        ax = axes[0, 1]
        epoch_loss = df.groupby('epoch')['loss'].mean()
        ax.plot(epoch_loss.index, epoch_loss.values, 'b-o', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Average Loss', fontsize=10)
        ax.set_title('Training Loss (per epoch)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        epoch_lr = df.groupby('epoch')['learning_rate'].first()
        ax.plot(epoch_lr.index, epoch_lr.values, 'r-o', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Learning Rate', fontsize=10)
        ax.set_title('Learning Rate Schedule', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Loss distribution
        ax = axes[1, 1]
        ax.hist(df['loss'].values, bins=50, alpha=0.7, color='blue')
        ax.set_xlabel('Loss', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Loss Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f'{self.encoder_name}_06_training_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_precision_recall_curve(self) -> Path:
        """Generate Precision-Recall curve."""
        if 'genuine_scores' not in self.raw_data:
            return None
        
        genuine = self.raw_data['genuine_scores']
        impostor = self.raw_data['impostor_scores']
        
        all_scores = np.concatenate([genuine, impostor])
        all_labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.encoder_name.upper()}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        path = self.output_dir / f'{self.encoder_name}_07_precision_recall.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_score_vs_rank(self) -> Path:
        """Generate score vs rank plot (CMC-like)."""
        if 'genuine_scores' not in self.raw_data:
            return None
        
        genuine = np.sort(self.raw_data['genuine_scores'])[::-1]  # Descending
        impostor = np.sort(self.raw_data['impostor_scores'])[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(len(genuine)), genuine, 'g-', linewidth=2, label='Genuine')
        ax.plot(range(len(impostor)), impostor, 'r-', linewidth=2, label='Impostor', alpha=0.7)
        ax.axhline(self.metrics.get('eer_threshold', 0.5), color='blue', linestyle='--', label='EER Threshold')
        
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title(f'Score vs Rank - {self.encoder_name.upper()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = self.output_dir / f'{self.encoder_name}_08_score_vs_rank.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_metrics_summary(self) -> Path:
        """Generate a visual summary of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Key metrics bar chart
        ax = axes[0, 0]
        key_metrics = {
            'EER (%)': self.metrics.get('eer', 0) * 100,
            'AUC': self.metrics.get('auc', 0),
            'd-prime': self.metrics.get('d_prime', 0),
            'Accuracy': self.metrics.get('accuracy_at_eer', 0)
        }
        bars = ax.bar(key_metrics.keys(), key_metrics.values(), color=['red', 'green', 'blue', 'orange'])
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title('Key Performance Metrics', fontsize=12)
        for bar, val in zip(bars, key_metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')
        
        # FAR at different FRR levels
        ax = axes[0, 1]
        far_at_fnr = {f'FRR={k}': self.metrics.get(f'far_at_fnr_{k}', 0) * 100 
                     for k in [0.001, 0.01, 0.05, 0.1, 0.2]}
        ax.bar(far_at_fnr.keys(), far_at_fnr.values(), color='crimson')
        ax.set_ylabel('FAR (%)', fontsize=10)
        ax.set_title('FAR at Different FRR Values', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Score statistics comparison
        ax = axes[1, 0]
        x = np.arange(4)
        width = 0.35
        genuine_stats = [
            self.metrics.get('genuine_mean', 0),
            self.metrics.get('genuine_std', 0),
            self.metrics.get('genuine_median', 0),
            self.metrics.get('genuine_q75', 0) - self.metrics.get('genuine_q25', 0)
        ]
        impostor_stats = [
            self.metrics.get('impostor_mean', 0),
            self.metrics.get('impostor_std', 0),
            self.metrics.get('impostor_median', 0),
            self.metrics.get('impostor_q75', 0) - self.metrics.get('impostor_q25', 0)
        ]
        ax.bar(x - width/2, genuine_stats, width, label='Genuine', color='green', alpha=0.7)
        ax.bar(x + width/2, impostor_stats, width, label='Impostor', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Std', 'Median', 'IQR'])
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title('Score Statistics', fontsize=12)
        ax.legend()
        
        # Sample counts
        ax = axes[1, 1]
        counts = {
            'Samples': self.metrics.get('n_total_samples', 0),
            'Identities': self.metrics.get('n_unique_identities', 0),
            'Genuine Pairs': self.metrics.get('n_genuine_pairs', 0),
            'Impostor Pairs': self.metrics.get('n_impostor_pairs', 0)
        }
        ax.bar(counts.keys(), counts.values(), color=['blue', 'orange', 'green', 'red'])
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Dataset Statistics', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        path = self.output_dir / f'{self.encoder_name}_09_metrics_summary.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    # =========================================================================
    # CSV EXPORTS
    # =========================================================================
    
    def save_all_csvs(self) -> Dict[str, Path]:
        """Save all data to CSV files."""
        csv_paths = {}
        
        # 1. Training history
        if self.training_history:
            df = pd.DataFrame(self.training_history)
            path = self.output_dir / f'{self.encoder_name}_training_history.csv'
            df.to_csv(path, index=False)
            csv_paths['training_history'] = path
        
        # 2. Main metrics
        if self.metrics:
            df = pd.DataFrame([self.metrics])
            path = self.output_dir / f'{self.encoder_name}_metrics.csv'
            df.to_csv(path, index=False)
            csv_paths['metrics'] = path
        
        # 3. Genuine scores
        if 'genuine_scores' in self.raw_data:
            df = pd.DataFrame({'score': self.raw_data['genuine_scores'], 'type': 'genuine'})
            path = self.output_dir / f'{self.encoder_name}_genuine_scores.csv'
            df.to_csv(path, index=False)
            csv_paths['genuine_scores'] = path
        
        # 4. Impostor scores
        if 'impostor_scores' in self.raw_data:
            df = pd.DataFrame({'score': self.raw_data['impostor_scores'], 'type': 'impostor'})
            path = self.output_dir / f'{self.encoder_name}_impostor_scores.csv'
            df.to_csv(path, index=False)
            csv_paths['impostor_scores'] = path
        
        # 5. All verification scores combined
        if 'genuine_scores' in self.raw_data and 'impostor_scores' in self.raw_data:
            genuine = self.raw_data['genuine_scores']
            impostor = self.raw_data['impostor_scores']
            df = pd.DataFrame({
                'score': np.concatenate([genuine, impostor]),
                'label': np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))]),
                'type': ['genuine'] * len(genuine) + ['impostor'] * len(impostor)
            })
            path = self.output_dir / f'{self.encoder_name}_all_verification_scores.csv'
            df.to_csv(path, index=False)
            csv_paths['all_verification_scores'] = path
        
        # 6. ROC curve data
        if 'fpr' in self.raw_data:
            df = pd.DataFrame({
                'fpr': self.raw_data['fpr'],
                'tpr': self.raw_data['tpr'],
                'fnr': self.raw_data['fnr'],
                'threshold': self.raw_data['thresholds']
            })
            path = self.output_dir / f'{self.encoder_name}_roc_data.csv'
            df.to_csv(path, index=False)
            csv_paths['roc_data'] = path
        
        # 7. Operating points summary
        operating_points = []
        for far in [0.0001, 0.001, 0.01, 0.05, 0.1]:
            for metric_key in [f'frr_at_far_{far}', f'threshold_at_far_{far}']:
                if metric_key in self.metrics:
                    operating_points.append({
                        'operating_point': f'FAR={far}',
                        'metric': metric_key.split('_at_')[0],
                        'value': self.metrics[metric_key]
                    })
        for fnr in [0.001, 0.01, 0.05, 0.1, 0.2]:
            if f'far_at_fnr_{fnr}' in self.metrics:
                operating_points.append({
                    'operating_point': f'FRR={fnr}',
                    'metric': 'far',
                    'value': self.metrics[f'far_at_fnr_{fnr}']
                })
        if operating_points:
            df = pd.DataFrame(operating_points)
            path = self.output_dir / f'{self.encoder_name}_operating_points.csv'
            df.to_csv(path, index=False)
            csv_paths['operating_points'] = path
        
        # 8. Score statistics
        score_stats = []
        for prefix in ['genuine', 'impostor']:
            stats_dict = {'distribution': prefix}
            for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis']:
                key = f'{prefix}_{stat}'
                if key in self.metrics:
                    stats_dict[stat] = self.metrics[key]
            score_stats.append(stats_dict)
        if score_stats:
            df = pd.DataFrame(score_stats)
            path = self.output_dir / f'{self.encoder_name}_score_statistics.csv'
            df.to_csv(path, index=False)
            csv_paths['score_statistics'] = path
        
        # 9. Per-epoch summary
        if self.training_history:
            df = pd.DataFrame(self.training_history)
            epoch_summary = df.groupby('epoch').agg({
                'loss': ['mean', 'min', 'max', 'std'],
                'learning_rate': 'first'
            }).reset_index()
            epoch_summary.columns = ['epoch', 'loss_mean', 'loss_min', 'loss_max', 'loss_std', 'learning_rate']
            path = self.output_dir / f'{self.encoder_name}_epoch_summary.csv'
            epoch_summary.to_csv(path, index=False)
            csv_paths['epoch_summary'] = path
        
        # 10. Metrics in long format (for easy plotting)
        if self.metrics:
            long_format = [{'metric': k, 'value': v} for k, v in self.metrics.items()]
            df = pd.DataFrame(long_format)
            path = self.output_dir / f'{self.encoder_name}_metrics_long.csv'
            df.to_csv(path, index=False)
            csv_paths['metrics_long'] = path
        
        logger.info(f"Saved {len(csv_paths)} CSV files")
        return csv_paths
    
    # =========================================================================
    # MASTER EXPORT FUNCTION
    # =========================================================================
    
    def generate_all_outputs(self) -> Dict[str, Any]:
        """Generate all outputs: metrics, plots, and CSVs."""
        outputs = {
            'plots': {},
            'csvs': {},
            'metrics': self.metrics
        }
        
        # Generate all plots
        plot_functions = [
            ('roc_curve', self.plot_roc_curve),
            ('det_curve', self.plot_det_curve),
            ('far_frr_curves', self.plot_far_frr_curves),
            ('score_distributions', self.plot_score_distributions),
            ('confusion_matrix', self.plot_confusion_matrix),
            ('training_curves', self.plot_training_curves),
            ('precision_recall', self.plot_precision_recall_curve),
            ('score_vs_rank', self.plot_score_vs_rank),
            ('metrics_summary', self.plot_metrics_summary),
        ]
        
        for name, func in plot_functions:
            try:
                path = func()
                if path:
                    outputs['plots'][name] = str(path)
                    logger.info(f"Generated plot: {name}")
            except Exception as e:
                logger.warning(f"Failed to generate {name}: {e}")
        
        # Save all CSVs
        outputs['csvs'] = {k: str(v) for k, v in self.save_all_csvs().items()}
        
        # Save master summary JSON
        summary = {
            'encoder': self.encoder_name,
            'n_metrics': len(self.metrics),
            'n_plots': len(outputs['plots']),
            'n_csvs': len(outputs['csvs']),
            'metrics': self.metrics,
            'outputs': outputs,
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / f'{self.encoder_name}_complete_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        outputs['summary_json'] = str(json_path)
        
        total_outputs = len(self.metrics) + len(outputs['plots']) + len(outputs['csvs'])
        logger.info(f"Generated {total_outputs} total outputs ({len(self.metrics)} metrics, {len(outputs['plots'])} plots, {len(outputs['csvs'])} CSVs)")
        
        return outputs
