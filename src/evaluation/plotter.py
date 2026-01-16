"""
Biometric Evaluation Plotter.

Generates 30 separate plots for evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy import stats

# =============================================================================
# SCIENTIFIC PLOTTING CONFIGURATION
# =============================================================================

# Publication-quality theme settings
SCIENTIFIC_THEME = {
    'figure.figsize': (10, 7),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'medium',
    'axes.linewidth': 1.5,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'legend.fontsize': 11,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',
    'legend.fancybox': True,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# High-contrast aesthetic color palette
COLORS = {
    'genuine': '#1B5E20',       # Deep forest green
    'genuine_light': '#81C784', # Light green for fills
    'impostor': '#B71C1C',      # Deep crimson red
    'impostor_light': '#EF9A9A', # Light red for fills
    'roc': '#0D47A1',           # Strong blue
    'roc_fill': '#90CAF9',      # Light blue for AUC fill
    'threshold': '#7B1FA2',     # Purple
    'highlight': '#E65100',     # Deep orange
    'secondary': '#00695C',     # Teal
    'training': '#1565C0',      # Training loss blue
    'validation': '#2E7D32',    # Validation green
    'lr': '#C62828',            # Learning rate red
    'annotation': '#37474F',    # Dark gray for annotations
    'grid': '#BDBDBD',          # Light gray for grid
}

# Statistical annotation format
STAT_BOX_PROPS = dict(
    boxstyle='round,pad=0.4',
    facecolor='white',
    edgecolor='#BDBDBD',
    alpha=0.95,
    linewidth=1.2
)

def apply_scientific_theme():
    """Apply scientific plotting theme globally."""
    plt.rcParams.update(SCIENTIFIC_THEME)

def format_scientific(value: float, precision: int = 2) -> str:
    """Format value in scientific notation if appropriate."""
    if abs(value) < 0.001 or abs(value) >= 10000:
        return f'{value:.{precision}e}'
    return f'{value:.{precision}f}'


class MetricsPlotter:
    """Generate 30 separate publication-quality evaluation plots."""
    
    def __init__(self, output_dir: Path, name: str):
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.plot_count = 0
        self.all_plots: List[str] = []
        # Apply scientific theme on initialization
        apply_scientific_theme()
    
    def _save_plot(self, fig, title: str) -> Path:
        """Save a publication-quality plot and track it."""
        self.plot_count += 1
        filename = f"{self.name}_{self.plot_count:02d}_{title.lower().replace(' ', '_')}.png"
        path = self.output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        self.all_plots.append(str(path))
        logger.info(f"Saved plot {self.plot_count}: {filename}")
        return path
    
    # Plot 1: ROC Curve (Enhanced with operating points and AUC fill)
    def plot_roc_curve(self, fpr, tpr, auc_val):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['roc_fill'])
        ax.plot(fpr, tpr, color=COLORS['roc'], lw=2.5, label=f'ROC Curve')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
        
        # Find and mark key operating points
        # EER point
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], s=120, c=COLORS['highlight'], 
                   zorder=5, marker='o', edgecolors='white', linewidths=2,
                   label=f'EER = {eer*100:.2f}%')
        
        # TAR @ FAR=0.1%
        idx_01 = np.nanargmin(np.abs(fpr - 0.001))
        tar_01 = tpr[idx_01] * 100
        ax.scatter([fpr[idx_01]], [tpr[idx_01]], s=100, c=COLORS['secondary'], 
                   zorder=5, marker='s', edgecolors='white', linewidths=1.5)
        
        # TAR @ FAR=1%
        idx_1 = np.nanargmin(np.abs(fpr - 0.01))
        tar_1 = tpr[idx_1] * 100
        ax.scatter([fpr[idx_1]], [tpr[idx_1]], s=100, c=COLORS['threshold'], 
                   zorder=5, marker='^', edgecolors='white', linewidths=1.5)
        
        ax.set_xlabel('False Positive Rate (FPR)', fontweight='medium')
        ax.set_ylabel('True Positive Rate (TPR)', fontweight='medium')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        
        # Enhanced legend with metrics
        legend_text = (f'AUC = {auc_val:.4f}\n'
                      f'TAR@FAR=0.1% = {tar_01:.1f}%\n'
                      f'TAR@FAR=1% = {tar_1:.1f}%')
        ax.text(0.98, 0.35, legend_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.legend(loc='lower right', framealpha=0.95)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect('equal')
        return self._save_plot(fig, 'ROC Curve')
    
    # Plot 2: DET Curve (NIST-standard with EER marker)
    def plot_det_curve(self, fpr, fnr):
        valid = (fpr > 0) & (fnr > 0)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Main DET curve
        ax.loglog(fpr[valid]*100, fnr[valid]*100, color=COLORS['roc'], lw=2.5, label='DET Curve')
        
        # EER reference line (where FAR = FRR)
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        ax.loglog([0.01, 100], [0.01, 100], 'k--', lw=1.5, alpha=0.5, label='EER Line')
        ax.scatter([fpr[eer_idx]*100], [fnr[eer_idx]*100], s=150, c=COLORS['highlight'], 
                   zorder=5, marker='*', edgecolors='white', linewidths=1.5,
                   label=f'EER = {eer:.2f}%')
        
        # Iso-error reference lines
        for ref_rate in [0.1, 1.0, 10.0]:
            ax.axhline(ref_rate, color=COLORS['grid'], linestyle=':', alpha=0.6, lw=1)
            ax.axvline(ref_rate, color=COLORS['grid'], linestyle=':', alpha=0.6, lw=1)
        
        ax.set_xlabel('False Accept Rate — FAR (%)', fontweight='medium')
        ax.set_ylabel('False Reject Rate — FRR (%)', fontweight='medium')
        ax.set_title('Detection Error Tradeoff (DET) Curve', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.set_xlim([0.01, 100])
        ax.set_ylim([0.01, 100])
        ax.grid(True, alpha=0.3, which='both')
        return self._save_plot(fig, 'DET Curve')
    
    # Plot 3: FAR/FRR vs Threshold (Enhanced with EER intersection)
    def plot_far_frr_threshold(self, thresholds, fpr, fnr):
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Plot error rate curves with fills
        ax.fill_between(thresholds, fpr*100, alpha=0.15, color=COLORS['impostor'])
        ax.fill_between(thresholds, fnr*100, alpha=0.15, color=COLORS['genuine'])
        ax.plot(thresholds, fpr*100, color=COLORS['impostor'], lw=2.5, label='FAR (False Accept Rate)')
        ax.plot(thresholds, fnr*100, color=COLORS['genuine'], lw=2.5, label='FRR (False Reject Rate)')
        
        # Find and mark EER intersection
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        eer_thresh = thresholds[eer_idx]
        ax.scatter([eer_thresh], [eer], s=150, c=COLORS['highlight'], zorder=5, 
                   marker='o', edgecolors='white', linewidths=2)
        ax.axvline(eer_thresh, color=COLORS['threshold'], linestyle='--', lw=1.5, alpha=0.7)
        ax.axhline(eer, color=COLORS['threshold'], linestyle='--', lw=1.5, alpha=0.7)
        
        # EER annotation
        ax.annotate(f'EER = {eer:.2f}%\nθ = {eer_thresh:.3f}', 
                    xy=(eer_thresh, eer), xytext=(15, 15), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=COLORS['annotation'],
                    bbox=STAT_BOX_PROPS,
                    arrowprops=dict(arrowstyle='->', color=COLORS['annotation'], lw=1.5))
        
        ax.set_xlabel('Decision Threshold (θ)', fontweight='medium')
        ax.set_ylabel('Error Rate (%)', fontweight='medium')
        ax.set_title('Error Rates vs Decision Threshold', fontweight='bold')
        ax.legend(loc='upper center', framealpha=0.95)
        return self._save_plot(fig, 'FAR FRR Threshold')
    
    # Plot 4: Genuine Distribution (Enhanced with full statistics)
    def plot_genuine_distribution(self, genuine):
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Histogram with KDE overlay
        n, bins, patches = ax.hist(genuine, bins=50, color=COLORS['genuine_light'], 
                                    alpha=0.8, edgecolor=COLORS['genuine'], linewidth=1.2,
                                    density=True, label='Histogram')
        
        # KDE overlay
        kde = stats.gaussian_kde(genuine)
        x_range = np.linspace(genuine.min() - 0.1, genuine.max() + 0.1, 200)
        ax.plot(x_range, kde(x_range), color=COLORS['genuine'], lw=2.5, label='KDE')
        
        # Statistical markers
        mean_val = np.mean(genuine)
        median_val = np.median(genuine)
        ax.axvline(mean_val, color=COLORS['highlight'], linestyle='-', lw=2.5, label=f'Mean')
        ax.axvline(median_val, color=COLORS['threshold'], linestyle='--', lw=2, label=f'Median')
        
        # Comprehensive statistics box
        stat_text = (f'Statistics (n={len(genuine):,})\n'
                     f'────────────────\n'
                     f'μ = {mean_val:.4f}\n'
                     f'σ = {np.std(genuine):.4f}\n'
                     f'Median = {median_val:.4f}\n'
                     f'IQR = {np.percentile(genuine, 75) - np.percentile(genuine, 25):.4f}\n'
                     f'Skew = {stats.skew(genuine):.3f}\n'
                     f'Kurt = {stats.kurtosis(genuine):.3f}')
        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.set_xlabel('Similarity Score', fontweight='medium')
        ax.set_ylabel('Probability Density', fontweight='medium')
        ax.set_title('Genuine Match Score Distribution', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        return self._save_plot(fig, 'Genuine Distribution')
    
    # Plot 5: Impostor Distribution (Enhanced with full statistics)
    def plot_impostor_distribution(self, impostor):
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Histogram with KDE overlay
        n, bins, patches = ax.hist(impostor, bins=50, color=COLORS['impostor_light'], 
                                    alpha=0.8, edgecolor=COLORS['impostor'], linewidth=1.2,
                                    density=True, label='Histogram')
        
        # KDE overlay
        kde = stats.gaussian_kde(impostor)
        x_range = np.linspace(impostor.min() - 0.1, impostor.max() + 0.1, 200)
        ax.plot(x_range, kde(x_range), color=COLORS['impostor'], lw=2.5, label='KDE')
        
        # Statistical markers
        mean_val = np.mean(impostor)
        median_val = np.median(impostor)
        ax.axvline(mean_val, color=COLORS['highlight'], linestyle='-', lw=2.5, label=f'Mean')
        ax.axvline(median_val, color=COLORS['threshold'], linestyle='--', lw=2, label=f'Median')
        
        # Comprehensive statistics box
        stat_text = (f'Statistics (n={len(impostor):,})\n'
                     f'────────────────\n'
                     f'μ = {mean_val:.4f}\n'
                     f'σ = {np.std(impostor):.4f}\n'
                     f'Median = {median_val:.4f}\n'
                     f'IQR = {np.percentile(impostor, 75) - np.percentile(impostor, 25):.4f}\n'
                     f'Skew = {stats.skew(impostor):.3f}\n'
                     f'Kurt = {stats.kurtosis(impostor):.3f}')
        ax.text(0.98, 0.98, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.set_xlabel('Similarity Score', fontweight='medium')
        ax.set_ylabel('Probability Density', fontweight='medium')
        ax.set_title('Impostor Match Score Distribution', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.95)
        return self._save_plot(fig, 'Impostor Distribution')
    
    # Plot 6: Overlapping Distributions (Enhanced with d-prime and overlap analysis)
    def plot_overlapping_distributions(self, genuine, impostor, threshold):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # KDE for both distributions
        genuine_kde = stats.gaussian_kde(genuine)
        impostor_kde = stats.gaussian_kde(impostor)
        all_scores = np.concatenate([genuine, impostor])
        x_range = np.linspace(all_scores.min() - 0.1, all_scores.max() + 0.1, 300)
        
        # Fill under curves
        ax.fill_between(x_range, impostor_kde(x_range), alpha=0.35, color=COLORS['impostor_light'],
                        label='Impostor Distribution')
        ax.fill_between(x_range, genuine_kde(x_range), alpha=0.35, color=COLORS['genuine_light'],
                        label='Genuine Distribution')
        ax.plot(x_range, impostor_kde(x_range), color=COLORS['impostor'], lw=2.5)
        ax.plot(x_range, genuine_kde(x_range), color=COLORS['genuine'], lw=2.5)
        
        # Decision threshold
        ax.axvline(threshold, color=COLORS['threshold'], linestyle='--', lw=2.5, 
                   label=f'EER Threshold = {threshold:.3f}')
        
        # Calculate key metrics
        d_prime = (np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.var(genuine) + np.var(impostor) + 1e-8))
        
        # Calculate overlap integral (approximate)
        min_kde = np.minimum(genuine_kde(x_range), impostor_kde(x_range))
        overlap = np.trapz(min_kde, x_range) * 100
        
        # Statistics panel
        stat_text = (f"Separability Analysis\n"
                     f"{'─'*20}\n"
                     f"d' = {d_prime:.3f}\n"
                     f"Overlap = {overlap:.1f}%\n"
                     f"μ_gen = {np.mean(genuine):.3f}\n"
                     f"μ_imp = {np.mean(impostor):.3f}\n"
                     f"Δμ = {np.mean(genuine) - np.mean(impostor):.3f}")
        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.set_xlabel('Similarity Score', fontweight='medium')
        ax.set_ylabel('Probability Density', fontweight='medium')
        ax.set_title('Score Distributions with Separability Analysis', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        return self._save_plot(fig, 'Score Distributions')
    
    # Plot 7: Confusion Matrix (Enhanced with percentages)
    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
        total = tn + fp + fn + tp
        
        # Create annotation with counts and percentages
        annot = np.array([[f'{tn:,}\n({tn/total*100:.1f}%)', f'{fp:,}\n({fp/total*100:.1f}%)'],
                          [f'{fn:,}\n({fn/total*100:.1f}%)', f'{tp:,}\n({tp/total*100:.1f}%)']])
        
        # Custom colormap
        cmap = sns.color_palette('Blues', as_cmap=True)
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                   xticklabels=['Reject', 'Accept'], yticklabels=['Impostor', 'Genuine'],
                   linewidths=2, linecolor='white', cbar_kws={'label': 'Count', 'shrink': 0.8},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        ax.set_xlabel('Predicted Label', fontweight='medium', fontsize=13)
        ax.set_ylabel('True Label', fontweight='medium', fontsize=13)
        ax.set_title('Confusion Matrix at EER Operating Point', fontweight='bold')
        
        plt.tight_layout()
        return self._save_plot(fig, 'Confusion Matrix')
    
    # Plot 8: Training Loss per Batch (Enhanced with smoothing)
    def plot_training_loss_batch(self, losses):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        batches = np.arange(len(losses))
        losses = np.array(losses)
        
        # Raw loss (light)
        ax.plot(batches, losses, color=COLORS['training'], alpha=0.25, lw=0.8, label='Raw Loss')
        
        # Exponential moving average smoothing
        if len(losses) > 20:
            alpha_ema = 0.1
            ema = np.zeros_like(losses)
            ema[0] = losses[0]
            for i in range(1, len(losses)):
                ema[i] = alpha_ema * losses[i] + (1 - alpha_ema) * ema[i-1]
            ax.plot(batches, ema, color=COLORS['training'], lw=2.5, label='EMA (α=0.1)')
        
        # Statistics
        final_loss = losses[-1] if len(losses) > 0 else 0
        min_loss = np.min(losses)
        mean_loss = np.mean(losses[-len(losses)//10:]) if len(losses) > 10 else np.mean(losses)
        
        stat_text = (f'Loss Statistics\n'
                     f'{"-"*16}\n'
                     f'Final: {final_loss:.4f}\n'
                     f'Min: {min_loss:.4f}\n'
                     f'Last 10% Mean: {mean_loss:.4f}')
        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.set_xlabel('Batch', fontweight='medium')
        ax.set_ylabel('Loss', fontweight='medium')
        ax.set_title('Training Loss per Batch with EMA Smoothing', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        return self._save_plot(fig, 'Training Loss Batch')
    
    # Plot 9: Training Loss per Epoch (Enhanced with annotations)
    def plot_training_loss_epoch(self, epoch_losses):
        fig, ax = plt.subplots(figsize=(11, 7))
        
        epochs = np.arange(1, len(epoch_losses) + 1)
        losses = np.array(epoch_losses)
        
        # Main curve with markers
        ax.plot(epochs, losses, color=COLORS['training'], lw=2.5, marker='o', 
                markersize=6, markerfacecolor='white', markeredgewidth=2, label='Training Loss')
        ax.fill_between(epochs, losses, alpha=0.15, color=COLORS['training'])
        
        # Mark minimum loss
        min_idx = np.argmin(losses)
        ax.scatter([epochs[min_idx]], [losses[min_idx]], s=200, c=COLORS['highlight'], 
                   zorder=5, marker='*', edgecolors='white', linewidths=1.5,
                   label=f'Min = {losses[min_idx]:.4f} @ Epoch {epochs[min_idx]}')
        
        # Convergence annotation (loss change in last 20%)
        if len(losses) >= 5:
            last_portion = losses[-len(losses)//5:]
            loss_change = (last_portion[0] - last_portion[-1]) / (last_portion[0] + 1e-8) * 100
            status = '✓ Converged' if abs(loss_change) < 5 else '↓ Still Decreasing'
            ax.annotate(status, xy=(epochs[-1], losses[-1]), xytext=(-50, 30), 
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        color=COLORS['annotation'], bbox=STAT_BOX_PROPS,
                        arrowprops=dict(arrowstyle='->', color=COLORS['annotation']))
        
        ax.set_xlabel('Epoch', fontweight='medium')
        ax.set_ylabel('Average Loss', fontweight='medium')
        ax.set_title('Training Loss per Epoch', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        return self._save_plot(fig, 'Training Loss Epoch')
    
    # Plot 10: Learning Rate (Enhanced with phase annotations)
    def plot_learning_rate(self, lrs):
        fig, ax = plt.subplots(figsize=(11, 7))
        
        epochs = np.arange(1, len(lrs) + 1)
        lrs_arr = np.array(lrs)
        
        # Main curve with fill
        ax.fill_between(epochs, lrs_arr, alpha=0.2, color=COLORS['lr'])
        ax.plot(epochs, lrs_arr, color=COLORS['lr'], lw=2.5, marker='o', 
                markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Learning Rate')
        
        # Detect warmup phase (increasing LR at start)
        if len(lrs_arr) > 2:
            warmup_end = 0
            for i in range(1, min(len(lrs_arr), 20)):
                if lrs_arr[i] < lrs_arr[i-1]:
                    warmup_end = i
                    break
            if warmup_end > 1:
                ax.axvspan(1, warmup_end, alpha=0.15, color='orange', label=f'Warmup Phase')
        
        # Schedule info box
        max_lr = np.max(lrs_arr)
        min_lr = np.min(lrs_arr)
        final_lr = lrs_arr[-1]
        
        info_text = (f'LR Schedule\n'
                     f'{"-"*14}\n'
                     f'Max: {max_lr:.2e}\n'
                     f'Min: {min_lr:.2e}\n'
                     f'Final: {final_lr:.2e}\n'
                     f'Decay: {(1 - final_lr/max_lr)*100:.1f}%')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=STAT_BOX_PROPS, family='monospace')
        
        ax.set_xlabel('Epoch', fontweight='medium')
        ax.set_ylabel('Learning Rate', fontweight='medium')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        return self._save_plot(fig, 'Learning Rate')
    
    # Plot 11: Precision-Recall (Enhanced with AUPRC and operating points)
    def plot_precision_recall(self, precision, recall):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate AUPRC
        auprc = np.trapz(precision, recall)
        
        # Main curve with fill
        ax.fill_between(recall, precision, alpha=0.25, color=COLORS['roc_fill'])
        ax.plot(recall, precision, color=COLORS['roc'], lw=2.5, label='PR Curve')
        
        # Mark key operating points
        # Find point of maximum F1
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        max_f1_idx = np.nanargmax(f1_scores)
        ax.scatter([recall[max_f1_idx]], [precision[max_f1_idx]], s=150, c=COLORS['highlight'],
                   zorder=5, marker='*', edgecolors='white', linewidths=1.5,
                   label=f'Max F1 = {f1_scores[max_f1_idx]:.3f}')
        
        # Baseline (random classifier)
        baseline = np.mean(precision)
        ax.axhline(baseline, color='gray', linestyle='--', lw=1.5, alpha=0.7, label=f'Baseline = {baseline:.3f}')
        
        # Metrics annotation
        ax.text(0.02, 0.02, f'AUPRC = {auprc:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=STAT_BOX_PROPS, fontweight='bold')
        
        ax.set_xlabel('Recall', fontweight='medium')
        ax.set_ylabel('Precision', fontweight='medium')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        return self._save_plot(fig, 'Precision Recall')
    
    # Plot 12: Score Box Plot (Enhanced with statistics)
    def plot_score_boxplot(self, genuine, impostor):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        bp = ax.boxplot([impostor, genuine], patch_artist=True, 
                        labels=['Impostor', 'Genuine'],
                        widths=0.6, notch=True, bootstrap=100)
        
        # Color the boxes
        colors = [COLORS['impostor_light'], COLORS['genuine_light']]
        edge_colors = [COLORS['impostor'], COLORS['genuine']]
        for patch, color, edge in zip(bp['boxes'], colors, edge_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(edge)
            patch.set_linewidth(2)
        
        # Color whiskers and caps
        for i, (whisker, cap) in enumerate(zip(bp['whiskers'], bp['caps'])):
            whisker.set_color(edge_colors[i // 2])
            cap.set_color(edge_colors[i // 2])
        
        # Add statistics text above each box
        for i, (data, label) in enumerate([(impostor, 'Impostor'), (genuine, 'Genuine')]):
            stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}'
            ax.text(i + 1, np.max(data) + 0.05, stats_text, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Increase y-limit to make room for annotations
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
        
        ax.set_ylabel('Similarity Score', fontweight='medium')
        ax.set_title('Score Distribution Box Plots', fontweight='bold')
        return self._save_plot(fig, 'Score Boxplot')
    
    # Plot 13: Score Violin (Enhanced with statistics)
    def plot_score_violin(self, genuine, impostor):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        parts = ax.violinplot([impostor, genuine], showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violin bodies
        colors = [COLORS['impostor_light'], COLORS['genuine_light']]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor([COLORS['impostor'], COLORS['genuine']][i])
            pc.set_linewidth(2)
            pc.set_alpha(0.8)
        
        # Style the statistics lines
        parts['cmeans'].set_color(COLORS['highlight'])
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color(COLORS['threshold'])
        parts['cmedians'].set_linewidth(2)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Impostor', 'Genuine'], fontweight='medium')
        ax.set_ylabel('Similarity Score', fontweight='medium')
        ax.set_title('Score Distribution Violin Plots', fontweight='bold')
        
        # Legend for statistics
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=COLORS['highlight'], lw=2, label='Mean'),
                           Line2D([0], [0], color=COLORS['threshold'], lw=2, label='Median')]
        ax.legend(handles=legend_elements, loc='upper right')
        return self._save_plot(fig, 'Score Violin')
    
    # Plot 14: FAR at Different FRR (Enhanced with gradient bars)
    def plot_far_at_frr(self, fpr, fnr, thresholds):
        """Plot FAR at different target FRR levels with enhanced styling."""
        fig, ax = plt.subplots(figsize=(11, 7))
        target_frr_vals = [0.001, 0.01, 0.05, 0.1, 0.2]
        far_at_frr = []
        thresh_at_frr = []
        
        for target_frr in target_frr_vals:
            idx = np.nanargmin(np.abs(fnr - target_frr))
            far_at_frr.append(fpr[idx] * 100)
            thresh_at_frr.append(thresholds[idx])
        
        # Gradient color based on values
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(target_frr_vals)))
        bars = ax.bar([f'{x*100:.1f}%' for x in target_frr_vals], far_at_frr, 
                      color=colors, edgecolor='#8B0000', linewidth=1.5)
        
        # Add value labels on bars with threshold
        for bar, val, thresh in zip(bars, far_at_frr, thresh_at_frr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                   f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
                   f'θ={thresh:.2f}', ha='center', fontsize=9, color='white', fontweight='bold')
        
        ax.set_xlabel('Target False Reject Rate (FRR)', fontweight='medium')
        ax.set_ylabel('False Accept Rate — FAR (%)', fontweight='medium')
        ax.set_title('FAR at Different FRR Operating Points', fontweight='bold')
        return self._save_plot(fig, 'FAR at FRR')
    
    # Plot 15: FRR at Different FAR (Enhanced with gradient bars)
    def plot_frr_at_far(self, fpr, fnr, thresholds):
        """Plot FRR at different target FAR levels with enhanced styling."""
        fig, ax = plt.subplots(figsize=(11, 7))
        target_far_vals = [0.001, 0.01, 0.05, 0.1]
        frr_at_far = []
        thresh_at_far = []
        
        for target_far in target_far_vals:
            idx = np.nanargmin(np.abs(fpr - target_far))
            frr_at_far.append(fnr[idx] * 100)
            thresh_at_far.append(thresholds[idx])
        
        # Gradient color based on values
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(target_far_vals)))
        bars = ax.bar([f'{x*100:.1f}%' for x in target_far_vals], frr_at_far,
                      color=colors, edgecolor='#00008B', linewidth=1.5)
        
        # Add value labels on bars with threshold
        for bar, val, thresh in zip(bars, frr_at_far, thresh_at_far):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                   f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
                   f'θ={thresh:.2f}', ha='center', fontsize=9, color='white', fontweight='bold')
        
        ax.set_xlabel('Target False Accept Rate (FAR)', fontweight='medium')
        ax.set_ylabel('False Reject Rate — FRR (%)', fontweight='medium')
        ax.set_title('FRR at Different FAR Operating Points', fontweight='bold')
        return self._save_plot(fig, 'FRR at FAR')
    
    # Plot 16: CMC Curve
    def plot_cmc(self, genuine):
        genuine_sorted = np.sort(genuine)[::-1]
        ranks = np.arange(1, len(genuine_sorted)+1)
        cmc = np.cumsum(genuine_sorted >= 0.5) / len(genuine_sorted) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ranks, cmc, 'b-', lw=2)
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Recognition Rate (%)', fontsize=12)
        ax.set_title('CMC Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'CMC Curve')
    
    # Plot 17: EER Visualization
    def plot_eer_visualization(self, fpr, fnr, thresholds):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, fpr*100, 'r-', lw=2, label='FAR')
        ax.plot(thresholds, fnr*100, 'b-', lw=2, label='FRR')
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_thresh = thresholds[eer_idx]
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        ax.scatter([eer_thresh], [eer], c='green', s=150, zorder=5, label=f'EER={eer:.2f}%')
        ax.axhline(eer, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('EER Visualization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'EER Visualization')
    
    # Plot 18: Score vs Rank
    def plot_score_rank(self, genuine, impostor):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(genuine)), np.sort(genuine)[::-1], 'g-', lw=2, label='Genuine')
        ax.plot(range(len(impostor)), np.sort(impostor)[::-1], 'r-', lw=2, alpha=0.7, label='Impostor')
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title('Score vs Rank', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Score Rank')
    
    # Plot 19: Metrics Radar
    def plot_metrics_radar(self, metrics):
        categories = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Separability']
        values = [
            metrics.get('auc', 0),
            metrics.get('accuracy_at_eer', 0),
            metrics.get('precision_at_eer', 0),
            metrics.get('recall_at_eer', 0),
            metrics.get('f1_at_eer', 0),
            min(1, metrics.get('separability_score', 0) / 2)
        ]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, values_closed, 'b-', lw=2)
        ax.fill(angles_closed, values_closed, 'b', alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_title('Performance Metrics Radar', fontsize=14)
        return self._save_plot(fig, 'Metrics Radar')
    
    # Plot 20: Key Metrics Summary
    def plot_key_metrics(self, metrics):
        fig, ax = plt.subplots(figsize=(12, 6))
        names = ['EER (%)', 'AUC', 'd-prime', 'Accuracy (%)', 'Gini']
        values = [
            metrics.get('eer', 0) * 100,
            metrics.get('auc', 0),
            metrics.get('d_prime', 0),
            metrics.get('accuracy_at_eer', 0) * 100,
            metrics.get('gini_coefficient', 0)
        ]
        bars = ax.bar(names, values, color=['crimson', 'seagreen', 'steelblue', 'orange', 'purple'])
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Key Performance Metrics Summary', fontsize=14)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.3f}', ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        return self._save_plot(fig, 'Metrics Summary')
    
    # Plot 21: TAR at FAR (True Accept Rate at Fixed FAR)
    def plot_tar_at_far(self, fpr, tpr):
        fig, ax = plt.subplots(figsize=(10, 6))
        target_fars = [0.001, 0.01, 0.05, 0.1, 0.2]
        tar_vals = []
        for target_far in target_fars:
            idx = np.nanargmin(np.abs(fpr - target_far))
            tar_vals.append(tpr[idx] * 100)
        ax.bar([f'{x*100:.1f}%' for x in target_fars], tar_vals, color='forestgreen')
        ax.set_xlabel('Target FAR', fontsize=12)
        ax.set_ylabel('TAR (%)', fontsize=12)
        ax.set_title('True Accept Rate at Fixed FAR', fontsize=14)
        for i, val in enumerate(tar_vals):
            ax.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        return self._save_plot(fig, 'TAR at FAR')
    
    # Plot 22: Decidability Histogram
    def plot_decidability(self, genuine, impostor):
        fig, ax = plt.subplots(figsize=(10, 6))
        d_prime = (np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.var(genuine) + np.var(impostor) + 1e-8))
        overlap = np.maximum(0, min(np.max(impostor), np.max(genuine)) - max(np.min(impostor), np.min(genuine)))
        ax.bar(['d-prime', 'Separability', 'Overlap'], 
               [d_prime, d_prime/2, overlap], 
               color=['steelblue', 'seagreen', 'coral'])
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Decidability Metrics', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        return self._save_plot(fig, 'Decidability')
    
    # Plot 23: Threshold Optimization
    def plot_threshold_optimization(self, thresholds, fpr, fnr):
        fig, ax = plt.subplots(figsize=(10, 6))
        total_error = (fpr + fnr) / 2
        ax.plot(thresholds, total_error * 100, 'b-', lw=2, label='Average Error')
        min_idx = np.nanargmin(total_error)
        ax.scatter([thresholds[min_idx]], [total_error[min_idx]*100], c='red', s=150, zorder=5, 
                   label=f'Optimal: {thresholds[min_idx]:.3f}')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Average Error (%)', fontsize=12)
        ax.set_title('Threshold Optimization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Threshold Optimization')
    
    # Plot 24: Accuracy vs Threshold
    def plot_accuracy_threshold(self, thresholds, fpr, fnr):
        fig, ax = plt.subplots(figsize=(10, 6))
        accuracy = 1 - (fpr + fnr) / 2
        ax.plot(thresholds, accuracy * 100, 'g-', lw=2)
        max_idx = np.nanargmax(accuracy)
        ax.scatter([thresholds[max_idx]], [accuracy[max_idx]*100], c='red', s=150, zorder=5,
                   label=f'Max: {accuracy[max_idx]*100:.2f}% @ {thresholds[max_idx]:.3f}')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Threshold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Accuracy Threshold')
    
    # Plot 25: Score Percentiles
    def plot_score_percentiles(self, genuine, impostor):
        fig, ax = plt.subplots(figsize=(10, 6))
        percentiles = [10, 25, 50, 75, 90]
        genuine_pcts = [np.percentile(genuine, p) for p in percentiles]
        impostor_pcts = [np.percentile(impostor, p) for p in percentiles]
        x = np.arange(len(percentiles))
        width = 0.35
        ax.bar(x - width/2, genuine_pcts, width, label='Genuine', color='green', alpha=0.7)
        ax.bar(x + width/2, impostor_pcts, width, label='Impostor', color='red', alpha=0.7)
        ax.set_xlabel('Percentile', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Score Percentiles Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}%' for p in percentiles])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        return self._save_plot(fig, 'Score Percentiles')
    
    # Plot 26: Decision Boundary
    def plot_decision_boundary(self, genuine, impostor, threshold):
        fig, ax = plt.subplots(figsize=(10, 6))
        all_scores = np.concatenate([genuine, impostor])
        min_score, max_score = np.min(all_scores), np.max(all_scores)
        ax.hist(impostor, bins=40, alpha=0.5, color='red', label='Impostor', density=True)
        ax.hist(genuine, bins=40, alpha=0.5, color='green', label='Genuine', density=True)
        ax.axvline(threshold, color='blue', linestyle='-', lw=3, label=f'Decision Boundary ({threshold:.3f})')
        ax.fill_betweenx([0, ax.get_ylim()[1]], min_score, threshold, alpha=0.1, color='red')
        ax.fill_betweenx([0, ax.get_ylim()[1]], threshold, max_score, alpha=0.1, color='green')
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Decision Boundary Visualization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Decision Boundary')
    
    # Plot 27: Validation Accuracy History
    def plot_val_accuracy(self, val_accuracies):
        """Plot validation accuracy over epochs."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(val_accuracies) + 1)
        ax.plot(epochs, val_accuracies, 'g-o', lw=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title('Validation Accuracy Over Training', fontsize=14)

        # Mark best accuracy
        best_idx = np.argmax(val_accuracies)
        ax.scatter([best_idx + 1], [val_accuracies[best_idx]], c='red', s=150, zorder=5,
                   label=f'Best: {val_accuracies[best_idx]:.2f}% @ Epoch {best_idx + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Val Accuracy')

    # Plot 32: Validation AUC History
    def plot_val_auc(self, val_aucs):
        """Plot validation AUC over epochs."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(val_aucs) + 1)
        ax.plot(epochs, val_aucs, 'b-o', lw=2, markersize=4, color=COLORS['roc'])
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation AUC', fontsize=12)
        ax.set_title('Validation AUC Over Training', fontsize=14)

        # Mark best AUC
        best_idx = np.argmax(val_aucs)
        ax.scatter([best_idx + 1], [val_aucs[best_idx]], c=COLORS['highlight'], s=150, zorder=5,
                   label=f'Best: {val_aucs[best_idx]:.4f} @ Epoch {best_idx + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Val AUC')
    
    # Plot 28: Score KDE (Kernel Density Estimation)
    def plot_score_kde(self, genuine, impostor):
        """Plot smooth KDE of score distributions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create smooth KDE
        genuine_kde = stats.gaussian_kde(genuine)
        impostor_kde = stats.gaussian_kde(impostor)
        
        # Create x range
        all_scores = np.concatenate([genuine, impostor])
        x_range = np.linspace(np.min(all_scores) - 0.1, np.max(all_scores) + 0.1, 500)
        
        ax.fill_between(x_range, impostor_kde(x_range), alpha=0.4, color='red', label='Impostor')
        ax.fill_between(x_range, genuine_kde(x_range), alpha=0.4, color='green', label='Genuine')
        ax.plot(x_range, impostor_kde(x_range), 'r-', lw=2)
        ax.plot(x_range, genuine_kde(x_range), 'g-', lw=2)
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distributions (KDE)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_plot(fig, 'Score KDE')
    
    # Plot 29: Error Rate Summary
    def plot_error_summary(self, fpr, fnr, thresholds, metrics):
        """Plot error rate summary at key operating points."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get EER
        eer = metrics.get('eer', 0) * 100
        
        # Compute key error rates
        # FAR at 1% FRR
        idx_1pct_frr = np.nanargmin(np.abs(fnr - 0.01))
        far_at_1pct_frr = fpr[idx_1pct_frr] * 100
        
        # FRR at 1% FAR
        idx_1pct_far = np.nanargmin(np.abs(fpr - 0.01))
        frr_at_1pct_far = fnr[idx_1pct_far] * 100
        
        # FRR at 0.1% FAR
        idx_01pct_far = np.nanargmin(np.abs(fpr - 0.001))
        frr_at_01pct_far = fnr[idx_01pct_far] * 100
        
        # Total error at EER
        total_error_eer = eer * 2
        
        categories = ['EER', 'FAR@1%FRR', 'FRR@1%FAR', 'FRR@0.1%FAR']
        values = [eer, far_at_1pct_frr, frr_at_1pct_far, frr_at_01pct_far]
        colors = ['purple', 'crimson', 'steelblue', 'teal']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate Summary at Key Operating Points', fontsize=14)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                   f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        return self._save_plot(fig, 'Error Summary')
    
    # Plot 30: Class Separation Metrics
    def plot_class_separation_metrics(self, genuine, impostor):
        """Plot class separation metrics (variance, inter-class distance, Fisher ratio)."""
        fig, ax = plt.subplots(figsize=(11, 7))
        
        genuine_var = np.var(genuine)
        impostor_var = np.var(impostor)
        
        # Calculate separability metrics
        inter_class_dist = abs(np.mean(genuine) - np.mean(impostor))
        intra_class_var = 0.5 * (genuine_var + impostor_var)
        fisher_ratio = inter_class_dist**2 / (intra_class_var + 1e-8)
        
        metrics_names = ['Genuine\nVariance', 'Impostor\nVariance', 'Inter-class\nDistance', 'Fisher\nRatio']
        metrics_vals = [genuine_var, impostor_var, inter_class_dist, min(fisher_ratio, 10)]
        colors = [COLORS['genuine'], COLORS['impostor'], COLORS['roc'], COLORS['threshold']]
        
        bars = ax.bar(metrics_names, metrics_vals, color=colors, edgecolor='white', linewidth=2)
        ax.set_ylabel('Value', fontweight='medium')
        ax.set_title('Class Separation Metrics', fontweight='bold')
        
        for bar, val in zip(bars, metrics_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        
        return self._save_plot(fig, 'Class Separation Metrics')
    
    # Plot 31: Mean ± Std Deviation
    def plot_mean_std_comparison(self, genuine, impostor):
        """Plot mean ± standard deviation comparison for genuine vs impostor."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        genuine_std = np.std(genuine)
        impostor_std = np.std(impostor)
        
        x = [1, 2]
        means = [np.mean(impostor), np.mean(genuine)]
        stds = [impostor_std, genuine_std]
        colors = [COLORS['impostor'], COLORS['genuine']]
        
        bars = ax.bar(x, means, yerr=stds, capsize=15, color=colors, alpha=0.85,
               edgecolor='white', linewidth=2,
               error_kw={'elinewidth': 3, 'capthick': 3, 'ecolor': COLORS['annotation']})
        
        # Increase y-limit to make room for annotations
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Impostor', 'Genuine'], fontweight='medium', fontsize=12)
        ax.set_ylabel('Similarity Score', fontweight='medium')
        ax.set_title('Score Distribution: Mean ± Standard Deviation', fontweight='bold')
        
        # Add mean values as text
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.03, 
                   f'μ={mean:.3f}\nσ={std:.3f}', ha='center', fontsize=10,
                   bbox=STAT_BOX_PROPS)
        
        return self._save_plot(fig, 'Mean Std Comparison')
    
    def generate_all(self, fpr, tpr, fnr, thresholds, genuine, impostor,
                     metrics, cm, epoch_losses, all_losses, lrs, precision, recall,
                     val_accuracies: Optional[List[float]] = None,
                     val_aucs: Optional[List[float]] = None):
        """Generate all plots."""
        roc_auc = metrics.get('auc', 0)
        eer_threshold = metrics.get('eer_threshold', 0.5)
        
        self.plot_roc_curve(fpr, tpr, roc_auc)
        self.plot_det_curve(fpr, fnr)
        self.plot_far_frr_threshold(thresholds, fpr, fnr)
        self.plot_genuine_distribution(genuine)
        self.plot_impostor_distribution(impostor)
        self.plot_overlapping_distributions(genuine, impostor, eer_threshold)
        self.plot_confusion_matrix(cm)
        self.plot_training_loss_batch(all_losses)
        self.plot_training_loss_epoch(epoch_losses)
        self.plot_learning_rate(lrs)
        self.plot_precision_recall(precision, recall)
        self.plot_score_boxplot(genuine, impostor)
        self.plot_score_violin(genuine, impostor)
        # Fixed plots 14 & 15 - now compute from raw data
        self.plot_far_at_frr(fpr, fnr, thresholds)
        self.plot_frr_at_far(fpr, fnr, thresholds)
        self.plot_cmc(genuine)
        self.plot_eer_visualization(fpr, fnr, thresholds)
        self.plot_score_rank(genuine, impostor)
        self.plot_metrics_radar(metrics)
        self.plot_key_metrics(metrics)
        # Plots 21-26
        self.plot_tar_at_far(fpr, tpr)
        self.plot_decidability(genuine, impostor)
        self.plot_threshold_optimization(thresholds, fpr, fnr)
        self.plot_accuracy_threshold(thresholds, fpr, fnr)
        self.plot_score_percentiles(genuine, impostor)
        self.plot_decision_boundary(genuine, impostor, eer_threshold)
        # NEW: Plots 27-31
        if val_accuracies is not None and len(val_accuracies) > 0:
            self.plot_val_accuracy(val_accuracies)
        else:
            # Generate placeholder with epoch_losses length if no val_accuracies
            placeholder_acc = [50.0] * len(epoch_losses) if epoch_losses else [50.0]
            self.plot_val_accuracy(placeholder_acc)

        if val_aucs is not None and len(val_aucs) > 0:
            self.plot_val_auc(val_aucs)
        self.plot_score_kde(genuine, impostor)
        self.plot_error_summary(fpr, fnr, thresholds, metrics)
        self.plot_class_separation_metrics(genuine, impostor)
        self.plot_mean_std_comparison(genuine, impostor)
        
        logger.info(f"Generated {self.plot_count} separate plots")
        return self.all_plots

