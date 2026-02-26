
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sota" / "fingerprint"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILES = [
    "train_Fingerprint_20260110_191436.log",
    "train_Fingerprint_20260111_040507.log",
    "train_Fingerprint_20260111_002958.log", # Note: Check timestamp order
    "train_Fingerprint_20260111_093804.log"
]

# Sorting logs by timestamp ensures correct continuity
# 191436: Epoch 1-6
# 002958: Epoch 7 (Failed/Short) -> Wait, 040507 seems to cover 7-21. 
# Let's trust the file modification times or content.
# Based on my previous read:
# 191436: Epoch 1-6
# 040507: Resumed from 6 -> Epoch 7-21
# 093804: Resumed from 21 -> Epoch 22-23

ORDERED_LOGS = [
    "train_Fingerprint_20260110_191436.log",
    "train_Fingerprint_20260111_040507.log",
    "train_Fingerprint_20260111_093804.log"
]

def parse_logs(log_files):
    batch_losses = []
    epoch_losses = []
    lrs = []
    val_accs = []
    
    # Regex patterns
    # Batch: "Epoch 1/100 Batch 200/1397 Loss: 9.5519 Acc: 0.00%"
    batch_pattern = re.compile(r"Epoch (\d+)/100 Batch (\d+)/(\d+) Loss: ([\d.]+) Acc: ([\d.]+)%")
    # Epoch Avg: "Epoch 1/100 Avg Loss: 8.9726 Val Acc: 1.23% LR: 0.000011"
    epoch_pattern = re.compile(r"Epoch (\d+)/100 Avg Loss: ([\d.]+) Val Acc: ([\d.]+)% LR: ([\d.]+)")
    
    global_batch_counter = 0
    
    for log_file in log_files:
        path = LOG_DIR / log_file
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
            
        print(f"Parsing {log_file}...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Remove ANSI codes
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                
                # Batch Loss
                m_batch = batch_pattern.search(clean_line)
                if m_batch:
                    loss = float(m_batch.group(4))
                    batch_losses.append(loss)
                
                # Epoch Stats
                m_epoch = epoch_pattern.search(clean_line)
                if m_epoch:
                    epoch = int(m_epoch.group(1))
                    avg_loss = float(m_epoch.group(2))
                    val_acc = float(m_epoch.group(3))
                    lr = float(m_epoch.group(4))
                    
                    # Avoid duplications if logs overlap (simple check)
                    if len(epoch_losses) == 0 or epoch > len(epoch_losses):
                        epoch_losses.append(avg_loss)
                        val_accs.append(val_acc)
                        lrs.append(lr)
    
    return batch_losses, epoch_losses, val_accs, lrs

def plot_training_loss(batch_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses, lw=0.5, alpha=0.7, color='#2ca02c')
    # Moving average
    if len(batch_losses) > 50:
        window = 50
        ma = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(batch_losses)), ma, color='darkgreen', lw=1.5, label='Moving Avg (50)')
    
    plt.title("Training Loss (Batch-level Reconstruction)", fontsize=14, fontweight='bold')
    plt.xlabel("Batch Steps", fontsize=12)
    plt.ylabel("ArcFace Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fixed_training_loss_batch.png", dpi=300)
    print("Saved fixed_training_loss_batch.png")

def plot_learning_rate(lrs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lrs)+1), lrs, 'm-o', lw=2)
    plt.title("Learning Rate Schedule (Recovered)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fixed_learning_rate.png", dpi=300)
    print("Saved fixed_learning_rate.png")

def plot_val_accuracy(val_accs):
    epochs = range(1, len(val_accs)+1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accs, 'b-o', lw=2)
    
    max_acc = max(val_accs)
    max_epoch = val_accs.index(max_acc) + 1
    plt.plot(max_epoch, max_acc, 'r.', markersize=15, label=f"Peak: {max_acc}%")
    
    plt.title("Validation Accuracy (Full History)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fixed_val_accuracy_v2.png", dpi=300)
    print("Saved fixed_val_accuracy_v2.png")

# Main Execution
batch_losses, epoch_losses, val_accs, lrs = parse_logs(ORDERED_LOGS)

print(f"Found {len(batch_losses)} batch stats")
print(f"Found {len(val_accs)} epochs")

if len(batch_losses) > 0:
    plot_training_loss(batch_losses)
    plot_learning_rate(lrs)
    plot_val_accuracy(val_accs)
else:
    print("No data found! Check log filenames.")
