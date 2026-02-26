
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Reconstructed Data from Logs
epochs = list(range(1, 24))
val_accs = [
    1.23, 90.59, 98.60, 98.97, 99.04, 99.32,  # Run 1 (191436.log)
    99.04, 99.29, 99.21, 99.44, 99.36, 99.48, 99.58, 99.57, 99.71, 99.60, 99.66, 99.60, 99.64, 99.64, 99.68, # Run 2 (040507.log)
    99.74, 99.68 # Run 3 (093804.log)
]

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "outputs" / "sota" / "fingerprint"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_accs, 'g-o', linewidth=2, markersize=6)
plt.title("Validation Accuracy Over Training (Recovered)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation Accuracy (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

# Highlight max
max_acc = max(val_accs)
max_epoch = epochs[val_accs.index(max_acc)]
plt.plot(max_epoch, max_acc, 'r.', markersize=20, label=f"Best: {max_acc}% @ Epoch {max_epoch}")
plt.legend()

output_path = output_dir / "fixed_val_accuracy.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved recovered plot to {output_path}")
