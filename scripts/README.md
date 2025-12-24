# Scripts

Helper utilities for dataset download, general training, etc.

## SOTA Training Scripts

SOTA training modules are now located in `src/models/`:

- `src/models/fingerprint/train_sota.py` - Fingerprint (ResNet50 + ArcFace + Cutout)
- `src/models/iris/train_sota.py` - Iris (ResNet50 + ArcFace + GridMask)
- `src/models/face/train_sota.py` - Face (to be implemented)

## Usage

### Fingerprint
```bash
.\venv\Scripts\python -m src.models.fingerprint.train_sota --epochs 5 --batch-size 48 --sample-fraction 0.2
```

### Iris
```bash
.\venv\Scripts\python -m src.models.iris.train_sota --epochs 5 --batch-size 48 --sample-fraction 0.2
```

## Hardware Requirements
- GPU with 4GB+ VRAM (RTX 3050 tested)
- 16GB+ RAM recommended
- Windows: `num_workers=0` required for DataLoader
