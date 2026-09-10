# Oil Spill Detection System

A deep learning project for industrial oil spill detection using satellite/aerial imagery. This system provides both **Classification** (detecting if a patch contains oil) and **Segmentation** (pixel-level masking of the spill).

## Features
- **Binary Classification**: Robustly differentiates oil spills from clean ocean textures using MobileNetV3-Large.
- **Hard Negative Sampling**: Specifically trained on challenging ocean textures like wave foam and coastlines.
- **3-Zone Labeling**: Clean supervision logic that eliminates ambiguous samples during training.
- **Semantic Segmentation**: Pixel-level spill delineation using U-Net with ResNet34 backbone.
- **Dual-Threshold Inference**: Flexible decision logic for identifying "Possible" vs "High Confidence" threats.

## Project Structure
```text
project_root/
├── src/
│   ├── data/       # Data preparation & split scripts
│   ├── training/   # Model training pipelines
│   └── inference/  # Prediction & visualization tools
├── data/           # (Ignored) Dataset splits & raw data
├── models/         # (Ignored) Trained .pth checkpoints
├── outputs/        # (Ignored) Per-epoch logs & visualizations
├── requirements.txt
└── README.md
```

## Setup
> **Requirement**: Python 3.10+ is required (tested on Python 3.12).

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
To build the classification dataset from segmentation splits with hard negative sampling:
```bash
python src/data/build_classification.py
```

### 2. Training
Standardized training for both pipelines:
```bash
python src/training/train_classification.py
python src/training/train_segmentation.py
```

### 3. Inference
Run prediction on a single image using the dual-threshold strategy:
```bash
python src/inference/predict.py --image path/to/image.png --mode dual
```

## Evaluation
- Detailed metrics (IoU, Dice, F1, Recall) are exported to `outputs/` after training.
- Visualization tools for error analysis are available in `src/tools/visualize_classification.py`.
