# NFCD: Normalizing Flow for Change Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.1+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Description

This project implements a **semi-supervised change detection framework** for remote sensing images using **Normalizing Flow (NF)** to generate pseudo labels. The framework employs a three-stage training strategy to effectively leverage both labeled and unlabeled data for improved change detection performance.

### Key Features

- 🔥 **Three-Stage Training Pipeline**: Progressive training strategy for optimal performance
- 🎯 **Normalizing Flow Integration**: Generates high-quality pseudo labels for unlabeled data
- 🏗️ **Multiple Backbone Support**: ResNet50, ResNet101, HRNet architectures
- 📊 **Multiple Dataset Support**: CDD, LEVIR-CD, WHU-CD datasets
- 🔄 **Semi-Supervised Learning**: Efficient utilization of limited labeled data
- 📈 **Consistency Regularization**: Feature alignment loss for robust predictions

## 🗂️ Project Structure

```
nfcd/
├── base/                   # Base classes for datasets, models, and trainers
│   ├── base_dataloader.py
│   ├── base_dataset.py
│   ├── base_model.py
│   └── base_trainer.py
├── configs/               # Configuration files for different datasets
│   ├── config_CDD.json
│   ├── config_LEVIR.json
│   └── config_WHU.json
├── dataloaders/           # Data loading and preprocessing
│   ├── CDDataset.py
│   └── ...
├── models/                # Model architectures
│   ├── backbones/        # Backbone networks (ResNet, HRNet)
│   ├── decoder.py        # Decoder modules
│   ├── encoder.py        # Encoder modules
│   ├── nf.py            # Normalizing Flow implementation
│   ├── NF_ResNet50_CD.py
│   └── ...
├── utils/                 # Utility functions
│   ├── helpers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── visualize.py
│   └── ...
├── train.py              # Training script
├── trainer.py            # Trainer implementation
├── inference.py          # Inference script
├── visual.py             # Visualization tools
└── requirements.txt      # Python dependencies
```

## 📦 Dataset Information

### Supported Datasets

1. **CDD (Change Detection Dataset)**
2. **LEVIR-CD** (LEVIR Change Detection Dataset)
3. **WHU-CD** (WHU Building Change Detection Dataset)

### Dataset Structure

Organize your dataset as follows:

```
DATA/
├── CDD/                  # or LEVIR/WHU
│   ├── A/               # Pre-change images
│   ├── B/               # Post-change images
│   ├── label/           # Ground truth labels
│   └── list/            # Train/val/test split files
│       ├── train_supervised.txt
│       ├── train_unsupervised.txt
│       ├── val.txt
│       └── test.txt
```

## 🛠️ Requirements

### Dependencies

```bash
Python >= 3.7
PyTorch >= 1.1.0
torchvision
numpy >= 1.16.3
matplotlib >= 3.1.1
opencv-python >= 4.1.1.26
tensorboard
tqdm >= 4.38.0
scikit-image >= 0.15.0
scipy
FrEIA  # For Normalizing Flow
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nfcd.git
cd nfcd
```

2. **Create a virtual environment (recommended)**
```bash
conda create -n nfcd python=3.8
conda activate nfcd
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install FrEIA (Normalizing Flow library)**
```bash
pip install git+https://github.com/VLL-HD/FrEIA.git
```

## 🚀 Usage Instructions

### 1. Configuration

Edit the configuration file for your dataset (e.g., `configs/config_CDD.json`):

```json
{
  "name": "NFCD",
  "percent": 5,                    // Percentage of labeled data (5%, 10%, 20%)
  "model": {
    "backbone": "ResNet50",        // Backbone: ResNet50, ResNet101, HRNet (NF kept for legacy configs)
    "confidence_thr": 0.95,        // Confidence threshold for pseudo labels
    "nf_weight": 0.7               // Weight for NF loss
  },
  "train_supervised": {
    "data_dir": "/path/to/dataset",
    "batch_size": 4,
    "crop_size": 256
  }
}
```

### 2. Training

The training process consists of three stages:

#### Stage 1: Base Model Training
```bash
python train.py \
    --config configs/config_CDD.json \
    --gpu 0 \
    --aug_type all
```

#### Stage 2: Normalizing Flow Training
The framework automatically trains the Normalizing Flow model after Stage 1.

#### Stage 3: Pseudo Label Refinement
The model generates and refines pseudo labels using the trained NF model.

**Control training stages** by modifying the `process` parameter in `config.json`:
- `[1]`: Only Stage 1
- `[1, 2]`: Stages 1 and 2
- `[1, 2, 3, 4]`: All stages (full pipeline)

### 3. Inference

Run inference on test data:

```bash
python inference.py \
    --config configs/config_CDD.json \
    --model /path/to/best_model.pth \
    --Dataset_Path /path/to/test/dataset \
    --save
```

### 4. Visualization

Visualize predictions:

```bash
python visual.py \
    --config configs/config_CDD.json \
    --model /path/to/best_model.pth \
    --Dataset_Path /path/to/dataset \
    --method NF
```

## 🧪 Methodology

### Three-Stage Training Strategy

#### **Stage 1: Supervised Baseline Training**
- Train the base change detection model using labeled data
- Apply consistency regularization between weak and strong augmentations
- Loss: `L_total = L_supervised + L_consistency + L_alignment`

#### **Stage 2: Normalizing Flow Training**
- Freeze the base model
- Train Normalizing Flow decoders on multi-scale features
- Learn probability distributions of unchanged pixels
- Generate anomaly scores for change detection

#### **Stage 3: Pseudo Label Generation & Refinement**
- Generate pseudo labels using trained NF model
- Apply confidence-based filtering
- Refine predictions using connected component analysis
- Fine-tune the model with combined labeled and pseudo-labeled data


## 📁 Model Checkpoints

Trained models are saved in:
```
outputs/
├── DATASET_NAME/
│   ├── stage1/
│   │   └── best_model_thr-0.95.pth
│   ├── stage2/
│   │   └── nf/best_model_nf_decoders.pth
│   ├── fake_labels/
│   │   ├── Label_batch_0.pt
│   │   ├── noLabel_batch_0.pt
│   │   └── ...
│   └── stage3/
│       └── weightXX/
│           └── best_model.pth
```
Pseudo labels are shared across all weight settings and live directly under `fake_labels`; only the stage3 checkpoints remain weight-specific.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
