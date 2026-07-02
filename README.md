# CLARA---Chest-Lung-Abnormilities-Recognition-AI
CLARA: Chest Lung Abnormality Recognition AI — Multi-label chest X-ray abnormality detection using DenseNet121,ResNet50, and EfficientNetB4

# CLARA — Chest Lung Abnormality Recognition AI

[![Conference](https://img.shields.io/badge/Conference-DMCE--GTC%202026-blue)]([(https://dmcegtc.csidmce.com)])
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-green)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
[![Python](https://img.shields.io/badge/Python-3.12-yellow)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

> **Accepted and presented at DMCE-GTC 2026 — International Conference on Computing and IT Advancements, organised in association with the Computer Society of India.**

---

## Overview

CLARA is a deep learning framework for automated **multi-label detection of 14 thoracic abnormalities** from chest X-ray images. It presents a systematic and fair comparative study of three CNN architectures — **DenseNet121**, **ResNet50**, and **EfficientNetB4** — trained on the NIH ChestX-ray14 dataset under strictly identical experimental conditions.

A key methodological contribution is the use of **strict patient-level data partitioning**, eliminating the information leakage prevalent in prior work where image-level random splits allow the same patient to appear in both training and test sets.

---

## Key Results

| Model | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| **DenseNet121** | **0.821** | **0.992** | 0.145 | 0.114 |
| ResNet50 | 0.764 | 0.676 | **0.709** | **0.173** |
| EfficientNetB4 | 0.705 | 0.684 | 0.626 | 0.147 |
| CheXNet (Reference) | 0.841 | N/A | N/A | N/A |

### DenseNet121 Surpasses CheXNet on Key Diseases

| Disease | CLARA (DenseNet121) | CheXNet |
|---|---|---|
| Effusion | **0.882** | 0.864 ✓ |
| Consolidation | **0.781** | 0.790 ✓ |
| Atelectasis | **0.806** | 0.809 ≈ |
| Fibrosis | **0.798** | 0.805 ≈ |

> ✓ Surpasses CheXNet &nbsp;&nbsp; ≈ Matches within 0.003

---

## Novel Contributions

- **Patient-level data split** — zero patient overlap between train/validation/test sets
- **Custom weighted binary cross-entropy** — per-disease weights (Hernia weight ≈ 476) compensating severe class imbalance
- **Morphological preprocessing** — erosion, dilation, and top-hat transform enhancing lung structure visibility
- **Adaptive per-disease thresholds** — learned on validation set, replacing fixed 0.5 cutoff used in all prior work
- **Fair 3-architecture comparison** — identical loss, optimizer, augmentation, and evaluation across all models
- **Grad-CAM interpretability** — clinically validated heatmaps for all 14 disease classes

---

## Dataset

**NIH ChestX-ray14** — [Download from Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

| Property | Value |
|---|---|
| Total Images | 112,120 |
| Unique Patients | 30,805 |
| Disease Labels | 14 |
| Train Split | 78,377 images (70% patients) |
| Validation Split | 11,134 images (10% patients) |
| Test Split | 22,609 images (20% patients) |

**14 Disease Classes:**
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

---

## System Architecture

```
Input X-ray (PNG)
      ↓
Preprocessing
  • Resize 320×320
  • Morphological: Erosion → Dilation → Top-Hat
  • Samplewise normalization (zero mean, unit variance)
  • Augmentation (train only): flip, rotate ±5°, zoom ±5%
      ↓
Base CNN (ImageNet pretrained, top removed)
  ┌──────────────┬──────────────┬─────────────────┐
  │ DenseNet121  │   ResNet50   │ EfficientNetB4  │
  │  121 layers  │  50 layers   │   47 layers     │
  │    7M params │   25M params │   19M params    │
  └──────────────┴──────────────┴─────────────────┘
      ↓
GlobalAveragePooling2D
(batch, 10, 10, 1024) → (batch, 1024)
      ↓
Dropout (0.1)
      ↓
Dense (14, sigmoid)
14 independent probabilities [0, 1]
      ↓
Output: Disease scores + Grad-CAM heatmaps
```

---

## Installation

```bash
git clone https://github.com/yourusername/clara-chest-xray.git
cd clara-chest-xray
pip install -r requirements.txt
```

---

## Usage

### 1. Setup Dataset on Kaggle
```python
# Dataset path on Kaggle
DATASET_ROOT = "/kaggle/input/datasets/organizations/nih-chest-xrays/data"
```

### 2. Run Notebooks in Order
```
notebooks/
├── clara_densenet121.ipynb     # Main model — best results
├── clara_resnet50.ipynb        # Balanced performance model
└── clara_efficientnetb4.ipynb  # Lightweight model
```

### 3. Custom Weighted Loss
```python
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            lp = -1 * K.mean(pos_weights[i] * y_true[:, i] *
                              K.log(y_pred[:, i] + epsilon))
            ln = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) *
                              K.log(1 - y_pred[:, i] + epsilon))
            loss += lp + ln
        return loss
    return weighted_loss
```

### 4. Adaptive Thresholds
```python
# Find optimal threshold per disease on validation set
for i, label in enumerate(labels):
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.05, 0.95, 0.01):
        yp = (val_preds[:, i] > thresh).astype(int)
        f1 = f1_score(val_true[:, i], yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    thresholds[label] = round(best_thresh, 2)
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001 → 0.0001) |
| Batch Size | 32 |
| Image Size | 320 × 320 |
| Hardware | 2× NVIDIA Tesla T4 (Kaggle) |
| Strategy | TensorFlow MirroredStrategy |
| DenseNet121 Epochs | 15 (8 + 7 continued) |
| ResNet50 Epochs | 15 (8 + 7 continued) |
| EfficientNetB4 Epochs | 15 (8 + 7 continued) |
| Early Stopping Patience | 8 |

---

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{clara2026,
  title     = {CLARA: Chest Lung Abnormality Recognition AI —
               A Comparative Study of DenseNet121, ResNet50,
               and EfficientNetB4 on NIH ChestX-ray14},
  author    = {Jadhav, Mansi and Mundas, Akshat and
               Nehate, Paras and Nehete, Khushi},
  booktitle = {DMCE-GTC 2026 International Conference on
               Computing and IT Advancements},
  year      = {2026},
  organization = {Computer Society of India}
}
```



## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Accepted and presented at DMCE-GTC 2026 — International Conference on Computing and IT Advancements, organised in association with the Computer Society of India.*
