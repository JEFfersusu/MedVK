# MedVK

**Official PyTorch implementation of "MedVK: Efficient Medical Image Classification via Decoupled Kolmogorov-Arnold Networks"**

<p align="center">
  <img src="asset/fig1.png" width="800px" />
</p>

---

## Overview

MedVK is a lightweight and expressive framework for medical image classification, built on a decoupled Kolmogorov–Arnold Network (KAN). Unlike traditional CNNs, Transformers, or Mamba-based models that rely on fixed activations and coupled feature modeling, MedVK introduces spline-driven nonlinearities and a multi-branch design to improve adaptability, interpretability, and efficiency, especially on complex or small-scale medical datasets.

**Key Challenges:**

❗ Fixed activation functions in CNNs/Transformers fail to adapt to diverse lesion characteristics and subtle anatomical features.

❗ Coupled spatial-channel modeling blurs local texture and global semantic boundaries, harming fine-grained classification.

❗ Overhead of attention-heavy models restricts real-world deployment in clinical or resource-constrained settings.

**Our Solution:**

✅ Replace fixed activations with B-spline-based KAN nonlinearities for adaptive, data-driven representation.

✅ Design a decoupled architecture (KANFormer) that separates spatial and channel-wise modeling into independent, specialized branches.

✅ Introduce a KANFusion module for hierarchical multi-scale feature aggregation with minimal cost.

✅ Provide three variants (Tiny, Small, Base) for flexible deployment across devices and constraints.


## Architecture

<div align="center">

![MedVK Framework](asset/fig2.png)
_*Overall Architecture of MedVK.*_

</div>



### Key Features

**Decoupled multi-branch design:** Explicitly separates spatial continuity from channel dependency.

**Spline-driven activations:** Enables data-adaptive modeling with smooth and interpretable nonlinearities.

**Ultra-efficient:** Achieves SOTA performance with up to 30× fewer GFLOPs than prior models.

**Model variants:** Choose from MedVK-T, MedVK-S, and MedVK-B based on your accuracy–efficiency needs.

**Robust across** modalities: Validated on X-ray, ultrasound, dermatoscopy, and retinal imaging.

**Interpretable:** Produces focused Grad-CAM heatmaps on lesion areas with improved localization.

###  Main Contributions

✨ Propose KANFormer, a decoupled vision architecture with spline-enhanced branches for spatial and channel modeling.

✨ Introduce MedVK, integrating KANFormer with a lightweight KANFusion module for effective multi-stage representation fusion.

✨ Achieve SOTA performance on six diverse medical image datasets while being 10–30× more efficient than transformer-based baselines.

✨ Provide a comprehensive ablation study and visualization analysis, validating both effectiveness and interpretability.

---

---

##  Installation

### Prerequisites

- Python 3.10 (Ubuntu22.04)
- CUDA 11.8
- PyTorch 2.12

### Step-by-Step Installation

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio
pip install timm==0.9.16 packaging==23.0

pip install pytest==8.3.5 chardet==4.0.0 yacs==0.1.8 termcolor==2.4.0
pip install scikit-learn==1.3.2 matplotlib==3.7.1
pip install SimpleITK scikit-image PyWavelets==1.4.1
```

---

##  Performance Results

MedVK achieves state-of-the-art performance across multiple medical imaging benchmarks. Results shown as **Tiny version** / **Large version**.

<div align="center">

| Dataset | Classes | Imaging Modality | F1-Score (%) | OA (%) | AUC (%) | Kappa (%) |
|:--------|:-------:|:--------------------------:|:----------------------:|:----------------------:|:------------------:|:------------------:|
| **[BloodMNIST](https://medmnist.com/)** | 8 | Blood Cell Microscope | 98.7 / 98.6 / 98.5 | 98.6 / 98.5 / 98.5 | 99.9 / 99.9 / 99.9 | 98.4 / 98.3 / 98.2 |
| **[BreastMNIST](https://medmnist.com/)** | 2 | Breast Ultrasound | 78.3 / 80.9 / 79.0 | 84.6 / 85.9 / 85.3 | 86.3 / 86.7 / 88.1 | 57.0 / 61.9 / 58.5 |
| **[DermaMNIST](https://medmnist.com/)** | 7 | Dermatoscope | 61.6 / 63.4 / 63.1 | 80.3 / 81.0 / 80.9 | 94.2 / 94.8 / 94.6 | 61.5 / 63.8 / 62.0 |
| **[PneumoniaMNIST](https://medmnist.com/)** | 2 | Chest X-Ray | 96.6 / 97.1 / 96.6 | 96.8 / 97.3 / 96.8 | 99.2 / 99.6 / 99.0 | 93.1 / 94.1 / 93.1 |
| **[RetinaMNIST](https://medmnist.com/)** | 5 | Fundus Camera | 39.9 / 36.2 / 39.9 | 56.5 / 57.0 / 57.8 | 76.0 / 74.3 / 75.7 | 37.3 / 37.6 / 40.0 |
| **[CPN X-ray](https://example.com/)** | 3 | Chest X-ray | 96.3 / 96.6 / 96.8 | 96.3 / 96.6 / 96.8 | 99.5 / 99.5 / 99.4 | 94.4 / 94.8 / 95.1 |

</div>

> **Note:** Pre-trained model weights will be released soon. Stay tuned for updates!

---

##  Quick Start

### Training Your Model

```bash
# Basic training command
python train.py \
    --model MedVK_T \
    --dataset PneumoniaMNIST \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### Model Evaluation

```bash
# Evaluate single model
python test.py \
    --dataset PneumoniaMNIST \
    --model MedVK_T \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 32
```
---

##  Visualization Results

### Attention Heatmaps

<div align="center">

![Grad-CAM Heatmap1](asset/fig3.png)
![Grad-CAM Heatmap2](asset/fig4.png)
_* Grad-CAM visualization showing model attention on medical images.*_
</div>

Our visualizations demonstrate that MedVK effectively focuses on clinically relevant regions, providing interpretable results for medical professionals.
