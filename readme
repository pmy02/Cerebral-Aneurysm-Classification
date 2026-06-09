<!-- Language switcher -->
**English** | [한국어](README.ko.md)

# Cerebral Aneurysm Classification

A deep-learning pipeline that detects the **presence** and **anatomical location** of cerebral aneurysms from anonymized cerebral angiography images.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c)
![Task](https://img.shields.io/badge/task-medical%20image%20classification-informational)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

> ⚠️ **Research / educational project (Jun–Jul 2023).** This is not a clinical decision-support tool and has not been validated for diagnostic use. Do not use it for patient care.

---

## Overview

Cerebral aneurysms are localized dilations of intracranial arteries whose rupture can be life-threatening, so early detection on angiography is clinically important. This project frames detection as two coupled learning problems on anonymized cerebral angiography images:

1. **Aneurysm detection** — a binary classification of whether an aneurysm is present.
2. **Location classification** — a multi-label classification of *which* arterial segments contain an aneurysm.

Each patient is represented by 8 angiography views. Because no single view exposes every arterial segment, the images and labels are split along the brain's anterior/posterior circulation before training.

<!-- TODO: migrate figures into a committed assets/ folder and use relative paths, e.g. ![Goal](assets/goal.png), so the README renders even outside this repo -->
![Project goal](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/aaab8df7-92f5-4f89-b43a-7a954261ea40)

## Task formulation

Brain circulation divides into anterior (internal carotid, ICA) and posterior (vertebral, VA) systems, so labels are grouped accordingly:

| Circulation | Backbone vessel | Segments (labels) |
|-------------|-----------------|-------------------|
| Anterior    | ICA             | ICA, AntChor, ACA, ACOM, MCA |
| Posterior   | VA              | VA, PICA, SCA, BA, PCA, PCOM |

This yields **four datasets**: a binary set and a multi-label set, each split into an anterior and a posterior subset.

## Pipeline

![Pipeline](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/becfd58a-7d74-4cb5-ac28-f3407dc530ad)

The end-to-end flow is: preprocess images → split by circulation → run binary detection → for positive cases, run multi-label location classification → aggregate per-patient predictions.

## Data and preprocessing

![Dataset construction](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/e0f9f236-abef-4d41-a3b9-b2c0a522d707)

**Label construction (multi-label / location).** Image paths are separated into anterior/posterior using the I/V marker. From `train.csv`, columns are reassigned per L/R and I/V into new anterior and posterior dataframes. Because each patient contributes 8 views split front/back, these frames are ~4× the original row count. Left/right are merged at the label level but resolved again at inference, since the same label can be positive or negative depending on the L/R view.

**Label construction (binary / detection).** For each row, the binary label is 1 if any location label is positive, else 0 — producing `binary_anterior` and `binary_posterior` to complement the multi-label `anterior` and `posterior` sets.

**Image preprocessing.** Training images came in three forms, handled as follows:
- *No margin or text* — used as-is.
- *Text + margin* — margins removed by pixel value; text sits at fixed coordinates, so a fixed region is masked and filled with the mean background color sampled from an adjacent region.
- *Margin only* — margin size varies per image, so horizontal/vertical lines through the center are scanned to find where gray begins and the image is cropped there.

> The underlying dataset is private, anonymized medical imaging and is **not** distributed here. See [Reproducibility](#reproducibility) for the expected input format.

## Method

![Model architecture](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/57600112-1e36-4d0b-9d0c-9761659d3bae)

**Binary detection (shared across anterior/posterior).**
- Backbone: **MedNet** — a ResNet-18 initialized with grayscale medical pretrained weights ([MedicalNet-Resnet18](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/blob/main/resnet_18.pth)).
- Loss: `BCELoss`. Augmentation: resize, normalize. Additional ops: `cv2.bitwise_not`, `cv2.medianBlur`.

**Multi-label location (separate models per circulation).**
- Anterior: **SwinV2** — `timm` pretrained `swinv2_cr_tiny_ns_224`.
- Posterior: **ResNet-18** — `torchvision` pretrained.
- Loss: **Asymmetric Loss** ([Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)). Augmentation: resize, normalize.

**Decision rule.**
1. Run binary detection; rows above threshold are treated as aneurysm-bearing, others have all location labels set to 0. The threshold is the median of binary predictions over `train.csv`.
2. For positive rows, run multi-label classification; each location is thresholded at the 90th percentile of that location's predictions over `train.csv` (21 thresholds).
3. Predictions have shape 9016×1 (8 views per patient). Anterior and posterior are each averaged over their 4 views, and the larger of the two averages decides the patient-level aneurysm score.

## Pretrained models

The `Model/` directory holds the trained checkpoints:

| File | Role | Backbone |
|------|------|----------|
| `MedNet_ant_binary.pt`        | Anterior binary detection  | ResNet-18 (MedicalNet) |
| `MedNet_pos_binary.pt`        | Posterior binary detection | ResNet-18 (MedicalNet) |
| `SwinNet_ant_multilabel.pt`   | Anterior location          | SwinV2 (timm) |
| `ResNet_pos_multilabel.pt`    | Posterior location         | ResNet-18 (torchvision) |
| `resnet_18.pth`               | MedicalNet base weights    | ResNet-18 |

> **Repo hygiene note.** `SwinNet_ant_multilabel.pt` and `resnet_18.pth` are tracked via Git LFS, but the other three checkpoints are committed as raw binaries with no `.gitattributes`. Track all model files with LFS consistently (a `.gitattributes` is provided) to keep the history small.

## Results

No headline metrics are reported yet for the full pipeline. Fill these in from your experiments:

<!-- TODO: replace with real numbers from your runs -->
| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Binary detection (anterior)  | MedNet  | F1 / AUROC | — |
| Binary detection (posterior) | MedNet  | F1 / AUROC | — |
| Location (anterior)          | SwinV2  | macro-F1   | — |
| Location (posterior)         | ResNet-18 | macro-F1 | — |

## Explainability

Grad-CAM was applied (on text/margin-removed images, without augmentation) to inspect which regions drove a ResNet-18's decisions. The heatmaps were interpretable but, applied without ground-truth accuracy as a reference, were used for qualitative inspection rather than rigorous attribution.

![Grad-CAM example](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/df6a2bb5-2efd-476e-8871-a1934169d5e7)

## Exploratory experiments

Approaches that were tried and what they showed — reported honestly, including negative results:

- **Rule-based crop** around positive-label locations. Dropped: when ≥2 aneurysms share a segment, the location label is still a single 1, so exact positions could not be recovered.
- **Split by position / angle / direction**, training per subset with weighted sampling and class weights. Class imbalance dominated for small classes; weighting gave only marginal gains.
- **Self-supervised pretraining.** An autoencoder (200 epochs) to learn imbalance-robust features before supervised binary classification; and contrastive learning over ResNet-50 features to pull same-class images together. Imbalance still limited supervised performance.
- **ResNet-18 with fc→conv head** (to preserve spatial information). F1 stayed at a constant 53.68% across epochs.
- **DenseNet** for stronger feature extraction.

## Repository structure

```
Cerebral-Aneurysm-Classification/
├── Model/                  # trained checkpoints (see Pretrained models)
├── .gitattributes          # Git LFS tracking for model files (recommended, provided)
├── README.md               # English (this file)
└── README.ko.md            # Korean
```

> **The training and inference code is not yet in the repository.** Adding it is the single highest-impact improvement — it turns this from a documented result into a reproducible project. A suggested layout: `src/` (data, models, losses), `scripts/` (`train.py`, `infer.py`, `gradcam.py`), `configs/`.

## Getting started

<!-- TODO: replace with real instructions once training/inference code is added -->
```bash
git clone https://github.com/pmy02/Cerebral-Aneurysm-Classification.git
cd Cerebral-Aneurysm-Classification
git lfs install && git lfs pull          # fetch LFS-tracked checkpoints
pip install -r requirements.txt          # provided skeleton; pin versions
```

Loading a checkpoint:
```python
import torch
# TODO: replace with the actual model class once code is committed
state = torch.load("Model/MedNet_ant_binary.pt", map_location="cpu")
```

## Reproducibility

To let others re-run the work, document (placeholders to fill in):
- **Environment** — Python and PyTorch versions, CUDA version. <!-- TODO -->
- **Dependencies** — `timm`, `torchvision`, `opencv-python`, etc., pinned in `requirements.txt`. <!-- TODO: pin -->
- **Data** — expected `train.csv` schema and image directory layout; how the (private) data is obtained. <!-- TODO -->
- **Hardware & runtime** — GPU model and approximate training time. <!-- TODO -->

## Citation

```bibtex
@misc{cerebral_aneurysm_classification_2023,
  title  = {Cerebral Aneurysm Classification},
  author = {<!-- TODO: author name(s) -->},
  year   = {2023},
  url    = {https://github.com/pmy02/Cerebral-Aneurysm-Classification}
}
```

## Acknowledgments

- [MedicalNet (TencentMedicalNet)](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18) — grayscale medical pretrained weights.
- [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) — Asymmetric Loss.
- [`timm`](https://github.com/huggingface/pytorch-image-models) and `torchvision` — model backbones.

## License

<!-- TODO: no LICENSE file is present. For a public portfolio repo, consider adding one (e.g., MIT) so others know how they may use the code and weights. -->

## Contact

<!-- TODO: name · academic email · GitHub / website -->
