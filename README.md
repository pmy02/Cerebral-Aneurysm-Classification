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

Metrics are computed with [`scripts/evaluate.py`](scripts/evaluate.py), which loads a
checkpoint, runs it on a held-out split, and prints the table below. Binary tasks
report F1/AUROC at the median-probability threshold; location tasks report macro-F1
at the per-class 90th-percentile threshold (the project's decision rule). Run it on
your data to fill in the values — the numbers below are intentionally left blank
rather than estimated:

```bash
python -m scripts.evaluate --config configs/binary_anterior.yaml \
    --checkpoint Model/MedNet_ant_binary.pt
```

<!-- Fill from scripts/evaluate.py output; do not estimate -->
| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Binary detection (anterior)  | MedNet  | F1 / AUROC | _run evaluate.py_ |
| Binary detection (posterior) | MedNet  | F1 / AUROC | _run evaluate.py_ |
| Location (anterior)          | SwinV2  | macro-F1   | _run evaluate.py_ |
| Location (posterior)         | ResNet-18 | macro-F1 | _run evaluate.py_ |

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
├── src/
│   ├── data.py             # dataset construction, anterior/posterior split, Dataset
│   ├── preprocessing.py    # margin/text removal and cropping
│   ├── models.py           # MedNet, SwinV2, ResNet-18 builders
│   ├── losses.py           # Asymmetric Loss (multi-label)
│   └── metrics.py          # F1 / AUROC / macro-F1 / mAP
├── scripts/
│   ├── train.py            # config-driven training
│   ├── evaluate.py         # checkpoint -> metrics table
│   ├── infer.py            # full decision-rule pipeline
│   └── gradcam.py          # Grad-CAM heatmaps
├── configs/                # one YAML per task/circulation
├── requirements.txt
├── .gitattributes          # Git LFS tracking for model files
├── README.md               # English (this file)
└── README.ko.md            # Korean
```

> The code under `src/` and `scripts/` reconstructs the documented method as a
> runnable skeleton. Verify each module against your original implementation and


## Getting started

```bash
git clone https://github.com/pmy02/Cerebral-Aneurysm-Classification.git
cd Cerebral-Aneurysm-Classification
git lfs install && git lfs pull          # fetch LFS-tracked checkpoints
pip install -r requirements.txt          # pin versions for reproducibility
```

Train, evaluate, or run inference (each is config-driven):
```bash
python -m scripts.train    --config configs/binary_anterior.yaml
python -m scripts.evaluate --config configs/binary_anterior.yaml \
    --checkpoint Model/MedNet_ant_binary.pt
python -m scripts.gradcam  --checkpoint Model/MedNet_ant_binary.pt --image path/to/frame.png
```

## Contact

Minyoung Park — [LinkedIn](https://www.linkedin.com/in/minyoung-park-672754237) · minyo0119@gmail.com
