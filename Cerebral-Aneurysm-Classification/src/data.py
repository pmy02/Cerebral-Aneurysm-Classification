"""Dataset construction and PyTorch Dataset classes.

Label scheme (see README "Task formulation"):
  Anterior (ICA) : ICA, AntChor, ACA, ACOM, MCA
  Posterior (VA) : VA, PICA, SCA, BA, PCA, PCOM

Each patient has 8 views. The raw ``train.csv`` is expanded into anterior and
posterior dataframes (~4x rows) where left/right are merged at the label level.

The exact column names of your ``train.csv`` are dataset-specific; the
constants and the ``build_circulation_frame`` mapping below are marked TODO so
you can align them with your schema.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ANTERIOR_LABELS = ["ICA", "AntChor", "ACA", "ACOM", "MCA"]
POSTERIOR_LABELS = ["VA", "PICA", "SCA", "BA", "PCA", "PCOM"]


def build_circulation_frame(df: pd.DataFrame, circulation: str) -> pd.DataFrame:
    """Build the anterior or posterior dataframe from the raw train dataframe.

    Args:
        df: raw dataframe loaded from train.csv.
        circulation: "anterior" or "posterior".

    Returns:
        A dataframe with an ``image_path`` column and one binary column per
        label in the chosen circulation.

    TODO: implement the L/R, I/V column reassignment for your actual schema.
    This stub documents the intended output shape; wire it to your columns.
    """
    labels = ANTERIOR_LABELS if circulation == "anterior" else POSTERIOR_LABELS
    raise NotImplementedError(
        "Align build_circulation_frame() with your train.csv columns. "
        f"Expected output columns: ['image_path', *{labels}]."
    )


def make_binary_labels(frame: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Add a 'sum' binary column: 1 if any location label is positive."""
    frame = frame.copy()
    frame["sum"] = (frame[label_cols].sum(axis=1) > 0).astype(int)
    return frame


class AngiographyDataset(Dataset):
    """Loads angiography frames and their labels (binary or multi-label)."""

    def __init__(self, frame: pd.DataFrame, label_cols: list[str],
                 image_root: str = "", transform=None, grayscale: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_root = image_root
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = os.path.join(self.image_root, str(row["image_path"]))
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            raise FileNotFoundError(path)
        if self.transform is not None:
            img = self.transform(image=img)["image"] if _is_albumentations(self.transform) \
                else self.transform(img)
        label = row[self.label_cols].to_numpy(dtype="float32")
        return img, label


def _is_albumentations(t) -> bool:
    return t.__class__.__module__.startswith("albumentations")
