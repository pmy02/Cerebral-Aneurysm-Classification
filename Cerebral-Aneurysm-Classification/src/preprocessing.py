"""Image preprocessing for cerebral angiography frames.

Three input forms were observed in the training data (see README):
  1. No margin and no text  -> used as-is.
  2. Text + margin          -> margins removed, fixed text region masked and
                               filled with the mean background color.
  3. Margin only            -> center-line scan to find the gray boundary, crop.

Coordinates for the fixed text region are dataset-specific. Confirm the values
marked TODO against your own images before relying on this.
"""
from __future__ import annotations

import cv2
import numpy as np


def strip_margin(image: np.ndarray, gray_threshold: int = 10) -> np.ndarray:
    """Crop uniform dark margins by scanning outward from the image center.

    Scans the central horizontal and vertical lines and crops where pixel
    intensity first rises above ``gray_threshold`` (i.e. content begins).

    Args:
        image: HxW grayscale or HxWxC array.
        gray_threshold: intensity below which a pixel is considered margin.

    Returns:
        The cropped image.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row, col = gray[h // 2, :], gray[:, w // 2]

    xs = np.where(row > gray_threshold)[0]
    ys = np.where(col > gray_threshold)[0]
    if xs.size == 0 or ys.size == 0:
        return image  # nothing detected; return unchanged
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    return image[y0 : y1 + 1, x0 : x1 + 1]


def remove_text_region(
    image: np.ndarray,
    # TODO: replace with the fixed text-box coordinates for your dataset.
    text_box: tuple[int, int, int, int] = (0, 0, 0, 0),
    sample_box: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Mask a fixed-position text region and fill it with mean background color.

    The text sits at fixed coordinates, so a fixed ``text_box`` is overwritten
    with the mean pixel value sampled from an adjacent ``sample_box``.

    Args:
        image: HxW or HxWxC array.
        text_box: (x0, y0, x1, y1) region containing the text.
        sample_box: (x0, y0, x1, y1) clean background region to sample.

    Returns:
        Image with the text region filled.
    """
    out = image.copy()
    sx0, sy0, sx1, sy1 = sample_box
    tx0, ty0, tx1, ty1 = text_box
    if (sx1 - sx0) <= 0 or (sy1 - sy0) <= 0:
        return out  # sample box not configured yet
    fill = image[sy0:sy1, sx0:sx1].reshape(-1, *image.shape[2:]).mean(axis=0)
    out[ty0:ty1, tx0:tx1] = fill.astype(image.dtype)
    return out


def preprocess(
    image: np.ndarray,
    has_text: bool = False,
    has_margin: bool = False,
    **kwargs,
) -> np.ndarray:
    """Dispatch to the appropriate preprocessing path for one frame."""
    out = image
    if has_text:
        out = remove_text_region(out, **kwargs)
    if has_margin:
        out = strip_margin(out)
    return out
