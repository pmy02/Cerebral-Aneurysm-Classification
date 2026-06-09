"""Grad-CAM heatmaps for qualitative inspection (see README "Explainability").

Uses pytorch-grad-cam. Run on text/margin-removed images without augmentation,
as in the original analysis.

Usage:
    python -m scripts.gradcam --checkpoint Model/MedNet_ant_binary.pt --image path.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from scripts.train import build_model  # reuse model builder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="gradcam.png")
    args = ap.parse_args()

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise SystemExit("pip install grad-cam") from e

    # TODO: pass the same cfg used to build the trained model.
    cfg = {"model": "mednet", "task": "binary"}
    model = build_model(cfg)
    from src.checkpoint import load_checkpoint
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    target_layers = [model.layer4[-1]]  # last conv block of ResNet-18
    cam = GradCAM(model=model, target_layers=target_layers)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    tensor = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
    grayscale_cam = cam(input_tensor=tensor)[0]
    rgb = np.repeat(img[..., None], 3, axis=2)
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    cv2.imwrite(args.out, overlay[:, :, ::-1])
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
