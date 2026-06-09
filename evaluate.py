"""Evaluate a trained checkpoint and print a metrics table.

This is what fills the Results table in the README with REAL numbers. It does
not invent anything -- run it on your held-out split.

Usage:
    python -m scripts.evaluate --config configs/binary_anterior.yaml \
        --checkpoint Model/MedNet_ant_binary.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.metrics import binary_metrics, multilabel_metrics, percentile_thresholds
from scripts.train import build_model


@torch.no_grad()
def collect_predictions(model, loader, device, task):
    model.eval()
    probs, gts = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        if task == "binary":
            logits = logits.squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        gts.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(gts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    # Loads your existing checkpoint and prints how many layers actually matched.
    load_checkpoint(model, args.checkpoint, map_location=device)

    # TODO: build eval_loader from src.data using your held-out split.
    eval_loader: DataLoader = cfg["__inject_eval_loader__"]  # placeholder

    y_prob, y_true = collect_predictions(model, eval_loader, device, cfg["task"])

    if cfg["task"] == "binary":
        # Decision threshold = median of predictions (see README decision rule).
        thr = float(np.median(y_prob))
        m = binary_metrics(y_true.squeeze(), y_prob.squeeze(), threshold=thr)
        print(f"| metric | value |\n|---|---|")
        for k, v in m.items():
            print(f"| {k} | {v:.4f} |")
    else:
        thr = percentile_thresholds(y_prob, q=cfg.get("threshold_percentile", 90.0))
        m = multilabel_metrics(y_true, y_prob, thresholds=thr,
                               label_names=cfg.get("label_names"))
        print(f"macro_f1={m['macro_f1']:.4f}  micro_f1={m['micro_f1']:.4f}  mAP={m['mAP']:.4f}")
        for name, f1 in m["per_class_f1"].items():
            print(f"  {name}: {f1:.4f}")


if __name__ == "__main__":
    main()
