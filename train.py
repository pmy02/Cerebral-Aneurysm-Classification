"""Config-driven training entry point.

Usage:
    python -m scripts.train --config configs/binary_anterior.yaml

The config selects the task ("binary" or "multilabel"), the model builder, the
data split, and hyperparameters. This is a generic loop; adapt the data loading
to your train.csv schema (see src/data.py).
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.losses import AsymmetricLoss
from src.models import build_mednet, build_resnet_multilabel, build_swin_multilabel


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]
    if name == "mednet":
        return build_mednet(num_classes=1, medicalnet_weights=cfg.get("medicalnet_weights"))
    if name == "swin_multilabel":
        return build_swin_multilabel(num_classes=cfg["num_classes"])
    if name == "resnet_multilabel":
        return build_resnet_multilabel(num_classes=cfg["num_classes"])
    raise ValueError(f"unknown model: {name}")


def build_criterion(cfg: dict):
    return torch.nn.BCEWithLogitsLoss() if cfg["task"] == "binary" else AsymmetricLoss()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))

    # TODO: build train_loader from src.data using your train.csv schema.
    train_loader: DataLoader = cfg["__inject_train_loader__"]  # placeholder

    for epoch in range(cfg.get("epochs", 30)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            if cfg["task"] == "binary":
                logits = logits.squeeze(1)
                labels = labels.squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), cfg["out_checkpoint"])
    print(f"saved checkpoint -> {cfg['out_checkpoint']}")


if __name__ == "__main__":
    main()
