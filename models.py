"""Model definitions.

Binary detection (anterior/posterior, shared architecture):
    MedNet -- ResNet-18 backbone initialized from MedicalNet grayscale weights.
Multi-label location:
    Anterior  -- SwinV2 (timm swinv2_cr_tiny_ns_224)
    Posterior -- ResNet-18 (torchvision pretrained)

NOTE: The MedicalNet checkpoint key layout must be confirmed against your file.
The original MedicalNet weights target a 3D ResNet; this project used them in a
2D, single-channel setting, so the load is done with strict=False and a warning.
Verify the mapping before training (see TODO in ``build_mednet``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _adapt_first_conv_to_grayscale(model: nn.Module) -> None:
    """Replace the 3-channel stem with a 1-channel conv (averaged weights)."""
    old = model.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    model.conv1 = new


def build_mednet(num_classes: int = 1, medicalnet_weights: str | None = None,
                 grayscale: bool = True) -> nn.Module:
    """ResNet-18 for binary detection, optionally seeded with MedicalNet weights."""
    model = torchvision.models.resnet18(weights=None)
    if grayscale:
        _adapt_first_conv_to_grayscale(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if medicalnet_weights:
        state = torch.load(medicalnet_weights, map_location="cpu")
        state = state.get("state_dict", state)
        # TODO: confirm key prefixes match (e.g. strip "module."); MedicalNet
        # layouts vary. strict=False lets mismatched/extra keys be skipped.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[build_mednet] missing={len(missing)} unexpected={len(unexpected)} "
                  f"keys when loading MedicalNet weights -- verify mapping.")
    return model


def build_swin_multilabel(num_classes: int, pretrained: bool = True) -> nn.Module:
    """SwinV2 (timm) for the anterior multi-label location task."""
    import timm  # imported lazily so the binary path doesn't require timm
    return timm.create_model(
        "swinv2_cr_tiny_ns_224", pretrained=pretrained, num_classes=num_classes
    )


def build_resnet_multilabel(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 (torchvision) for the posterior multi-label location task."""
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
