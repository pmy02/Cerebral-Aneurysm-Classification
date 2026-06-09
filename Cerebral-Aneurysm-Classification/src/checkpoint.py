"""Checkpoint loading with explicit key-match reporting.

src/models.py reconstructs the documented architectures. For your *existing*
trained checkpoints to actually be used, their layer names must match this
architecture. This helper reports how many parameters truly load, so you can
confirm the existing weights are used rather than silently skipped by
``strict=False``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def load_checkpoint(model: nn.Module, path: str, map_location: str = "cpu",
                    strict: bool = False) -> nn.Module:
    """Load weights into ``model`` and print a key-match summary.

    Args:
        model: an instantiated model (architecture must match the checkpoint).
        path: path to a .pt/.pth state dict (or dict with 'state_dict').
        map_location: torch device mapping.
        strict: if True, raise on any mismatch instead of skipping.

    Returns:
        The model with loaded weights.
    """
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip a common "module." prefix from DataParallel-saved checkpoints.
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    matched = model_keys & ckpt_keys
    print(f"[load_checkpoint] {path}: matched {len(matched)}/{len(model_keys)} "
          f"model parameters ({len(ckpt_keys - model_keys)} unused checkpoint keys)")
    if len(model_keys) and len(matched) < 0.5 * len(model_keys):
        print("[load_checkpoint] WARNING: <50% of layers matched. The reconstructed "
              "architecture likely differs from the one that produced this checkpoint. "
              "Align src/models.py with your original model before trusting these weights.")

    model.load_state_dict(state, strict=strict)
    return model
