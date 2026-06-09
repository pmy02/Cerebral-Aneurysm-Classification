"""Loss functions.

AsymmetricLoss is a faithful reimplementation of the multi-label loss from
Alibaba-MIIL/ASL (MIT License), used for the location (multi-label) task.
Binary detection uses ``torch.nn.BCELoss`` / ``BCEWithLogitsLoss`` directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification.

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label
    Classification" (ICCV 2021), https://github.com/Alibaba-MIIL/ASL
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (B, C) raw model outputs.
            targets: (B, C) multi-hot labels in {0, 1}.
        """
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= torch.pow(1 - pt, gamma)
        return -loss.sum(dim=1).mean()
