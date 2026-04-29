"""Damage-model losses and bridge import for the teammate Siamese model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

SIAMESE_DIR = Path(__file__).resolve().parents[2] / "Siamese U-Net Transformer"
if str(SIAMESE_DIR) not in sys.path:
    sys.path.insert(0, str(SIAMESE_DIR))

from model_siamese import SiameseUNetTransformer


SiameseUNet = SiameseUNetTransformer


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).reshape(-1)
        target = target.float().reshape(-1)
        intersection = (pred * target).sum()
        return 1 - (
            (2 * intersection + self.smooth)
            / (pred.sum() + target.sum() + self.smooth)
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "alpha", torch.as_tensor(alpha, dtype=torch.float32), persistent=False
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.alpha if self.alpha.numel() > 0 else None
        ce = F.cross_entropy(logits, targets, reduction="none", weight=weight)
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class DamageAssessmentLoss(nn.Module):
    """
    Combined loss for joint localisation and damage classification.

    total = seg_weight * (BCE_loc + Dice_loc)
          + cls_weight * (Focal_dmg + Dice_dmg)
    """

    def __init__(
        self,
        seg_weight: float = 0.30,
        cls_weight: float = 0.70,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.5,
        num_classes: int = 5,
    ):
        super().__init__()
        self.seg_w = seg_weight
        self.cls_w = cls_weight
        self.num_classes = num_classes
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.bce = nn.BCEWithLogitsLoss()

        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights, persistent=False)

    def _multiclass_dice_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
        target_1h = target_1h.float()

        dims = (0, 2, 3)
        intersection = (probs * target_1h).sum(dim=dims)
        cardinality = (probs + target_1h).sum(dim=dims)
        dice_per_class = 1 - ((2 * intersection + 1.0) / (cardinality + 1.0))

        if self.class_weights.numel() == self.num_classes:
            norm_w = self.class_weights / self.class_weights.sum().clamp_min(1e-6)
            return (dice_per_class * norm_w).sum()
        return dice_per_class.mean()

    def forward(
        self,
        damage_logits: torch.Tensor,
        loc_logits: torch.Tensor,
        post_mask: torch.Tensor,
        pre_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        building_mask = (pre_mask > 0).float().unsqueeze(1)
        bce_loc = self.bce(loc_logits, building_mask)
        dice_loc = self.dice(loc_logits, building_mask)
        loc_loss = bce_loc + dice_loc

        focal_dmg = self.focal(damage_logits, post_mask)
        dice_dmg = self._multiclass_dice_loss(damage_logits, post_mask)
        cls_loss = focal_dmg + dice_dmg

        total = self.seg_w * loc_loss + self.cls_w * cls_loss
        return total, {
            "loc_loss": float(loc_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "focal_dmg": float(focal_dmg.item()),
            "dice_loc": float(dice_loc.item()),
            "dice_dmg": float(dice_dmg.item()),
            "total_loss": float(total.item()),
        }


__all__ = [
    "SiameseUNet",
    "SiameseUNetTransformer",
    "DamageAssessmentLoss",
]
