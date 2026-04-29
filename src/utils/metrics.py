"""
src/utils/metrics.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Damage Assessment Metrics

Beyond simple acre counts, this module computes:

  1. Structural Damage Index (SDI)     – weighted severity score [0,1]
  2. Per-class pixel area (acres / km²)
  3. Building-level damage statistics  – % buildings per category
  4. Change Detection IoU              – overlap between pre/post masks
  5. Damage F1 Score (xView2 protocol) – 30% loc + 70% cls
  6. Normalised Difference Index (NDI) – proxy for NDVI-like vegetation change
     (using RGB channels as B=blue, G=green surrogates)
  7. Damage Severity Distribution      – histogram with visual breakdown

The SDI is inspired by FEMA's Rapid Visual Screening methodology and the
Joint Damage Scale used in the xBD dataset.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
LABEL_NAMES = ["background", "no-damage", "minor", "major", "destroyed"]
LABEL_IDS   = [0, 1, 2, 3, 4]

# Weighted severity for SDI (background excluded)
SDI_WEIGHTS = {
    0: 0.00,   # background
    1: 0.00,   # no-damage
    2: 0.25,   # minor
    3: 0.75,   # major
    4: 1.00,   # destroyed
}

# Ground Sample Distance (metres per pixel) for xBD / NOAA imagery
DEFAULT_GSD_M  = 0.5          # 0.5 m/px  (DigitalGlobe ~0.5 m GSD)
M2_PER_ACRE    = 4046.86
M2_PER_KM2     = 1_000_000


# ─────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────
@dataclass
class DamageReport:
    """Structured damage assessment report for a single image pair."""

    # ── Pixel counts ──────────────────────────
    total_pixels:     int   = 0
    building_pixels:  int   = 0
    affected_pixels:  int   = 0    # any damage class ≥ 2

    # ── Area estimates ────────────────────────
    total_area_km2:        float = 0.0
    building_area_km2:     float = 0.0
    affected_area_km2:     float = 0.0
    affected_area_acres:   float = 0.0

    # ── Per-class pixel counts & areas ───────
    class_pixel_counts: Dict[str, int]   = field(default_factory=dict)
    class_areas_km2:    Dict[str, float] = field(default_factory=dict)
    class_areas_acres:  Dict[str, float] = field(default_factory=dict)

    # ── Structural Damage Index ───────────────
    sdi:               float = 0.0    # [0, 1]
    sdi_interpretation: str  = ""

    # ── Classification metrics ────────────────
    iou_per_class:  Dict[str, float] = field(default_factory=dict)
    mean_iou:       float = 0.0
    f1_loc:         float = 0.0    # localisation F1
    f1_cls:         float = 0.0    # damage classification F1
    f1_combined:    float = 0.0    # 0.3*loc + 0.7*cls

    # ── Building-level stats (if polygons available) ──
    total_buildings:     int   = 0
    pct_no_damage:       float = 0.0
    pct_minor:           float = 0.0
    pct_major:           float = 0.0
    pct_destroyed:       float = 0.0

    # ── Image-level change signal ─────────────
    mean_change_magnitude: float = 0.0   # mean |pre - post| pixel diff
    ndi_change:            float = 0.0   # Normalised Difference Index

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DISASTER DAMAGE ASSESSMENT REPORT",
            "=" * 60,
            f"  Total scene area   : {self.total_area_km2:.3f} km²  "
            f"({self.total_area_km2 * 1e6 / M2_PER_ACRE:.1f} acres)",
            f"  Buildings detected : {self.building_area_km2:.3f} km²",
            f"  Affected area      : {self.affected_area_km2:.4f} km²  "
            f"({self.affected_area_acres:.2f} acres)",
            "",
            "  Damage Breakdown:",
        ]
        for cls in LABEL_NAMES[1:]:
            pix = self.class_pixel_counts.get(cls, 0)
            km2 = self.class_areas_km2.get(cls, 0.0)
            ac  = self.class_areas_acres.get(cls, 0.0)
            lines.append(f"    {cls:<12} : {pix:>8} px  |  "
                         f"{km2:.4f} km²  |  {ac:.2f} acres")
        lines += [
            "",
            f"  Structural Damage Index (SDI) : {self.sdi:.4f}",
            f"  Interpretation               : {self.sdi_interpretation}",
            "",
            f"  Mean IoU       : {self.mean_iou:.4f}",
            f"  F1 Combined    : {self.f1_combined:.4f}  "
            f"(loc={self.f1_loc:.4f}  cls={self.f1_cls:.4f})",
            "",
            f"  Image change Δ  : {self.mean_change_magnitude:.4f}",
            f"  NDI change      : {self.ndi_change:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Core metric functions
# ─────────────────────────────────────────────
def pixel_to_area(
    num_pixels: int,
    gsd_m: float = DEFAULT_GSD_M,
) -> Tuple[float, float]:
    """Convert pixel count → (km², acres)."""
    m2  = num_pixels * (gsd_m ** 2)
    return m2 / M2_PER_KM2, m2 / M2_PER_ACRE


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Compute per-class IoU and mean IoU.

    pred, target: (H, W) integer arrays in [0, num_classes-1]
    Returns:
        iou_per_class: (num_classes,)
        mean_iou:      float
    """
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum()
        union = (p | t).sum()
        iou[c] = inter / (union + 1e-8)
    return iou, iou.mean()


def compute_f1(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 5,
) -> Tuple[float, float, float]:
    """
    Compute xView2-style combined F1.
    Returns: (f1_loc, f1_cls, f1_combined)
    """
    # Localisation: binary (building vs background)
    pred_loc   = (pred > 0).astype(int)
    target_loc = (target > 0).astype(int)
    tp_loc = ((pred_loc == 1) & (target_loc == 1)).sum()
    fp_loc = ((pred_loc == 1) & (target_loc == 0)).sum()
    fn_loc = ((pred_loc == 0) & (target_loc == 1)).sum()
    f1_loc = (2 * tp_loc) / (2 * tp_loc + fp_loc + fn_loc + 1e-8)

    # Damage classification (weighted F1 over classes 1-4)
    f1_classes = []
    for c in range(1, num_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        f1_c = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_classes.append(f1_c)
    f1_cls = float(np.mean(f1_classes))

    f1_combined = 0.3 * f1_loc + 0.7 * f1_cls
    return float(f1_loc), f1_cls, float(f1_combined)


def compute_sdi(pred_mask: np.ndarray) -> Tuple[float, str]:
    """
    Structural Damage Index (SDI).

    SDI = Σ_i  w_i * |pixels_i| / |total_building_pixels|

    where w_i is the damage weight for class i.

    Returns:
        sdi:             float in [0, 1]
        interpretation:  human-readable severity string
    """
    building_pixels = (pred_mask > 0).sum()
    if building_pixels == 0:
        return 0.0, "No buildings detected"

    weighted_sum = sum(
        SDI_WEIGHTS[c] * (pred_mask == c).sum()
        for c in SDI_WEIGHTS
        if c > 0
    )
    sdi = float(weighted_sum / building_pixels)

    if sdi < 0.05:
        label = "Negligible — minimal or no structural damage"
    elif sdi < 0.20:
        label = "Minor — localised light damage, structures largely intact"
    elif sdi < 0.40:
        label = "Moderate — noticeable damage, partial structural compromise"
    elif sdi < 0.65:
        label = "Severe — major structural damage, significant losses"
    else:
        label = "Catastrophic — widespread destruction"

    return sdi, label


def compute_ndi_change(
    pre_img: np.ndarray,
    post_img: np.ndarray,
) -> float:
    """
    Normalised Difference Index change proxy using green and blue channels.
    NDI = (G - B) / (G + B + ε)   (approximates vegetation/water change)
    Returns mean absolute change in NDI between pre and post images.

    pre_img, post_img: (H, W, 3) float32 arrays in [0, 1]
    """
    g_pre, b_pre  = pre_img[..., 1],  pre_img[..., 2]
    g_post, b_post = post_img[..., 1], post_img[..., 2]

    ndi_pre  = (g_pre  - b_pre)  / (g_pre  + b_pre  + 1e-8)
    ndi_post = (g_post - b_post) / (g_post + b_post + 1e-8)
    return float(np.abs(ndi_pre - ndi_post).mean())


# ─────────────────────────────────────────────
#  High-level assessment function
# ─────────────────────────────────────────────
def assess_damage(
    pred_damage:    np.ndarray,             # (H, W) int, predicted damage mask
    pred_loc:       np.ndarray,             # (H, W) int, predicted building mask (binary)
    gt_damage:      Optional[np.ndarray],   # (H, W) int, ground truth (or None)
    pre_img:        Optional[np.ndarray],   # (H, W, 3) float [0,1]
    post_img:       Optional[np.ndarray],   # (H, W, 3) float [0,1]
    gsd_m:          float = DEFAULT_GSD_M,
) -> DamageReport:
    """
    Full damage assessment pipeline.

    Args:
        pred_damage:  model's per-pixel damage prediction  (0-4)
        pred_loc:     model's building localisation mask   (0-1)
        gt_damage:    ground-truth damage mask (optional, for metrics)
        pre_img:      pre-disaster RGB image  [0,1]
        post_img:     post-disaster RGB image [0,1]
        gsd_m:        ground sample distance in metres/pixel

    Returns:
        DamageReport with all metrics populated
    """
    report = DamageReport()
    H, W   = pred_damage.shape
    report.total_pixels = H * W
    report.total_area_km2, _ = pixel_to_area(H * W, gsd_m)

    # ── Building pixels ──────────────────────────────────────────────
    building_pix = (pred_damage > 0).sum()
    report.building_pixels = int(building_pix)
    report.building_area_km2, _ = pixel_to_area(building_pix, gsd_m)

    # ── Affected pixels (any damage ≥ minor) ─────────────────────────
    affected_pix = (pred_damage >= 2).sum()
    report.affected_pixels = int(affected_pix)
    report.affected_area_km2, report.affected_area_acres = pixel_to_area(
        affected_pix, gsd_m
    )

    # ── Per-class breakdown ──────────────────────────────────────────
    for cls_id, cls_name in zip(LABEL_IDS, LABEL_NAMES):
        pix = int((pred_damage == cls_id).sum())
        km2, acres = pixel_to_area(pix, gsd_m)
        report.class_pixel_counts[cls_name] = pix
        report.class_areas_km2[cls_name]    = km2
        report.class_areas_acres[cls_name]  = acres

    # ── Building level stats ─────────────────────────────────────────
    total_bld = max(int((pred_damage > 0).sum()), 1)
    report.total_buildings = total_bld
    report.pct_no_damage   = 100 * (pred_damage == 1).sum() / total_bld
    report.pct_minor       = 100 * (pred_damage == 2).sum() / total_bld
    report.pct_major       = 100 * (pred_damage == 3).sum() / total_bld
    report.pct_destroyed   = 100 * (pred_damage == 4).sum() / total_bld

    # ── Structural Damage Index ──────────────────────────────────────
    report.sdi, report.sdi_interpretation = compute_sdi(pred_damage)

    # ── Metrics vs ground truth ──────────────────────────────────────
    if gt_damage is not None:
        iou_arr, report.mean_iou = compute_iou(pred_damage, gt_damage)
        for i, name in enumerate(LABEL_NAMES):
            report.iou_per_class[name] = float(iou_arr[i])
        report.f1_loc, report.f1_cls, report.f1_combined = compute_f1(
            pred_damage, gt_damage
        )

    # ── Image-level change magnitude ─────────────────────────────────
    if pre_img is not None and post_img is not None:
        report.mean_change_magnitude = float(
            np.abs(pre_img.astype(float) - post_img.astype(float)).mean()
        )
        report.ndi_change = compute_ndi_change(pre_img, post_img)

    return report


# ─────────────────────────────────────────────
#  Batch evaluation (for val / test loops)
# ─────────────────────────────────────────────
def batch_metrics(
    pred_logits: torch.Tensor,  # (B, 5, H, W)
    gt_masks:    torch.Tensor,  # (B, H, W)
) -> Dict[str, float]:
    """Quick per-batch metrics for training loop monitoring."""
    pred = pred_logits.argmax(dim=1)
    gt = gt_masks.long()
    num_classes = pred_logits.shape[1]
    eps = 1e-8

    class_ids = torch.arange(num_classes, device=pred.device).view(1, num_classes, 1, 1)
    pred_oh = pred.unsqueeze(1) == class_ids
    gt_oh = gt.unsqueeze(1) == class_ids

    intersection = (pred_oh & gt_oh).sum(dim=(2, 3)).float()
    union = (pred_oh | gt_oh).sum(dim=(2, 3)).float()
    miou = (intersection / (union + eps)).mean(dim=1)

    pred_loc = pred > 0
    gt_loc = gt > 0
    tp_loc = (pred_loc & gt_loc).sum(dim=(1, 2)).float()
    fp_loc = (pred_loc & ~gt_loc).sum(dim=(1, 2)).float()
    fn_loc = (~pred_loc & gt_loc).sum(dim=(1, 2)).float()
    f1_loc = (2 * tp_loc) / (2 * tp_loc + fp_loc + fn_loc + eps)

    damage_ids = torch.arange(1, num_classes, device=pred.device).view(1, num_classes - 1, 1, 1)
    pred_damage = pred.unsqueeze(1) == damage_ids
    gt_damage = gt.unsqueeze(1) == damage_ids
    tp_cls = (pred_damage & gt_damage).sum(dim=(2, 3)).float()
    fp_cls = (pred_damage & ~gt_damage).sum(dim=(2, 3)).float()
    fn_cls = (~pred_damage & gt_damage).sum(dim=(2, 3)).float()
    f1_cls = ((2 * tp_cls) / (2 * tp_cls + fp_cls + fn_cls + eps)).mean(dim=1)

    mean_miou = float(miou.mean().item())
    mean_f1_loc = float(f1_loc.mean().item())
    mean_f1_cls = float(f1_cls.mean().item())
    return {
        "mIoU": mean_miou,
        "f1_loc": mean_f1_loc,
        "f1_cls": mean_f1_cls,
        "f1_comb": 0.3 * mean_f1_loc + 0.7 * mean_f1_cls,
    }
