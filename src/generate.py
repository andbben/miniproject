"""
Generation helpers for the GAN pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.data.datasets import DISASTER_TO_IDX
    from src.models.gan import ImageTranslationGenerator, PreImageGenerator
except ModuleNotFoundError:
    from data.datasets import DISASTER_TO_IDX
    from models.gan import ImageTranslationGenerator, PreImageGenerator


def load_pre_generator(cfg: dict, device: torch.device) -> PreImageGenerator:
    gcfg = cfg["gan"]
    pgcfg = gcfg["pre_gan"]
    model = PreImageGenerator(
        latent_dim=pgcfg["latent_dim"],
        num_disaster_types=len(cfg["generate"]["disaster_types"]),
        disaster_emb_dim=pgcfg["disaster_emb_dim"],
        base_ch=pgcfg["base_ch"],
    ).to(device)
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / "pre_gan" / "best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["G"])
        print(f"[G_pre] Loaded from {ckpt_path}")
    else:
        print("[G_pre] No checkpoint found - using random weights (demo mode).")
    model.eval()
    return model


def load_post_generator(cfg: dict, device: torch.device) -> ImageTranslationGenerator:
    gcfg = cfg["gan"]
    pogcfg = gcfg["post_gan"]
    model = ImageTranslationGenerator(
        in_channels=3,
        num_disaster_types=len(cfg["generate"]["disaster_types"]),
        disaster_emb_dim=pogcfg["disaster_emb_dim"],
        base_ch=pogcfg["base_ch"],
        dropout=0.0,
    ).to(device)
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / "post_gan" / "best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["G"])
        print(f"[G_post] Loaded from {ckpt_path}")
    else:
        print("[G_post] No checkpoint found - using random weights (demo mode).")
    model.eval()
    return model


def load_generators(
    cfg: dict, device: torch.device
) -> Tuple[PreImageGenerator, ImageTranslationGenerator]:
    return load_pre_generator(cfg, device), load_post_generator(cfg, device)


@torch.inference_mode()
def generate_pairs(
    g_pre: PreImageGenerator,
    g_post: ImageTranslationGenerator,
    disaster: str,
    cfg: dict,
    device: torch.device,
    num_samples: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    disaster_idx = torch.full(
        (num_samples,),
        DISASTER_TO_IDX.get(disaster, 0),
        dtype=torch.long,
        device=device,
    )
    pre_img = g_pre.sample(
        batch_size=num_samples,
        disaster_idx=disaster_idx,
        device=device,
        temperature=cfg["generate"].get("temperature", 1.0),
    )
    post_img = g_post(pre_img, disaster_idx)
    return pre_img, post_img


@torch.inference_mode()
def generate_pair(
    g_pre: PreImageGenerator,
    g_post: ImageTranslationGenerator,
    disaster: str,
    cfg: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pre_img, post_img = generate_pairs(
        g_pre=g_pre,
        g_post=g_post,
        disaster=disaster,
        cfg=cfg,
        device=device,
        num_samples=1,
    )
    return pre_img, post_img
