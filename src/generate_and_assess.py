"""
Legacy entrypoint kept for compatibility.

This script now performs generation only. Detailed damage assessment has been
moved to the teammate model workspace:

  Siamese U-Net Transformer/assess_siamese.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.data.datasets import DISASTER_TO_IDX, SyntheticDisasterDataset
    from src.generate import generate_pairs, load_generators
except ModuleNotFoundError:
    from data.datasets import DISASTER_TO_IDX, SyntheticDisasterDataset
    from generate import generate_pairs, load_generators


DENORM = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    tensor = DENORM(tensor.detach().cpu())
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = tensor.clamp(0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def load_real_pair(pre_path: str, post_path: str, image_size: int, device: torch.device):
    resize = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pre_t = norm(resize(to_tensor(Image.open(pre_path).convert("RGB")))).unsqueeze(0).to(device)
    post_t = norm(resize(to_tensor(Image.open(post_path).convert("RGB")))).unsqueeze(0).to(device)
    return pre_t, post_t


def save_generation_panel(pre_rgb: np.ndarray, post_rgb: np.ndarray, output_path: Path, source_tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0d1117")
    titles = ["Pre-Disaster", "Post-Disaster"]
    for axis, image, title in zip(axes, [pre_rgb, post_rgb], titles):
        axis.imshow(image)
        axis.set_title(title, color="white")
        axis.set_facecolor("#161b22")
        axis.axis("off")
    fig.suptitle(f"Generated Pair - {source_tag}", color="white", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaster GAN generation-only pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--disaster",
        default="hurricane",
        choices=list(DISASTER_TO_IDX.keys()) + ["unknown"],
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output", default="outputs/generated")
    parser.add_argument("--pre_image", default=None)
    parser.add_argument("--post_image", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--xbd_test", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_pre = None
    generated_post = None
    using_real_pair = bool(args.pre_image and args.post_image)
    using_generated = not (args.demo or args.xbd_test or using_real_pair)

    if using_generated:
        g_pre, g_post = load_generators(cfg, device)
        generated_pre, generated_post = generate_pairs(
            g_pre=g_pre,
            g_post=g_post,
            disaster=args.disaster,
            cfg=cfg,
            device=device,
            num_samples=args.num_samples,
        )

    for sample_idx in range(args.num_samples):
        if args.demo:
            dataset = SyntheticDisasterDataset(
                num_samples=max(args.num_samples, 10),
                image_size=cfg["data"]["image_size"],
            )
            sample = dataset[sample_idx % len(dataset)]
            pre_t = sample["pre_image"].unsqueeze(0).to(device)
            post_t = sample["post_image"].unsqueeze(0).to(device)
            source_tag = f"synthetic_{sample_idx:02d}"
        elif args.xbd_test:
            img_dir = Path(cfg["data"]["xbd_root"]) / "test" / "images"
            all_pre = sorted(img_dir.glob("*_pre_disaster.png"))
            if not all_pre:
                raise FileNotFoundError(f"No test images found in {img_dir}")
            random.seed(sample_idx)
            chosen = random.choice(all_pre)
            stem = chosen.stem.replace("_pre_disaster", "")
            post_path = img_dir / f"{stem}_post_disaster.png"
            if not post_path.exists():
                post_path = chosen
            pre_t, post_t = load_real_pair(str(chosen), str(post_path), cfg["data"]["image_size"], device)
            source_tag = stem
        elif using_real_pair:
            pre_t, post_t = load_real_pair(args.pre_image, args.post_image, cfg["data"]["image_size"], device)
            source_tag = Path(args.pre_image).stem
        else:
            pre_t = generated_pre[sample_idx : sample_idx + 1]
            post_t = generated_post[sample_idx : sample_idx + 1]
            source_tag = f"{args.disaster}_sample{sample_idx:02d}"

        pre_rgb = tensor_to_rgb(pre_t.squeeze(0))
        post_rgb = tensor_to_rgb(post_t.squeeze(0))

        pre_path = out_dir / f"{source_tag}_pre.png"
        post_path = out_dir / f"{source_tag}_post.png"
        Image.fromarray(pre_rgb).save(pre_path)
        Image.fromarray(post_rgb).save(post_path)
        save_generation_panel(pre_rgb, post_rgb, out_dir / f"{source_tag}_pair.png", source_tag)
        print(f"Saved generated pair -> {source_tag}")


if __name__ == "__main__":
    main()
