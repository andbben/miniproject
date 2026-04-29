"""
Training entrypoint for the three-phase disaster pipeline.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.data.datasets import SyntheticDisasterDataset, XBDDataset
    from src.models.damage_assessment import DamageAssessmentLoss, SiameseUNet
    from src.models.gan import (
        GANCriterion,
        ImageTranslationGenerator,
        PatchDiscriminator,
        PreImageGenerator,
    )
    from src.utils.metrics import batch_metrics
except ModuleNotFoundError:
    from data.datasets import SyntheticDisasterDataset, XBDDataset
    from models.damage_assessment import DamageAssessmentLoss, SiameseUNet
    from models.gan import GANCriterion, ImageTranslationGenerator, PatchDiscriminator, PreImageGenerator
    from utils.metrics import batch_metrics


def configure_gpu() -> torch.device:
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_properties(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"[Device] CUDA - {gpu.name}")
        print(f"[GPU] VRAM={gpu.total_memory / 1e9:.1f} GB | SMs={gpu.multi_processor_count}")
        print("[GPU] AMP enabled")
        return device

    if torch.backends.mps.is_available():
        print("[Device] Apple MPS")
        return torch.device("mps")

    print("[Device] CPU")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(state: dict, path: Path, tag: str = "best") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / f"{tag}.pt"
    torch.save(state, fpath)
    print(f"  [ckpt] saved -> {fpath}")
    return fpath


def get_num_workers(cfg: dict, use_gpu: bool) -> int:
    if not use_gpu:
        return 0
    cores = os.cpu_count() or 4
    auto_workers = min(max(cores - 2, 2), 8)
    requested = int(cfg["data"].get("num_workers", auto_workers))
    return max(0, min(requested, auto_workers))


def build_dataloaders(cfg: dict, batch_size: int, use_synthetic: bool = False):
    img_size = cfg["data"]["image_size"]
    use_gpu = torch.cuda.is_available()
    num_workers = get_num_workers(cfg, use_gpu)

    if use_synthetic or not Path(cfg["data"]["xbd_root"]).exists():
        print("[Data] Using synthetic dataset")
        full_ds = SyntheticDisasterDataset(num_samples=1000, image_size=img_size)
        n_train = int(0.8 * len(full_ds))
        n_val = int(0.1 * len(full_ds))
        n_test = len(full_ds) - n_train - n_val
        train_ds, val_ds, _ = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg["project"]["seed"]),
        )
    else:
        print(f"[Data] Loading xBD from {cfg['data']['xbd_root']}")
        full_train = XBDDataset(
            cfg["data"]["xbd_root"],
            split="train",
            image_size=img_size,
            augment=False,
        )
        n_val = int(cfg["data"]["val_split"] * len(full_train))
        n_train = len(full_train) - n_val
        train_idx, val_idx = random_split(
            range(len(full_train)),
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg["project"]["seed"]),
        )
        train_base = XBDDataset(
            cfg["data"]["xbd_root"],
            split="train",
            image_size=img_size,
            augment=True,
        )
        val_base = XBDDataset(
            cfg["data"]["xbd_root"],
            split="train",
            image_size=img_size,
            augment=False,
        )
        train_ds = Subset(train_base, train_idx.indices)
        val_ds = Subset(val_base, val_idx.indices)
        print(f"[Data] total={len(full_train)} train={n_train} val={n_val}")

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_gpu,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    print(
        f"[DataLoader] workers={num_workers} pin_memory={use_gpu} batch_size={batch_size}"
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


class DevicePrefetchLoader:
    """Overlap host-to-device copies with compute on single-GPU training."""

    def __init__(self, loader: DataLoader, device: torch.device, use_stream: bool = True):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" and use_stream else None
        )
        self._next_batch: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.loader)

    def _move_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        moved = tensor.to(self.device, non_blocking=True)
        if (
            self.device.type == "cuda"
            and tensor.ndim == 4
            and torch.is_floating_point(tensor)
            and key.endswith("_image")
        ):
            moved = moved.contiguous(memory_format=torch.channels_last)
        return moved

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = self._move_tensor(key, value)
            else:
                moved[key] = value
        return moved

    def _record_stream(self, value: Any) -> None:
        if self.stream is None:
            return
        if torch.is_tensor(value):
            value.record_stream(torch.cuda.current_stream(self.device))
            return
        if isinstance(value, dict):
            for nested in value.values():
                self._record_stream(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                self._record_stream(nested)

    def _preload(self, iterator) -> None:
        try:
            batch = next(iterator)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is None:
            self._next_batch = self._move_batch(batch)
            return

        with torch.cuda.stream(self.stream):
            self._next_batch = self._move_batch(batch)

    def __iter__(self):
        iterator = iter(self.loader)
        self._preload(iterator)
        while self._next_batch is not None:
            if self.stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = self._next_batch
            self._record_stream(batch)
            self._preload(iterator)
            yield batch


def maybe_prefetch_loader(loader: DataLoader, device: torch.device, enabled: bool):
    return DevicePrefetchLoader(loader, device, use_stream=enabled)


def try_compile(model: nn.Module, label: str) -> nn.Module:
    if os.name == "nt":
        print(f"[Compile] {label} skipped on Windows")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print(f"[Compile] {label} compiled")
        return compiled
    except Exception as exc:
        print(f"[Compile] {label} skipped ({type(exc).__name__})")
        return model


def move_images(batch: dict, keys: list[str], device: torch.device) -> list[torch.Tensor]:
    tensors = []
    for key in keys:
        tensor = batch[key].to(device, non_blocking=True)
        if device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        tensors.append(tensor)
    return tensors


def maybe_channels_last(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def amp_context(use_amp: bool):
    if use_amp:
        return autocast(device_type="cuda", enabled=True)
    return nullcontext()


def build_wandb_config(base_cfg: dict, phase: str, extra: dict) -> dict:
    return {
        "phase": phase,
        "project_name": base_cfg["project"]["name"],
        "seed": base_cfg["project"]["seed"],
        **extra,
    }


def start_wandb_run(cfg: dict, phase: str, group: str, extra_config: dict):
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    if wandb is None:
        print("[wandb] package not installed; continuing without run tracking")
        return None
    try:
        run = wandb.init(
            entity=wcfg.get("entity"),
            project=wcfg.get("project"),
            group=group,
            job_type=phase,
            name=f"{phase}-{time.strftime('%Y%m%d-%H%M%S')}",
            config=build_wandb_config(cfg, phase, extra_config),
            mode=wcfg.get("mode", "online"),
        )
        return run
    except Exception as exc:
        print(f"[wandb] init failed ({type(exc).__name__}); continuing without tracking")
        return None


def log_wandb(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def finish_wandb_run(run, summary: Optional[dict] = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[key] = value
    run.finish()


def maybe_log_checkpoint_artifact(run, checkpoint_path: Path, artifact_name: str) -> None:
    if run is None or wandb is None or not checkpoint_path.exists():
        return
    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)


def train_pre_gan(
    cfg: dict,
    device: torch.device,
    use_synthetic: bool = False,
    run_group: str = "disaster-gan",
):
    print("\n" + "=" * 60)
    print("PHASE 1 - Pre-Disaster GAN")
    print("=" * 60)

    gcfg = cfg["gan"]
    pgcfg = gcfg["pre_gan"]
    use_amp = device.type == "cuda"
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / "pre_gan"
    num_disaster_types = len(cfg["generate"]["disaster_types"])

    run = start_wandb_run(
        cfg,
        phase="pre_gan",
        group=run_group,
        extra_config={
            "epochs": pgcfg["epochs"],
            "batch_size": pgcfg["batch_size"],
            "g_lr": pgcfg["g_lr"],
            "d_lr": pgcfg["d_lr"],
            "latent_dim": pgcfg["latent_dim"],
        },
    )

    G = PreImageGenerator(
        latent_dim=pgcfg["latent_dim"],
        num_disaster_types=num_disaster_types,
        disaster_emb_dim=pgcfg["disaster_emb_dim"],
        base_ch=pgcfg["base_ch"],
    ).to(device)
    D = PatchDiscriminator(
        in_channels=3,
        num_disaster_types=num_disaster_types,
        base_ch=gcfg["disc_base_ch"],
        n_scales=gcfg["disc_n_scales"],
    ).to(device)
    G = maybe_channels_last(G, device)
    D = maybe_channels_last(D, device)
    G = try_compile(G, "G_pre")
    D = try_compile(D, "D_pre")

    criterion = GANCriterion(lambda_adv=gcfg["lambda_adv"], lambda_perc=gcfg["lambda_perc"])
    opt_G = torch.optim.AdamW(
        G.parameters(),
        lr=pgcfg["g_lr"],
        betas=(0.5, 0.999),
        weight_decay=1e-5,
        fused=(device.type == "cuda"),
    )
    opt_D = torch.optim.AdamW(
        D.parameters(),
        lr=pgcfg["d_lr"],
        betas=(0.5, 0.999),
        weight_decay=1e-5,
        fused=(device.type == "cuda"),
    )
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=pgcfg["epochs"])
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=pgcfg["epochs"])
    scaler = GradScaler("cuda", enabled=use_amp)
    train_loader, _ = build_dataloaders(cfg, pgcfg["batch_size"], use_synthetic)

    best_g_loss = float("inf")
    best_checkpoint = None

    try:
        for epoch in range(1, pgcfg["epochs"] + 1):
            epoch_start = time.time()
            G.train()
            D.train()
            sums = {"d_loss": 0.0, "g_adv": 0.0, "g_perc": 0.0, "g_total": 0.0}
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}[pre-GAN]", leave=False):
                real_img = move_images(batch, ["pre_image"], device)[0]
                disaster_idx = batch["disaster_idx"].to(device, non_blocking=True)
                batch_size = real_img.size(0)

                opt_D.zero_grad(set_to_none=True)
                with amp_context(use_amp):
                    z = torch.randn(batch_size, unwrap_model(G).latent_dim, device=device)
                    fake_img = G(z, disaster_idx).detach()
                    real_preds, _ = D(real_img, disaster_idx)
                    fake_preds, _ = D(fake_img, disaster_idx)
                    d_loss = criterion.d_loss(real_preds, fake_preds)
                if torch.isfinite(d_loss):
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(opt_D)
                    nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                    scaler.step(opt_D)
                    scaler.update()

                opt_G.zero_grad(set_to_none=True)
                with amp_context(use_amp):
                    z = torch.randn(batch_size, unwrap_model(G).latent_dim, device=device)
                    fake_img = G(z, disaster_idx)
                    fake_preds, _ = D(fake_img, disaster_idx)
                    g_loss, info = criterion.g_loss_pre(fake_preds, fake_img, real_img)
                if torch.isfinite(g_loss):
                    scaler.scale(g_loss).backward()
                    scaler.unscale_(opt_G)
                    nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                    scaler.step(opt_G)
                    scaler.update()

                sums["d_loss"] += float(d_loss.item())
                sums["g_adv"] += float(info["g_adv"])
                sums["g_perc"] += float(info["g_perc"])
                sums["g_total"] += float(info["g_total"])
                n_batches += 1

            sched_G.step()
            sched_D.step()
            denom = max(n_batches, 1)
            epoch_time = time.time() - epoch_start
            g_avg = sums["g_total"] / denom

            print(
                f"  Ep {epoch:3d}/{pgcfg['epochs']} | "
                f"D={sums['d_loss']/denom:.4f} G={g_avg:.4f} "
                f"(adv={sums['g_adv']/denom:.4f} perc={sums['g_perc']/denom:.4f}) | "
                f"{epoch_time:.1f}s"
            )

            log_wandb(
                run,
                {
                    "epoch": epoch,
                    "pre_gan/d_loss": sums["d_loss"] / denom,
                    "pre_gan/g_total": g_avg,
                    "pre_gan/g_adv": sums["g_adv"] / denom,
                    "pre_gan/g_perc": sums["g_perc"] / denom,
                    "pre_gan/g_lr": opt_G.param_groups[0]["lr"],
                    "pre_gan/d_lr": opt_D.param_groups[0]["lr"],
                    "pre_gan/epoch_time_sec": epoch_time,
                },
            )

            if g_avg < best_g_loss:
                best_g_loss = g_avg
                best_checkpoint = save_checkpoint(
                    {
                        "epoch": epoch,
                        "G": unwrap_model(G).state_dict(),
                        "D": unwrap_model(D).state_dict(),
                        "g_loss": g_avg,
                    },
                    checkpoint_dir,
                    "best",
                )

        save_checkpoint(
            {"epoch": pgcfg["epochs"], "G": unwrap_model(G).state_dict()},
            checkpoint_dir,
            "last",
        )
    finally:
        if cfg.get("wandb", {}).get("log_checkpoints", False) and best_checkpoint is not None:
            maybe_log_checkpoint_artifact(run, best_checkpoint, "pre-gan-best")
        finish_wandb_run(run, {"best_g_loss": best_g_loss})

    print(f"\nPre-GAN done. Best G loss: {best_g_loss:.4f}")
    return G, D


def train_post_gan(
    cfg: dict,
    device: torch.device,
    use_synthetic: bool = False,
    run_group: str = "disaster-gan",
):
    print("\n" + "=" * 60)
    print("PHASE 2 - Post-Disaster GAN")
    print("=" * 60)

    gcfg = cfg["gan"]
    pogcfg = gcfg["post_gan"]
    use_amp = device.type == "cuda"
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / "post_gan"
    num_disaster_types = len(cfg["generate"]["disaster_types"])

    run = start_wandb_run(
        cfg,
        phase="post_gan",
        group=run_group,
        extra_config={
            "epochs": pogcfg["epochs"],
            "batch_size": pogcfg["batch_size"],
            "g_lr": pogcfg["g_lr"],
            "d_lr": pogcfg["d_lr"],
            "dropout": pogcfg["dropout"],
        },
    )

    G = ImageTranslationGenerator(
        in_channels=3,
        num_disaster_types=num_disaster_types,
        disaster_emb_dim=pogcfg["disaster_emb_dim"],
        base_ch=pogcfg["base_ch"],
        dropout=pogcfg["dropout"],
    ).to(device)
    D = PatchDiscriminator(
        in_channels=6,
        num_disaster_types=num_disaster_types,
        base_ch=gcfg["disc_base_ch"],
        n_scales=gcfg["disc_n_scales"],
    ).to(device)
    G = maybe_channels_last(G, device)
    D = maybe_channels_last(D, device)
    G = try_compile(G, "G_post")
    D = try_compile(D, "D_post")

    criterion = GANCriterion(
        lambda_adv=gcfg["lambda_adv"],
        lambda_l1=gcfg["lambda_l1"],
        lambda_fm=gcfg["lambda_fm"],
        lambda_perc=gcfg["lambda_perc"],
    )
    opt_G = torch.optim.AdamW(
        G.parameters(),
        lr=pogcfg["g_lr"],
        betas=(0.5, 0.999),
        weight_decay=1e-5,
        fused=(device.type == "cuda"),
    )
    opt_D = torch.optim.AdamW(
        D.parameters(),
        lr=pogcfg["d_lr"],
        betas=(0.5, 0.999),
        weight_decay=1e-5,
        fused=(device.type == "cuda"),
    )
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=pogcfg["epochs"])
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=pogcfg["epochs"])
    scaler = GradScaler("cuda", enabled=use_amp)
    train_loader, val_loader = build_dataloaders(cfg, pogcfg["batch_size"], use_synthetic)

    best_g_loss = float("inf")
    best_checkpoint = None

    try:
        for epoch in range(1, pogcfg["epochs"] + 1):
            epoch_start = time.time()
            G.train()
            D.train()
            sums = {
                "d_loss": 0.0,
                "g_adv": 0.0,
                "g_l1": 0.0,
                "g_fm": 0.0,
                "g_perc": 0.0,
                "g_total": 0.0,
            }
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}[post-GAN]", leave=False):
                real_pre, real_post = move_images(batch, ["pre_image", "post_image"], device)
                disaster_idx = batch["disaster_idx"].to(device, non_blocking=True)

                opt_D.zero_grad(set_to_none=True)
                with amp_context(use_amp):
                    fake_post = G(real_pre, disaster_idx).detach()
                    real_pair = torch.cat([real_pre, real_post], dim=1)
                    fake_pair = torch.cat([real_pre, fake_post], dim=1)
                    real_preds, _ = D(real_pair, disaster_idx)
                    fake_preds, _ = D(fake_pair, disaster_idx)
                    d_loss = criterion.d_loss(real_preds, fake_preds)
                if torch.isfinite(d_loss):
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(opt_D)
                    nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                    scaler.step(opt_D)
                    scaler.update()

                opt_G.zero_grad(set_to_none=True)
                with amp_context(use_amp):
                    fake_post = G(real_pre, disaster_idx)
                    fake_pair = torch.cat([real_pre, fake_post], dim=1)
                    real_pair = torch.cat([real_pre, real_post], dim=1)
                    fake_preds, fake_feats = D(fake_pair, disaster_idx, return_features=True)
                    _real_preds, real_feats = D(real_pair, disaster_idx, return_features=True)
                    g_loss, info = criterion.g_loss_post(
                        fake_preds,
                        fake_feats,
                        real_feats,
                        fake_post,
                        real_post,
                    )
                if torch.isfinite(g_loss):
                    scaler.scale(g_loss).backward()
                    scaler.unscale_(opt_G)
                    nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                    scaler.step(opt_G)
                    scaler.update()

                sums["d_loss"] += float(d_loss.item())
                for key in ["g_adv", "g_l1", "g_fm", "g_perc", "g_total"]:
                    sums[key] += float(info[key])
                n_batches += 1

            sched_G.step()
            sched_D.step()
            denom = max(n_batches, 1)

            G.eval()
            val_l1 = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pre, post = move_images(batch, ["pre_image", "post_image"], device)
                    disaster_idx = batch["disaster_idx"].to(device, non_blocking=True)
                    with amp_context(use_amp):
                        fake = G(pre, disaster_idx)
                    val_l1 += torch.nn.functional.l1_loss(fake, post).item()
            val_l1 /= max(len(val_loader), 1)
            G.train()

            epoch_time = time.time() - epoch_start
            g_avg = sums["g_total"] / denom
            print(
                f"  Ep {epoch:3d}/{pogcfg['epochs']} | "
                f"D={sums['d_loss']/denom:.4f} G={g_avg:.4f} "
                f"(L1={sums['g_l1']/denom:.4f} fm={sums['g_fm']/denom:.4f}) | "
                f"val_L1={val_l1:.4f} | {epoch_time:.1f}s"
            )

            log_wandb(
                run,
                {
                    "epoch": epoch,
                    "post_gan/d_loss": sums["d_loss"] / denom,
                    "post_gan/g_total": g_avg,
                    "post_gan/g_adv": sums["g_adv"] / denom,
                    "post_gan/g_l1": sums["g_l1"] / denom,
                    "post_gan/g_fm": sums["g_fm"] / denom,
                    "post_gan/g_perc": sums["g_perc"] / denom,
                    "post_gan/val_l1": val_l1,
                    "post_gan/g_lr": opt_G.param_groups[0]["lr"],
                    "post_gan/d_lr": opt_D.param_groups[0]["lr"],
                    "post_gan/epoch_time_sec": epoch_time,
                },
            )

            if g_avg < best_g_loss:
                best_g_loss = g_avg
                best_checkpoint = save_checkpoint(
                    {
                        "epoch": epoch,
                        "G": unwrap_model(G).state_dict(),
                        "D": unwrap_model(D).state_dict(),
                        "val_l1": val_l1,
                    },
                    checkpoint_dir,
                    "best",
                )

        save_checkpoint(
            {"epoch": pogcfg["epochs"], "G": unwrap_model(G).state_dict()},
            checkpoint_dir,
            "last",
        )
    finally:
        if cfg.get("wandb", {}).get("log_checkpoints", False) and best_checkpoint is not None:
            maybe_log_checkpoint_artifact(run, best_checkpoint, "post-gan-best")
        finish_wandb_run(run, {"best_g_loss": best_g_loss})

    print(f"\nPost-GAN done. Best G loss: {best_g_loss:.4f}")
    return G, D


def train_damage_model(
    cfg: dict,
    device: torch.device,
    use_synthetic: bool = False,
    run_group: str = "disaster-gan",
):
    print("\n" + "=" * 60)
    print("PHASE 3 - Siamese U-Net Transformer")
    print("=" * 60)

    dcfg = cfg["damage_model"]
    tcfg = cfg["train"]
    use_amp = device.type == "cuda"
    use_prefetch = use_amp and tcfg.get("damage_use_cuda_prefetch", True)
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / "damage"

    run = start_wandb_run(
        cfg,
        phase="damage_model",
        group=run_group,
        extra_config={
            "epochs": tcfg["damage_epochs"],
            "batch_size": tcfg["damage_batch_size"],
            "learning_rate": tcfg["damage_lr"],
            "architecture": dcfg.get("architecture", "siamese_unet_transformer"),
            "transformer_heads": dcfg.get("transformer_heads", 8),
            "transformer_layers": dcfg.get("transformer_layers", 2),
            "cuda_prefetch": use_prefetch,
        },
    )

    model = SiameseUNet(
        in_channels=dcfg["in_channels"],
        num_damage_classes=dcfg["num_classes"],
        base_channels=dcfg.get("base_channels", 64),
        dropout=dcfg.get("dropout", 0.15),
        transformer_heads=dcfg.get("transformer_heads", 8),
        transformer_layers=dcfg.get("transformer_layers", 2),
        transformer_mlp_ratio=dcfg.get("transformer_mlp_ratio", 2.0),
    ).to(device)
    model = maybe_channels_last(model, device)
    model = try_compile(model, "SiameseUNetTransformer")

    criterion = DamageAssessmentLoss(
        seg_weight=tcfg["seg_weight"],
        cls_weight=tcfg["cls_weight"],
        class_weights=torch.tensor(dcfg.get("class_weights", []), dtype=torch.float32)
        if dcfg.get("class_weights")
        else None,
        focal_gamma=dcfg.get("focal_gamma", 2.5),
        num_classes=dcfg["num_classes"],
    )
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["damage_lr"],
        weight_decay=tcfg["damage_weight_decay"],
        fused=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = GradScaler("cuda", enabled=use_amp)
    train_loader, val_loader = build_dataloaders(cfg, tcfg["damage_batch_size"], use_synthetic)
    train_batches = maybe_prefetch_loader(train_loader, device, enabled=use_prefetch)
    val_batches = maybe_prefetch_loader(val_loader, device, enabled=use_prefetch)

    if device.type == "cuda":
        print(f"[Damage] CUDA prefetch={'on' if use_prefetch else 'off'}")

    best_f1 = 0.0
    best_checkpoint = None

    try:
        for epoch in range(1, tcfg["damage_epochs"] + 1):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_batches, total=len(train_loader), desc=f"Ep{epoch:03d}[damage]", leave=False):
                pre_img = batch["pre_image"]
                post_img = batch["post_image"]
                pre_mask = batch["pre_mask"]
                post_mask = batch["post_mask"]

                optimizer.zero_grad(set_to_none=True)
                with amp_context(use_amp):
                    out = model(pre_img, post_img)
                    loss, info = criterion(
                        out["damage_logits"],
                        out["loc_logits"],
                        post_mask,
                        pre_mask,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += info["total_loss"]

            model.eval()
            val_metrics = {"mIoU": 0.0, "f1_loc": 0.0, "f1_cls": 0.0, "f1_comb": 0.0}
            with torch.inference_mode():
                for batch in val_batches:
                    pre_img = batch["pre_image"]
                    post_img = batch["post_image"]
                    post_mask = batch["post_mask"]
                    with amp_context(use_amp):
                        out = model(pre_img, post_img)
                    metrics = batch_metrics(out["damage_logits"], post_mask)
                    for key in val_metrics:
                        val_metrics[key] += metrics[key]

            for key in val_metrics:
                val_metrics[key] /= max(len(val_loader), 1)

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["f1_comb"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"  LR reduced: {old_lr:.2e} -> {new_lr:.2e}")

            epoch_time = time.time() - epoch_start
            mean_train_loss = epoch_loss / max(len(train_loader), 1)
            print(
                f"  Ep {epoch:3d}/{tcfg['damage_epochs']} | "
                f"loss={mean_train_loss:.4f} | "
                f"mIoU={val_metrics['mIoU']:.4f} F1={val_metrics['f1_comb']:.4f} | "
                f"{epoch_time:.1f}s"
            )

            log_wandb(
                run,
                {
                    "epoch": epoch,
                    "damage/train_loss": mean_train_loss,
                    "damage/val_mIoU": val_metrics["mIoU"],
                    "damage/val_f1_loc": val_metrics["f1_loc"],
                    "damage/val_f1_cls": val_metrics["f1_cls"],
                    "damage/val_f1_comb": val_metrics["f1_comb"],
                    "damage/lr": new_lr,
                    "damage/epoch_time_sec": epoch_time,
                },
            )

            if val_metrics["f1_comb"] > best_f1:
                best_f1 = val_metrics["f1_comb"]
                best_checkpoint = save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": unwrap_model(model).state_dict(),
                        "val_f1": best_f1,
                    },
                    checkpoint_dir,
                    "best",
                )
    finally:
        if cfg.get("wandb", {}).get("log_checkpoints", False) and best_checkpoint is not None:
            maybe_log_checkpoint_artifact(run, best_checkpoint, "damage-model-best")
        finish_wandb_run(run, {"best_f1_comb": best_f1})

    print(f"\nDamage model done. Best F1: {best_f1:.4f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaster GAN training")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="1=Pre GAN 2=Post GAN 3=Damage (same trainer as Siamese U-Net Transformer/train_siamese.py)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--all", action="store_true", help="Run all three phases")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = configure_gpu()

    torch.manual_seed(cfg["project"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["project"]["seed"])

    run_group = f"disaster-pipeline-{time.strftime('%Y%m%d-%H%M%S')}"

    if args.all:
        train_pre_gan(cfg, device, args.synthetic, run_group)
        train_post_gan(cfg, device, args.synthetic, run_group)
        train_damage_model(cfg, device, args.synthetic, run_group)
    elif args.phase == 1:
        train_pre_gan(cfg, device, args.synthetic, run_group)
    elif args.phase == 2:
        train_post_gan(cfg, device, args.synthetic, run_group)
    elif args.phase == 3:
        print(
            "[Phase 3] Using the same shared trainer that backs "
            "'Siamese U-Net Transformer/train_siamese.py'."
        )
        train_damage_model(cfg, device, args.synthetic, run_group)
    else:
        raise ValueError("Unknown phase. Use 1, 2, or 3.")


if __name__ == "__main__":
    main()
