"""
src/models/gan.py
─────────────────────────────────────────────────────────────────────────────
Conditional GAN pipeline for Disaster Satellite Image Generation

Replaces the VQ-VAE + ARM (PixelCNN) two-stage pipeline with two paired GANs:

  Stage 1 ─ Pre-Disaster GAN
    G_pre  : (z ∈ R^latent_dim, disaster_idx) → pre-disaster image (256×256)
    D_pre  : (image, disaster_idx)             → real / fake score

  Stage 2 ─ Post-Disaster GAN (Pix2Pix-style)
    G_post : (pre_image, disaster_idx)         → post-disaster image
    D_post : (pre_image ⊕ image, disaster_idx) → real / fake score

Architecture highlights
  • Generators     — Residual blocks + AdaIN conditioning
                     Pre-GAN uses DCGAN-style progressive upsample
                     Post-GAN uses U-Net encoder/decoder with skip connections
  • Discriminator  — Multi-scale PatchGAN (2 scales) with spectral normalisation
  • Loss           — LSGAN + L1 (post only) + perceptual VGG16 + feature matching
  • AMP-compatible — all ops safe under torch.cuda.amp.autocast
  • RTX 2070 Super — batch_size 16 (pre) / 8 (post) at 256×256 in FP16

References
  Pix2Pix  : Isola et al. CVPR 2017  https://arxiv.org/abs/1611.07004
  LSGAN    : Mao et al.  ICCV 2017  https://arxiv.org/abs/1611.04076
  PatchGAN : Li & Wand   ECCV 2016  https://arxiv.org/abs/1601.04589
  SpecNorm : Miyato et al.ICLR 2018 https://arxiv.org/abs/1802.05957
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def SN(module: nn.Module) -> nn.Module:
    """Apply spectral normalisation — keeps discriminator Lipschitz."""
    return nn.utils.spectral_norm(module)


# ─────────────────────────────────────────────
#  Conditioning: Adaptive Instance Normalisation
# ─────────────────────────────────────────────

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation for feature-level conditioning.
    Takes a conditioning vector and predicts per-channel scale + shift.

    x    : (B, C, H, W)   — feature map to normalise
    cond : (B, cond_dim)  — conditioning vector
    out  : (B, C, H, W)   — normalised + modulated feature map
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        # One linear layer predicts scale and shift together
        self.proj = nn.Linear(cond_dim, channels * 2)
        # Initialise to identity: scale=1, shift=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        params = self.proj(cond).unsqueeze(-1).unsqueeze(-1)   # (B, 2C, 1, 1)
        scale, shift = params.chunk(2, dim=1)
        return x * (1.0 + scale) + shift


# ─────────────────────────────────────────────
#  Generator building blocks
# ─────────────────────────────────────────────

class ResBlockAdaIN(nn.Module):
    """
    Residual block with AdaIN conditioning and optional nearest-neighbour upsample.
    Used in the Pre-image Generator.
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        cond_dim: int,
        upsample: bool = False,
    ):
        super().__init__()
        self.upsample = upsample

        self.adain1 = AdaIN(in_ch,  cond_dim)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.adain2 = AdaIN(out_ch, cond_dim)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # inplace=False is required for AMP/FP16 correctness:
        # autograd needs to re-read the pre-activation tensor during backward,
        # but inplace ops overwrite it before that can happen.
        h = F.relu(self.adain1(x, cond))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.adain2(h, cond))
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Encoder downsampling block: Conv stride-2 + InstanceNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlockAdaIN(nn.Module):
    """
    Decoder upsampling block for the Post-image U-Net.
    Transpose conv to upsample → concatenate skip → AdaIN condition → conv.

    in_ch   : channels from previous decoder layer
    skip_ch : channels from matching encoder skip connection
    out_ch  : output channels
    cond_dim: conditioning vector size
    """

    def __init__(
        self,
        in_ch:    int,
        skip_ch:  int,
        out_ch:   int,
        cond_dim: int,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        fused_ch    = out_ch + skip_ch
        self.adain  = AdaIN(fused_ch, cond_dim)
        self.conv   = nn.Conv2d(fused_ch, out_ch, 3, padding=1)
        self.drop   = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        # Protect against off-by-one from padding
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        h = torch.cat([x, skip], dim=1)
        h = F.relu(self.adain(h, cond))
        h = self.drop(h)
        return self.conv(h)


# ─────────────────────────────────────────────
#  Stage 1 — Pre-Disaster Image Generator
# ─────────────────────────────────────────────

class PreImageGenerator(nn.Module):
    """
    Generates pre-disaster satellite images from Gaussian noise.

    Input : z (B, latent_dim), disaster_idx (B,)
    Output: image (B, 3, 256, 256) — unbounded, ImageNet-normalised space

    Architecture (6 upsamples, 4×4 → 256×256):
        FC  : [z ‖ d_emb] → reshape (B, 512, 4, 4)
        ×6  : ResBlockAdaIN with upsample
        Out : InstanceNorm → Conv2d(3)
    """

    def __init__(
        self,
        latent_dim:         int = 128,
        num_disaster_types: int = 7,
        disaster_emb_dim:   int = 64,
        base_ch:            int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim        = latent_dim + disaster_emb_dim

        self.disaster_emb = nn.Embedding(num_disaster_types + 1, disaster_emb_dim)
        self.fc           = nn.Linear(cond_dim, 4 * 4 * base_ch)

        # Channel schedule: 512 → 256 → 256 → 128 → 64 → 32 → 32
        ch = [base_ch, 256, 256, 128, 64, 32, 32]
        self.blocks = nn.ModuleList([
            ResBlockAdaIN(ch[i], ch[i + 1], cond_dim, upsample=True)
            for i in range(len(ch) - 1)
        ])

        self.out_norm = nn.InstanceNorm2d(ch[-1], affine=True)
        self.out_conv = nn.Conv2d(ch[-1], 3, 3, padding=1)
        self._base_ch  = base_ch
        self._cond_dim = cond_dim

    def forward(
        self,
        z:            torch.Tensor,   # (B, latent_dim)
        disaster_idx: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        d_emb = self.disaster_emb(disaster_idx)             # (B, D)
        cond  = torch.cat([z, d_emb], dim=1)                # (B, cond_dim)
        h = self.fc(cond).view(-1, self._base_ch, 4, 4)

        for block in self.blocks:
            h = block(h, cond)

        h = F.relu(self.out_norm(h))
        return self.out_conv(h)                              # (B, 3, 256, 256)

    @torch.no_grad()
    def sample(
        self,
        batch_size:   int,
        disaster_idx: torch.Tensor,
        device:       torch.device,
        temperature:  float = 1.0,
    ) -> torch.Tensor:
        """Sample `batch_size` images for a given disaster type."""
        z = torch.randn(batch_size, self.latent_dim, device=device) * temperature
        return self(z, disaster_idx)


# ─────────────────────────────────────────────
#  Stage 2 — Post-Disaster Image Generator
# ─────────────────────────────────────────────

class ImageTranslationGenerator(nn.Module):
    """
    Translates a pre-disaster image into a post-disaster image.
    Conditioned on disaster type via AdaIN in every decoder block.

    Architecture: U-Net encoder/decoder (Pix2Pix style)
        Encoder  : 5 DownBlocks  (256 → 8)
        Bottleneck: 2 residual convs + identity residual
        Decoder  : 5 UpBlockAdaIN (8 → 256) with skip connections

    Input : pre_image (B, 3, H, W), disaster_idx (B,)
    Output: post_image (B, 3, H, W) — same spatial resolution
    """

    def __init__(
        self,
        in_channels:        int   = 3,
        num_disaster_types: int   = 7,
        disaster_emb_dim:   int   = 128,
        base_ch:            int   = 64,
        dropout:            float = 0.5,
    ):
        super().__init__()
        self.disaster_emb = nn.Embedding(num_disaster_types + 1, disaster_emb_dim)
        c        = base_ch
        cond_dim = disaster_emb_dim

        # ── Encoder ─────────────────────────────────────────────────
        # 256→128→64→32→16→8  (5 downsamples)
        self.enc1 = DownBlock(in_channels, c,    norm=False)  # no IN on first layer
        self.enc2 = DownBlock(c,     c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.enc4 = DownBlock(c * 4, c * 8)
        self.enc5 = DownBlock(c * 8, c * 8)

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c * 8, c * 8, 3, padding=1),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(c * 8, c * 8, 3, padding=1),
        )

        # ── Decoder ──────────────────────────────────────────────────
        # in_ch, skip_ch, out_ch, cond_dim, dropout
        self.dec4 = UpBlockAdaIN(c * 8, c * 8, c * 8, cond_dim, dropout)  # 8→16
        self.dec3 = UpBlockAdaIN(c * 8, c * 4, c * 4, cond_dim, dropout)  # 16→32
        self.dec2 = UpBlockAdaIN(c * 4, c * 2, c * 2, cond_dim)           # 32→64
        self.dec1 = UpBlockAdaIN(c * 2, c,     c,     cond_dim)           # 64→128

        # Final upsample 128 → 256
        self.out = nn.ConvTranspose2d(c, in_channels, 4, stride=2, padding=1)

    def forward(
        self,
        pre_img:      torch.Tensor,   # (B, 3, H, W)
        disaster_idx: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        cond = self.disaster_emb(disaster_idx)  # (B, cond_dim)

        # Encode
        e1 = self.enc1(pre_img)    # (B,  c,   128, 128)
        e2 = self.enc2(e1)         # (B, 2c,   64,  64)
        e3 = self.enc3(e2)         # (B, 4c,   32,  32)
        e4 = self.enc4(e3)         # (B, 8c,   16,  16)
        e5 = self.enc5(e4)         # (B, 8c,   8,   8)

        # Bottleneck (residual)
        bn = e5 + self.bottleneck(e5)

        # Decode with skip connections + AdaIN conditioning
        d = self.dec4(bn, e4, cond)   # (B, 8c,  16,  16)
        d = self.dec3(d,  e3, cond)   # (B, 4c,  32,  32)
        d = self.dec2(d,  e2, cond)   # (B, 2c,  64,  64)
        d = self.dec1(d,  e1, cond)   # (B,  c,  128, 128)

        return self.out(d)             # (B, 3,   256, 256)


# ─────────────────────────────────────────────
#  Multi-Scale PatchGAN Discriminator
# ─────────────────────────────────────────────

class _SingleScaleDisc(nn.Module):
    """5-layer PatchGAN at a single scale (≈ 70×70 receptive field)."""

    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()

        def block(ic, oc, stride, norm=True):
            layers = [SN(nn.Conv2d(ic, oc, 4, stride=stride, padding=1))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.net = nn.Sequential(
            *block(in_ch,      base_ch,     stride=2, norm=False),
            *block(base_ch,    base_ch * 2, stride=2),
            *block(base_ch*2,  base_ch * 4, stride=2),
            *block(base_ch*4,  base_ch * 8, stride=1),
            SN(nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1)),
        )
        # Also expose intermediate features for feature-matching loss
        # We split net into 5 stages for feature access
        nets = list(self.net.children())
        # Group: [first_block, 2nd, 3rd, 4th, last_conv]
        self.stages = nn.ModuleList([
            nn.Sequential(*nets[:3]),    # stage 0
            nn.Sequential(*nets[3:6]),   # stage 1
            nn.Sequential(*nets[6:9]),   # stage 2
            nn.Sequential(*nets[9:12]),  # stage 3
            nn.Sequential(*nets[12:]),   # stage 4 (output)
        ])

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        if return_features:
            return feats[-1], feats[:-1]
        return feats[-1], []


class PatchDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    Operates at 2 scales (original + 0.5×) for richer gradient signal.

    Stage 1 input  : image (3 channels)
    Stage 2 input  : pre_image ⊕ target_image (6 channels) — conditional

    Disaster type is injected by tiling a projected embedding as extra channels.
    """

    def __init__(
        self,
        in_channels:        int = 3,    # 3 for pre-GAN, 6 for post-GAN
        num_disaster_types: int = 7,
        disaster_emb_dim:   int = 16,   # small — used as extra input channels
        n_scales:           int = 2,
        base_ch:            int = 64,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.disaster_emb = nn.Embedding(num_disaster_types + 1, disaster_emb_dim)

        total_in = in_channels + disaster_emb_dim
        self.discs = nn.ModuleList([
            _SingleScaleDisc(total_in, base_ch) for _ in range(n_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(
        self,
        img:          torch.Tensor,    # (B, C, H, W)
        disaster_idx: torch.Tensor,    # (B,)
        return_features: bool = False,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Returns:
            predictions : list of patch tensors, one per scale
            features    : list of feature lists (for feature matching)
        """
        d_emb = self.disaster_emb(disaster_idx)   # (B, D)
        preds, feats_all = [], []

        x = img
        for disc in self.discs:
            B, _, H, W = x.shape
            d_spatial   = d_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            x_in        = torch.cat([x, d_spatial], dim=1)
            pred, feats = disc(x_in, return_features)
            preds.append(pred)
            feats_all.append(feats)
            x = self.downsample(x)

        return preds, feats_all


# ─────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss.
    D loss : 0.5 * [mean((D(real) - 1)²) + mean(D(fake)²)]
    G loss : 0.5 *  mean((D(fake) - 1)²)

    More stable training than vanilla BCE; avoids vanishing gradients.
    """

    def discriminator_loss(
        self,
        real_preds: List[torch.Tensor],
        fake_preds: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for rp, fp in zip(real_preds, fake_preds):
            loss = loss + 0.5 * ((rp - 1.0).pow(2).mean() + fp.pow(2).mean())
        return loss / len(real_preds)

    def generator_loss(
        self,
        fake_preds: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for fp in fake_preds:
            loss = loss + 0.5 * (fp - 1.0).pow(2).mean()
        return loss / len(fake_preds)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching: penalise distance between discriminator intermediate
    features for real vs fake images. Stabilises generator training by
    providing a dense gradient signal.  Weight λ=10 following Pix2Pix.
    """

    def forward(
        self,
        real_feats: List[List[torch.Tensor]],
        fake_feats: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        loss = 0.0
        count = 0
        for r_scale, f_scale in zip(real_feats, fake_feats):
            for rf, ff in zip(r_scale, f_scale):
                loss  = loss + F.l1_loss(ff, rf.detach())
                count += 1
        return loss / max(count, 1)


class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss using relu2_2 and relu4_2 features.
    Loaded lazily so it does not slow down model init.
    Frozen — no gradients through VGG.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self._vgg   = None
        self._device = device

    def _build(self, device: torch.device):
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Extract features up to relu4_2 (index 23 in features)
        self._vgg   = nn.Sequential(*list(vgg.features.children())[:24]).to(device)
        for p in self._vgg.parameters():
            p.requires_grad_(False)
        self._slice1 = list(range(0,  5))   # relu1_2
        self._slice2 = list(range(5,  10))  # relu2_2
        self._slice3 = list(range(10, 24))  # relu4_2

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for i, layer in enumerate(self._vgg):
            x = layer(x)
            if i in (4, 9, 23):
                feats.append(x)
        return feats

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        device = fake.device
        if self._vgg is None:
            self._build(device)

        # VGG expects [0,1] input — clamp & denormalize from ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        fake_n = (fake * std + mean).clamp(0, 1)
        real_n = (real * std + mean).clamp(0, 1)

        fake_feats = self._extract(fake_n)
        real_feats = self._extract(real_n)

        loss = sum(F.l1_loss(ff, rf.detach())
                   for ff, rf in zip(fake_feats, real_feats))
        return loss / len(fake_feats)


# ─────────────────────────────────────────────
#  Combined GAN Loss (convenience wrapper)
# ─────────────────────────────────────────────

class GANCriterion:
    """
    Aggregates all generator losses into a single scalar.

    stage1 (pre-GAN):
        G_loss = λ_adv * LSGAN_G + λ_perc * Perceptual

    stage2 (post-GAN):
        G_loss = λ_adv * LSGAN_G + λ_l1 * L1 + λ_fm * FeatMatch + λ_perc * Perceptual
    """

    def __init__(
        self,
        lambda_adv:  float = 1.0,
        lambda_l1:   float = 10.0,   # L1 for post-image paired supervision
        lambda_fm:   float = 10.0,   # feature matching weight
        lambda_perc: float = 0.1,    # perceptual weight (VGG)
    ):
        self.lsgan    = LSGANLoss()
        self.fm       = FeatureMatchingLoss()
        self.perc     = PerceptualLoss()
        self.λ_adv    = lambda_adv
        self.λ_l1     = lambda_l1
        self.λ_fm     = lambda_fm
        self.λ_perc   = lambda_perc

    def d_loss(
        self,
        real_preds: List[torch.Tensor],
        fake_preds: List[torch.Tensor],
    ) -> torch.Tensor:
        return self.lsgan.discriminator_loss(real_preds, fake_preds)

    def g_loss_pre(
        self,
        fake_preds: List[torch.Tensor],
        fake_img:   torch.Tensor,
        real_img:   torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        adv  = self.lsgan.generator_loss(fake_preds)
        perc = self.perc(fake_img, real_img)
        total = self.λ_adv * adv + self.λ_perc * perc
        return total, {
            "g_adv":  adv.item(),
            "g_perc": perc.item(),
            "g_total": total.item(),
        }

    def g_loss_post(
        self,
        fake_preds:  List[torch.Tensor],
        fake_feats:  List[List[torch.Tensor]],
        real_feats:  List[List[torch.Tensor]],
        fake_img:    torch.Tensor,
        real_img:    torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        adv  = self.lsgan.generator_loss(fake_preds)
        fm   = self.fm(real_feats, fake_feats)
        l1   = F.l1_loss(fake_img, real_img)
        perc = self.perc(fake_img, real_img)
        total = (self.λ_adv  * adv
               + self.λ_fm   * fm
               + self.λ_l1   * l1
               + self.λ_perc * perc)
        return total, {
            "g_adv":   adv.item(),
            "g_fm":    fm.item(),
            "g_l1":    l1.item(),
            "g_perc":  perc.item(),
            "g_total": total.item(),
        }
