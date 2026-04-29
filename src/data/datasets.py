"""
src/data/datasets.py
─────────────────────────────────────────────────────────────────────────────
Data-loading utilities for:
  1. xBD dataset  (pre/post disaster pairs + building damage polygons)
  2. NOAA Emergency Response Imagery  (aerial post-disaster tiles)

xBD download: https://xview2.org/dataset  (free, registration required)
NOAA viewer:  https://storms.ngs.noaa.gov
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────
#  Label constants  (Joint Damage Scale)
# ─────────────────────────────────────────────
DAMAGE_LABELS = {
    "background":   0,
    "no-damage":    1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed":    4,
}
LABEL_COLORS = {
    0: (0,   0,   0),    # background  – black
    1: (0, 255,   0),    # no-damage   – green
    2: (255, 255, 0),    # minor       – yellow
    3: (255, 165, 0),    # major       – orange
    4: (255,   0, 0),    # destroyed   – red
}

DISASTER_TYPES = [
    "hurricane", "wildfire", "flood",
    "earthquake", "tornado", "tsunami", "volcanic_eruption",
]
DISASTER_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DISASTER_TYPES)}


# ─────────────────────────────────────────────
#  Transform helpers
# ─────────────────────────────────────────────
def get_transforms(image_size: int = 256, augment: bool = True):
    """Return image and mask transforms (applied together for alignment)."""
    img_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        img_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ] + img_ops
    return transforms.Compose(img_ops)


def mask_transform(mask: np.ndarray, image_size: int = 256) -> torch.Tensor:
    mask = cv2.resize(mask, (image_size, image_size),
                      interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask).long()


# ─────────────────────────────────────────────
#  xBD Dataset
# ─────────────────────────────────────────────
class XBDDataset(Dataset):
    """
    Loads pre/post disaster image pairs from the xBD dataset.

    Directory layout expected:
        <root>/
          train/
            images/
              <event>_<id>_pre_disaster.png
              <event>_<id>_post_disaster.png
            labels/
              <event>_<id>_pre_disaster.json
              <event>_<id>_post_disaster.json
          test/  (same structure)

    Each JSON label file follows the xBD annotation format:
        { "features": { "xy": [ { "properties": { "uid": ..., "subtype": "no-damage" }, "wkt": "..." }, ... ] } }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",           # "train" | "val" | "test"
        image_size: int = 256,
        augment: bool = True,
        disaster_filter: Optional[List[str]] = None,
        return_masks: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.return_masks = return_masks
        self.disaster_filter = disaster_filter

        self.img_transform = get_transforms(image_size, self.augment)

        # Collect all unique event IDs
        img_dir = self.root / split / "images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"xBD image directory not found: {img_dir}\n"
                "Download xBD from https://xview2.org/dataset"
            )

        pre_paths = sorted(img_dir.glob("*_pre_disaster.png"))
        self.samples: List[Dict] = []

        for pre_path in pre_paths:
            stem = pre_path.stem.replace("_pre_disaster", "")
            post_path = img_dir / f"{stem}_post_disaster.png"
            pre_lbl  = self.root / split / "labels" / f"{stem}_pre_disaster.json"
            post_lbl = self.root / split / "labels" / f"{stem}_post_disaster.json"

            if not post_path.exists():
                continue

            # Infer disaster type from filename (xBD convention: <event>_<id>)
            disaster = self._infer_disaster(stem)
            if disaster_filter and disaster not in disaster_filter:
                continue

            self.samples.append({
                "pre_img":   pre_path,
                "post_img":  post_path,
                "pre_lbl":   pre_lbl if pre_lbl.exists() else None,
                "post_lbl":  post_lbl if post_lbl.exists() else None,
                "disaster":  disaster,
                "stem":      stem,
            })

    def _infer_disaster(self, stem: str) -> str:
        stem_l = stem.lower()
        for d in DISASTER_TYPES:
            if d.replace("_", "") in stem_l.replace("_", ""):
                return d
        return "unknown"

    def _load_damage_mask(self, json_path: Path, is_post: bool) -> np.ndarray:
        """Convert xBD JSON polygon annotations → per-pixel damage mask."""
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        if json_path is None or not json_path.exists():
            return mask
        try:
            with open(json_path) as f:
                data = json.load(f)
            from shapely import wkt
            for feat in data.get("features", {}).get("xy", []):
                subtype = feat["properties"].get("subtype", "no-damage")
                label   = DAMAGE_LABELS.get(subtype, 1)
                polygon = wkt.loads(feat["wkt"])
                coords  = np.array(polygon.exterior.coords, dtype=np.int32)
                if is_post:
                    cv2.fillPoly(mask, [coords], label)
                else:
                    cv2.fillPoly(mask, [coords], 1)  # pre: just building mask
        except Exception:
            pass
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        pre_img  = Image.open(s["pre_img"]).convert("RGB")
        post_img = Image.open(s["post_img"]).convert("RGB")

        # Shared spatial flip seed for alignment
        seed = random.randint(0, 2**32)

        def transform_img(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.img_transform(img)

        pre_t  = transform_img(pre_img)
        post_t = transform_img(post_img)

        disaster_idx = torch.tensor(
            DISASTER_TO_IDX.get(s["disaster"], 0), dtype=torch.long
        )

        sample = {
            "pre_image":    pre_t,
            "post_image":   post_t,
            "disaster_idx": disaster_idx,
            "disaster":     s["disaster"],
            "stem":         s["stem"],
        }

        if self.return_masks:
            pre_mask  = self._load_damage_mask(s["pre_lbl"],  is_post=False)
            post_mask = self._load_damage_mask(s["post_lbl"], is_post=True)
            sample["pre_mask"]  = mask_transform(pre_mask,  self.image_size)
            sample["post_mask"] = mask_transform(post_mask, self.image_size)

        return sample


# ─────────────────────────────────────────────
#  NOAA Imagery Fetcher
# ─────────────────────────────────────────────
class NOAAImageryFetcher:
    """
    Fetch georeferenced aerial tiles from NOAA's Emergency Response
    Imagery WMTS service: https://storms.ngs.noaa.gov

    Usage:
        fetcher = NOAAImageryFetcher(event="florence")
        tile    = fetcher.get_tile(lat=34.2, lon=-77.9, zoom=16)
    """

    BASE_URL = "https://storms.ngs.noaa.gov/storms/{event}/tileserver.php"
    KNOWN_EVENTS = [
        "florence", "michael", "harvey", "irma", "dorian",
        "ida", "ian", "laura", "helene", "ian2022",
    ]

    def __init__(self, event: str = "florence", save_dir: str = "./data/noaa"):
        self.event    = event
        self.save_dir = Path(save_dir) / event
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session  = None

    def _lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to XYZ tile indices."""
        import math
        lat_r = math.radians(lat)
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
        return x, y

    def get_tile(
        self,
        lat: float,
        lon: float,
        zoom: int = 16,
        layer: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Download a single 256×256 tile and return as np.ndarray (H,W,3).
        Returns None if tile is unavailable or network is blocked.
        """
        import requests
        x, y = self._lat_lon_to_tile(lat, lon, zoom)
        url  = f"https://storms.ngs.noaa.gov/storms/{self.event}/tiles/{zoom}/{x}/{y}.jpg"
        cache_path = self.save_dir / f"{zoom}_{x}_{y}.jpg"

        if cache_path.exists():
            img = cv2.imread(str(cache_path))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                cache_path.write_bytes(resp.content)
                img_arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        except Exception as e:
            print(f"[NOAAFetcher] Could not fetch tile: {e}")
        return None

    def build_mosaic(
        self,
        lat_center: float,
        lon_center: float,
        zoom: int = 16,
        radius: int = 2,
    ) -> Optional[np.ndarray]:
        """
        Build a (2*radius+1)×(2*radius+1) tile mosaic centred on lat/lon.
        Returns a large np.ndarray suitable for cropping / patching.
        """
        cx, cy = self._lat_lon_to_tile(lat_center, lon_center, zoom)
        rows = []
        for dy in range(-radius, radius + 1):
            row_tiles = []
            for dx in range(-radius, radius + 1):
                tile = self.get_tile.__func__(self, lat_center, lon_center, zoom)  # reuse
                # Simplified: fetch by direct XYZ
                import requests
                url  = f"https://storms.ngs.noaa.gov/storms/{self.event}/tiles/{zoom}/{cx+dx}/{cy+dy}.jpg"
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        arr = np.frombuffer(resp.content, dtype=np.uint8)
                        t   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        t   = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
                        row_tiles.append(t)
                        continue
                except Exception:
                    pass
                row_tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))
            rows.append(np.concatenate(row_tiles, axis=1))
        return np.concatenate(rows, axis=0)


# ─────────────────────────────────────────────
#  Synthetic demo dataset (no download needed)
# ─────────────────────────────────────────────
class SyntheticDisasterDataset(Dataset):
    """
    Generates synthetic pre/post disaster image pairs on-the-fly for
    unit-testing and demo purposes when no real data is downloaded.

    Post-disaster images simulate damage by:
        - Adding gray/brown noise patches (rubble)
        - Dimming fire-affected regions
        - Flooding low-elevation areas with blue overlay
    """

    DISASTER_SIMS = ["hurricane", "wildfire", "flood", "earthquake", "tornado"]

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 256,
        seed: int = 42,
    ):
        self.n          = num_samples
        self.size       = image_size
        self.rng        = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng  = np.random.RandomState(idx)
        dtype = np.uint8

        # ── Generate pre-disaster image ──────────────────────────────
        # Simple procedural terrain: gradient + noise texture
        pre = np.zeros((self.size, self.size, 3), dtype=dtype)
        # Green/brown base (land)
        pre[:, :, 1] = rng.randint(80, 160, (self.size, self.size))  # G channel
        pre[:, :, 0] = rng.randint(50, 120, (self.size, self.size))  # R channel
        pre[:, :, 2] = rng.randint(20,  80, (self.size, self.size))  # B channel
        # Overlay roads/structures as white-ish rectangles
        num_buildings = rng.randint(5, 20)
        for _ in range(num_buildings):
            x1, y1 = rng.randint(0, self.size-20, 2)
            w  = rng.randint(10, 40)
            h  = rng.randint(10, 40)
            color = rng.randint(180, 240, 3).astype(dtype)
            pre[y1:y1+h, x1:x1+w] = color

        # ── Disaster type ────────────────────────────────────────────
        disaster_idx = idx % len(self.DISASTER_SIMS)
        disaster     = self.DISASTER_SIMS[disaster_idx]

        # ── Generate post-disaster image ─────────────────────────────
        post = pre.copy().astype(np.float32)
        damage_mask = np.zeros((self.size, self.size), dtype=np.uint8)

        if disaster == "wildfire":
            # Char regions dark + orange tint
            n_patches = rng.randint(3, 8)
            for _ in range(n_patches):
                x1, y1 = rng.randint(0, self.size-40, 2)
                r = rng.randint(20, 60)
                Y, X = np.ogrid[:self.size, :self.size]
                circle = (X - x1)**2 + (Y - y1)**2 <= r**2
                post[circle, 0] = np.clip(post[circle, 0] * 0.4 + 80, 0, 255)
                post[circle, 1] = np.clip(post[circle, 1] * 0.2, 0, 255)
                post[circle, 2] = np.clip(post[circle, 2] * 0.1, 0, 255)
                sev = rng.choice([2, 3, 4], p=[0.2, 0.4, 0.4])
                damage_mask[circle] = sev

        elif disaster == "flood":
            # Blue overlay on lower half
            flood_h = rng.randint(self.size//3, 2*self.size//3)
            post[flood_h:, :, 2] = np.clip(post[flood_h:, :, 2] * 0.3 + 120, 0, 255)
            post[flood_h:, :, 0] = np.clip(post[flood_h:, :, 0] * 0.3, 0, 255)
            post[flood_h:, :, 1] = np.clip(post[flood_h:, :, 1] * 0.4, 0, 255)
            damage_mask[flood_h:, :] = rng.choice([1, 2, 3], size=(self.size - flood_h, self.size))

        elif disaster in ("hurricane", "tornado"):
            # Scattered debris patches
            n_patches = rng.randint(5, 15)
            for _ in range(n_patches):
                x1, y1 = rng.randint(0, self.size-30, 2)
                w = rng.randint(15, 50); h = rng.randint(15, 50)
                post[y1:y1+h, x1:x1+w] = rng.randint(80, 140, 3)
                damage_mask[y1:y1+h, x1:x1+w] = rng.choice([2, 3, 4])

        elif disaster == "earthquake":
            # Ground cracks: horizontal/vertical stripes dimmed
            for _ in range(rng.randint(3, 8)):
                orientation = rng.choice(["h", "v"])
                pos = rng.randint(0, self.size)
                if orientation == "h":
                    post[pos:pos+3, :] = 30
                    damage_mask[pos:pos+3, :] = 3
                else:
                    post[:, pos:pos+3] = 30
                    damage_mask[:, pos:pos+3] = 3

        post = np.clip(post, 0, 255).astype(dtype)

        # ── To tensors ───────────────────────────────────────────────
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        to_t = transforms.ToTensor()

        pre_t  = norm(to_t(Image.fromarray(pre)))
        post_t = norm(to_t(Image.fromarray(post)))

        return {
            "pre_image":    pre_t,
            "post_image":   post_t,
            "pre_mask":     torch.zeros(self.size, self.size, dtype=torch.long),
            "post_mask":    torch.from_numpy(damage_mask).long(),
            "disaster_idx": torch.tensor(disaster_idx, dtype=torch.long),
            "disaster":     disaster,
            "stem":         f"synthetic_{idx:05d}",
        }
