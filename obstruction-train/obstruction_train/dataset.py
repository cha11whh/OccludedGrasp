import io
import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & ~eroded


def bbox_from_masks(mask_a: np.ndarray, mask_b: np.ndarray, pad_frac: float = 0.20):
    mask = mask_a | mask_b
    ys, xs = np.where(mask)
    if xs.size == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    h, w = mask.shape
    pad = int(max(x2 - x1, y2 - y1) * pad_frac)
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def crop_resize(arr: np.ndarray, box, size: int, interpolation) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def normalize_depth(depth: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    depth = depth.astype(np.float32)
    if valid is None:
        vals = depth[np.isfinite(depth)]
    else:
        vals = depth[valid & np.isfinite(depth)]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_contact_heatmap(point, box, size: int, sigma: float = 5.0) -> np.ndarray:
    heat = np.zeros((size, size), dtype=np.float32)
    if not point:
        return heat
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return heat
    px = (float(point["x"]) - x1) * size / (x2 - x1)
    py = (float(point["y"]) - y1) * size / (y2 - y1)
    if px < 0 or py < 0 or px >= size or py >= size:
        return heat
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
    return heat.astype(np.float32)


class UnoBenchPairDataset(Dataset):
    def __init__(
        self,
        unobench_root,
        pair_jsonl,
        image_size=224,
        pad_frac=0.20,
        external_depth_dir=None,
        depth_transform="identity",
    ):
        self.root = Path(unobench_root)
        self.rows = read_jsonl(Path(pair_jsonl))
        self.image_size = int(image_size)
        self.pad_frac = float(pad_frac)
        self.external_depth_dir = Path(external_depth_dir) if external_depth_dir else None
        self.depth_transform = depth_transform
        self._images_zip = None
        self._meta_zip = None

    def __len__(self):
        return len(self.rows)

    @property
    def images_zip(self):
        if self._images_zip is None:
            self._images_zip = zipfile.ZipFile(self.root / "images.zip")
        return self._images_zip

    @property
    def meta_zip(self):
        if self._meta_zip is None:
            self._meta_zip = zipfile.ZipFile(self.root / "meta_data" / "annotations_meta.zip")
        return self._meta_zip

    def read_rgb(self, member: str):
        with self.images_zip.open(member) as f:
            data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(member)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def read_npz(self, member: str):
        with self.meta_zip.open(member) as f:
            return np.load(io.BytesIO(f.read()))

    def read_external_depth(self, image_path: str):
        if self.external_depth_dir is None:
            return None
        stem = Path(image_path).stem
        path = self.external_depth_dir / f"{stem}_raw.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path).astype(np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb = self.read_rgb(row["image_path"])
        meta = self.read_npz(row["npz_path"])
        depth = self.read_external_depth(row["image_path"])
        if depth is None:
            depth = meta["depth"].astype(np.float32)
        inst = meta["instances_objects"].astype(np.int32)

        if rgb.shape[:2] != inst.shape[:2]:
            rgb = cv2.resize(rgb, (inst.shape[1], inst.shape[0]), interpolation=cv2.INTER_LINEAR)
        if depth.shape[:2] != inst.shape[:2]:
            depth = cv2.resize(depth, (inst.shape[1], inst.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_a = inst == int(row["obj_a"])
        mask_b = inst == int(row["obj_b"])
        box = bbox_from_masks(mask_a, mask_b, self.pad_frac)

        rgb_crop = crop_resize(rgb, box, self.image_size, cv2.INTER_LINEAR).astype(np.float32) / 255.0
        depth_crop = crop_resize(depth, box, self.image_size, cv2.INTER_LINEAR)
        ma = crop_resize(mask_a.astype(np.uint8), box, self.image_size, cv2.INTER_NEAREST).astype(bool)
        mb = crop_resize(mask_b.astype(np.uint8), box, self.image_size, cv2.INTER_NEAREST).astype(bool)
        ba = boundary(ma)
        bb = boundary(mb)

        valid = ma | mb
        depth_crop = normalize_depth(depth_crop, valid)
        if self.depth_transform == "inverse":
            depth_crop = 1.0 - depth_crop
        elif self.depth_transform != "identity":
            raise ValueError(f"unknown depth_transform: {self.depth_transform}")

        x = np.concatenate(
            [
                rgb_crop.transpose(2, 0, 1),
                depth_crop[None, ...],
                ma[None, ...].astype(np.float32),
                mb[None, ...].astype(np.float32),
                ba[None, ...].astype(np.float32),
                bb[None, ...].astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        label = int(row["label"])
        if label == 0:
            depth_order = 0
        elif label == 1:
            depth_order = 1
        else:
            depth_order = 2

        heat = make_contact_heatmap(row.get("point"), box, self.image_size)
        ratio = np.array([float(row.get("mask_ratio", 0.0))], dtype=np.float32)

        return {
            "image": torch.from_numpy(x),
            "relation": torch.tensor(label, dtype=torch.long),
            "depth_order": torch.tensor(depth_order, dtype=torch.long),
            "ratio": torch.from_numpy(ratio),
            "contact_heatmap": torch.from_numpy(heat[None, ...]),
        }
