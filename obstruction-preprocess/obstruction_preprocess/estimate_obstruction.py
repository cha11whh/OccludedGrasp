import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
]


@dataclass
class ObjectMask:
    obj_id: int
    name: str
    modal_path: Path
    amodal_path: Path
    modal: np.ndarray
    amodal: np.ndarray
    score: Optional[float] = None
    mask_type: str = "unknown"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 0


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()).strip("_")


def severity_from_ratio(ratio: float) -> str:
    if ratio < 0.10:
        return "slightly"
    if ratio < 0.40:
        return "partially"
    if ratio < 0.70:
        return "mostly"
    return "heavily"


def bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    return mask & ~erode(mask, radius)


def depth_resize(depth: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if depth.shape[:2] == shape:
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


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


def transform_depth(depth: np.ndarray, transform: str) -> np.ndarray:
    depth = normalize_depth(depth)
    if transform == "identity":
        return depth
    if transform == "inverse":
        return 1.0 - depth
    raise ValueError(f"unknown depth_transform: {transform}")


def crop_resize(arr: np.ndarray, box: Tuple[int, int, int, int], size: int, interpolation: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def pair_bbox(mask_a: np.ndarray, mask_b: np.ndarray, pad_frac: float) -> Tuple[int, int, int, int]:
    mask = mask_a | mask_b
    ys, xs = np.where(mask)
    h, w = mask.shape
    if xs.size == 0:
        return 0, 0, w, h
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(max(x2 - x1, y2 - y1) * pad_frac)
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def resize_long_side(image: np.ndarray, max_side: int, interpolation: int) -> Tuple[np.ndarray, float, float]:
    if max_side <= 0:
        return image, 1.0, 1.0
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= max_side:
        return image, 1.0, 1.0
    scale = float(max_side) / float(current)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, float(w) / float(new_w), float(h) / float(new_h)


def median_depth(depth: Optional[np.ndarray], mask: np.ndarray) -> Optional[float]:
    if depth is None or not np.any(mask):
        return None
    vals = depth[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def depth_support_score(blocker_depth: Optional[float], blocked_depth: Optional[float], closer: str) -> Optional[float]:
    if blocker_depth is None or blocked_depth is None:
        return None
    delta = blocker_depth - blocked_depth
    if closer == "larger":
        return float(delta)
    if closer == "smaller":
        return float(-delta)
    return float(abs(delta))


def squash_positive(value: Optional[float], scale: float) -> float:
    if value is None:
        return 0.0
    if scale <= 0:
        return 1.0 if value > 0 else 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def resolve_modal_metadata(modal_dir: Path, modal_metadata: Path) -> Dict[str, dict]:
    items = read_json(modal_metadata)
    out = {}
    for idx, item in enumerate(items):
        name = item.get("class_name") or item.get("prompt") or f"object_{idx}"
        out[name] = {
            "path": modal_dir / item["mask"],
            "score": item.get("score"),
            "index": idx,
        }
    return out


def load_objects(
    image_shape: Tuple[int, int],
    modal_dir: Path,
    modal_metadata: Path,
    amodal_dir: Path,
    amodal_metadata: Path,
) -> List[ObjectMask]:
    modal_by_name = resolve_modal_metadata(modal_dir, modal_metadata)
    amodal_items = read_json(amodal_metadata)
    objects = []

    for item in amodal_items:
        name = item["class_name"]
        modal_info = modal_by_name.get(name)
        if modal_info is None:
            raise KeyError(f"No modal mask found for class name: {name}")

        modal_path = Path(modal_info["path"])
        amodal_path = amodal_dir / item["mask"]
        modal = load_mask(modal_path, image_shape)
        amodal = load_mask(amodal_path, image_shape)

        # The amodal/final mask can occasionally be smaller than the visible mask.
        # Keep the visible part in the amodal support to avoid impossible subtraction.
        amodal = amodal | modal

        objects.append(
            ObjectMask(
                obj_id=int(item["id"]),
                name=name,
                modal_path=modal_path,
                amodal_path=amodal_path,
                modal=modal,
                amodal=amodal,
                score=item.get("score", modal_info.get("score")),
                mask_type=item.get("mask_type", "unknown"),
            )
        )

    objects.sort(key=lambda o: o.obj_id)
    return objects


def dedupe_objects(objects: List[ObjectMask], iou_thresh: float) -> Tuple[List[ObjectMask], List[dict]]:
    if iou_thresh > 1.0:
        return objects, []

    order = sorted(
        objects,
        key=lambda o: (float(o.score if o.score is not None else -1.0), int(o.amodal.sum())),
        reverse=True,
    )
    kept: List[ObjectMask] = []
    duplicates: List[dict] = []

    for obj in order:
        matched = None
        matched_iou = 0.0
        for keep in kept:
            iou = mask_iou(obj.modal, keep.modal)
            if iou >= iou_thresh and iou > matched_iou:
                matched = keep
                matched_iou = iou
        if matched is None:
            kept.append(obj)
        else:
            duplicates.append(
                {
                    "removed_id": obj.obj_id,
                    "removed_name": obj.name,
                    "kept_id": matched.obj_id,
                    "kept_name": matched.name,
                    "modal_iou": round(matched_iou, 6),
                    "reason": f"modal IoU >= {iou_thresh}",
                }
            )

    kept.sort(key=lambda o: o.obj_id)
    duplicates.sort(key=lambda d: (d["kept_id"], d["removed_id"]))
    return kept, duplicates


def estimate_pair(
    blocker: ObjectMask,
    blocked: ObjectMask,
    depth: Optional[np.ndarray],
    contact_radius: int,
    min_hidden_area: int,
    min_contact_area: int,
    min_contact_fraction: float,
    min_relation_confidence: float,
    depth_closer: str,
    depth_scale: float,
) -> Optional[dict]:
    if blocker.obj_id == blocked.obj_id:
        return None

    hidden = blocked.amodal & ~blocked.modal
    hidden_area = int(hidden.sum())
    amodal_area = int(blocked.amodal.sum())
    if hidden_area < min_hidden_area or amodal_area == 0:
        return None

    blocker_near_hidden = blocker.modal & dilate(hidden, contact_radius)
    hidden_near_blocker = hidden & dilate(blocker.modal, contact_radius)
    support = blocker_near_hidden | hidden_near_blocker
    contact_area = int(support.sum())
    if contact_area < min_contact_area:
        return None

    hidden_contact_area = int(hidden_near_blocker.sum())
    blocker_contact_area = int(blocker_near_hidden.sum())
    contact_fraction_of_hidden = float(hidden_contact_area) / float(max(1, hidden_area))
    contact_fraction_of_blocker = float(blocker_contact_area) / float(max(1, int(blocker.modal.sum())))
    contact_score = max(contact_fraction_of_hidden, contact_fraction_of_blocker)
    if contact_score < min_contact_fraction:
        return None

    ratio = float(hidden_near_blocker.sum()) / float(amodal_area)

    contact = support
    point = mask_center(contact)
    if point is None:
        return None

    blocked_contact = blocked.modal & dilate(contact, contact_radius)
    if not np.any(blocked_contact):
        blocked_contact = boundary(blocked.modal, max(1, contact_radius // 2)) & dilate(contact, contact_radius * 2)

    blocker_d = median_depth(depth, blocker.modal & dilate(contact, contact_radius))
    blocked_d = median_depth(depth, blocked_contact)
    depth_score = depth_support_score(blocker_d, blocked_d, depth_closer)
    depth_order_score = squash_positive(depth_score, depth_scale)

    # Edge existence is based on geometric contact plus optional depth-order support.
    # Obstruction ratio is kept as an edge feature for downstream policy/RL, not as
    # the graph-building criterion.
    relation_confidence = 0.75 * min(1.0, contact_score / 0.08) + 0.25 * depth_order_score
    if depth_score is None:
        relation_confidence = min(1.0, contact_score / 0.08)
    if relation_confidence < min_relation_confidence:
        return None

    return {
        "blocker": blocker.obj_id,
        "blocked": blocked.obj_id,
        "subject": blocker.obj_id,
        "object": blocked.obj_id,
        "relation": "obstructs",
        "blocker_name": blocker.name,
        "blocked_name": blocked.name,
        "mask_ratio": round(ratio, 6),
        "hidden_area": hidden_area,
        "amodal_area": amodal_area,
        "contact_area": contact_area,
        "degree": severity_from_ratio(ratio),
        "point": {"x": int(point[0]), "y": int(point[1])},
        "contact": {
            "area": contact_area,
            "hidden_contact_area": hidden_contact_area,
            "blocker_contact_area": blocker_contact_area,
            "fraction_of_hidden": round(contact_fraction_of_hidden, 6),
            "fraction_of_blocker": round(contact_fraction_of_blocker, 6),
            "score": round(float(contact_score), 6),
            "radius": contact_radius,
        },
        "depth": {
            "blocker_median": blocker_d,
            "blocked_visible_median": blocked_d,
            "closer_direction": depth_closer,
            "support_score": depth_score,
            "order_score": round(float(depth_order_score), 6),
        },
        "edge_features": {
            "obstruction_ratio": round(ratio, 6),
            "degree": severity_from_ratio(ratio),
            "hidden_area": hidden_area,
            "amodal_area": amodal_area,
            "contact_area": contact_area,
            "contact_score": round(float(contact_score), 6),
            "depth_order_score": round(float(depth_order_score), 6),
        },
        "graph_decision": {
            "criterion": "contact_near_hidden_region_with_optional_depth_order",
            "used_obstruction_ratio_for_edge": False,
            "min_contact_area": min_contact_area,
            "min_contact_fraction": min_contact_fraction,
            "min_relation_confidence": min_relation_confidence,
        },
        "confidence": round(float(relation_confidence), 6),
        "relation_confidence": round(float(relation_confidence), 6),
        "_contact_mask": contact,
        "_hidden_near_blocker": hidden_near_blocker,
    }


def load_relation_model(checkpoint: Path, train_root: Path, model_name: str, image_size: int, device: str):
    if str(train_root) not in sys.path:
        sys.path.insert(0, str(train_root))
    import torch
    from obstruction_train.model import build_model

    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = build_model(model_name, image_size=image_size).to(torch_device)
    ckpt = torch.load(checkpoint, map_location=torch_device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, torch_device, torch


def make_pair_tensor(
    rgb: np.ndarray,
    depth_raw: Optional[np.ndarray],
    obj_a: ObjectMask,
    obj_b: ObjectMask,
    image_size: int,
    pad_frac: float,
    depth_transform: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    box = pair_bbox(obj_a.modal, obj_b.modal, pad_frac)
    rgb_crop = crop_resize(rgb, box, image_size, cv2.INTER_LINEAR).astype(np.float32) / 255.0

    if depth_raw is None:
        depth_crop = np.zeros((image_size, image_size), dtype=np.float32)
    else:
        depth_crop = crop_resize(depth_raw, box, image_size, cv2.INTER_LINEAR)

    ma = crop_resize(obj_a.modal.astype(np.uint8), box, image_size, cv2.INTER_NEAREST).astype(bool)
    mb = crop_resize(obj_b.modal.astype(np.uint8), box, image_size, cv2.INTER_NEAREST).astype(bool)
    ba = boundary(ma)
    bb = boundary(mb)

    if depth_raw is not None:
        depth_crop = normalize_depth(depth_crop, ma | mb)
        if depth_transform == "inverse":
            depth_crop = 1.0 - depth_crop
        elif depth_transform != "identity":
            raise ValueError(f"unknown depth_transform: {depth_transform}")

    x = np.concatenate(
        [
            rgb_crop.transpose(2, 0, 1),
            depth_crop[None, ...].astype(np.float32),
            ma[None, ...].astype(np.float32),
            mb[None, ...].astype(np.float32),
            ba[None, ...].astype(np.float32),
            bb[None, ...].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return x, box


def heatmap_point(heatmap: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    x1, y1, x2, y2 = box
    px = x1 + (float(x) + 0.5) * max(1, x2 - x1) / float(heatmap.shape[1])
    py = y1 + (float(y) + 0.5) * max(1, y2 - y1) / float(heatmap.shape[0])
    return int(round(px)), int(round(py))


def model_pair_record(
    blocker: ObjectMask,
    blocked: ObjectMask,
    relation_confidence: float,
    no_relation_confidence: float,
    predicted_ratio: float,
    point: Tuple[int, int],
    depth: Optional[np.ndarray],
    depth_transform: str,
    contact_radius: int,
    model_class: int,
) -> dict:
    hidden = blocked.amodal & ~blocked.modal
    hidden_area = int(hidden.sum())
    amodal_area = int(blocked.amodal.sum())
    near = dilate(blocker.modal, contact_radius)
    hidden_near_blocker = hidden & near
    if np.any(hidden_near_blocker):
        support = hidden_near_blocker | (blocker.modal & dilate(hidden_near_blocker, contact_radius))
    else:
        support = (blocker.modal & dilate(blocked.modal, contact_radius)) | (blocked.modal & dilate(blocker.modal, contact_radius))
    if not np.any(support):
        support = blocker.modal | blocked.modal

    contact_area = int(support.sum())
    hidden_contact_area = int(hidden_near_blocker.sum())
    blocker_contact_area = int((blocker.modal & dilate(hidden if np.any(hidden) else blocked.modal, contact_radius)).sum())
    contact_score = float(contact_area) / float(max(1, int((blocker.modal | blocked.modal).sum())))
    blocker_d = median_depth(depth, blocker.modal & dilate(support, contact_radius))
    blocked_d = median_depth(depth, blocked.modal & dilate(support, contact_radius))
    mask_ratio = float(hidden_near_blocker.sum()) / float(max(1, amodal_area))
    if mask_ratio <= 0:
        mask_ratio = float(predicted_ratio)

    return {
        "blocker": blocker.obj_id,
        "blocked": blocked.obj_id,
        "subject": blocker.obj_id,
        "object": blocked.obj_id,
        "relation": "obstructs",
        "blocker_name": blocker.name,
        "blocked_name": blocked.name,
        "mask_ratio": round(mask_ratio, 6),
        "hidden_area": hidden_area,
        "amodal_area": amodal_area,
        "contact_area": contact_area,
        "degree": severity_from_ratio(mask_ratio),
        "point": {"x": int(point[0]), "y": int(point[1])},
        "contact": {
            "area": contact_area,
            "hidden_contact_area": hidden_contact_area,
            "blocker_contact_area": blocker_contact_area,
            "fraction_of_hidden": round(float(hidden_contact_area) / float(max(1, hidden_area)), 6),
            "fraction_of_blocker": round(float(blocker_contact_area) / float(max(1, int(blocker.modal.sum()))), 6),
            "score": round(float(contact_score), 6),
            "radius": contact_radius,
        },
        "depth": {
            "blocker_median": blocker_d,
            "blocked_visible_median": blocked_d,
            "closer_direction": "model_input",
            "support_score": None if blocker_d is None or blocked_d is None else blocker_d - blocked_d,
            "order_score": None,
            "depth_transform": depth_transform,
        },
        "edge_features": {
            "obstruction_ratio": round(mask_ratio, 6),
            "predicted_obstruction_ratio": round(float(predicted_ratio), 6),
            "degree": severity_from_ratio(mask_ratio),
            "hidden_area": hidden_area,
            "amodal_area": amodal_area,
            "contact_area": contact_area,
            "contact_score": round(float(contact_score), 6),
            "relation_model_confidence": round(float(relation_confidence), 6),
            "no_relation_confidence": round(float(no_relation_confidence), 6),
        },
        "graph_decision": {
            "criterion": "pair_relation_transformer",
            "used_obstruction_ratio_for_edge": False,
            "model_class": int(model_class),
        },
        "confidence": round(float(relation_confidence), 6),
        "relation_confidence": round(float(relation_confidence), 6),
        "_contact_mask": support,
        "_hidden_near_blocker": hidden_near_blocker if np.any(hidden_near_blocker) else support,
    }


def estimate_pairs_with_model(
    rgb: np.ndarray,
    depth_raw: Optional[np.ndarray],
    depth_for_stats: Optional[np.ndarray],
    objects: List[ObjectMask],
    checkpoint: Path,
    train_root: Path,
    model_name: str,
    image_size: int,
    pad_frac: float,
    depth_transform: str,
    device: str,
    batch_size: int,
    min_confidence: float,
    contact_radius: int,
) -> List[dict]:
    model, torch_device, torch = load_relation_model(checkpoint, train_root, model_name, image_size, device)
    pair_meta = []
    tensors = []
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            x, box = make_pair_tensor(rgb, depth_raw, obj_a, obj_b, image_size, pad_frac, depth_transform)
            tensors.append(x)
            pair_meta.append((obj_a, obj_b, box))

    pairs = []
    if not tensors:
        return pairs

    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(np.stack(tensors[start : start + batch_size])).to(torch_device)
            out = model(batch)
            probs = torch.softmax(out["relation_logits"], dim=1).detach().cpu().numpy()
            ratios = out["ratio"].detach().cpu().numpy().reshape(-1)
            heatmaps = torch.sigmoid(out["contact_heatmap_logits"]).detach().cpu().numpy()[:, 0]
            for offset, prob in enumerate(probs):
                cls = int(np.argmax(prob))
                if cls == 2:
                    continue
                conf = float(prob[cls])
                if conf < min_confidence:
                    continue
                obj_a, obj_b, box = pair_meta[start + offset]
                if cls == 0:
                    blocker, blocked = obj_a, obj_b
                else:
                    blocker, blocked = obj_b, obj_a
                pairs.append(
                    model_pair_record(
                        blocker=blocker,
                        blocked=blocked,
                        relation_confidence=conf,
                        no_relation_confidence=float(prob[2]),
                        predicted_ratio=float(ratios[offset]),
                        point=heatmap_point(heatmaps[offset], box),
                        depth=depth_for_stats,
                        depth_transform=depth_transform,
                        contact_radius=contact_radius,
                        model_class=cls,
                    )
                )
    return pairs


def build_paths(objects: List[ObjectMask], pairs: List[dict]) -> List[dict]:
    incoming: Dict[int, List[int]] = {o.obj_id: [] for o in objects}
    for p in pairs:
        incoming[p["blocked"]].append(p["blocker"])

    def paths_to_top(target: int, seen: Optional[List[int]] = None) -> List[List[int]]:
        seen = list(seen or [])
        if target in seen:
            return [[target]]
        blockers = incoming.get(target, [])
        if not blockers:
            return [[target]]
        paths = []
        for b in blockers:
            for parent_path in paths_to_top(b, seen + [target]):
                paths.append(parent_path + [target])
        return paths

    results = []
    for obj in objects:
        paths = paths_to_top(obj.obj_id)
        if paths == [[obj.obj_id]]:
            paths = []
        results.append(
            {
                "scene_object": obj.obj_id,
                "target_object": obj.obj_id,
                "target_name": obj.name,
                "occlusion_paths": paths,
            }
        )
    return results


def overlay_objects(rgb: np.ndarray, objects: List[ObjectMask]) -> np.ndarray:
    out = rgb.copy()
    for idx, obj in enumerate(objects):
        color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.uint8)
        mask = obj.modal
        out[mask] = (0.55 * out[mask] + 0.45 * color).astype(np.uint8)
        c = mask_center(mask)
        if c:
            cv2.putText(out, f"{obj.obj_id}:{obj.name}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, cv2.LINE_AA)
    return out


def overlay_obstructions(rgb: np.ndarray, pairs: List[dict]) -> np.ndarray:
    out = rgb.copy()
    for idx, p in enumerate(pairs):
        color = PALETTE[idx % len(PALETTE)]
        pt = (p["point"]["x"], p["point"]["y"])
        cv2.circle(out, pt, 18, color, -1)
        cv2.putText(
            out,
            f'{p["blocker"]}->{p["blocked"]} {p["mask_ratio"]:.2f}',
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def save_pair_masks(out_dir: Path, pairs: List[dict], shape: Tuple[int, int]):
    hidden_dir = out_dir / "hidden_regions"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        mask = p.pop("_hidden_near_blocker")
        p.pop("_contact_mask", None)
        rel = Path("hidden_regions") / f'blocked_{p["blocked"]}_by_{p["blocker"]}.png'
        cv2.imwrite(str(out_dir / rel), (mask.astype(np.uint8) * 255))
        p["mask_path"] = str(rel)


def main():
    parser = argparse.ArgumentParser(description="Estimate UnoGrasp-style obstruction metadata from RGB-D and masks.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", default=None, help="Depth map path, .npy or image. Optional.")
    parser.add_argument("--modal-dir", required=True)
    parser.add_argument("--modal-metadata", required=True)
    parser.add_argument("--amodal-dir", required=True)
    parser.add_argument("--amodal-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-id", default="scene")
    parser.add_argument("--view-id", default="0")
    parser.add_argument("--contact-radius", type=int, default=24)
    parser.add_argument("--min-hidden-area", type=int, default=200)
    parser.add_argument("--min-contact-area", type=int, default=80)
    parser.add_argument("--min-contact-fraction", type=float, default=0.001)
    parser.add_argument("--min-relation-confidence", type=float, default=0.01)
    parser.add_argument("--min-ratio", type=float, default=0.0, help="Deprecated: mask ratio is now saved as an edge feature, not used to build graph edges.")
    parser.add_argument("--dedupe-iou", type=float, default=0.98, help="Merge nearly identical modal masks. Use >1 to disable.")
    parser.add_argument("--depth-closer", choices=["larger", "smaller", "absolute"], default="larger")
    parser.add_argument("--depth-transform", choices=["identity", "inverse"], default="identity")
    parser.add_argument("--depth-scale", type=float, default=0.25, help="Depth score normalization scale for relation confidence.")
    parser.add_argument("--relation-mode", choices=["heuristic", "model"], default="heuristic")
    parser.add_argument("--relation-model-checkpoint", default="/home/ubuntu22-zy/occlusion/obstruction-train/outputs/pair_transformer_v1_base_da_vitl_ft_e1/best.pt")
    parser.add_argument("--relation-model-root", default="/home/ubuntu22-zy/occlusion/obstruction-train")
    parser.add_argument("--relation-model-name", default="pair_transformer_base", choices=["pair_transformer_small", "pair_transformer_base", "pair_transformer_large"])
    parser.add_argument("--relation-model-image-size", type=int, default=224)
    parser.add_argument("--relation-model-pad-frac", type=float, default=0.20)
    parser.add_argument("--relation-model-batch-size", type=int, default=64)
    parser.add_argument("--relation-model-device", default="cuda")
    parser.add_argument("--relation-model-min-confidence", type=float, default=0.50)
    parser.add_argument("--save-visualizations", type=str2bool, default=True)
    parser.add_argument("--process-max-side", type=int, default=0, help="Downscale RGB/depth/masks for graph estimation. 0 keeps full resolution.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_bgr = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(args.rgb)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb, point_scale_x, point_scale_y = resize_long_side(rgb, args.process_max_side, cv2.INTER_AREA)
    shape = rgb.shape[:2]

    depth_raw = None
    depth = None
    if args.depth:
        depth_path = Path(args.depth)
        if depth_path.suffix.lower() == ".npy":
            depth_raw = np.load(depth_path).astype(np.float32)
        else:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise FileNotFoundError(depth_path)
            depth_raw = depth_img.astype(np.float32)
        depth_raw = depth_resize(depth_raw, shape)
        depth = transform_depth(depth_raw, args.depth_transform)

    objects = load_objects(
        shape,
        Path(args.modal_dir),
        Path(args.modal_metadata),
        Path(args.amodal_dir),
        Path(args.amodal_metadata),
    )
    objects, duplicates = dedupe_objects(objects, args.dedupe_iou)

    if args.relation_mode == "model":
        pairs = estimate_pairs_with_model(
            rgb=rgb,
            depth_raw=depth_raw,
            depth_for_stats=depth,
            objects=objects,
            checkpoint=Path(args.relation_model_checkpoint),
            train_root=Path(args.relation_model_root),
            model_name=args.relation_model_name,
            image_size=args.relation_model_image_size,
            pad_frac=args.relation_model_pad_frac,
            depth_transform=args.depth_transform,
            device=args.relation_model_device,
            batch_size=args.relation_model_batch_size,
            min_confidence=args.relation_model_min_confidence,
            contact_radius=args.contact_radius,
        )
    else:
        pairs = []
        for blocked in objects:
            for blocker in objects:
                pair = estimate_pair(
                    blocker=blocker,
                    blocked=blocked,
                    depth=depth,
                    contact_radius=args.contact_radius,
                    min_hidden_area=args.min_hidden_area,
                    min_contact_area=args.min_contact_area,
                    min_contact_fraction=args.min_contact_fraction,
                    min_relation_confidence=args.min_relation_confidence,
                    depth_closer=args.depth_closer,
                    depth_scale=args.depth_scale,
            )
                if pair is not None:
                    pairs.append(pair)

    pairs.sort(key=lambda x: (x["blocked"], -x["relation_confidence"], x["blocker"]))
    save_pair_masks(out_dir, pairs, shape)

    occ_detail = []
    for p in pairs:
        item = dict(p)
        item["point"] = {
            "x": int(round(float(item["point"]["x"]) * point_scale_x)),
            "y": int(round(float(item["point"]["y"]) * point_scale_y)),
        }
        item.update(
            {
                "scene_id": str(args.scene_id),
                "view_id": str(args.view_id),
                "obj1": int(p["blocker"]),
                "obj2": int(p["blocked"]),
            }
        )
        occ_detail.append(item)

    objects_json = []
    for obj in objects:
        objects_json.append(
            {
                "id": obj.obj_id,
                "name": obj.name,
                "score": obj.score,
                "mask_type": obj.mask_type,
                "modal_path": str(obj.modal_path),
                "amodal_path": str(obj.amodal_path),
                "modal_area": int(obj.modal.sum()),
                "amodal_area": int(obj.amodal.sum()),
                "hidden_area": int((obj.amodal & ~obj.modal).sum()),
                "depth_transform": args.depth_transform,
                "relation_mode": args.relation_mode,
                "bbox": (
                    None
                    if bbox(obj.amodal) is None
                    else [
                        int(round(bbox(obj.amodal)[0] * point_scale_x)),
                        int(round(bbox(obj.amodal)[1] * point_scale_y)),
                        int(round(bbox(obj.amodal)[2] * point_scale_x)),
                        int(round(bbox(obj.amodal)[3] * point_scale_y)),
                    ]
                ),
            }
        )

    paths = build_paths(objects, pairs)

    write_json(out_dir / "objects.json", objects_json)
    write_json(out_dir / "duplicate_objects.json", duplicates)
    write_json(out_dir / "occ_detail.json", occ_detail)
    write_json(out_dir / "occlusion_paths.json", paths)

    if args.save_visualizations:
        cv2.imwrite(str(out_dir / "overlay_objects.png"), cv2.cvtColor(overlay_objects(rgb, objects), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "overlay_obstructions.png"), cv2.cvtColor(overlay_obstructions(rgb, pairs), cv2.COLOR_RGB2BGR))

    print(f"Loaded {len(objects)} objects")
    if duplicates:
        print(f"Merged {len(duplicates)} duplicate objects")
        for d in duplicates:
            print(f'  removed obj {d["removed_id"]} ({d["removed_name"]}) -> kept obj {d["kept_id"]} ({d["kept_name"]}), IoU={d["modal_iou"]}')
    print(f"Estimated {len(occ_detail)} obstruction relations")
    print(f"Wrote: {out_dir / 'occ_detail.json'}")
    for p in occ_detail:
        print(
            f'  obj {p["obj1"]} ({p["blocker_name"]}) -> obj {p["obj2"]} '
            f'({p["blocked_name"]}): contact={p["contact"]["score"]:.4f}, '
            f'ratio(feature)={p["mask_ratio"]:.4f}, degree(feature)={p["degree"]}, '
            f'point=({p["point"]["x"]},{p["point"]["y"]}), conf={p["relation_confidence"]:.3f}'
        )


if __name__ == "__main__":
    main()
