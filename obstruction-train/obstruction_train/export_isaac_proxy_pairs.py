"""Export Isaac feedback-loop captures as approximate obstruction-model pairs.

Labels are derived independently from simulator RGB-D instance masks: overlapping
projected bounding boxes plus median depth determine the front/behind direction.
This is a low-fidelity proxy dataset, not human ground truth or amodal annotation.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_capture(cycle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    rgb = np.load(cycle_dir / "rgb.npy")
    depth = np.load(cycle_dir / "depth.npy").squeeze()
    instances = np.load(cycle_dir / "instance_segmentation.npy").squeeze().astype(np.int32)
    info = json.loads((cycle_dir / "camera_info.json").read_text(encoding="utf-8"))
    labels = {int(key): str(value) for key, value in info.get("idToLabels", {}).items()}
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or instances.ndim != 2 or depth.ndim != 2:
        raise ValueError(f"invalid capture shapes in {cycle_dir}: rgb={rgb.shape}, depth={depth.shape}, ids={instances.shape}")
    if rgb.shape[:2] != depth.shape or depth.shape != instances.shape:
        raise ValueError(f"RGB/depth/instance dimensions differ in {cycle_dir}")
    return rgb.astype(np.uint8), depth.astype(np.float32), instances, labels


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _pair_rows(
    image_id: int,
    image_path: str,
    npz_path: str,
    depth: np.ndarray,
    instances: np.ndarray,
    object_ids: list[int],
    *,
    overlap_threshold: float,
    depth_margin_m: float,
) -> list[dict[str, Any]]:
    boxes = {object_id: _bbox(instances == object_id) for object_id in object_ids}
    medians = {}
    for object_id in object_ids:
        values = depth[(instances == object_id) & np.isfinite(depth) & (depth > 0)]
        if values.size:
            medians[object_id] = float(np.median(values))
    rows = []
    for index, first_id in enumerate(object_ids):
        for second_id in object_ids[index + 1 :]:
            first_box, second_box = boxes[first_id], boxes[second_id]
            if first_box is None or second_box is None:
                continue
            x0 = max(first_box[0], second_box[0])
            y0 = max(first_box[1], second_box[1])
            x1 = min(first_box[2], second_box[2])
            y1 = min(first_box[3], second_box[3])
            overlap_area = max(0, x1 - x0) * max(0, y1 - y0)
            first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
            second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
            if first_id in medians and second_id in medians and overlap_area > 0:
                # Smaller camera depth is closer. The pair is positive only when
                # there is meaningful projected overlap and a visible depth gap.
                if medians[first_id] + depth_margin_m < medians[second_id]:
                    ratio = overlap_area / max(second_area, 1)
                    if ratio >= overlap_threshold:
                        rows.append({"image_id": image_id, "image_path": image_path, "npz_path": npz_path,
                                     "scene_id": image_id, "view_id": 0, "obj_a": first_id, "obj_b": second_id,
                                     "label": 0, "mask_ratio": float(ratio), "mask_area": int(overlap_area),
                                     "point": [(x0 + x1) // 2, (y0 + y1) // 2]})
                        rows.append({"image_id": image_id, "image_path": image_path, "npz_path": npz_path,
                                     "scene_id": image_id, "view_id": 0, "obj_a": second_id, "obj_b": first_id,
                                     "label": 1, "mask_ratio": float(ratio), "mask_area": int(overlap_area),
                                     "point": [(x0 + x1) // 2, (y0 + y1) // 2]})
                        continue
                elif medians[second_id] + depth_margin_m < medians[first_id]:
                    ratio = overlap_area / max(first_area, 1)
                    if ratio >= overlap_threshold:
                        rows.append({"image_id": image_id, "image_path": image_path, "npz_path": npz_path,
                                     "scene_id": image_id, "view_id": 0, "obj_a": second_id, "obj_b": first_id,
                                     "label": 0, "mask_ratio": float(ratio), "mask_area": int(overlap_area),
                                     "point": [(x0 + x1) // 2, (y0 + y1) // 2]})
                        rows.append({"image_id": image_id, "image_path": image_path, "npz_path": npz_path,
                                     "scene_id": image_id, "view_id": 0, "obj_a": first_id, "obj_b": second_id,
                                     "label": 1, "mask_ratio": float(ratio), "mask_area": int(overlap_area),
                                     "point": [(x0 + x1) // 2, (y0 + y1) // 2]})
                        continue
            # Include both pair orders so the NONE class has no directional bias.
            for obj_a, obj_b in ((first_id, second_id), (second_id, first_id)):
                rows.append({"image_id": image_id, "image_path": image_path, "npz_path": npz_path,
                             "scene_id": image_id, "view_id": 0, "obj_a": obj_a, "obj_b": obj_b,
                             "label": 2, "mask_ratio": 0.0, "mask_area": 0, "point": None})
    return rows


def export_captures(capture_root: Path, output_root: Path, *, val_ratio: float = 0.2,
                    seed: int = 1337, overlap_threshold: float = 0.03,
                    depth_margin_m: float = 0.015) -> dict[str, int]:
    capture_dirs = sorted(path for path in capture_root.glob("cycle_*") if (path / "rgb.npy").is_file())
    if not capture_dirs:
        raise FileNotFoundError(f"no cycle_*/rgb.npy captures under {capture_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta_data").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(output_root / "images.zip", "w", compression=zipfile.ZIP_DEFLATED) as image_zip, \
         zipfile.ZipFile(output_root / "meta_data" / "annotations_meta.zip", "w", compression=zipfile.ZIP_DEFLATED) as meta_zip:
        for image_id, cycle_dir in enumerate(capture_dirs):
            rgb, depth, instances, labels = _read_capture(cycle_dir)
            object_ids = sorted(object_id for object_id, label in labels.items()
                                if label.startswith("/World/Clutter/") and np.any(instances == object_id))
            if len(object_ids) < 2:
                continue
            image_path = f"images/image_{image_id:06d}.png"
            npz_path = f"annotations_meta/ALL_NPZ/scene{image_id}_view0.npz"
            image_buffer = io.BytesIO()
            Image.fromarray(rgb).save(image_buffer, format="PNG")
            image_zip.writestr(image_path, image_buffer.getvalue())
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, depth=depth, instances_objects=instances)
            meta_zip.writestr(npz_path, npz_buffer.getvalue())
            rows.extend(_pair_rows(image_id, image_path, npz_path, depth, instances, object_ids,
                                   overlap_threshold=overlap_threshold, depth_margin_m=depth_margin_m))

    if not rows:
        raise ValueError("no object-pair examples were generated; capture more overlapping proxy scenes")
    label_counts = {str(label): sum(row["label"] == label for row in rows) for label in range(3)}
    if label_counts["0"] + label_counts["1"] == 0:
        raise ValueError(
            "no directional occlusion positives were derived; add captures with projected object overlap "
            "and a measurable depth gap before training (do not train on a NONE-only dataset)"
        )
    image_ids = sorted({int(row["image_id"]) for row in rows})
    if len(image_ids) < 2:
        raise ValueError("at least two distinct captured scenes are required for a train/validation split")
    random.Random(seed).shuffle(image_ids)
    n_val = max(1, min(len(image_ids) - 1, round(len(image_ids) * val_ratio)))
    val_ids = set(image_ids[:n_val])
    for split_name, selected in (("train", [row for row in rows if row["image_id"] not in val_ids]),
                                 ("val", [row for row in rows if row["image_id"] in val_ids])):
        with (output_root / f"{split_name}_pairs.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row) + "\n")
    summary = {"scenes": len(image_ids), "pairs": len(rows), "label_counts": label_counts,
               "train_pairs": sum(row["image_id"] not in val_ids for row in rows),
               "val_pairs": sum(row["image_id"] in val_ids for row in rows)}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True, help="Isaac output directory containing cycle_*/ RGB-D captures")
    parser.add_argument("--out-root", type=Path, required=True, help="Root consumed by UnoBenchPairDataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overlap-threshold", type=float, default=0.03)
    parser.add_argument("--depth-margin-m", type=float, default=0.015)
    args = parser.parse_args()
    print(json.dumps(export_captures(args.capture_root, args.out_root, val_ratio=args.val_ratio, seed=args.seed,
                                     overlap_threshold=args.overlap_threshold, depth_margin_m=args.depth_margin_m), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
