import argparse
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_intersection(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def find_amodal_candidates(masks, overlap_thresh):
    bboxes = [mask_bbox(mask) for mask in masks]
    candidates = {}
    for i, box_i in enumerate(bboxes):
        if box_i is None:
            continue
        area_i = bbox_area(box_i)
        if area_i == 0:
            continue
        overlaps = []
        for j, box_j in enumerate(bboxes):
            if i == j or box_j is None:
                continue
            overlap = bbox_intersection(box_i, box_j) / area_i
            if overlap >= overlap_thresh:
                overlaps.append({"id": j, "bbox_overlap": overlap})
        if overlaps:
            candidates[i] = overlaps
    return candidates


def save_mask(mask, path):
    Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB").save(path)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_modal_masks(modal_masks_dir, img_basename):
    mask_dir = os.path.join(modal_masks_dir, img_basename)
    metadata_path = os.path.join(mask_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    masks = []
    class_names = []
    pred_scores = []
    for item in metadata:
        mask_path = os.path.join(mask_dir, item["mask"])
        mask = np.array(Image.open(mask_path).convert("L"))
        masks.append(mask > 0)
        class_names.append(item["class_name"])
        pred_scores.append(float(item.get("score", 1.0)))

    if not masks:
        return None
    return np.stack(masks, axis=0), class_names, pred_scores


def morph(mask, op, radius):
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    src = mask.astype(np.uint8)
    if op == "dilate":
        out = cv2.dilate(src, kernel, iterations=1)
    elif op == "close":
        out = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"unknown morphology op: {op}")
    return out.astype(bool)


def contour_hull(mask):
    src = mask.astype(np.uint8)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    hull_mask = np.zeros_like(src)
    for contour in contours:
        if cv2.contourArea(contour) < 16:
            continue
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(hull_mask, hull, 1)
    return hull_mask.astype(bool)


def bbox_fill(mask, pad_fraction=0.04):
    box = mask_bbox(mask)
    out = np.zeros_like(mask, dtype=bool)
    if box is None:
        return out
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = box
    pad = int(round(max(x1 - x0, y1 - y0) * pad_fraction))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    out[y0:y1, x0:x1] = True
    return out


def is_rectangular_class(name):
    tokens = name.lower()
    return any(word in tokens for word in ["box", "book", "carton", "package", "board"])


def geometric_complete_mask(mask_id, masks, class_names, candidates, radius=36, max_added_fraction=0.65):
    modal = masks[mask_id].astype(bool)
    if not np.any(modal):
        return modal, {
            "mode": "geometric",
            "added_area": 0,
            "added_fraction_of_modal": 0.0,
            "candidate_blockers": [],
        }

    blocker_ids = [int(item["id"]) for item in candidates.get(mask_id, [])]
    blocker_union = np.zeros_like(modal, dtype=bool)
    for blocker_id in blocker_ids:
        blocker_union |= masks[blocker_id].astype(bool)

    closed = morph(modal, "close", max(2, radius // 4))
    hull = contour_hull(closed)
    if is_rectangular_class(class_names[mask_id]):
        shape_prior = bbox_fill(closed, pad_fraction=0.03)
    else:
        shape_prior = hull | (bbox_fill(closed, pad_fraction=0.02) & morph(hull, "dilate", max(2, radius // 4)))

    if np.any(blocker_union):
        blocker_near = morph(blocker_union, "dilate", radius)
        modal_near = morph(modal, "dilate", radius * 2)
        hidden_support = blocker_near & modal_near
        hidden = shape_prior & hidden_support & ~modal
    else:
        hidden = shape_prior & ~modal

    max_added = int(round(max_added_fraction * max(1, int(modal.sum()))))
    added_area = int(hidden.sum())
    if added_area > max_added:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hidden.astype(np.uint8), connectivity=8)
        components = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            cx, cy = centroids[label]
            dist = cv2.pointPolygonTest(
                cv2.convexHull(np.column_stack(np.where(modal)[::-1]).astype(np.int32)),
                (float(cx), float(cy)),
                True,
            )
            components.append((area, dist, label))
        components.sort(key=lambda x: (x[1], x[0]), reverse=True)
        limited = np.zeros_like(hidden)
        used = 0
        for area, _, label in components:
            if used >= max_added:
                break
            component = labels == label
            limited |= component
            used += area
        hidden = limited
        added_area = int(hidden.sum())

    completed = modal | hidden
    return completed, {
        "mode": "geometric",
        "added_area": added_area,
        "added_fraction_of_modal": round(float(added_area) / float(max(1, int(modal.sum()))), 6),
        "candidate_blockers": blocker_ids,
        "radius": radius,
        "max_added_fraction": max_added_fraction,
        "shape_prior": "bbox" if is_rectangular_class(class_names[mask_id]) else "convex_hull",
    }


def main_cli():
    parser = argparse.ArgumentParser(description="Fast modal-first inference with optional amodal completion")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--img_filenames_txt", required=True)
    parser.add_argument("--classes_txt", required=True)
    parser.add_argument("--modal_masks_dir", required=True)
    parser.add_argument("--output_dir", default="fast_modal_first_output")
    parser.add_argument("--amodal_classes_txt", help="Optional class allowlist for amodal completion")
    parser.add_argument("--bbox_overlap_thresh", type=float, default=0.02)
    parser.add_argument("--run_amodal", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--amodal_completion_mode", choices=["iterative", "geometric"], default="iterative")
    parser.add_argument("--geometric_radius", type=int, default=36)
    parser.add_argument("--geometric_max_added_fraction", type=float, default=0.65)
    parser.add_argument("--gdino_config", default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino_ckpt", default="Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", default="Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--instaorder_ckpt", default="InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar")
    parser.add_argument("--lama_config_path", default="lama/big-lama/config.yaml")
    parser.add_argument("--lama_ckpt_path", default="lama/big-lama/models/best.ckpt")
    parser.add_argument("--sd_model_id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--mc_timestep", type=int, default=35)
    parser.add_argument("--mc_clean_bkgd_img", default="images/gray_wallpaper.jpeg")
    parser.add_argument("--sd_num_inference_steps", type=int, default=20)
    parser.add_argument("--max_iter_id", type=int, default=1)
    parser.add_argument("--complete_boundary_objects", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()

    img_filenames = read_txt(args.img_filenames_txt)
    classes = read_txt(args.classes_txt)
    amodal_classes = set(read_txt(args.amodal_classes_txt)) if args.amodal_classes_txt else None
    os.makedirs(args.output_dir, exist_ok=True)

    models = None
    for img_filename in img_filenames:
        img_basename = os.path.splitext(img_filename)[0]
        img_path = os.path.join(args.input_dir, img_filename)
        img_pil = Image.open(img_path).convert("RGB")
        img = np.array(img_pil)
        output_img_dir = os.path.join(args.output_dir, img_basename)

        if os.path.exists(output_img_dir):
            shutil.rmtree(output_img_dir)
        for subdir in ["amodal_completions", "amodal_segmentations", "final_segmentations"]:
            os.makedirs(os.path.join(output_img_dir, subdir), exist_ok=True)

        modal_masks = load_modal_masks(args.modal_masks_dir, img_basename)
        if modal_masks is None:
            print(f"{img_filename}: no precomputed modal masks found")
            continue
        masks, class_names, pred_scores = modal_masks

        candidates = find_amodal_candidates(masks, args.bbox_overlap_thresh)
        if amodal_classes is not None:
            candidates = {
                mask_id: overlaps
                for mask_id, overlaps in candidates.items()
                if class_names[mask_id] in amodal_classes
            }
        metadata = []
        offsets = {}

        for mask_id, query_mask in enumerate(masks):
            class_name = class_names[mask_id]
            mask_relpath = f"final_segmentations/{class_name}_{mask_id}.png"
            save_mask(query_mask, os.path.join(output_img_dir, mask_relpath))
            metadata.append({
                "id": mask_id,
                "class_name": class_name,
                "score": pred_scores[mask_id],
                "mask_type": "modal_fallback",
                "mask": mask_relpath,
                "offset": [0, 0],
                "amodal_candidate": mask_id in candidates,
                "candidate_overlaps": candidates.get(mask_id, []),
            })

        if args.run_amodal and candidates and args.amodal_completion_mode == "geometric":
            for mask_id in sorted(candidates):
                class_name = class_names[mask_id]
                completed, completion_info = geometric_complete_mask(
                    mask_id,
                    masks,
                    class_names,
                    candidates,
                    radius=args.geometric_radius,
                    max_added_fraction=args.geometric_max_added_fraction,
                )
                if int((completed & ~masks[mask_id].astype(bool)).sum()) == 0:
                    continue

                final_relpath = f"final_segmentations/{class_name}_{mask_id}.png"
                save_mask(completed, os.path.join(output_img_dir, final_relpath))
                save_mask(completed, os.path.join(output_img_dir, "amodal_segmentations", f"{class_name}_{mask_id}.png"))

                metadata[mask_id].update({
                    "class_name": class_name,
                    "mask_type": "amodal_geometric",
                    "mask": final_relpath,
                    "offset": [0, 0],
                    "completion": completion_info,
                })

        if args.run_amodal and candidates and args.amodal_completion_mode == "iterative":
            import main

            classes_for_image = sorted(set(classes + class_names))
            if models is None:
                models = main.load_models(
                    args.gdino_config,
                    args.gdino_ckpt,
                    args.sd_model_id,
                    args.instaorder_ckpt,
                    args.lama_config_path,
                    args.lama_ckpt_path,
                    args.mc_timestep,
                )
            gdino_model, sd_inpaint_model, instaorder_model, lama_model = models

            for mask_id in sorted(candidates):
                query_obj = main.QueryObject(img_path, img, img_pil, mask_id, masks[mask_id], output_img_dir)
                while query_obj.run_iter:
                    query_obj = main.run_iteration(
                        query_obj,
                        args.output_dir,
                        masks,
                        classes_for_image,
                        class_names,
                        pred_scores,
                        gdino_model,
                        args.sam_ckpt,
                        instaorder_model,
                        sd_inpaint_model,
                        lama_model,
                        args.mc_timestep,
                        args.mc_clean_bkgd_img,
                        args.sd_num_inference_steps,
                        args.complete_boundary_objects,
                        save_interm=True,
                    )
                    if query_obj.iter_id >= args.max_iter_id:
                        break

                if query_obj.amodal_segmentation is None or query_obj.iter_id == 0:
                    continue

                class_name = query_obj.query_class
                x_offset, y_offset = main.compute_offset(
                    query_obj.query_mask_canvas,
                    query_obj.init_outpaint_mask_canvas,
                    query_obj.amodal_segmentation,
                )
                offsets[f"{class_name}_{mask_id}"] = [x_offset, y_offset]

                final_relpath = f"final_segmentations/{class_name}_{mask_id}.png"
                save_mask(query_obj.amodal_segmentation, os.path.join(output_img_dir, final_relpath))
                save_mask(query_obj.amodal_segmentation, os.path.join(output_img_dir, "amodal_segmentations", f"{class_name}_{mask_id}.png"))
                query_obj.amodal_completion.save(os.path.join(output_img_dir, "amodal_completions", f"{class_name}_{mask_id}.jpg"), quality=90)

                metadata[mask_id].update({
                    "class_name": class_name,
                    "mask_type": "amodal",
                    "mask": final_relpath,
                    "offset": [x_offset, y_offset],
                })

        with open(os.path.join(output_img_dir, "mask_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(output_img_dir, "offsets.json"), "w") as f:
            json.dump(offsets, f, indent=2, sort_keys=True)
        print(f"{img_filename}: saved {len(metadata)} masks, amodal candidates={sorted(candidates)}")


if __name__ == "__main__":
    main_cli()
