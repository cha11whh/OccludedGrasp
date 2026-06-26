import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def norm_text(s: str) -> str:
    return " ".join((s or "").lower().replace("_", " ").split())


def resolve_target(objects: Dict[int, dict], target: Optional[str]) -> Optional[dict]:
    if not target:
        return None
    try:
        target_id = int(target)
        if target_id in objects:
            return objects[target_id]
    except ValueError:
        pass

    q = norm_text(target)
    exact = [o for o in objects.values() if norm_text(o.get("name", "")) == q]
    if exact:
        return exact[0]

    partial = [o for o in objects.values() if q in norm_text(o.get("name", "")) or norm_text(o.get("name", "")) in q]
    if partial:
        partial.sort(key=lambda o: len(o.get("name", "")))
        return partial[0]

    names = ", ".join(f'{o["id"]}:{o.get("name", "")}' for o in objects.values())
    raise KeyError(f"Cannot find target '{target}'. Available objects: {names}")


def load_mask(path: Path, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if shape and img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 0


def load_depth(path: Optional[Path], shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        depth = img.astype(np.float32)
    if shape and depth.shape[:2] != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def normalize_depth(depth: np.ndarray, valid: Optional[np.ndarray] = None, inverse: bool = True) -> np.ndarray:
    if valid is None:
        vals = depth[np.isfinite(depth)]
    else:
        vals = depth[valid & np.isfinite(depth)]
    if vals.size == 0:
        out = np.zeros_like(depth, dtype=np.float32)
    else:
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            out = np.zeros_like(depth, dtype=np.float32)
        else:
            out = np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return 1.0 - out if inverse else out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return mask & ~erode(mask, radius)


def mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def bbox_area(obj: dict) -> int:
    box = obj.get("bbox")
    if not box:
        return 0
    x1, y1, x2, y2 = box
    return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))


def compactness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter <= 0:
        return 0.0
    return float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))


def depth_stats(depth: Optional[np.ndarray], mask: np.ndarray) -> dict:
    if depth is None or not np.any(mask):
        return {"median": None, "std": None, "stability": 0.5}
    vals = depth[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"median": None, "std": None, "stability": 0.5}
    std = float(np.std(vals))
    return {
        "median": float(np.median(vals)),
        "std": std,
        "stability": float(np.clip(1.0 - std / 0.25, 0.0, 1.0)),
    }


def build_relation_maps(relations: List[dict], min_confidence: float) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    incoming: Dict[int, List[dict]] = {}
    outgoing: Dict[int, List[dict]] = {}
    for rel in relations:
        if float(rel.get("confidence", rel.get("relation_confidence", 0.0))) < min_confidence:
            continue
        blocker = int(rel["blocker"])
        blocked = int(rel["blocked"])
        incoming.setdefault(blocked, []).append(rel)
        outgoing.setdefault(blocker, []).append(rel)
    for mapping in (incoming, outgoing):
        for rels in mapping.values():
            rels.sort(key=lambda r: (-float(r.get("confidence", 0.0)), -float(r.get("mask_ratio", 0.0))))
    return incoming, outgoing


def trace_target_path_ids(target_id: Optional[int], incoming: Dict[int, List[dict]], max_depth: int = 8) -> List[int]:
    if target_id is None:
        return []
    path = [target_id]
    current = target_id
    seen = {target_id}
    for _ in range(max_depth):
        rels = [r for r in incoming.get(current, []) if int(r["blocker"]) not in seen]
        if not rels:
            break
        rel = rels[0]
        current = int(rel["blocker"])
        path.append(current)
        seen.add(current)
    return path


def target_release_gain(obj_id: int, target_id: Optional[int], outgoing: Dict[int, List[dict]], target_path: List[int]) -> float:
    gain = 0.0
    path_set = set(target_path)
    for rel in outgoing.get(obj_id, []):
        blocked = int(rel["blocked"])
        conf = float(rel.get("confidence", rel.get("relation_confidence", 0.0)))
        ratio = float(rel.get("mask_ratio", 0.0))
        if target_id is not None and blocked == target_id:
            gain += 2.0 * conf + 2.0 * ratio
        elif blocked in path_set:
            gain += 1.2 * conf + ratio
        else:
            gain += 0.3 * conf + 0.3 * ratio
    return float(gain)


def blocked_grasp_region_ratio(obj_id: int, masks: Dict[int, np.ndarray], incoming: Dict[int, List[dict]], radius: int) -> float:
    mask = masks[obj_id]
    if not np.any(mask):
        return 1.0
    grasp_region = boundary(mask, radius) | erode(mask, max(1, radius * 2))
    if not np.any(grasp_region):
        grasp_region = mask
    blocked_region = np.zeros_like(mask, dtype=bool)
    for rel in incoming.get(obj_id, []):
        blocker = int(rel["blocker"])
        if blocker in masks:
            blocked_region |= dilate(masks[blocker], radius)
    return float((grasp_region & blocked_region).sum()) / float(max(1, grasp_region.sum()))


def score_object_action(
    obj: dict,
    masks: Dict[int, np.ndarray],
    depth: Optional[np.ndarray],
    incoming: Dict[int, List[dict]],
    outgoing: Dict[int, List[dict]],
    target_id: Optional[int],
    target_path: List[int],
    grasp_region_radius: int,
    step_penalty: float,
) -> dict:
    obj_id = int(obj["id"])
    mask = masks[obj_id]
    area = float(mask.sum())
    scene_area = float(mask.size)
    area_score = float(np.clip(math.log1p(area) / math.log1p(max(2.0, scene_area * 0.12)), 0.0, 1.0))
    box_area = max(1, bbox_area(obj))
    fill_score = float(np.clip(area / float(box_area), 0.0, 1.0))
    shape_score = compactness(mask)
    dstat = depth_stats(depth, mask)

    in_degree = len(incoming.get(obj_id, []))
    out_degree = len(outgoing.get(obj_id, []))
    top_accessible = in_degree == 0
    is_target = target_id is not None and obj_id == target_id
    is_direct_target_blocker = any(int(r["blocked"]) == target_id for r in outgoing.get(obj_id, [])) if target_id is not None else False
    is_on_target_path = obj_id in set(target_path[1:])
    release_gain = target_release_gain(obj_id, target_id, outgoing, target_path)
    blocked_region = blocked_grasp_region_ratio(obj_id, masks, incoming, grasp_region_radius)

    success_prior = 0.0
    if is_target and top_accessible:
        success_prior += 10.0
    elif is_target:
        success_prior += 2.0
    elif is_direct_target_blocker:
        success_prior += 4.0
    elif is_on_target_path:
        success_prior += 3.0
    elif top_accessible:
        success_prior += 1.2
    else:
        success_prior += 0.3

    graph_progress = 0.0
    if is_direct_target_blocker:
        graph_progress += 2.5
    if is_on_target_path:
        graph_progress += 1.5
    graph_progress += min(2.0, 0.45 * out_degree)
    graph_progress -= 1.2 * in_degree

    affordance = 1.2 * area_score + 0.8 * fill_score + 0.6 * shape_score + 0.8 * float(dstat["stability"])
    collision_penalty = -2.0 * blocked_region - 0.4 * in_degree
    efficiency = step_penalty
    target_priority = 1.0 if is_target else (0.7 if is_direct_target_blocker else (0.4 if is_on_target_path else 0.0))
    target_release = min(4.0, release_gain)

    reward = success_prior + graph_progress + affordance + collision_penalty + efficiency + target_priority + target_release
    center = mask_center(mask)
    terms = {
        "success_prior": round(float(success_prior), 6),
        "graph_progress": round(float(graph_progress), 6),
        "affordance": round(float(affordance), 6),
        "collision_penalty": round(float(collision_penalty), 6),
        "efficiency": round(float(efficiency), 6),
        "target_priority": round(float(target_priority), 6),
        "target_release_gain": round(float(target_release), 6),
        "blocked_grasp_region_ratio": round(float(blocked_region), 6),
    }

    role = "target" if is_target else ("direct_target_blocker" if is_direct_target_blocker else ("target_path_blocker" if is_on_target_path else "scene_object"))
    return {
        "action_type": "grasp",
        "object_id": obj_id,
        "object_name": obj.get("name", str(obj_id)),
        "role": role,
        "top_accessible": top_accessible,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "grasp_point": None if center is None else {"x": center[0], "y": center[1]},
        "reward": round(float(reward), 6),
        "terms": terms,
        "features": {
            "area_score": round(area_score, 6),
            "fill_score": round(fill_score, 6),
            "shape_score": round(shape_score, 6),
            "depth_median": dstat["median"],
            "depth_std": dstat["std"],
            "depth_stability": round(float(dstat["stability"]), 6),
            "release_gain_raw": round(float(release_gain), 6),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build reward-ranked grasp actions from obstruction graph outputs.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--target", default=None, help="Optional target object id or name.")
    parser.add_argument("--depth", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--depth-inverse", type=lambda v: str(v).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--grasp-region-radius", type=int, default=18)
    parser.add_argument("--step-penalty", type=float, default=-0.1)
    args = parser.parse_args()

    object_list = read_json(Path(args.objects))
    objects = {int(o["id"]): o for o in object_list}
    relations = read_json(Path(args.relations))

    first_mask = load_mask(Path(object_list[0]["modal_path"])) if object_list else None
    shape = first_mask.shape if first_mask is not None else None
    masks = {int(o["id"]): load_mask(Path(o["modal_path"]), shape) for o in object_list}
    depth_raw = load_depth(Path(args.depth), shape) if args.depth else None
    all_mask = np.zeros(shape, dtype=bool) if shape else None
    if all_mask is not None:
        for m in masks.values():
            all_mask |= m
    depth = normalize_depth(depth_raw, all_mask, inverse=args.depth_inverse) if depth_raw is not None else None

    target = resolve_target(objects, args.target)
    target_id = None if target is None else int(target["id"])
    incoming, outgoing = build_relation_maps(relations, args.min_confidence)
    target_path = trace_target_path_ids(target_id, incoming)

    actions = [
        score_object_action(
            obj=o,
            masks=masks,
            depth=depth,
            incoming=incoming,
            outgoing=outgoing,
            target_id=target_id,
            target_path=target_path,
            grasp_region_radius=args.grasp_region_radius,
            step_penalty=args.step_penalty,
        )
        for o in object_list
    ]
    actions.sort(key=lambda a: (-float(a["reward"]), int(a["object_id"])))

    result = {
        "target": None if target is None else {"id": target_id, "name": target.get("name", str(target_id))},
        "target_path": target_path,
        "reward_definition": {
            "reward": "success_prior + graph_progress + affordance + collision_penalty + efficiency + target_priority + target_release_gain",
            "efficiency_step_penalty": args.step_penalty,
            "blocked_grasp_region_ratio": "fraction of an object's candidate grasp region occupied by incoming blockers",
        },
        "ranked_actions": actions,
    }

    out_dir = Path(args.out_dir)
    write_json(out_dir / "ranked_actions.json", result)
    write_json(out_dir / "candidates.json", actions)
    write_json(out_dir / "reward_terms.json", result["reward_definition"])

    print(f"Wrote: {out_dir / 'ranked_actions.json'}")
    for action in actions[: min(5, len(actions))]:
        print(
            f"  reward={action['reward']:.3f} obj {action['object_id']} "
            f"({action['object_name']}), role={action['role']}, top_accessible={action['top_accessible']}"
        )


if __name__ == "__main__":
    main()
