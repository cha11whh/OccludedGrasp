import argparse
import json
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


def load_mask(path: Path, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if shape and img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 0


def bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def center_from_mask(mask: np.ndarray) -> Optional[dict]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return {"x": int(round(float(xs.mean()))), "y": int(round(float(ys.mean())))}


def choose_action(ranked_actions: dict, force_object_id: Optional[int], min_reward: float) -> dict:
    actions = ranked_actions.get("ranked_actions", ranked_actions if isinstance(ranked_actions, list) else [])
    if not actions:
        raise ValueError("No ranked actions found")
    if force_object_id is not None:
        for action in actions:
            if int(action["object_id"]) == int(force_object_id):
                return action
        raise KeyError(f"object_id {force_object_id} is not in ranked actions")
    for action in actions:
        if float(action.get("reward", 0.0)) >= min_reward:
            return action
    return actions[0]


def candidate_center_px(candidate: dict) -> Optional[Tuple[int, int]]:
    if "center_px" in candidate:
        p = candidate["center_px"]
        return int(round(float(p[0]))), int(round(float(p[1])))
    if "pixel" in candidate:
        p = candidate["pixel"]
        return int(round(float(p["x"]))), int(round(float(p["y"])))
    return None


def select_grasp_candidate(
    candidates: List[dict],
    mask: np.ndarray,
    action_reward: float,
    reward_weight: float,
    min_grasp_score: float,
) -> Optional[dict]:
    best = None
    best_score = -1e9
    for cand in candidates:
        score = float(cand.get("score", cand.get("grasp_score", 0.0)))
        if score < min_grasp_score:
            continue
        px = candidate_center_px(cand)
        if px is not None:
            x, y = px
            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                continue
            if not mask[y, x]:
                continue
        collision = float(cand.get("collision_score", 0.0))
        combined = score + reward_weight * action_reward - collision
        if combined > best_score:
            best_score = combined
            best = dict(cand)
            best["combined_score"] = round(float(combined), 6)
    return best


def build_request(
    action: dict,
    obj: dict,
    mask: np.ndarray,
    backend: str,
    rgb: Optional[Path],
    depth: Optional[Path],
    intrinsics: Optional[Path],
    candidates: Optional[List[dict]],
    reward_weight: float,
    min_grasp_score: float,
) -> dict:
    selected = None
    if candidates:
        selected = select_grasp_candidate(candidates, mask, float(action.get("reward", 0.0)), reward_weight, min_grasp_score)

    return {
        "backend": backend,
        "status": "ready_for_execution" if selected else "needs_grasp_backend",
        "action": action,
        "object": {
            "id": int(obj["id"]),
            "name": obj.get("name", str(obj["id"])),
            "modal_path": obj.get("modal_path"),
            "amodal_path": obj.get("amodal_path"),
            "bbox": obj.get("bbox") or bbox_from_mask(mask),
            "center_px": center_from_mask(mask),
        },
        "sensor_inputs": {
            "rgb": None if rgb is None else str(rgb),
            "metric_depth": None if depth is None else str(depth),
            "camera_intrinsics": None if intrinsics is None else str(intrinsics),
            "note": "Use metric depth from a calibrated RGB-D camera for real 6-DoF grasping; relative Depth Anything depth is for reasoning only.",
        },
        "grasp_backend_request": {
            "object_mask_path": obj.get("modal_path"),
            "restrict_grasps_to_object_mask": True,
            "sort_by": "combined(grasp_score, action_reward, collision)",
            "expected_output_fields": [
                "pose_camera",
                "width",
                "score",
                "collision_score",
                "center_px",
            ],
        },
        "selected_grasp": selected,
        "robot_command": None
        if selected is None
        else {
            "frame": "camera",
            "pose_camera": selected.get("pose_camera"),
            "width": selected.get("width"),
            "pregrasp_offset_m": selected.get("pregrasp_offset_m", 0.08),
            "speed": selected.get("speed", "normal"),
            "close_gripper": True,
            "lift_after_grasp_m": selected.get("lift_after_grasp_m", 0.10),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert ranked grasp actions into a lower-level grasp-network/robot request.")
    parser.add_argument("--ranked-actions", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["anygrasp", "contact_graspnet", "vgn", "external"], default="anygrasp")
    parser.add_argument("--rgb", default=None)
    parser.add_argument("--metric-depth", default=None)
    parser.add_argument("--camera-intrinsics", default=None)
    parser.add_argument("--grasp-candidates", default=None, help="Optional JSON candidates produced by a grasp network.")
    parser.add_argument("--force-object-id", type=int, default=None)
    parser.add_argument("--min-reward", type=float, default=-999.0)
    parser.add_argument("--reward-weight", type=float, default=0.10)
    parser.add_argument("--min-grasp-score", type=float, default=0.0)
    args = parser.parse_args()

    ranked = read_json(Path(args.ranked_actions))
    objects = {int(o["id"]): o for o in read_json(Path(args.objects))}
    action = choose_action(ranked, args.force_object_id, args.min_reward)
    obj_id = int(action["object_id"])
    if obj_id not in objects:
        raise KeyError(f"object_id {obj_id} is missing from objects.json")
    obj = objects[obj_id]
    mask = load_mask(Path(obj["modal_path"]))

    candidates = None
    if args.grasp_candidates:
        data = read_json(Path(args.grasp_candidates))
        candidates = data.get("grasps", data if isinstance(data, list) else [])

    request = build_request(
        action=action,
        obj=obj,
        mask=mask,
        backend=args.backend,
        rgb=Path(args.rgb) if args.rgb else None,
        depth=Path(args.metric_depth) if args.metric_depth else None,
        intrinsics=Path(args.camera_intrinsics) if args.camera_intrinsics else None,
        candidates=candidates,
        reward_weight=args.reward_weight,
        min_grasp_score=args.min_grasp_score,
    )
    write_json(Path(args.out), request)
    print(f"Selected object {obj_id} ({obj.get('name', obj_id)}) via action reward {action.get('reward')}")
    print(f"handoff_status={request['status']}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
