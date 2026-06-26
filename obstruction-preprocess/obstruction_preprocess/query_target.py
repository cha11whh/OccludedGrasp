import argparse
import json
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
]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def norm_text(s: str) -> str:
    return " ".join((s or "").lower().replace("_", " ").split())


def object_center_from_bbox(obj: dict) -> Optional[Tuple[int, int]]:
    box = obj.get("bbox")
    if not box:
        return None
    x1, y1, x2, y2 = box
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def resolve_target(objects: Dict[int, dict], target: str) -> dict:
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


def build_incoming(relations: List[dict], min_confidence: float, min_ratio: float) -> Dict[int, List[dict]]:
    incoming: Dict[int, List[dict]] = {}
    for rel in relations:
        ratio = float(rel.get("mask_ratio", 0.0))
        conf = float(rel.get("confidence", 1.0))
        if ratio < min_ratio or conf < min_confidence:
            continue
        blocked = int(rel["blocked"])
        incoming.setdefault(blocked, []).append(rel)

    for blocked, rels in incoming.items():
        rels.sort(key=lambda r: (-float(r.get("confidence", 0.0)), -float(r.get("mask_ratio", 0.0)), int(r["blocker"])))
    return incoming


def relation_step(rel: dict, objects: Dict[int, dict]) -> dict:
    blocked_id = int(rel["blocked"])
    blocker_id = int(rel["blocker"])
    return {
        "blocked": {
            "id": blocked_id,
            "name": objects.get(blocked_id, {}).get("name", rel.get("blocked_name", str(blocked_id))),
        },
        "blocker": {
            "id": blocker_id,
            "name": objects.get(blocker_id, {}).get("name", rel.get("blocker_name", str(blocker_id))),
        },
        "ratio": rel.get("mask_ratio"),
        "degree": rel.get("degree"),
        "contact_point": rel.get("point"),
        "contact_score": (rel.get("contact") or {}).get("score"),
        "confidence": rel.get("confidence"),
        "depth": rel.get("depth"),
    }


def trace_paths(
    target_id: int,
    objects: Dict[int, dict],
    incoming: Dict[int, List[dict]],
    max_depth: int,
) -> List[dict]:
    paths = []

    def dfs(current_id: int, object_ids: List[int], steps: List[dict], seen: set):
        if len(object_ids) > max_depth:
            paths.append(make_path(object_ids, steps, objects, stopped_reason="max_depth"))
            return

        rels = incoming.get(current_id, [])
        rels = [r for r in rels if int(r["blocker"]) not in seen]
        if not rels:
            paths.append(make_path(object_ids, steps, objects, stopped_reason="top_accessible"))
            return

        for rel in rels:
            blocker_id = int(rel["blocker"])
            dfs(
                blocker_id,
                object_ids + [blocker_id],
                steps + [relation_step(rel, objects)],
                seen | {blocker_id},
            )

    dfs(target_id, [target_id], [], {target_id})
    return paths


def make_path(object_ids: List[int], steps: List[dict], objects: Dict[int, dict], stopped_reason: str) -> dict:
    top_id = object_ids[-1]
    return {
        "object_ids": object_ids,
        "object_names": [objects.get(i, {}).get("name", str(i)) for i in object_ids],
        "steps": steps,
        "top_object": {"id": top_id, "name": objects.get(top_id, {}).get("name", str(top_id))},
        "stopped_reason": stopped_reason,
    }


def build_think(paths: List[dict], target: dict) -> str:
    if len(paths) == 1 and not paths[0]["steps"]:
        return f'<think>{target["name"]} at object {target["id"]} is not obstructed and can be grasped directly.</think>'

    chunks = []
    for idx, path in enumerate(paths, start=1):
        parts = []
        for step in path["steps"]:
            blocked = step["blocked"]
            blocker = step["blocker"]
            point = step.get("contact_point") or {}
            ratio = step.get("ratio")
            contact = step.get("contact_score")
            degree = step.get("degree")
            if ratio is None:
                ratio_txt = ""
            else:
                ratio_txt = f"; obstruction-ratio feature {float(ratio) * 100:.1f}%"
            if contact is None:
                contact_score_txt = ""
            else:
                contact_score_txt = f" with contact score {float(contact):.3f}"
            contact_txt = ""
            if "x" in point and "y" in point:
                contact_txt = f" at contact point ({point['x']}, {point['y']})"
            degree_txt = f" ({degree})" if degree else ""
            parts.append(
                f"{blocked['name']} object {blocked['id']} is obstructed by "
                f"{blocker['name']} object {blocker['id']}{contact_txt}{contact_score_txt}{ratio_txt}{degree_txt}."
            )
        top = path["top_object"]
        parts.append(f"{top['name']} object {top['id']} has no blocker above it and is directly graspable.")
        prefix = f"Path {idx}: " if len(paths) > 1 else ""
        chunks.append(prefix + " ".join(parts))
    return "<think>" + " ".join(chunks) + "</think>"


def unique_top_objects(paths: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for path in paths:
        top = path["top_object"]
        if top["id"] not in seen:
            seen.add(top["id"])
            out.append(top)
    return out


def draw_query(rgb_path: Path, objects: Dict[int, dict], paths: List[dict], out_path: Path):
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rgb_path)

    for path_idx, path in enumerate(paths):
        color = PALETTE[path_idx % len(PALETTE)]
        centers = []
        for obj_id in path["object_ids"]:
            c = object_center_from_bbox(objects.get(obj_id, {}))
            centers.append(c)
            if c:
                name = objects.get(obj_id, {}).get("name", str(obj_id))
                cv2.circle(img, c, 18, color, -1)
                cv2.putText(img, f"{obj_id}:{name}", (c[0] + 16, c[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        for a, b in zip(centers, centers[1:]):
            if a and b:
                cv2.arrowedLine(img, a, b, color, 5, cv2.LINE_AA, tipLength=0.04)

        for step in path["steps"]:
            point = step.get("contact_point") or {}
            if "x" in point and "y" in point:
                pt = (int(point["x"]), int(point["y"]))
                cv2.drawMarker(img, pt, color, markerType=cv2.MARKER_CROSS, markerSize=32, thickness=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Trace target-centric obstruction paths from estimated relations.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--target", required=True, help="Target object id or name.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rgb", default=None, help="Optional RGB image for visualization.")
    parser.add_argument("--vis", default=None, help="Optional visualization output path.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-ratio", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    object_list = read_json(Path(args.objects))
    objects = {int(o["id"]): o for o in object_list}
    relations = read_json(Path(args.relations))

    target = resolve_target(objects, args.target)
    incoming = build_incoming(relations, args.min_confidence, args.min_ratio)
    paths = trace_paths(int(target["id"]), objects, incoming, args.max_depth)
    answer = unique_top_objects(paths)
    think = build_think(paths, target)

    result = {
        "target": {"id": int(target["id"]), "name": target.get("name", str(target["id"]))},
        "paths": paths,
        "answer": answer,
        "think": think,
        "answer_text": "<answer>" + json.dumps(answer, ensure_ascii=False) + "</answer>",
    }
    write_json(Path(args.out), result)

    if args.rgb and args.vis:
        draw_query(Path(args.rgb), objects, paths, Path(args.vis))

    print(think)
    print(result["answer_text"])
    print(f"Wrote: {args.out}")
    if args.vis:
        print(f"Wrote visualization: {args.vis}")


if __name__ == "__main__":
    main()
