"""Isaac Lab tabletop grasp-order visualization.

This is the Isaac Lab counterpart of ``sim_grasp_experiment.py``. It keeps the
planner contract the same: read objects, obstruction relations, and ranked
actions, then execute either a target-reaching or table-clearing grasp sequence.

Run this script through Isaac Lab, for example:

    ./isaaclab.sh -p /home/ubuntu22-zy/occlusion/obstruction-preprocess/obstruction_preprocess/isaac_lab_tabletop_grasp.py --headless ...

The first version uses scripted kinematic pick/carry motions for the selected
objects. This gives a much more realistic renderer and scene substrate than
PyBullet while keeping the grasp-order integration simple. A robot arm/gripper
controller can replace ``move_prim`` later without changing the planner inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def action_by_object(ranked: dict) -> Dict[int, dict]:
    return {int(action["object_id"]): action for action in ranked.get("ranked_actions", [])}


def make_fallback_action(obj: dict) -> dict:
    obj_id = int(obj["id"])
    return {
        "action_type": "grasp",
        "object_id": obj_id,
        "object_name": obj.get("name", str(obj_id)),
        "role": "unranked",
        "reward": 0.0,
    }


def incoming_count(relations: List[dict], object_id: int, removed: set) -> int:
    return sum(
        1
        for rel in relations
        if int(rel["blocked"]) == int(object_id)
        and int(rel["blocker"]) not in removed
        and int(rel["blocked"]) not in removed
    )


def active_incoming(relations: List[dict], object_id: int, removed: set) -> List[int]:
    return [
        int(rel["blocker"])
        for rel in relations
        if int(rel["blocked"]) == int(object_id)
        and int(rel["blocker"]) not in removed
        and int(rel["blocked"]) not in removed
    ]


def active_outgoing_count(relations: List[dict], object_id: int, removed: set) -> int:
    return sum(
        1
        for rel in relations
        if int(rel["blocker"]) == int(object_id)
        and int(rel["blocked"]) not in removed
        and int(rel["blocker"]) not in removed
    )


def blocker_closure(relations: List[dict], target_id: int, removed: set) -> set:
    closure = set()
    stack = active_incoming(relations, target_id, removed)
    while stack:
        obj_id = stack.pop()
        if obj_id in closure or obj_id in removed:
            continue
        closure.add(obj_id)
        stack.extend(active_incoming(relations, obj_id, removed))
    return closure


def choose_next_object(
    task_mode: str,
    objects: Dict[int, dict],
    relations: List[dict],
    ranked_actions: Dict[int, dict],
    removed: set,
    target_id: Optional[int],
) -> Optional[int]:
    remaining = [obj_id for obj_id in objects if obj_id not in removed]
    if not remaining:
        return None

    def reward(obj_id: int) -> float:
        return float(ranked_actions.get(obj_id, {}).get("reward", 0.0))

    def top_accessible(obj_id: int) -> bool:
        return incoming_count(relations, obj_id, removed) == 0

    if task_mode == "clear_table":
        candidates = [obj_id for obj_id in remaining if top_accessible(obj_id)]
        if not candidates:
            candidates = remaining
        return max(candidates, key=lambda obj_id: (reward(obj_id), active_outgoing_count(relations, obj_id, removed)))

    if target_id is None:
        raise ValueError("--task-mode reach_target requires ranked_actions target or --target-object-id")
    if target_id in removed:
        return None
    blockers = blocker_closure(relations, target_id, removed)
    if not blockers:
        return target_id
    candidates = [obj_id for obj_id in blockers if top_accessible(obj_id)]
    if not candidates:
        candidates = list(blockers)
    return max(candidates, key=lambda obj_id: (active_outgoing_count(relations, obj_id, removed), reward(obj_id)))


def infer_object_style(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("thermos", "bottle", "cup", "mug", "can", "jar")):
        return "cylinder"
    if any(token in text for token in ("cable", "cord", "wire", "charger", "remote")):
        return "bar"
    if any(token in text for token in ("book", "notebook", "paper", "cardboard", "box")):
        return "flat_box"
    if any(token in text for token in ("pouch", "bag", "cloth", "fabric")):
        return "pouch"
    return "box"


def object_geometry(name: str) -> Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]:
    style = infer_object_style(name)
    if style == "cylinder":
        return style, (0.055, 0.055, 0.15), (0.85, 0.08, 0.05)
    if style == "bar":
        return style, (0.13, 0.03, 0.025), (0.03, 0.03, 0.03)
    if style == "flat_box":
        return style, (0.18, 0.11, 0.035), (0.90, 0.90, 0.88)
    if style == "pouch":
        return style, (0.24, 0.15, 0.06), (0.78, 0.66, 0.42)
    return style, (0.12, 0.10, 0.08), (0.25, 0.35, 0.85)


def front_occlusion_xy(index: int) -> Tuple[float, float]:
    positions = [
        (-0.18, -0.20),
        (0.00, -0.22),
        (0.18, -0.19),
        (-0.10, -0.02),
        (0.10, 0.00),
        (-0.27, -0.12),
        (0.27, -0.10),
        (-0.26, 0.05),
        (-0.05, 0.08),
        (0.20, 0.08),
        (0.00, 0.22),
        (0.30, 0.20),
    ]
    return positions[index % len(positions)]


def add_demo_props(objects: Dict[int, dict]) -> None:
    names = ["green mug", "silver can", "blue notebook", "yellow cup", "red can", "thin book", "black remote"]
    for i, name in enumerate(names):
        obj_id = 1000 + i
        objects[obj_id] = {"id": obj_id, "name": name, "is_demo_prop": True}


def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Lab tabletop grasp-order simulation.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--ranked-actions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-mode", choices=["reach_target", "clear_table"], default="reach_target")
    parser.add_argument("--target-object-id", type=int, default=None)
    parser.add_argument("--add-demo-props", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-frames", type=int, default=60)
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--table-top-z", type=float, default=0.75)
    parser.add_argument("--save-usd", default=None)
    parser.add_argument("--camera-front", action="store_true", default=True)
    parser.add_argument("--close-app", action="store_true", help="Call simulation_app.close() before exit. Disabled by default because headless Kit shutdown can hang on some systems.")
    return parser


def main():
    parser = parse_args()

    # Isaac Lab must launch the Omniverse app before importing most simulator modules.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    print("[INFO] Launching Isaac Lab app...", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[INFO] Isaac Lab app launched.", flush=True)

    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom
    import omni.usd

    print("[INFO] Reading planner inputs...", flush=True)
    objects_list = read_json(args.objects)
    objects = {int(o["id"]): o for o in objects_list}
    if args.add_demo_props:
        add_demo_props(objects)
    relations = read_json(args.relations)
    ranked = read_json(args.ranked_actions)
    ranked_actions = action_by_object(ranked)
    target = ranked.get("target") or {}
    target_id = args.target_object_id if args.target_object_id is not None else (None if target.get("id") is None else int(target["id"]))
    if target_id is not None and (not target or target.get("id") is None):
        target = {"id": target_id, "name": objects[target_id].get("name", str(target_id))}

    print("[INFO] Creating SimulationContext...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    print("[INFO] SimulationContext created.", flush=True)
    if args.camera_front:
        sim.set_camera_view([0.0, -1.8, 1.02], [0.0, -0.02, 0.86])
    else:
        sim.set_camera_view([1.1, -1.1, 1.8], [0.0, 0.0, 0.8])

    print("[INFO] Spawning scene...", flush=True)
    # Scene.
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.86, 0.86, 0.82))
    cfg_light.func("/World/Light", cfg_light, translation=(0.6, -0.6, 3.0))
    sim_utils.create_prim("/World/Objects", "Xform")

    table_cfg = sim_utils.MeshCuboidCfg(
        size=(1.05, 0.82, 0.06),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.45, 0.28)),
    )
    table_cfg.func("/World/TableTop", table_cfg, translation=(0.0, 0.0, args.table_top_z - 0.03))

    object_paths: Dict[int, str] = {}
    object_positions: Dict[int, Tuple[float, float, float]] = {}
    for index, (obj_id, obj) in enumerate(objects.items()):
        name = obj.get("name", str(obj_id))
        style, dims, color = object_geometry(name)
        x, y = front_occlusion_xy(index)
        z = args.table_top_z + dims[2] / 2.0
        prim_path = f"/World/Objects/obj_{obj_id}"
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        if style == "cylinder":
            radius = dims[0]
            height = dims[2]
            cfg = sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=material,
            )
        else:
            cfg = sim_utils.MeshCuboidCfg(
                size=dims,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=material,
            )
        cfg.func(prim_path, cfg, translation=(x, y, z))
        object_paths[obj_id] = prim_path
        object_positions[obj_id] = (x, y, z)

    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    print("[INFO] Simulation reset done.", flush=True)
    for _ in range(args.settle_frames):
        sim.step()
    print("[INFO] Initial settle done.", flush=True)

    stage = omni.usd.get_context().get_stage()

    def move_prim(prim_path: str, xyz: Tuple[float, float, float]):
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*xyz))

    removed = set()
    executed = []
    before_incoming = None if target_id is None else incoming_count(relations, target_id, removed=set())
    max_steps = args.max_steps if args.max_steps is not None else len(objects)

    for step_index in range(max_steps):
        next_id = choose_next_object(args.task_mode, objects, relations, ranked_actions, removed, target_id)
        if next_id is None:
            break
        action = ranked_actions.get(next_id, make_fallback_action(objects[next_id]))
        print(f"[INFO] Step {step_index + 1}: grasp {action['object_name']}", flush=True)
        prim_path = object_paths[next_id]
        start = object_positions[next_id]
        lift = (start[0], start[1], start[2] + 0.35)
        drop = (2.0 + 0.12 * step_index, 2.0, -1.0)
        for frame in range(args.step_frames):
            t = (frame + 1) / float(args.step_frames)
            if t < 0.55:
                a = t / 0.55
                xyz = (start[0], start[1], start[2] + (lift[2] - start[2]) * a)
            else:
                a = (t - 0.55) / 0.45
                xyz = (lift[0] + (drop[0] - lift[0]) * a, lift[1] + (drop[1] - lift[1]) * a, lift[2] + (drop[2] - lift[2]) * a)
            move_prim(prim_path, xyz)
            sim.step()
        removed.add(next_id)
        executed.append(
            {
                "step": step_index + 1,
                "object_id": next_id,
                "object_name": action["object_name"],
                "role": action.get("role"),
                "reward": float(action.get("reward", 0.0)),
                "success": True,
                "target_incoming_after_step": None if target_id is None else incoming_count(relations, target_id, removed=removed),
            }
        )
        if args.task_mode == "reach_target" and target_id in removed:
            break

    for _ in range(args.settle_frames):
        sim.step()
    print("[INFO] Final settle done.", flush=True)

    after_incoming = None if target_id is None else incoming_count(relations, target_id, removed=removed)
    task_complete = len(removed) == len(objects) if args.task_mode == "clear_table" else (target_id in removed if target_id is not None else False)
    if args.save_usd:
        omni.usd.get_context().save_as_stage(args.save_usd)

    write_json(
        args.out,
        {
            "simulator": "isaac_lab_tabletop_scripted",
            "task_mode": args.task_mode,
            "target": target,
            "task_complete": task_complete,
            "target_incoming_before": before_incoming,
            "target_incoming_after": after_incoming,
            "target_released": after_incoming is not None and before_incoming is not None and after_incoming < before_incoming,
            "removed_objects": sorted(removed),
            "executed_sequence": executed,
            "object_paths": object_paths,
            "save_usd": args.save_usd,
        },
    )
    print(f"[INFO] Isaac Lab tabletop task_complete={task_complete} steps={len(executed)}")
    print(f"[INFO] Wrote: {args.out}")
    if args.close_app:
        simulation_app.close()


if __name__ == "__main__":
    main()
