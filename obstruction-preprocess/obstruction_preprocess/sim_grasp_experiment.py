import argparse
import json
import sys
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


def write_video(path: Path, frames: List[np.ndarray], fps: float):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_gif(path: Path, frames: List[np.ndarray], duration_ms: int):
    if not frames:
        return
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def add_pybullet_path(path: Optional[Path]):
    if path:
        sys.path.insert(0, str(path))


def load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img > 0


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def choose_action(ranked: dict, force_object_id: Optional[int] = None) -> dict:
    actions = ranked.get("ranked_actions", [])
    if not actions:
        raise ValueError("ranked_actions.json has no ranked_actions")
    if force_object_id is not None:
        for action in actions:
            if int(action["object_id"]) == int(force_object_id):
                return action
        raise KeyError(f"object_id {force_object_id} is not in ranked actions")
    return actions[0]


def action_by_object(ranked: dict) -> Dict[int, dict]:
    return {int(action["object_id"]): action for action in ranked.get("ranked_actions", [])}


def make_fallback_action(obj: dict) -> dict:
    obj_id = int(obj["id"])
    return {
        "action_type": "grasp",
        "object_id": obj_id,
        "object_name": obj.get("name", str(obj_id)),
        "role": "unranked",
        "top_accessible": True,
        "reward": 0.0,
    }


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


def object_dims_from_mask(mask: np.ndarray, scene_scale_m: float, min_height_m: float, max_height_m: float) -> Tuple[float, float, float]:
    box = mask_bbox(mask)
    if box is None:
        return 0.04, 0.04, min_height_m
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    sx = max(0.03, (x2 - x1) / float(max(w, h)) * scene_scale_m)
    sy = max(0.03, (y2 - y1) / float(max(w, h)) * scene_scale_m)
    area_ratio = float(mask.sum()) / float(max(1, (x2 - x1) * (y2 - y1)))
    sz = min(max_height_m, max(min_height_m, 0.025 + 0.04 * area_ratio))
    return sx, sy, sz


def infer_object_style(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("thermos", "bottle", "cup", "mug", "can", "jar")):
        return "cylinder"
    if any(token in text for token in ("cable", "cord", "wire", "charger")):
        return "cable"
    if any(token in text for token in ("book", "notebook", "paper", "cardboard", "box")):
        return "flat_box"
    if any(token in text for token in ("pouch", "bag", "cloth", "fabric")):
        return "pouch"
    return "box"


def styled_dims(name: str, mask: np.ndarray, scene_scale_m: float) -> Tuple[str, float, float, float]:
    sx, sy, sz = object_dims_from_mask(mask, scene_scale_m, min_height_m=0.025, max_height_m=0.10)
    style = infer_object_style(name)
    if style == "cylinder":
        diameter = max(0.045, min(0.09, (sx + sy) * 0.36))
        height = max(0.11, min(0.19, max(sx, sy) * 0.95))
        return style, diameter, diameter, height
    if style == "cable":
        length = max(0.09, min(0.18, max(sx, sy)))
        width = max(0.012, min(0.025, min(sx, sy) * 0.45))
        return style, length, width, 0.018
    if style == "flat_box":
        return style, max(0.07, sx), max(0.045, sy), max(0.025, min(0.06, sz * 0.75))
    if style == "pouch":
        return style, max(0.09, sx), max(0.06, sy), max(0.035, min(0.075, sz))
    return style, sx, sy, sz


def object_xy_from_mask(mask: np.ndarray, scene_scale_m: float) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0, 0.0
    h, w = mask.shape[:2]
    scale = scene_scale_m / float(max(w, h))
    x = (float(xs.mean()) - w / 2.0) * scale
    y = -(float(ys.mean()) - h / 2.0) * scale
    return x, y


def front_occlusion_xy(index: int) -> Tuple[float, float]:
    positions = [
        (-0.18, -0.20),
        (0.00, -0.22),
        (0.18, -0.19),
        (-0.10, -0.02),
        (0.10, 0.00),
        (-0.22, 0.14),
        (0.02, 0.16),
        (0.24, 0.13),
    ]
    return positions[index % len(positions)]


def topological_heights(objects: Dict[int, dict], relations: List[dict]) -> Dict[int, int]:
    incoming: Dict[int, List[int]] = {}
    for rel in relations:
        incoming.setdefault(int(rel["blocker"]), [])
        incoming.setdefault(int(rel["blocked"]), []).append(int(rel["blocker"]))

    memo: Dict[int, int] = {}

    def height(obj_id: int, seen: set) -> int:
        if obj_id in memo:
            return memo[obj_id]
        if obj_id in seen:
            return 0
        blockers = incoming.get(obj_id, [])
        if not blockers:
            memo[obj_id] = 0
        else:
            memo[obj_id] = 1 + max(height(b, seen | {obj_id}) for b in blockers)
        return memo[obj_id]

    for obj_id in objects:
        height(obj_id, set())
    # For simulation z, blockers should be higher than objects they block.
    max_h = max(memo.values()) if memo else 0
    return {obj_id: max_h - h for obj_id, h in memo.items()}


def create_table(p, table_top_z: float, table_size: Tuple[float, float], table_thickness: float):
    sx, sy = table_size
    top_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2.0, sy / 2.0, table_thickness / 2.0])
    top_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[sx / 2.0, sy / 2.0, table_thickness / 2.0],
        rgbaColor=[0.72, 0.58, 0.42, 1.0],
    )
    table = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=top_col,
        baseVisualShapeIndex=top_vis,
        basePosition=[0.0, 0.0, table_top_z - table_thickness / 2.0],
    )
    leg_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, table_top_z / 2.0])
    leg_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, table_top_z / 2.0],
        rgbaColor=[0.48, 0.36, 0.23, 1.0],
    )
    for lx in (-sx / 2.0 + 0.06, sx / 2.0 - 0.06):
        for ly in (-sy / 2.0 + 0.06, sy / 2.0 - 0.06):
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_col,
                baseVisualShapeIndex=leg_vis,
                basePosition=[lx, ly, table_top_z / 2.0],
            )
    return table


def create_box_body(p, half_extents: List[float], position: List[float], color: List[float]):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=position)


def create_cylinder_body(p, radius: float, height: float, position: List[float], color: List[float]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=position)


def create_demo_props(p, table_top_z: float):
    props = [
        ("green mug", "cylinder", [-0.27, -0.12], [0.09, 0.055], [0.05, 0.55, 0.20, 1.0]),
        ("silver can", "cylinder", [0.27, -0.10], [0.075, 0.12], [0.78, 0.78, 0.74, 1.0]),
        ("blue notebook", "box", [-0.26, 0.05], [0.11, 0.07, 0.012], [0.05, 0.16, 0.75, 1.0]),
        ("yellow cup", "cylinder", [-0.05, 0.08], [0.05, 0.10], [0.95, 0.82, 0.22, 1.0]),
        ("red can", "cylinder", [0.20, 0.08], [0.045, 0.115], [0.85, 0.08, 0.05, 1.0]),
        ("thin book", "box", [0.00, 0.22], [0.12, 0.08, 0.014], [0.55, 0.08, 0.62, 1.0]),
        ("black remote", "box", [0.30, 0.20], [0.07, 0.025, 0.012], [0.02, 0.02, 0.02, 1.0]),
    ]
    created = []
    for name, shape, xy, dims, color in props:
        if shape == "cylinder":
            radius, height = dims
            body = create_cylinder_body(p, radius, height, [xy[0], xy[1], table_top_z + height / 2.0], color)
            created.append({"name": name, "body": body, "style": "cylinder", "position": [xy[0], xy[1]]})
        else:
            hx, hy, hz = dims
            body = create_box_body(p, [hx, hy, hz], [xy[0], xy[1], table_top_z + hz], color)
            created.append({"name": name, "body": body, "style": "box", "position": [xy[0], xy[1]]})
    return created


def make_body(p, obj: dict, mask: np.ndarray, xy: Tuple[float, float], z_layer: int, scene_scale_m: float, table_top_z: float):
    name = obj.get("name", str(obj.get("id", "")))
    style, sx, sy, sz = styled_dims(name, mask, scene_scale_m)
    z = table_top_z + sz / 2.0 + z_layer * 0.055
    color_id = int(obj["id"]) % 6
    colors = [
        [0.8, 0.1, 0.1, 1.0],
        [0.1, 0.3, 0.9, 1.0],
        [0.9, 0.8, 0.6, 1.0],
        [0.9, 0.9, 0.9, 1.0],
        [0.1, 0.1, 0.1, 1.0],
        [0.1, 0.7, 0.3, 1.0],
    ][color_id]
    if style == "cylinder":
        radius = sx / 2.0
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=sz)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=sz, rgbaColor=colors)
    else:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2.0, sy / 2.0, sz / 2.0])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx / 2.0, sy / 2.0, sz / 2.0], rgbaColor=colors)
    body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[xy[0], xy[1], z])
    return body, {"style": style, "size_m": [sx, sy, sz], "initial_pose": [xy[0], xy[1], z]}


def render_frame(p, width: int, height: int, text: str, camera_preset: str) -> np.ndarray:
    if camera_preset == "front_occlusion":
        eye = [0.0, -1.05, 0.47]
        target = [0.0, 0.02, 0.42]
        fov = 35.0
    else:
        eye = [0.0, -0.78, 0.96]
        target = [0.0, 0.0, 0.34]
        fov = 48.0
    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=float(width) / float(height),
        nearVal=0.01,
        farVal=3.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, 0), (width, 54), (245, 245, 245), -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)
    return frame


def save_visual_frame(path: Path, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def incoming_count(relations: List[dict], object_id: int, removed: set) -> int:
    return sum(
        1
        for rel in relations
        if int(rel["blocked"]) == int(object_id)
        and int(rel["blocker"]) not in removed
        and int(rel["blocked"]) not in removed
    )


def run_simulation(args):
    add_pybullet_path(Path(args.pybullet_path) if args.pybullet_path else None)
    import pybullet as p
    import pybullet_data

    objects_list = read_json(Path(args.objects))
    objects = {int(o["id"]): o for o in objects_list}
    relations = read_json(Path(args.relations))
    ranked = read_json(Path(args.ranked_actions))
    first_action = choose_action(ranked, args.force_object_id)
    ranked_actions = action_by_object(ranked)
    target = ranked.get("target") or {}
    target_id = args.target_object_id if args.target_object_id is not None else (None if target.get("id") is None else int(target["id"]))
    if target_id is not None and (not target or target.get("id") is None):
        target = {"id": target_id, "name": objects[target_id].get("name", str(target_id))}

    masks = {obj_id: load_mask(Path(obj["modal_path"])) for obj_id, obj in objects.items()}
    z_layers = topological_heights(objects, relations)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.setTimeStep(1.0 / 240.0)
    create_table(p, args.table_top_z, (args.table_size_x, args.table_size_y), args.table_thickness)
    demo_props = create_demo_props(p, args.table_top_z) if args.add_demo_props else []

    frames: List[np.ndarray] = []
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("frame_*.png"):
            old_frame.unlink()

    def capture(label: str):
        if not args.visualize:
            return
        frame = render_frame(p, args.render_width, args.render_height, label, args.camera_preset)
        frame_idx = len(frames)
        frames.append(frame)
        if frames_dir:
            save_visual_frame(frames_dir / f"frame_{frame_idx:04d}.png", frame)

    body_by_obj = {}
    sim_objects = {}
    for demo_index, prop in enumerate(demo_props):
        obj_id = 1000 + demo_index
        objects[obj_id] = {"id": obj_id, "name": prop["name"]}
        body_by_obj[obj_id] = prop["body"]
        sim_objects[obj_id] = {
            "name": prop["name"],
            "body": prop["body"],
            "style": prop["style"],
            "is_demo_prop": True,
            "initial_pose": prop["position"],
        }
    for index, (obj_id, obj) in enumerate(objects.items()):
        if obj_id in body_by_obj:
            continue
        if args.layout_mode == "front_occlusion":
            xy = front_occlusion_xy(index)
            z_layer = 0
        else:
            xy = object_xy_from_mask(masks[obj_id], args.scene_scale_m)
            z_layer = z_layers.get(obj_id, 0)
        body, meta = make_body(p, obj, masks[obj_id], xy, z_layer, args.scene_scale_m, args.table_top_z)
        body_by_obj[obj_id] = body
        sim_objects[obj_id] = {
            "name": obj.get("name", str(obj_id)),
            "body": body,
            **meta,
        }

    for _ in range(120):
        p.stepSimulation()
    capture("settled scene")

    before_incoming = None if target_id is None else incoming_count(relations, target_id, removed=set())
    removed = set()
    executed_sequence = []
    step_index = 0

    def execute_grasp(grasp_id: int, action: dict) -> bool:
        nonlocal step_index
        if grasp_id not in body_by_obj:
            return False
        body = body_by_obj[grasp_id]
        pos, orn = p.getBasePositionAndOrientation(body)
        # Kinematic top-down "grasp": lift the selected object if it is top-accessible in the planned graph.
        top_accessible = incoming_count(relations, grasp_id, removed=removed) == 0
        if top_accessible:
            capture(f"step {step_index + 1}: grasp {action['object_name']}")
            steps = max(2, int(args.lift_steps))
            for step in range(steps):
                alpha = float(step + 1) / float(steps)
                lift_z = pos[2] + args.lift_m * alpha
                p.resetBasePositionAndOrientation(body, [pos[0], pos[1], lift_z], orn)
                p.stepSimulation()
                if step % max(1, args.capture_every) == 0 or step == steps - 1:
                    capture(f"step {step_index + 1}: lifting {action['object_name']}  {alpha:.0%}")
            carry_target = [
                args.drop_origin_x + args.drop_stride_x * step_index,
                args.drop_origin_y + args.drop_stride_y * (step_index % 2),
                pos[2] + args.lift_m,
            ]
            for step in range(steps // 2):
                alpha = float(step + 1) / float(max(1, steps // 2))
                x = pos[0] + (carry_target[0] - pos[0]) * alpha
                y = pos[1] + (carry_target[1] - pos[1]) * alpha
                z = pos[2] + args.lift_m
                p.resetBasePositionAndOrientation(body, [x, y, z], orn)
                p.stepSimulation()
                if step % max(1, args.capture_every) == 0 or step == steps // 2 - 1:
                    capture(f"step {step_index + 1}: carrying away {action['object_name']}  {alpha:.0%}")
            p.resetBasePositionAndOrientation(body, [2.0 + 0.1 * step_index, 2.0, -1.0], orn)
            removed.add(grasp_id)
            executed_sequence.append(
                {
                    "step": step_index + 1,
                    "object_id": grasp_id,
                    "object_name": action["object_name"],
                    "reward": float(action.get("reward", 0.0)),
                    "role": action.get("role"),
                    "success": True,
                    "target_incoming_after_step": None if target_id is None else incoming_count(relations, target_id, removed=removed),
                }
            )
            step_index += 1
            return True
        else:
            capture(f"step {step_index + 1}: {action['object_name']} is blocked")
            executed_sequence.append(
                {
                    "step": step_index + 1,
                    "object_id": grasp_id,
                    "object_name": action["object_name"],
                    "reward": float(action.get("reward", 0.0)),
                    "role": action.get("role"),
                    "success": False,
                    "reason": "not_top_accessible",
                }
            )
            step_index += 1
            return False

    if args.force_object_id is not None:
        execute_grasp(int(first_action["object_id"]), first_action)
    else:
        max_steps = args.max_steps if args.max_steps is not None else len(objects)
        while len(removed) < len(objects) and step_index < max_steps:
            next_id = choose_next_object(args.task_mode, objects, relations, ranked_actions, removed, target_id)
            if next_id is None:
                break
            action = ranked_actions.get(next_id, make_fallback_action(objects[next_id]))
            execute_grasp(next_id, action)
            if args.task_mode == "reach_target" and target_id in removed:
                break

    for _ in range(120):
        p.stepSimulation()
    capture("after grasp")

    after_incoming = None if target_id is None else incoming_count(relations, target_id, removed=removed)
    target_released = False
    if target_id is not None and before_incoming is not None and after_incoming is not None:
        target_released = after_incoming < before_incoming
    task_complete = len(removed) == len(objects) if args.task_mode == "clear_table" else (target_id in removed if target_id is not None else False)

    final_poses = {}
    for obj_id, body in body_by_obj.items():
        pos, orn = p.getBasePositionAndOrientation(body)
        final_poses[obj_id] = {"position": list(pos), "orientation_xyzw": list(orn)}

    p.disconnect()
    video_path = None
    if args.visualize and args.video:
        video_path = str(Path(args.video))
        write_video(Path(args.video), frames, fps=args.fps)
    gif_path = None
    if args.visualize and args.gif:
        gif_path = str(Path(args.gif))
        write_gif(Path(args.gif), frames, duration_ms=args.gif_duration_ms)
    reward = 0.0
    reward += 4.0 * sum(1 for step in executed_sequence if step.get("success"))
    reward += -5.0 * sum(1 for step in executed_sequence if not step.get("success"))
    reward += 3.0 if target_released else 0.0
    reward += -0.1 * len(executed_sequence)

    return {
        "simulator": "pybullet_tabletop_semantic_objects",
        "note": "This validates grasp-order handoff and top-down removal logic, not final real robot contact dynamics.",
        "task_mode": args.task_mode,
        "selected_action": first_action,
        "executed_sequence": executed_sequence,
        "target": target,
        "task_complete": task_complete,
        "grasp_success": bool(executed_sequence and executed_sequence[-1].get("success")),
        "target_incoming_before": before_incoming,
        "target_incoming_after": after_incoming,
        "target_released": target_released,
        "removed_objects": sorted(list(removed)),
        "sim_reward": round(float(reward), 6),
        "visualization": {
            "enabled": bool(args.visualize),
            "frames_dir": str(frames_dir) if frames_dir else None,
            "video": video_path,
            "gif": gif_path,
            "frame_count": len(frames),
            "camera_preset": args.camera_preset,
            "layout_mode": args.layout_mode,
        },
        "objects": sim_objects,
        "demo_props": demo_props,
        "final_poses": final_poses,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight PyBullet grasp-order simulation from ranked actions.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--ranked-actions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pybullet-path", default="/home/ubuntu22-zy/occlusion/.deps/pybullet")
    parser.add_argument("--scene-scale-m", type=float, default=0.60)
    parser.add_argument("--table-top-z", type=float, default=0.32)
    parser.add_argument("--table-size-x", type=float, default=0.96)
    parser.add_argument("--table-size-y", type=float, default=0.78)
    parser.add_argument("--table-thickness", type=float, default=0.045)
    parser.add_argument("--camera-preset", choices=["tabletop", "front_occlusion"], default="tabletop")
    parser.add_argument("--layout-mode", choices=["mask", "front_occlusion"], default="mask")
    parser.add_argument("--add-demo-props", action="store_true")
    parser.add_argument("--lift-m", type=float, default=0.25)
    parser.add_argument("--carry-dx-m", type=float, default=0.24)
    parser.add_argument("--carry-dy-m", type=float, default=-0.18)
    parser.add_argument("--drop-origin-x", type=float, default=0.30)
    parser.add_argument("--drop-origin-y", type=float, default=-0.22)
    parser.add_argument("--drop-stride-x", type=float, default=-0.09)
    parser.add_argument("--drop-stride-y", type=float, default=0.08)
    parser.add_argument("--lift-steps", type=int, default=72)
    parser.add_argument("--capture-every", type=int, default=3)
    parser.add_argument("--task-mode", choices=["reach_target", "clear_table"], default="reach_target")
    parser.add_argument("--target-object-id", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--gif", default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--force-object-id", type=int, default=None)
    args = parser.parse_args()

    result = run_simulation(args)
    write_json(Path(args.out), result)
    print(f"selected object {result['selected_action']['object_id']} ({result['selected_action']['object_name']})")
    print(f"task_mode={result['task_mode']} task_complete={result['task_complete']} steps={len(result['executed_sequence'])}")
    print(f"grasp_success={result['grasp_success']} target_released={result['target_released']} reward={result['sim_reward']}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
