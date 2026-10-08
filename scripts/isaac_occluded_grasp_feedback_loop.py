#!/usr/bin/env python3
"""Same-scene occlusion -> Pi05 -> Franka proxy -> re-observe feedback smoke.

This validates the persistent simulation/inference/control loop with a Franka
proxy and geometric clutter. Pi05-DROID actions are executed as bounded Panda
joint velocities; the base checkpoint is not fine-tuned for this proxy task.
"""
import argparse
import json
import os
import time
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--target", default="GreenMugProxy")
parser.add_argument("--cycles", type=int, default=2)
parser.add_argument("--ik-steps-per-waypoint", type=int, default=120)
parser.add_argument("--grasp-approach-distance", type=float, default=0.05)
parser.add_argument("--grasp-lift-height", type=float, default=0.08)
parser.add_argument("--grasp-palm-offset", type=float, default=0.14)
parser.add_argument("--grasp-y-offset", type=float, default=0.0)
parser.add_argument("--control-source", choices=("pi05", "visual_ik"), default="pi05")
parser.add_argument("--pi05-actions-per-cycle", type=int, default=2)
parser.add_argument("--pi05-max-joint-velocity", type=float, default=0.3)
parser.add_argument("--pi05-gripper-one-is-open", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--success-lift-threshold", type=float, default=0.06)
parser.add_argument("--gripper-close-steps", type=int, default=30)
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--warmup-steps", type=int, default=12)
parser.add_argument("--debug-control", action="store_true")
parser.add_argument("--record-expert", action="store_true", help="Record 15-Hz visual_ik demonstrations; export only episodes that pass grasp success checks.")
parser.add_argument("--skip-model-inference", action="store_true")
parser.add_argument("--grasp-point-source", choices=("visual", "oracle_xy", "oracle_xyz"), default="visual")
parser.add_argument("--grasp-orientation", choices=("top_down", "current"), default="top_down")
def environment_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


parser.add_argument("--model-python", type=Path, default=environment_path("OPENPI_MODEL_PYTHON"))
parser.add_argument("--pi05-checkpoint", type=Path, default=environment_path("PI05_CHECKPOINT"))
parser.add_argument("--relation-checkpoint", type=Path, default=environment_path("OCCLUDED_GRASP_CHECKPOINT"))
parser.add_argument("--relation-model-root", type=Path, default=environment_path("OCCLUDED_GRASP_MODEL_ROOT"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.cycles, args.ik_steps_per_waypoint, args.gripper_close_steps, args.width, args.height, args.pi05_actions_per_cycle) <= 0 or min(args.grasp_approach_distance, args.grasp_lift_height, args.grasp_palm_offset, args.success_lift_threshold, args.pi05_max_joint_velocity) <= 0 or args.warmup_steps < 0:
    parser.error("cycle/IK/close/image settings and grasp distances must be positive; warmup must be non-negative")
if not args.skip_model_inference:
    required_paths = {
        "--model-python/OPENPI_MODEL_PYTHON": args.model_python,
        "--pi05-checkpoint/PI05_CHECKPOINT": args.pi05_checkpoint,
        "--relation-checkpoint/OCCLUDED_GRASP_CHECKPOINT": args.relation_checkpoint,
        "--relation-model-root/OCCLUDED_GRASP_MODEL_ROOT": args.relation_model_root,
    }
    for option, path in required_paths.items():
        if path is None or not path.exists():
            parser.error(f"{option} must point to an existing path")
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_matrix, quat_inv, subtract_frame_transforms
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from openpi.policies.pi05_droid_action import DROID_CONTROL_FREQUENCY_HZ, integrate_joint_velocity_action

OPENPI_ROOT = SCRIPT_DIR.parent
PREPROCESS_ROOT = OPENPI_ROOT.parent / "obstruction-preprocess"
CAMERA_POSITION_W = None
CAMERA_QUAT_ROS = None


def static_box(path, size, position, color, label):
    cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                              semantic_tags=[("class", label)])
    cfg.func(path, cfg, translation=position)


def clutter_box(name, size, position, color):
    cfg = RigidObjectCfg(
        prim_path=f"/World/Clutter/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0,
                friction_combine_mode="max", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            semantic_tags=[("class", name)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )
    return RigidObject(cfg=cfg)


def run_stage(name, command, env, logs):
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    (logs / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} returned {result.returncode}: {result.stderr[-8000:]}")
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)


def capture(sim, camera, wrist_camera, out_dir, warmup):
    for _ in range(args.warmup_steps if warmup else 2):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        wrist_camera.update(dt=sim.get_physics_dt())
    sim.step()
    camera.update(dt=sim.get_physics_dt(), force_recompute=True)
    wrist_camera.update(dt=sim.get_physics_dt(), force_recompute=True)
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    wrist_rgb = wrist_camera.data.output["rgb"][0].detach().cpu().numpy()
    depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    seg = camera.data.output["instance_segmentation_fast"][0].detach().cpu().numpy()
    if seg.ndim == 3:
        seg = seg[..., 0]
    np.save(out_dir / "rgb.npy", rgb)
    np.save(out_dir / "depth.npy", depth)
    np.save(out_dir / "instance_segmentation.npy", seg.astype(np.int32))
    rgb_path = out_dir / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)
    wrist_rgb_path = out_dir / "wrist_rgb.png"
    Image.fromarray(wrist_rgb).save(wrist_rgb_path)
    # Copy metadata now because later camera updates overwrite camera.data.info.
    info = camera.data.info[0].get("instance_segmentation_fast") or {}
    info["configured_pose_ros"] = {"position_w": CAMERA_POSITION_W[0].detach().cpu().tolist(),
                                    "quaternion_wxyz": CAMERA_QUAT_ROS[0].detach().cpu().tolist(),
                                    "intrinsic_matrix": camera.data.intrinsic_matrices[0].detach().cpu().tolist()}
    (out_dir / "camera_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return rgb_path, wrist_rgb_path, out_dir / "depth.npy", info.get("idToLabels", {})


def pixel_to_world_grasp_point(camera, cycle_dir, labels, object_name):
    """Back-project the selected instance-mask centroid with measured depth."""
    segmentation = np.load(cycle_dir / "instance_segmentation.npy")
    depth = np.load(cycle_dir / "depth.npy")
    instance_id = next((int(key) for key, label in labels.items() if label.endswith("/" + object_name)), None)
    if instance_id is None:
        raise RuntimeError(f"selected object {object_name!r} has no camera instance ID")
    ys, xs = np.nonzero(segmentation == instance_id)
    if len(xs) < 20:
        raise RuntimeError(f"selected object {object_name!r} has too few visible pixels: {len(xs)}")
    instance_depth = depth[ys, xs]
    valid_mask = np.isfinite(instance_depth) & (instance_depth > 0.05)
    if not np.any(valid_mask):
        raise RuntimeError(f"selected object {object_name!r} has no valid depth")
    valid_xs = xs[valid_mask].astype(np.float32)
    valid_ys = ys[valid_mask].astype(np.float32)
    valid_depth = instance_depth[valid_mask].astype(np.float32)
    intrinsic = camera.data.intrinsic_matrices[0]
    fx, fy = float(intrinsic[0, 0].item()), float(intrinsic[1, 1].item())
    cx, cy = float(intrinsic[0, 2].item()), float(intrinsic[1, 2].item())
    points_camera = np.stack(((valid_xs - cx) * valid_depth / fx,
                              (valid_ys - cy) * valid_depth / fy,
                              valid_depth), axis=-1)
    median_point_camera = np.median(points_camera, axis=0).astype(np.float32)
    point_camera = torch.as_tensor(median_point_camera[None, :], device=camera.device)
    u, v = float(np.median(valid_xs)), float(np.median(valid_ys))
    z = float(median_point_camera[2])
    quat_ros = CAMERA_QUAT_ROS
    camera_position_w = CAMERA_POSITION_W
    if quat_ros is None or camera_position_w is None or not torch.isfinite(camera_position_w).all() or not torch.isfinite(quat_ros).all():
        raise RuntimeError("configured camera world pose is unavailable/non-finite")
    point_world = camera_position_w + quat_apply(quat_ros, point_camera)
    forward_world = quat_apply(quat_ros, torch.tensor([[0.0, 0.0, 1.0]], device=camera.device))
    return point_world[0], forward_world[0], {"instance_id": instance_id, "pixel_uv": [u, v], "depth_m": z,
                                              "camera_point_xyz_m": point_camera[0].cpu().tolist(),
                                              "world_surface_point_xyz_m": point_world[0].cpu().tolist()}


def execute_pi05_action_chunk(sim, robot, clutter_assets, actions, arm_ids, finger_ids, joint_limits):
    """Execute unnormalized Pi05-DROID joint velocities at the dataset control rate."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise ValueError(f"expected finite Pi05 actions with shape [T, 8], got {actions.shape}")
    action_count = min(args.pi05_actions_per_cycle, len(actions))
    physics_steps = max(1, round(1.0 / (DROID_CONTROL_FREQUENCY_HZ * sim.get_physics_dt())))
    limits_np = joint_limits[0].detach().cpu().numpy()
    execution_log = []
    for action_index, action in enumerate(actions[:action_count]):
        current_np = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        target_np, gripper_target = integrate_joint_velocity_action(
            current_np, action, limits_np,
            max_joint_velocity=args.pi05_max_joint_velocity,
            gripper_one_is_open=args.pi05_gripper_one_is_open,
        )
        target = torch.as_tensor(target_np, device=robot.device)[None, :]
        fingers = torch.full((1, len(finger_ids)), gripper_target, dtype=torch.float32, device=robot.device)
        robot.set_joint_position_target(target, joint_ids=arm_ids)
        robot.set_joint_position_target(fingers, joint_ids=finger_ids)
        for _ in range(physics_steps):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
            for asset in clutter_assets.values():
                asset.update(sim.get_physics_dt())
        velocity_clipped_mask = (np.abs(action[:7]) > args.pi05_max_joint_velocity)
        execution_log.append({
            "chunk_index": action_index,
            "raw_joint_velocity": action[:7].tolist(),
            "velocity_clipped_mask": velocity_clipped_mask.tolist(),
            "velocity_clipped_fraction": float(np.mean(velocity_clipped_mask)),
            "raw_gripper_position": float(action[7]),
            "target_joint_position": target_np.tolist(),
            "measured_joint_position": robot.data.joint_pos[0, arm_ids].detach().cpu().tolist(),
            "gripper_target_m": gripper_target,
            "physics_steps": physics_steps,
        })
    return {
        "control_frequency_hz": DROID_CONTROL_FREQUENCY_HZ,
        "actions_executed": action_count,
        "max_joint_velocity": args.pi05_max_joint_velocity,
        "gripper_one_is_open": args.pi05_gripper_one_is_open,
        "action_log": execution_log,
    }


def move_ee_to(sim, robot, camera, wrist_camera, clutter_assets, controller, arm_ids, ee_body_id, jacobian_body_id,
               target_position_w, target_quaternion_w, joint_limits, finger_ids, gripper_target_m, steps,
               expert_samples=None, expert_frame_dir=None):
    """Move Panda hand to a world-frame pose using frame-consistent DLS differential IK."""
    if args.debug_control:
        print("[control-debug] move_ee_to entered", flush=True)
        print("[control-debug] move before robot.update", flush=True)
    robot.update(dt=sim.get_physics_dt())
    if args.debug_control:
        print("[control-debug] move after robot.update", flush=True)
    root_pose = robot.data.root_pose_w
    target_position_b, target_quaternion_b = subtract_frame_transforms(
        root_pose[:, :3], root_pose[:, 3:7], target_position_w[None, :], target_quaternion_w[None, :]
    )
    if args.debug_control:
        print("[control-debug] move after frame transform", flush=True)
    controller.reset()
    if args.debug_control:
        print("[control-debug] move after controller.reset", flush=True)
    controller.set_command(torch.cat((target_position_b, target_quaternion_b), dim=-1))
    if args.debug_control:
        print("[control-debug] move after set_command", flush=True)
    start_position_w = robot.data.body_pose_w[0, ee_body_id, :3].detach().cpu().tolist()
    if args.debug_control:
        print("[control-debug] move after start pose cpu sync", flush=True)
    target_position_w_list = target_position_w.detach().cpu().tolist()
    last_position_error = float("inf")
    physics_steps_per_dataset_frame = max(1, round(1.0 / (DROID_CONTROL_FREQUENCY_HZ * sim.get_physics_dt())))
    for step_index in range(steps):
        debug_step = args.debug_control and step_index == 0
        sample_boundary = (expert_samples is not None and expert_frame_dir is not None
                           and step_index % physics_steps_per_dataset_frame == 0
                           and step_index + physics_steps_per_dataset_frame <= steps)
        if sample_boundary:
            segment_start_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()
            finger_state = float(torch.clamp(robot.data.joint_pos[0, finger_ids].mean() / 0.04, 0.0, 1.0).item())
            external_image = camera.data.output["rgb"][0].detach().cpu().numpy()
            wrist_image = wrist_camera.data.output["rgb"][0].detach().cpu().numpy()
            frame_id = len(expert_samples)
            external_rel = Path("expert_frames") / f"{frame_id:06d}_external.png"
            wrist_rel = Path("expert_frames") / f"{frame_id:06d}_wrist.png"
            expert_frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(external_image).save(expert_frame_dir.parent / external_rel)
            Image.fromarray(wrist_image).save(expert_frame_dir.parent / wrist_rel)
        if debug_step:
            print("[control-debug] before jacobian", flush=True)
        # PhysX Jacobians are world-frame; convert both linear/angular rows to robot-root frame,
        # matching Isaac Lab's verified Franka DLS controller test.
        jacobian_world = robot.root_physx_view.get_jacobians()[:, jacobian_body_id, :, arm_ids]
        if debug_step:
            print("[control-debug] after jacobian", flush=True)
        base_rot_inv = matrix_from_quat(quat_inv(root_pose[:, 3:7]))
        jacobian = jacobian_world.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_inv, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_inv, jacobian[:, 3:, :])
        hand_pose_w = robot.data.body_pose_w[:, ee_body_id]
        hand_position_b, hand_quaternion_b = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], hand_pose_w[:, :3], hand_pose_w[:, 3:7]
        )
        arm_position = robot.data.joint_pos[:, arm_ids]
        if debug_step:
            print("[control-debug] before controller.compute", flush=True)
        desired = controller.compute(hand_position_b, hand_quaternion_b, jacobian, arm_position)
        if debug_step:
            print("[control-debug] after controller.compute", flush=True)
        desired = torch.where(torch.isfinite(desired), desired, arm_position)
        desired = arm_position + torch.clamp(desired - arm_position, -0.04, 0.04)
        desired = torch.maximum(torch.minimum(desired, joint_limits[:, :, 1]), joint_limits[:, :, 0])
        robot.set_joint_position_target(desired, joint_ids=arm_ids)
        fingers = torch.full((1, 2), gripper_target_m, dtype=torch.float32, device=robot.device)
        robot.set_joint_position_target(fingers, joint_ids=finger_ids)
        robot.write_data_to_sim()
        if debug_step:
            print("[control-debug] before sim.step", flush=True)
            step_started = time.perf_counter()
        render_expert_frame = (expert_samples is not None and (step_index + 1) % physics_steps_per_dataset_frame == 0)
        sim.step(render=render_expert_frame)
        if render_expert_frame:
            camera.update(dt=sim.get_physics_dt(), force_recompute=True)
            wrist_camera.update(dt=sim.get_physics_dt(), force_recompute=True)
        if debug_step:
            print(f"[control-debug] after sim.step seconds={time.perf_counter() - step_started:.6f}", flush=True)
        robot.update(sim.get_physics_dt())
        if debug_step:
            print("[control-debug] after robot.update", flush=True)
        for asset in clutter_assets.values():
            asset.update(sim.get_physics_dt())
        if sample_boundary and (step_index + 1) % physics_steps_per_dataset_frame == 0:
            segment_end_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()
            elapsed = physics_steps_per_dataset_frame * sim.get_physics_dt()
            joint_velocity = (segment_end_q - segment_start_q) / elapsed
            gripper_position = float(np.clip(gripper_target_m / 0.04, 0.0, 1.0))
            expert_samples.append({
                "exterior_image_1_left": external_rel.as_posix(),
                "wrist_image_left": wrist_rel.as_posix(),
                "state": np.concatenate((segment_start_q, [finger_state])).astype(np.float32).tolist(),
                "action": np.concatenate((joint_velocity, [gripper_position])).astype(np.float32).tolist(),
            })
        if debug_step:
            print("[control-debug] after rigid updates", flush=True)
        root_pose = robot.data.root_pose_w
        hand_pose_w = robot.data.body_pose_w[:, ee_body_id]
        hand_position_b, _ = subtract_frame_transforms(root_pose[:, :3], root_pose[:, 3:7],
                                                        hand_pose_w[:, :3], hand_pose_w[:, 3:7])
        last_position_error = float(torch.linalg.vector_norm(hand_position_b - target_position_b).item())
    final_position_w = robot.data.body_pose_w[0, ee_body_id, :3].detach().cpu().tolist()
    return {"position_error_root_m": last_position_error, "start_hand_world_m": start_position_w,
            "target_hand_world_m": target_position_w_list, "final_hand_world_m": final_position_w}


def execute_visual_grasp(sim, robot, camera, clutter_assets, controller, arm_ids, ee_body_id, jacobian_body_id,
                         joint_limits, finger_ids, target_world, camera_forward, selected_object, args,
                         cycle_dir, wrist_camera):
    if args.debug_control:
        print("[control-debug] execute_visual_grasp entered", flush=True)
        print("[control-debug] before initial robot.update", flush=True)
    robot.update(dt=sim.get_physics_dt())
    if args.debug_control:
        print("[control-debug] after initial robot.update", flush=True)
    root_pose = robot.data.root_pose_w
    hand_pose_w = robot.data.body_pose_w[:, ee_body_id]
    # Preserve the current hand orientation in world coordinates for all IK waypoints.
    if args.grasp_orientation == "top_down":
        target_quaternion_w = torch.tensor([0.0, 0.7071068, 0.7071068, 0.0], device=robot.device)
    else:
        target_quaternion_w = hand_pose_w[0, 3:7].clone()
    if args.debug_control:
        local_x = quat_apply(target_quaternion_w[None, :], torch.tensor([[1.0, 0.0, 0.0]], device=robot.device))[0]
        local_y = quat_apply(target_quaternion_w[None, :], torch.tensor([[0.0, 1.0, 0.0]], device=robot.device))[0]
        local_z = quat_apply(target_quaternion_w[None, :], torch.tensor([[0.0, 0.0, 1.0]], device=robot.device))[0]
        print(f"[control-debug] hand_quat_wxyz={target_quaternion_w.detach().cpu().tolist()}", flush=True)
        print(f"[control-debug] hand_axes_world x={local_x.cpu().tolist()} y={local_y.cpu().tolist()} z={local_z.cpu().tolist()}", flush=True)
    target_world = target_world.clone()
    # The visual mask/depth gives the visible front surface. Move slightly into the object
    # along camera-forward so the two Panda fingers can straddle its width.
    grasp_center = target_world + camera_forward * 0.055
    asset = clutter_assets[selected_object]
    expert_samples = [] if args.record_expert else None
    expert_frame_dir = cycle_dir / "expert_frames" if args.record_expert else None
    if args.grasp_point_source in ("oracle_xy", "oracle_xyz"):
        grasp_center = grasp_center.clone()
        grasp_center[:2] = asset.data.root_pos_w[0, :2]
    if args.grasp_point_source == "oracle_xyz":
        grasp_center[2] = asset.data.root_pos_w[0, 2]
    grasp_center[1] += args.grasp_y_offset
    contact = grasp_center + torch.tensor([0.0, 0.0, args.grasp_palm_offset], device=robot.device)
    approach = contact + torch.tensor([0.0, 0.0, args.grasp_approach_distance], device=robot.device)
    lift = contact + torch.tensor([0.0, 0.0, args.grasp_lift_height], device=robot.device)
    if args.debug_control:
        print("[control-debug] before selected object read", flush=True)
    before_position = asset.data.root_pos_w[0].clone()
    if args.debug_control:
        print("[control-debug] after selected object read", flush=True)
    hand_initial_world = robot.data.body_pose_w[0, ee_body_id, :3].detach().cpu().tolist()
    measured = {}
    object_positions_by_stage = {}
    if args.debug_control:
        print("[control-debug] before first move_ee_to", flush=True)
    measured["open_and_approach"] = move_ee_to(sim, robot, camera, wrist_camera, clutter_assets, controller, arm_ids,
                                                          ee_body_id, jacobian_body_id, approach,
                                                          target_quaternion_w, joint_limits, finger_ids, 0.04,
                                                          args.ik_steps_per_waypoint,
                                                          expert_samples, expert_frame_dir)
    object_positions_by_stage["after_approach"] = asset.data.root_pos_w[0].detach().cpu().tolist()
    measured["descend"] = move_ee_to(sim, robot, camera, wrist_camera, clutter_assets, controller, arm_ids,
                                              ee_body_id, jacobian_body_id, contact, target_quaternion_w,
                                              joint_limits, finger_ids, 0.04, args.ik_steps_per_waypoint,
                                              expert_samples, expert_frame_dir)
    object_positions_by_stage["after_descend"] = asset.data.root_pos_w[0].detach().cpu().tolist()
    measured["close"] = move_ee_to(sim, robot, camera, wrist_camera, clutter_assets, controller, arm_ids,
                                             ee_body_id, jacobian_body_id, contact, target_quaternion_w,
                                             joint_limits, finger_ids, 0.0, args.gripper_close_steps,
                                             expert_samples, expert_frame_dir)
    object_positions_by_stage["after_close"] = asset.data.root_pos_w[0].detach().cpu().tolist()
    left_finger_body_id = robot.find_bodies("panda_leftfinger")[0][0]
    right_finger_body_id = robot.find_bodies("panda_rightfinger")[0][0]
    left_finger_position = robot.data.body_pos_w[0, left_finger_body_id].clone()
    right_finger_position = robot.data.body_pos_w[0, right_finger_body_id].clone()
    finger_midpoint_after_close = 0.5 * (left_finger_position + right_finger_position)
    finger_joint_positions_after_close = robot.data.joint_pos[0, finger_ids].clone()
    measured["lift"] = move_ee_to(sim, robot, camera, wrist_camera, clutter_assets, controller, arm_ids,
                                            ee_body_id, jacobian_body_id, lift, target_quaternion_w,
                                            joint_limits, finger_ids, 0.0, args.ik_steps_per_waypoint,
                                            expert_samples, expert_frame_dir)
    object_positions_by_stage["after_lift"] = asset.data.root_pos_w[0].detach().cpu().tolist()
    after_position = asset.data.root_pos_w[0].clone()
    vertical_lift = float((after_position[2] - before_position[2]).item())
    ik_reached_contact = measured["descend"]["position_error_root_m"] <= 0.04
    final_hand_position = robot.data.body_pose_w[0, ee_body_id, :3]
    object_to_hand_after = float(torch.linalg.vector_norm(after_position - final_hand_position).item())
    grasp_success = (vertical_lift >= args.success_lift_threshold and ik_reached_contact
                     and object_to_hand_after <= 0.20)
    return {"success": grasp_success, "ik_reached_contact": ik_reached_contact,
            "hand_initial_world_m": hand_initial_world,
            "object_before_world_m": before_position.detach().cpu().tolist(),
            "object_after_world_m": after_position.detach().cpu().tolist(), "object_vertical_displacement_m": vertical_lift,
            "object_to_hand_after_m": object_to_hand_after,
            "success_threshold_m": args.success_lift_threshold, "ik_reached_contact": ik_reached_contact,
            "waypoint_diagnostics": measured, "object_positions_by_stage": object_positions_by_stage,
            "expert_sample_count": len(expert_samples) if expert_samples is not None else 0,
            "expert_samples": expert_samples if expert_samples is not None else [],
            "left_finger_after_close_world_m": left_finger_position.detach().cpu().tolist(),
            "right_finger_after_close_world_m": right_finger_position.detach().cpu().tolist(),
            "finger_midpoint_after_close_world_m": finger_midpoint_after_close.detach().cpu().tolist(),
            "finger_joint_positions_after_close_m": finger_joint_positions_after_close.detach().cpu().tolist(),
            "success_definition": "contact IK reached, object rose by the configured threshold, and remained within 0.20 m of the hand"}


def perceive_plan_infer(cycle_dir, rgb_path, wrist_rgb_path, depth_path, labels, model_env, robot_state):
    masks = cycle_dir / "masks"
    (masks / "modal").mkdir(parents=True, exist_ok=True)
    (masks / "amodal").mkdir(parents=True, exist_ok=True)
    seg = np.load(cycle_dir / "instance_segmentation.npy")
    objects = []
    for instance_id, label in sorted(((int(key), value) for key, value in labels.items()), key=lambda item: item[0]):
        if not label.startswith("/World/Clutter/"):
            continue
        name = label.rsplit("/", 1)[-1]
        mask = (seg == instance_id).astype(np.uint8) * 255
        if not np.any(mask):
            continue
        filename = f"{len(objects):03d}_{name}.png"
        Image.fromarray(mask).save(masks / "modal" / filename)
        # No amodal completion model is wired yet; explicitly use this as a proxy.
        Image.fromarray(mask).save(masks / "amodal" / filename)
        objects.append({"id": len(objects), "class_name": name, "mask": filename, "score": 1.0,
                        "mask_type": "sim_instance_segmentation"})
    if len(objects) < 2:
        raise RuntimeError(f"only {len(objects)} proxy clutter objects are visible")
    if args.skip_model_inference:
        selected = next((obj for obj in objects if obj["class_name"] == args.target), objects[0])
        ranked = {"ranked_actions": [{
            "object_id": selected["id"],
            "object_name": selected["class_name"],
            "role": "diagnostic_direct_target",
            "reward": 0.0,
        }]}
        actions = np.zeros((15, 8), dtype=np.float32)
        np.save(cycle_dir / "pi05_actions.npy", actions)
        return ranked, actions
    modal_json, amodal_json = masks / "modal_metadata.json", masks / "amodal_metadata.json"
    modal_json.write_text(json.dumps([{"class_name": obj["class_name"], "mask": obj["mask"], "score": 1.0} for obj in objects], indent=2), encoding="utf-8")
    amodal_json.write_text(json.dumps(objects, indent=2), encoding="utf-8")
    logs = cycle_dir / "logs"
    logs.mkdir(exist_ok=True)
    relation_dir = cycle_dir / "relations"
    run_stage("obstruction", [str(args.model_python), "-m", "obstruction_preprocess.estimate_obstruction",
                               "--rgb", str(rgb_path), "--depth", str(depth_path),
                               "--modal-dir", str(masks / "modal"), "--modal-metadata", str(modal_json),
                               "--amodal-dir", str(masks / "amodal"), "--amodal-metadata", str(amodal_json),
                               "--scene-id", "persistent-proxy", "--view-id", cycle_dir.name,
                               "--relation-mode", "model", "--relation-model-checkpoint", str(args.relation_checkpoint),
                               "--relation-model-root", str(args.relation_model_root), "--relation-model-name", "pair_transformer_base",
                               "--relation-model-device", "cuda", "--relation-model-min-confidence", "0.0",
                               "--depth-transform", "inverse", "--dedupe-iou", "1.01",
                               "--save-visualizations", "false", "--out-dir", str(relation_dir)], model_env, logs)
    ranked_dir = cycle_dir / "ranked"
    run_stage("rank_actions", [str(args.model_python), "-m", "obstruction_preprocess.rl_affordance_preprocess",
                                "--objects", str(relation_dir / "objects.json"), "--relations", str(relation_dir / "occ_detail.json"),
                                "--target", args.target, "--depth", str(depth_path), "--out-dir", str(ranked_dir)], model_env, logs)
    ranked_path = ranked_dir / "ranked_actions.json"
    action_path = cycle_dir / "pi05_actions.npy"
    state_path = cycle_dir / "pi05_state.npy"
    np.save(state_path, np.asarray(robot_state, dtype=np.float32))
    run_stage("pi05", [str(args.model_python), str(SCRIPT_DIR / "occluded_grasp_pi05_smoke.py"),
                        "--ranked-actions", str(ranked_path), "--image", str(rgb_path),
                        "--wrist-image", str(wrist_rgb_path), "--state", str(state_path),
                        "--out", str(cycle_dir / "pi05_summary.json"), "--checkpoint", str(args.pi05_checkpoint),
                        "--actions-out", str(action_path)], model_env, logs)
    return json.loads(ranked_path.read_text(encoding="utf-8")), np.load(action_path)


def main():
    output_root = args.out_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_env = os.environ.copy()
    paths = [str(OPENPI_ROOT / "src"), str(PREPROCESS_ROOT), str(args.relation_model_root)]
    model_env["PYTHONPATH"] = os.pathsep.join(paths + [model_env.get("PYTHONPATH", "")])
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    static_box("/World/Cabinet/Back", (0.96, 0.06, 1.45), (0.0, 0.35, 1.10), (0.48, 0.34, 0.22), "cabinet")
    static_box("/World/Cabinet/LeftWall", (0.06, 0.66, 1.50), (-0.51, 0.0, 1.10), (0.55, 0.40, 0.25), "cabinet")
    static_box("/World/Cabinet/RightWall", (0.06, 0.66, 1.50), (0.51, 0.0, 1.10), (0.55, 0.40, 0.25), "cabinet")
    static_box("/World/Cabinet/Bottom", (1.08, 0.70, 0.08), (0.0, 0.0, 0.36), (0.55, 0.40, 0.25), "cabinet")
    static_box("/World/Cabinet/Top", (1.08, 0.70, 0.08), (0.0, 0.0, 1.84), (0.55, 0.40, 0.25), "cabinet")
    static_box("/World/Cabinet/Shelf0", (0.96, 0.62, 0.045), (0.0, 0.02, 0.93), (0.64, 0.48, 0.30), "shelf")
    static_box("/World/Cabinet/Shelf1", (0.96, 0.62, 0.045), (0.0, 0.02, 1.40), (0.64, 0.48, 0.30), "shelf")
    clutter_specs = [
        ("GreenMugProxy", (0.02, 0.04, 0.15), (-0.10, -0.15, 1.0275), (0.10, 0.65, 0.25)),
        ("BlueBoxProxy", (0.25, 0.18, 0.22), (0.06, 0.04, 1.05), (0.20, 0.35, 0.85)),
        ("RedBoxProxy", (0.19, 0.18, 0.28), (0.25, 0.10, 1.08), (0.85, 0.18, 0.12)),
        ("YellowBoxProxy", (0.24, 0.18, 0.23), (-0.20, 0.08, 1.52), (0.85, 0.70, 0.12)),
    ]
    clutter = {name: clutter_box(name, size, pos, color) for name, size, pos, color in clutter_specs}
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, -0.72, 0.70)
    robot_cfg.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
    robot = Articulation(cfg=robot_cfg)
    sim_utils.create_prim("/World/CameraRoot", "Xform")
    camera = Camera(CameraCfg(prim_path="/World/CameraRoot/Camera", update_period=0.0,
                             width=args.width, height=args.height,
                             data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
                             colorize_instance_segmentation=False,
                             spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                                              horizontal_aperture=20.955, clipping_range=(0.05, 10.0))))
    wrist_camera = Camera(CameraCfg(
        prim_path="/World/Robot/panda_hand/wrist_cam", update_period=0.0,
        width=args.width, height=args.height, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    ))
    sim.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())
    global CAMERA_POSITION_W, CAMERA_QUAT_ROS
    CAMERA_POSITION_W = torch.tensor([[0.0, -1.8, 1.25]], dtype=torch.float32, device=sim.device)
    camera_target_w = torch.tensor([[0.0, 0.10, 1.22]], dtype=torch.float32, device=sim.device)
    camera_forward = torch.nn.functional.normalize(camera_target_w - CAMERA_POSITION_W, dim=-1)
    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=sim.device)
    camera_right = torch.nn.functional.normalize(torch.cross(camera_forward, world_up, dim=-1), dim=-1)
    camera_down = torch.nn.functional.normalize(torch.cross(camera_forward, camera_right, dim=-1), dim=-1)
    camera_rotation_ros = torch.stack((camera_right[0], camera_down[0], camera_forward[0]), dim=-1)[None, ...]
    CAMERA_QUAT_ROS = quat_from_matrix(camera_rotation_ros)
    camera.set_world_poses(CAMERA_POSITION_W, CAMERA_QUAT_ROS, convention="ros")
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    finger_ids = [i for i, name in enumerate(robot.joint_names) if name.startswith("panda_finger_joint")]
    if len(finger_ids) != 2:
        raise RuntimeError(f"expected two finger joints; got {finger_ids}")
    limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]
    ee_body_id = robot.find_bodies("panda_hand")[0][0]
    jacobian_body_id = ee_body_id - 1  # Panda is fixed-base; Jacobian excludes root.
    robot.update(dt=sim.get_physics_dt())
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device,
    )
    records = []
    for cycle in range(args.cycles):
        cycle_dir = output_root / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        before = {name: asset.data.root_pos_w[0].detach().cpu().tolist() for name, asset in clutter.items()}
        rgb_path, wrist_rgb_path, depth_path, labels = capture(
            sim, camera, wrist_camera, cycle_dir, warmup=(cycle == 0)
        )
        robot_state = np.concatenate((
            robot.data.joint_pos[0, arm_ids].detach().cpu().numpy(),
            [float(torch.clamp(robot.data.joint_pos[0, finger_ids].mean() / 0.04, 0.0, 1.0).item())],
        )).astype(np.float32)
        ranked, action_chunk = perceive_plan_infer(
            cycle_dir, rgb_path, wrist_rgb_path, depth_path, labels, model_env, robot_state
        )
        selected = ranked["ranked_actions"][0]
        selected_name = str(selected["object_name"])
        if selected_name not in clutter:
            raise RuntimeError(f"ranker selected unknown proxy object {selected_name!r}")
        grasp_point_world, camera_forward_world, grasp_observation = pixel_to_world_grasp_point(
            camera, cycle_dir, labels, selected_name
        )
        control_objects_before = {
            name: asset.data.root_pos_w[0].detach().cpu().tolist() for name, asset in clutter.items()
        }
        hand_before = robot.data.body_pos_w[0, ee_body_id].detach().cpu().numpy()
        print(f"[cycle {cycle}] selected={selected_name} target={args.target} Pi05 chunk={action_chunk.shape}; control_source={args.control_source}", flush=True)
        pi05_execution = None
        grasp_result = None
        if args.control_source == "pi05":
            pi05_execution = execute_pi05_action_chunk(
                sim, robot, clutter, action_chunk, arm_ids, finger_ids, limits
            )
        else:
            grasp_result = execute_visual_grasp(sim, robot, camera, clutter, controller, arm_ids,
                                                ee_body_id, jacobian_body_id, limits, finger_ids,
                                                grasp_point_world, camera_forward_world, selected_name, args,
                                                cycle_dir, wrist_camera)
            if args.record_expert:
                expert_manifest = {
                    "success": bool(grasp_result["success"]),
                    "task": f"Clear access to {args.target} by grasping {selected_name}.",
                    "frames": grasp_result.get("expert_samples", []),
                    "success_threshold_m": args.success_lift_threshold,
                    "provenance": "Isaac Lab visual_ik expert; only success=true episodes may be used for Pi05 training.",
                }
                if expert_manifest["success"] and len(expert_manifest["frames"]) >= 2:
                    (cycle_dir / "expert_episode_manifest.json").write_text(json.dumps(expert_manifest, indent=2) + "\n", encoding="utf-8")
                else:
                    expert_manifest["success"] = False
                    (cycle_dir / "expert_candidate_rejected.json").write_text(json.dumps(expert_manifest, indent=2) + "\n", encoding="utf-8")
        after = {name: asset.data.root_pos_w[0].detach().cpu().tolist() for name, asset in clutter.items()}
        hand_after = robot.data.body_pos_w[0, ee_body_id].detach().cpu().numpy()
        object_before = np.asarray(control_objects_before[selected_name], dtype=np.float32)
        object_after = np.asarray(after[selected_name], dtype=np.float32)
        hand_object_distance_before = float(np.linalg.norm(hand_before - object_before))
        hand_object_distance_after = float(np.linalg.norm(hand_after - object_after))
        if grasp_result is not None:
            grasp_success = grasp_result["success"]
        else:
            selected_lift = after[selected_name][2] - before[selected_name][2]
            selected_position = torch.as_tensor(after[selected_name], device=robot.device)
            hand_position = robot.data.body_pos_w[0, ee_body_id]
            object_to_hand = float(torch.linalg.vector_norm(selected_position - hand_position).item())
            grasp_success = selected_lift >= args.success_lift_threshold and object_to_hand <= 0.20
            pi05_execution["selected_object_vertical_displacement_m"] = selected_lift
            pi05_execution["selected_object_to_hand_after_m"] = object_to_hand
        record = {"cycle": cycle, "selected": selected, "ranked_actions": ranked["ranked_actions"],
                  "control_source": args.control_source, "pi05_state": robot_state.tolist(),
                  "hand_object_distance_before_m": hand_object_distance_before,
                  "hand_object_distance_after_m": hand_object_distance_after,
                  "hand_object_distance_delta_m": hand_object_distance_after - hand_object_distance_before,
                  "grasp_observation": grasp_observation, "robot_state_after": robot.data.joint_pos[0, arm_ids].detach().cpu().tolist(),
                  "objects_before": before, "objects_at_control_start": control_objects_before,
                  "objects_after": after, "visual_ik_grasp_attempt": grasp_result,
                  "pi05_execution": pi05_execution, "grasp_success": grasp_success,
                  "pi05_action_chunk_shape": list(action_chunk.shape),
                  "pi05_actions_used_for_joint_control": args.control_source == "pi05"}
        records.append(record)
        (cycle_dir / "cycle_result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    distance_deltas = [record["hand_object_distance_delta_m"] for record in records]
    clipped_fractions = [
        action["velocity_clipped_fraction"]
        for record in records if record["pi05_execution"] is not None
        for action in record["pi05_execution"]["action_log"]
    ]
    evaluation = {
        "cycles": len(records),
        "success_rate": sum(bool(record["grasp_success"]) for record in records) / len(records),
        "distance_improvement_rate": sum(delta < 0 for delta in distance_deltas) / len(distance_deltas),
        "mean_hand_object_distance_delta_m": float(np.mean(distance_deltas)),
        "mean_velocity_clipped_fraction": float(np.mean(clipped_fractions)) if clipped_fractions else None,
        "selected_objects": [record["selected"]["object_name"] for record in records],
    }
    final = {"schema": "occluded_grasp_pi05_feedback.v3", "simulator": "Isaac Sim 5.1 / Isaac Lab 2.3.2",
             "robot": "Franka Panda proxy", "target": args.target, "evaluation": evaluation, "cycles": records,
             "caveats": ["Scene, robot, camera and dynamic clutter objects persist across cycles; each cycle uses a new camera RGB-D and instance mask.",
                         "The selected instance ID mask centroid and camera depth are back-projected to a world grasp point; masks are simulator oracle segmentation.",
                         "Geometric clutter/cabinet are proxies; modal masks are reused as pseudo-amodal masks.",
                         "Pi05-DROID actions are unnormalized 7D Panda joint velocities plus 1D gripper position and are executed at 15 Hz with safety limits when control_source=pi05.",
                         "The visual-point DLS open/approach/close/lift primitive remains available only as an explicit comparison control source.",
                         "Grasp success is only a proxy metric: selected rigid-object root vertical displacement exceeds threshold; it is not a contact-force or true task-success detector.",
                         "This is not JAKA K1 geometry, calibration, or sim-to-real readiness."]}
    (output_root / "feedback_summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output_root / "feedback_summary.json"), "cycles": len(records),
                      "selected": [item["selected"]["object_name"] for item in records]}, indent=2), flush=True)


try:
    main()
except BaseException:
    app.close()
    raise
else:
    os._exit(0)
