"""Run a bounded, measurement-backed dual-Panda IsaacLab readiness scene."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
REPORT_SCHEMA_VERSION = 2
PLATFORM = "isaaclab_dual_franka_panda_fixed_base"
PHYSICS_DT = 1.0 / 60.0
PHASE_STEPS = 120
CUBE_SIZE_M = (0.15, 0.15, 0.15)
CAMERA_POSITION = (0.0, -2.075, 1.0)
CAMERA_IMAGE_SIZE = (120, 160)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report_atomically(report_path: str | os.PathLike, report: dict) -> None:
    """Replace the destination with one complete JSON document or leave it unchanged."""
    destination = Path(report_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _as_list(value) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _step(sim, scene, count: int) -> None:
    for _ in range(count):
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())


def _rotate_vector(quaternion, vector):
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (component / norm for component in (w, x, y, z))
    qv = (x, y, z)
    cross1 = (qv[1] * vector[2] - qv[2] * vector[1], qv[2] * vector[0] - qv[0] * vector[2], qv[0] * vector[1] - qv[1] * vector[0])
    cross2 = (qv[1] * cross1[2] - qv[2] * cross1[1], qv[2] * cross1[0] - qv[0] * cross1[2], qv[0] * cross1[1] - qv[1] * cross1[0])
    return [component + 2.0 * (w * first + second) for component, first, second in zip(vector, cross1, cross2)]


def _ray_box_depth(camera_pose, intrinsics, pixel_rc, cube_center, cube_dimensions, cube_orientation, front_axis, front_sign):
    row, column = pixel_rc
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    ray_camera = [(column - cx) / fx, (row - cy) / fy, 1.0]
    cube_inverse = [cube_orientation[0], -cube_orientation[1], -cube_orientation[2], -cube_orientation[3]]
    origin_local = _rotate_vector(cube_inverse, [camera_pose[index] - cube_center[index] for index in range(3)])
    direction_local = _rotate_vector(cube_inverse, _rotate_vector(camera_pose[3:], ray_camera))
    entry, exit_ = -math.inf, math.inf
    entry_faces = []
    for axis, (origin, direction, dimension) in enumerate(zip(origin_local, direction_local, cube_dimensions)):
        if abs(direction) < 1e-12:
            if abs(origin) > dimension / 2.0:
                raise RuntimeError("camera center pixel misses the known cube")
            continue
        first = (-dimension / 2.0 - origin) / direction
        second = (dimension / 2.0 - origin) / direction
        near, far = min(first, second), max(first, second)
        near_sign = -1 if first < second else 1
        if near > entry + 1e-9:
            entry = near
            entry_faces = [(axis, near_sign)]
        elif abs(near - entry) <= 1e-9:
            entry_faces.append((axis, near_sign))
        exit_ = min(exit_, far)
    if entry <= 0 or exit_ < entry or (front_axis, front_sign) not in entry_faces:
        raise RuntimeError("camera center pixel does not hit the known cube front face")
    return entry


def _orientation_drift(initial_quaternion, final_quaternion):
    initial_norm = math.sqrt(sum(value * value for value in initial_quaternion))
    final_norm = math.sqrt(sum(value * value for value in final_quaternion))
    dot = sum(a * b for a, b in zip(initial_quaternion, final_quaternion)) / (initial_norm * final_norm)
    return 2.0 * math.acos(min(1.0, abs(dot)))


def _robot_evidence(robot, arm_ids, finger_ids, initial, target, measured, opened, closed, reopened, base_initial):
    joint_ids = arm_ids + finger_ids
    final_positions = _as_list(robot.data.joint_pos[0, joint_ids])
    base_final = _as_list(robot.data.root_pos_w[0]) + _as_list(robot.data.root_quat_w[0])
    joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids, :].detach().cpu().tolist()
    return {
        "asset_uri": str(robot.cfg.spawn.usd_path),
        "fixed_base_measured": bool(robot.is_fixed_base),
        "self_collision_enabled": bool(robot.cfg.spawn.articulation_props.enabled_self_collisions),
        "gravity_enabled": not bool(robot.cfg.spawn.rigid_props.disable_gravity),
        "contact_sensors_enabled": bool(robot.cfg.spawn.activate_contact_sensors),
        "joint_names": ARM_JOINT_NAMES + FINGER_JOINT_NAMES,
        "joint_limits": [[float(pair[0]), float(pair[1])] for pair in joint_limits],
        "initial_joint_positions": initial + opened,
        "final_joint_positions": final_positions,
        "base_pose_initial": base_initial,
        "base_pose_final": base_final,
        "base_translation_drift_m": math.dist(base_initial[:3], base_final[:3]),
        "base_orientation_drift_rad": _orientation_drift(base_initial[3:], base_final[3:]),
        "arm_initial_positions": initial,
        "arm_target_positions": target,
        "arm_measured_positions": measured,
        "arm_tracking_error_rad": [actual - commanded for actual, commanded in zip(measured, target)],
        "finger_open_positions": opened,
        "finger_closed_positions": closed,
        "finger_reopened_positions": reopened,
    }


def run_readiness(device: str) -> dict:
    """Create the scene, exercise real joint targets, and return sensor evidence."""
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_assets import FRANKA_PANDA_CFG

    def panda_config(name: str, x_position: float):
        config = FRANKA_PANDA_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/{name}")
        config.init_state.pos = (x_position, 0.0, 0.0)
        config.spawn.activate_contact_sensors = True
        return config

    pitch = math.atan2(CAMERA_POSITION[2] - CUBE_SIZE_M[2] / 2.0, 2.0)
    camera_angle = -(math.pi / 2.0 + pitch)

    @configclass
    class DualPandaSceneCfg(InteractiveSceneCfg):
        ground = sim_utils.GroundPlaneCfg()
        dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        left_panda = panda_config("LeftPanda", -1.2)
        right_panda = panda_config("RightPanda", 1.2)
        known_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KnownCube",
            spawn=sim_utils.CuboidCfg(
                size=CUBE_SIZE_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CUBE_SIZE_M[2] / 2.0)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ReadinessCamera",
            update_period=0.0,
            height=CAMERA_IMAGE_SIZE[0],
            width=CAMERA_IMAGE_SIZE[1],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=CAMERA_POSITION,
                rot=(math.cos(camera_angle / 2.0), math.sin(camera_angle / 2.0), 0.0, 0.0),
                convention="ros",
            ),
        )

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(4.0, 4.0, 3.0), target=(0.0, 0.0, 0.0))
    scene = InteractiveScene(DualPandaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    dt = sim.get_physics_dt()

    robots = {side: scene[f"{side}_panda"] for side in ("left", "right")}
    joint_ids = {}
    for side, robot in robots.items():
        arm_ids, arm_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        finger_ids, finger_names = robot.find_joints(FINGER_JOINT_NAMES, preserve_order=True)
        if arm_names != ARM_JOINT_NAMES or finger_names != FINGER_JOINT_NAMES:
            raise RuntimeError(f"{side} Panda articulation does not expose the expected 7+2 joints")
        if not robot.cfg.spawn.activate_contact_sensors or not robot.cfg.spawn.articulation_props.enabled_self_collisions:
            raise RuntimeError(f"{side} Panda contact sensors or self-collision are disabled")
        joint_ids[side] = (arm_ids, finger_ids)

    _step(sim, scene, PHASE_STEPS)
    initial, opened, base_initial, targets = {}, {}, {}, {}
    for side, robot in robots.items():
        arm_ids, finger_ids = joint_ids[side]
        initial[side] = _as_list(robot.data.joint_pos[0, arm_ids])
        opened[side] = _as_list(robot.data.joint_pos[0, finger_ids])
        base_initial[side] = _as_list(robot.data.root_pos_w[0]) + _as_list(robot.data.root_quat_w[0])
        targets[side] = initial[side].copy()
        targets[side][0] += 0.04
        command = torch.tensor([targets[side]], dtype=robot.data.joint_pos.dtype, device=robot.device)
        robot.set_joint_position_target(command, joint_ids=arm_ids)
    _step(sim, scene, PHASE_STEPS)
    arm_measured = {
        side: _as_list(robots[side].data.joint_pos[0, joint_ids[side][0]])
        for side in robots
    }

    closed = {}
    for side, robot in robots.items():
        finger_ids = joint_ids[side][1]
        command = torch.full((1, 2), 0.006, dtype=robot.data.joint_pos.dtype, device=robot.device)
        robot.set_joint_position_target(command, joint_ids=finger_ids)
    _step(sim, scene, PHASE_STEPS)
    for side, robot in robots.items():
        closed[side] = _as_list(robot.data.joint_pos[0, joint_ids[side][1]])
        command = torch.full((1, 2), 0.04, dtype=robot.data.joint_pos.dtype, device=robot.device)
        robot.set_joint_position_target(command, joint_ids=joint_ids[side][1])
    _step(sim, scene, PHASE_STEPS)
    reopened = {
        side: _as_list(robots[side].data.joint_pos[0, joint_ids[side][1]])
        for side in robots
    }

    camera = scene["camera"]
    rgb = camera.data.output["rgb"][0]
    depth = camera.data.output["distance_to_image_plane"][0]
    if rgb.ndim != 3 or depth.ndim != 3 or depth.shape[-1] != 1:
        raise RuntimeError(f"camera returned unexpected RGB-D shapes: {tuple(rgb.shape)}, {tuple(depth.shape)}")
    center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
    sample_pixel = [center_y, center_x]
    measured_depth = float(depth[center_y, center_x, 0].item())
    if not math.isfinite(measured_depth) or measured_depth <= 0.1:
        raise RuntimeError("camera center pixel has no finite depth on the known cube")
    camera_pose = _as_list(camera.data.pos_w[0]) + _as_list(camera.data.quat_w_ros[0])
    cube_center = _as_list(scene["known_cube"].data.root_pos_w[0])
    camera_matrix = camera.data.intrinsic_matrices[0].detach().cpu().tolist()
    cube_orientation = _as_list(scene["known_cube"].data.root_quat_w[0])
    front_face_axis, front_face_sign = 1, -1
    expected_depth = _ray_box_depth(
        camera_pose,
        camera_matrix,
        sample_pixel,
        cube_center,
        CUBE_SIZE_M,
        cube_orientation,
        front_face_axis,
        front_face_sign,
    )
    robot_reports = {
        side: _robot_evidence(
            robots[side], *joint_ids[side], initial[side], targets[side], arm_measured[side],
            opened[side], closed[side], reopened[side], base_initial[side]
        )
        for side in robots
    }

    return {
        "platform": PLATFORM,
        "scope": "readiness_only",
        "grasp_success_claimed": False,
        "collision_path_tested": False,
        "scene": {
            "ground_present": True,
            "gravity_enabled": any(float(component) != 0.0 for component in sim_cfg.gravity),
            "contact_sensors_enabled": all(item["contact_sensors_enabled"] for item in robot_reports.values()),
            "self_collision_enabled": all(item["self_collision_enabled"] for item in robot_reports.values()),
            "steps": PHASE_STEPS * 4,
            "physics_dt_s": dt,
            "known_geometry": {
                "kind": "rigid_cube",
                "dimensions_m": list(CUBE_SIZE_M),
                "center_world_m": cube_center,
                "orientation_world_wxyz": cube_orientation,
                "front_face_axis_index": front_face_axis,
                "front_face_sign": front_face_sign,
                "sample_pixel_rc": sample_pixel,
                "expected_depth_m": expected_depth,
                "measured_depth_m": measured_depth,
                "absolute_error_m": abs(expected_depth - measured_depth),
            },
        },
        "robots": robot_reports,
        "camera": {
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "frame": "world",
            "depth_data_type": "distance_to_image_plane",
            "depth_units": "m",
            "optical_frame": "ros",
            "optical_axis": "+Z",
            "quaternion_order": "wxyz",
            "pose_world": camera_pose,
            "intrinsics_matrix": camera_matrix,
            "rgb_finite_fraction": float(torch.isfinite(rgb.to(torch.float32)).float().mean().item()),
            "depth_finite_fraction": float(torch.isfinite(depth.to(torch.float32)).float().mean().item()),
        },
    }


def _parse_request_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_known_args(argv)[0]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    request = _parse_request_args(argv)
    report_path = Path(request.report).expanduser().resolve()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": request.run_id,
        "status": "error",
        "error": "worker did not complete",
        "started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now(),
    }
    try:
        write_report_atomically(report_path, report)
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--report", required=True)
        parser.add_argument("--run-id", required=True)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        if not args.headless or not args.enable_cameras:
            raise ValueError("worker requires --headless and --enable_cameras")

        app_launcher = AppLauncher(args)
        try:
            evidence = run_readiness(args.device)
        finally:
            app_launcher.app.close()
        report.update(evidence)
        report.update(status="complete", error=None, completed_at_utc=_utc_now())
        write_report_atomically(report_path, report)
        print(f"readiness complete: run_id={request.run_id} report={report_path}")
        return 0
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", completed_at_utc=_utc_now())
        try:
            write_report_atomically(report_path, report)
        except OSError as write_error:
            print(f"readiness worker failed and error report could not be saved: {write_error}", file=sys.stderr)
        print(f"readiness worker failed: {report['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())