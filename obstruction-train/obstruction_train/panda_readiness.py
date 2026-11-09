"""Bounded, auditable readiness preflight for a dual Franka Panda scene."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid

from obstruction_train.budget_runner import run_bounded


ISAAC_PYTHON = "/home/media/miniforge3/envs/isaaclab/bin/python"
ISAACLAB_SOURCE = Path("/home/media/zy/IsaacLab/source")
WORKER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "isaac_dual_panda_readiness.py"
PLATFORM = "isaaclab_dual_franka_panda_fixed_base"
DEFAULT_RUN_TIMEOUT_SECONDS = 300.0
GPU_BUDGET_SECONDS = 14400.0
MAX_REPORT_AGE_SECONDS = 900.0
ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]


def validate_readiness(report, *, run_id):
    """Validate and return fresh, complete physical evidence for ``run_id``."""
    if not isinstance(report, dict) or not report:
        raise ValueError("readiness report is incomplete")
    if report.get("status") == "error":
        raise ValueError("readiness report records a worker error")
    if not isinstance(run_id, str) or not run_id or report.get("run_id") != run_id:
        raise ValueError("readiness report run_id does not match request")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 2:
        raise ValueError("readiness report schema is malformed")
    if report.get("status") != "complete" or report.get("error") is not None:
        raise ValueError("readiness report is incomplete or contains an error")

    _ensure_finite_json(report)
    started = _parse_timestamp(report.get("started_at_utc"))
    completed = _parse_timestamp(report.get("completed_at_utc"))
    now = datetime.now(timezone.utc)
    if completed < started or started.timestamp() > now.timestamp() + 5 or completed.timestamp() > now.timestamp() + 5:
        raise ValueError("readiness report timestamps are malformed")
    if now.timestamp() - completed.timestamp() > MAX_REPORT_AGE_SECONDS:
        raise ValueError("readiness report is stale")

    if report.get("platform") != PLATFORM:
        raise ValueError("readiness report platform is missing or contradictory")
    if report.get("scope") != "readiness_only" or report.get("grasp_success_claimed") is not False:
        raise ValueError("readiness report makes a contradictory success claim")
    if report.get("collision_path_tested") is not False:
        raise ValueError("readiness report must not claim collision-path validation")

    scene = _mapping(report.get("scene"), "scene")
    _expect_true(scene, "ground_present", "scene")
    _expect_true(scene, "gravity_enabled", "scene")
    _expect_true(scene, "contact_sensors_enabled", "scene")
    _expect_true(scene, "self_collision_enabled", "scene")
    if _integer(scene.get("steps"), "scene.steps") < 240:
        raise ValueError("readiness evidence is incomplete: insufficient simulation steps")
    if _number(scene.get("physics_dt_s"), "scene.physics_dt_s") <= 0:
        raise ValueError("readiness evidence is malformed: physics_dt_s")

    camera = _mapping(report.get("camera"), "camera")
    _validate_camera(camera)

    geometry = _mapping(scene.get("known_geometry"), "scene.known_geometry")
    if geometry.get("kind") != "rigid_cube":
        raise ValueError("readiness evidence is incomplete: known rigid geometry")
    dimensions = _vector(geometry.get("dimensions_m"), 3, "known_geometry.dimensions_m")
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("readiness evidence is malformed: cube dimensions")
    _vector(geometry.get("center_world_m"), 3, "known_geometry.center_world_m")
    cube_orientation = _vector(geometry.get("orientation_world_wxyz"), 4, "known_geometry.orientation_world_wxyz")
    _unit_quaternion(cube_orientation, "known_geometry.orientation_world_wxyz")
    front_axis = _integer(geometry.get("front_face_axis_index"), "known_geometry.front_face_axis_index")
    front_sign = _integer(geometry.get("front_face_sign"), "known_geometry.front_face_sign")
    if front_axis not in (0, 1, 2) or front_sign not in (-1, 1):
        raise ValueError("readiness evidence is malformed: known geometry front face")
    expected_from_geometry = _ray_box_depth(camera, geometry)
    expected_depth = _number(geometry.get("expected_depth_m"), "known_geometry.expected_depth_m")
    measured_depth = _number(geometry.get("measured_depth_m"), "known_geometry.measured_depth_m")
    depth_error = _number(geometry.get("absolute_error_m"), "known_geometry.absolute_error_m")
    if abs(expected_depth - expected_from_geometry) > 1e-4:
        raise ValueError("readiness evidence contradicts camera/cube geometry depth")
    if expected_depth <= 0 or measured_depth <= 0 or abs(abs(expected_depth - measured_depth) - depth_error) > 1e-4:
        raise ValueError("readiness evidence is malformed: known geometry depth comparison")
    if depth_error > 0.25:
        raise ValueError("readiness evidence contradicts known metric geometry")

    robots = _mapping(report.get("robots"), "robots")
    for side in ("left", "right"):
        _validate_robot(_mapping(robots.get(side), f"robots.{side}"), side)

    return report


def _validate_robot(robot, side):
    prefix = f"robots.{side}"
    if not isinstance(robot.get("asset_uri"), str) or not robot["asset_uri"].strip():
        raise ValueError(f"readiness evidence is incomplete: {prefix}.asset_uri")
    _expect_true(robot, "fixed_base_measured", prefix)
    _expect_true(robot, "self_collision_enabled", prefix)
    _expect_true(robot, "gravity_enabled", prefix)
    _expect_true(robot, "contact_sensors_enabled", prefix)

    names = robot.get("joint_names")
    if not isinstance(names, list) or len(names) != 9 or len(set(names)) != 9:
        raise ValueError(f"readiness evidence is malformed: {prefix}.joint_names")
    if names[:7] != ARM_JOINT_NAMES or names[7:] != FINGER_JOINT_NAMES:
        raise ValueError(f"readiness evidence contradicts Panda joint order: {prefix}")
    limits = robot.get("joint_limits")
    if not isinstance(limits, list) or len(limits) != 9:
        raise ValueError(f"readiness evidence is malformed: {prefix}.joint_limits")
    limit_pairs = [_vector(pair, 2, f"{prefix}.joint_limits[{index}]") for index, pair in enumerate(limits)]
    if any(low >= high for low, high in limit_pairs):
        raise ValueError(f"readiness evidence is malformed: {prefix}.joint_limits")
    initial_all = _vector(robot.get("initial_joint_positions"), 9, f"{prefix}.initial_joint_positions")
    final_all = _vector(robot.get("final_joint_positions"), 9, f"{prefix}.final_joint_positions")
    for index, (low, high) in enumerate(limit_pairs):
        if not low - 1e-4 <= initial_all[index] <= high + 1e-4 or not low - 1e-4 <= final_all[index] <= high + 1e-4:
            raise ValueError(f"readiness evidence contradicts joint limits: {prefix}.{names[index]}")

    initial_pose = _vector(robot.get("base_pose_initial"), 7, f"{prefix}.base_pose_initial")
    final_pose = _vector(robot.get("base_pose_final"), 7, f"{prefix}.base_pose_final")
    _unit_quaternion(initial_pose[3:], f"{prefix}.base_pose_initial quaternion")
    _unit_quaternion(final_pose[3:], f"{prefix}.base_pose_final quaternion")
    drift = _number(robot.get("base_translation_drift_m"), f"{prefix}.base_translation_drift_m")
    expected_drift = math.dist(initial_pose[:3], final_pose[:3])
    if abs(drift - expected_drift) > 1e-4 or drift > 0.01:
        raise ValueError(f"readiness evidence contradicts fixed-base drift: {prefix}")
    orientation_drift = _number(robot.get("base_orientation_drift_rad"), f"{prefix}.base_orientation_drift_rad")
    expected_orientation_drift = _quaternion_distance(initial_pose[3:], final_pose[3:])
    if abs(orientation_drift - expected_orientation_drift) > 1e-4 or orientation_drift > 0.01:
        raise ValueError(f"readiness evidence contradicts fixed-base orientation drift: {prefix}")

    arm_initial = _vector(robot.get("arm_initial_positions"), 7, f"{prefix}.arm_initial_positions")
    arm_target = _vector(robot.get("arm_target_positions"), 7, f"{prefix}.arm_target_positions")
    arm_measured = _vector(robot.get("arm_measured_positions"), 7, f"{prefix}.arm_measured_positions")
    arm_error = _vector(robot.get("arm_tracking_error_rad"), 7, f"{prefix}.arm_tracking_error_rad")
    if any(abs((measured - target) - error) > 1e-4 for measured, target, error in zip(arm_measured, arm_target, arm_error)):
        raise ValueError(f"readiness evidence is malformed: {prefix}.arm_tracking_error_rad")
    if max(abs(error) for error in arm_error) > 0.08:
        raise ValueError(f"readiness evidence contradicts joint-target tracking: {prefix}")
    if not any(abs(target - initial) >= 0.02 and abs(measured - initial) >= 0.01 for initial, target, measured in zip(arm_initial, arm_target, arm_measured)):
        raise ValueError(f"readiness evidence lacks measured arm motion: {prefix}")

    opened = _vector(robot.get("finger_open_positions"), 2, f"{prefix}.finger_open_positions")
    closed = _vector(robot.get("finger_closed_positions"), 2, f"{prefix}.finger_closed_positions")
    reopened = _vector(robot.get("finger_reopened_positions"), 2, f"{prefix}.finger_reopened_positions")
    for label, positions in (("open", opened), ("closed", closed), ("reopened", reopened)):
        for finger_index, position in enumerate(positions, start=7):
            low, high = limit_pairs[finger_index]
            if not low - 1e-4 <= position <= high + 1e-4:
                raise ValueError(f"readiness evidence contradicts joint limits: {prefix}.{names[finger_index]} ({label})")
    if any(open_value - close_value < 0.015 or reopen_value - close_value < 0.015 for open_value, close_value, reopen_value in zip(opened, closed, reopened)):
        raise ValueError(f"readiness evidence lacks measured finger close/open motion: {prefix}")


def _validate_camera(camera):
    rgb_shape = _shape(camera.get("rgb_shape"), "camera.rgb_shape")
    depth_shape = _shape(camera.get("depth_shape"), "camera.depth_shape")
    if len(rgb_shape) != 3 or len(depth_shape) != 3 or rgb_shape[:2] != depth_shape[:2] or rgb_shape[2] < 3 or depth_shape[2] != 1:
        raise ValueError("readiness evidence has malformed RGB-D shapes")
    if camera.get("frame") != "world" or camera.get("depth_data_type") != "distance_to_image_plane" or camera.get("depth_units") != "m":
        raise ValueError("readiness evidence has malformed camera frame/depth metadata")
    if camera.get("optical_frame") != "ros" or camera.get("optical_axis") != "+Z" or camera.get("quaternion_order") != "wxyz":
        raise ValueError("readiness evidence has malformed optical camera convention")
    pose = _vector(camera.get("pose_world"), 7, "camera.pose_world")
    _unit_quaternion(pose[3:], "camera.pose_world quaternion")
    matrix = camera.get("intrinsics_matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("readiness evidence has malformed camera intrinsics")
    rows = [_vector(row, 3, f"camera.intrinsics_matrix[{index}]") for index, row in enumerate(matrix)]
    if rows[0][0] <= 0 or rows[1][1] <= 0 or abs(rows[2][2] - 1) > 1e-4:
        raise ValueError("readiness evidence has malformed camera intrinsics")
    if abs(rows[0][1]) > 1e-8 or abs(rows[1][0]) > 1e-8 or abs(rows[2][0]) > 1e-8 or abs(rows[2][1]) > 1e-8:
        raise ValueError("readiness evidence has unsupported camera intrinsics")
    for key in ("rgb_finite_fraction", "depth_finite_fraction"):
        coverage = _number(camera.get(key), f"camera.{key}")
        if not 0.0 <= coverage <= 1.0 or coverage < 0.9:
            raise ValueError(f"readiness evidence has insufficient finite camera coverage: {key}")


def _ray_box_depth(camera, geometry):
    pixel = geometry.get("sample_pixel_rc")
    if not isinstance(pixel, list) or len(pixel) != 2 or any(type(value) is not int for value in pixel):
        raise ValueError("readiness evidence is malformed: known geometry sample_pixel_rc")
    height, width = _shape(camera.get("depth_shape"), "camera.depth_shape")[:2]
    row, column = pixel
    if not 0 <= row < height or not 0 <= column < width:
        raise ValueError("readiness evidence is malformed: known geometry sample pixel is outside the image")

    intrinsics = camera["intrinsics_matrix"]
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    ray_camera = [(column - cx) / fx, (row - cy) / fy, 1.0]
    pose = camera["pose_world"]
    camera_rotation = pose[3:]
    cube_rotation = geometry["orientation_world_wxyz"]
    cube_inverse = [cube_rotation[0], -cube_rotation[1], -cube_rotation[2], -cube_rotation[3]]
    origin_local = _rotate_vector(cube_inverse, [pose[index] - geometry["center_world_m"][index] for index in range(3)])
    direction_local = _rotate_vector(cube_inverse, _rotate_vector(camera_rotation, ray_camera))
    half_extents = [dimension / 2.0 for dimension in geometry["dimensions_m"]]
    entry, exit_ = -math.inf, math.inf
    entry_faces = []
    for axis, (origin, direction, half_extent) in enumerate(zip(origin_local, direction_local, half_extents)):
        if abs(direction) < 1e-12:
            if abs(origin) > half_extent:
                raise ValueError("readiness sample ray misses the known cube")
            continue
        first = (-half_extent - origin) / direction
        second = (half_extent - origin) / direction
        near, far = min(first, second), max(first, second)
        near_sign = -1 if first < second else 1
        if near > entry + 1e-9:
            entry = near
            entry_faces = [(axis, near_sign)]
        elif abs(near - entry) <= 1e-9:
            entry_faces.append((axis, near_sign))
        exit_ = min(exit_, far)
    if entry <= 0 or exit_ < entry or (geometry["front_face_axis_index"], geometry["front_face_sign"]) not in entry_faces:
        raise ValueError("readiness sample pixel does not hit the declared cube front face")
    return entry


def _unit_quaternion(quaternion, path):
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"readiness evidence has malformed quaternion: {path}")


def _quaternion_distance(first, second):
    first_norm = math.sqrt(sum(component * component for component in first))
    second_norm = math.sqrt(sum(component * component for component in second))
    dot = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    return 2.0 * math.acos(min(1.0, abs(dot)))


def _rotate_vector(quaternion, vector):
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (component / norm for component in (w, x, y, z))
    qv = (x, y, z)
    cross1 = (qv[1] * vector[2] - qv[2] * vector[1], qv[2] * vector[0] - qv[0] * vector[2], qv[0] * vector[1] - qv[1] * vector[0])
    cross2 = (qv[1] * cross1[2] - qv[2] * cross1[1], qv[2] * cross1[0] - qv[0] * cross1[2], qv[0] * cross1[1] - qv[1] * cross1[0])
    return [component + 2.0 * (w * first + second) for component, first, second in zip(vector, cross1, cross2)]


def _shape(value, path):
    if not isinstance(value, list) or any(type(dimension) is not int or dimension <= 0 for dimension in value):
        raise ValueError(f"readiness evidence is malformed: {path}")
    return value


def _mapping(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"readiness evidence is incomplete: {path}")
    return value


def _expect_true(mapping, key, path):
    if mapping.get(key) is not True:
        raise ValueError(f"readiness evidence is incomplete or contradictory: {path}.{key}")


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"readiness evidence is malformed: {path}")
    return float(value)


def _integer(value, path):
    if type(value) is not int:
        raise ValueError(f"readiness evidence is malformed: {path}")
    return value


def _vector(value, length, path):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"readiness evidence is malformed: {path}")
    return [_number(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _ensure_finite_json(value, path="report"):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"readiness report is malformed: {path} key")
            _ensure_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"readiness report contains non-finite evidence: {path}")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"readiness report is malformed: {path}")


def _parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("readiness report timestamps are malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("readiness report timestamps are malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("readiness report timestamps are malformed")
    return parsed.astimezone(timezone.utc)


def run_preflight(*, report_path, ledger_path, timeout_seconds=DEFAULT_RUN_TIMEOUT_SECONDS):
    """Launch the real Isaac worker under the persistent 14,400-second budget."""
    report_file = Path(report_path).expanduser().resolve()
    run_id = uuid.uuid4().hex
    source_paths = [ISAACLAB_SOURCE / "isaaclab", ISAACLAB_SOURCE / "isaaclab_assets"]
    missing_sources = [str(path) for path in source_paths if not path.is_dir()]
    if missing_sources:
        raise RuntimeError(f"IsaacLab source paths are unavailable: {missing_sources}")
    pythonpath = os.pathsep.join([str(path) for path in source_paths] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    command = [
        "/usr/bin/env",
        f"PYTHONPATH={pythonpath}",
        ISAAC_PYTHON,
        str(WORKER_PATH),
        "--report",
        str(report_file),
        "--run-id",
        run_id,
        "--headless",
        "--enable_cameras",
    ]
    result = run_bounded(
        command,
        ledger_path=ledger_path,
        timeout_seconds=timeout_seconds,
        budget_seconds=GPU_BUDGET_SECONDS,
    )
    if result.timed_out:
        raise RuntimeError(f"Isaac readiness worker exceeded {timeout_seconds} seconds")
    if result.returncode != 0:
        raise RuntimeError(f"Isaac readiness worker failed with exit code {result.returncode}: {result.stderr[-2000:]}")
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Isaac worker did not produce a readable readiness report") from error
    return validate_readiness(report, run_id=run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run bounded dual-Panda Isaac readiness audit.")
    parser.add_argument("--report", required=True, help="Atomic JSON readiness report destination")
    parser.add_argument("--ledger", required=True, help="Persistent GPU-budget ledger path")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_RUN_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(report_path=args.report, ledger_path=args.ledger, timeout_seconds=args.timeout_seconds)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"readiness preflight failed: {error}", file=sys.stderr)
        return 1
    print(f"readiness complete: run_id={report['run_id']} report={Path(args.report).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())