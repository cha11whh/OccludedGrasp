from datetime import datetime, timedelta, timezone
import copy

import pytest

from obstruction_train.panda_readiness import validate_readiness


def _complete_report():
    started = datetime.now(timezone.utc) - timedelta(seconds=2)
    arms = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    robot = {
        "asset_uri": "omniverse://assets/Robots/FrankaEmika/panda_instanceable.usd",
        "fixed_base_measured": True,
        "self_collision_enabled": True,
        "gravity_enabled": True,
        "contact_sensors_enabled": True,
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)]
        + ["panda_finger_joint1", "panda_finger_joint2"],
        "joint_limits": [[-3.1, 3.1]] * 7 + [[0.0, 0.04]] * 2,
        "initial_joint_positions": arms + [0.04, 0.04],
        "final_joint_positions": [0.028] + arms[1:] + [0.04, 0.04],
        "base_pose_initial": [-1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "base_pose_final": [-1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "base_translation_drift_m": 0.0,
        "base_orientation_drift_rad": 0.0,
        "arm_initial_positions": arms,
        "arm_target_positions": [0.03] + arms[1:],
        "arm_measured_positions": [0.028] + arms[1:],
        "arm_tracking_error_rad": [-0.002] + [0.0] * 6,
        "finger_open_positions": [0.04, 0.04],
        "finger_closed_positions": [0.006, 0.006],
        "finger_reopened_positions": [0.039, 0.039],
    }
    return {
        "schema_version": 2,
        "run_id": "run-expected",
        "status": "complete",
        "error": None,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": "isaaclab_dual_franka_panda_fixed_base",
        "scope": "readiness_only",
        "grasp_success_claimed": False,
        "collision_path_tested": False,
        "scene": {
            "ground_present": True,
            "gravity_enabled": True,
            "contact_sensors_enabled": True,
            "self_collision_enabled": True,
            "steps": 240,
            "physics_dt_s": 1 / 60,
            "known_geometry": {
                "kind": "rigid_cube",
                "dimensions_m": [0.15, 0.15, 0.15],
                "center_world_m": [0.0, 0.0, 0.075],
                "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                "front_face_axis_index": 1,
                "front_face_sign": -1,
                "sample_pixel_rc": [60, 80],
                "expected_depth_m": 2.1862345367,
                "measured_depth_m": 2.1962345367,
                "absolute_error_m": 0.01,
            },
        },
        "robots": {"left": robot, "right": copy.deepcopy(robot)},
        "camera": {
            "rgb_shape": [120, 160, 4],
            "depth_shape": [120, 160, 1],
            "frame": "world",
            "depth_data_type": "distance_to_image_plane",
            "depth_units": "m",
            "optical_frame": "ros",
            "optical_axis": "+Z",
            "quaternion_order": "wxyz",
            "pose_world": [0.0, -2.075, 1.0, 0.5459517853, -0.8378165957, 0.0, 0.0],
            "intrinsics_matrix": [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
            "rgb_finite_fraction": 1.0,
            "depth_finite_fraction": 1.0,
        },
    }


def test_validator_rejects_an_incomplete_report():
    with pytest.raises(ValueError, match="incomplete"):
        validate_readiness({}, run_id="run-expected")


def test_validator_rejects_worker_error_reports():
    report = {"run_id": "run-expected", "status": "error", "error": "asset load failed"}
    with pytest.raises(ValueError, match="error"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_evidence_for_a_different_run():
    report = {"run_id": "run-other", "status": "complete"}
    with pytest.raises(ValueError, match="run_id"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_stale_completion_evidence():
    completed = datetime.now(timezone.utc) - timedelta(hours=1)
    report = {
        "schema_version": 2,
        "run_id": "run-expected",
        "status": "complete",
        "error": None,
        "started_at_utc": (completed - timedelta(seconds=2)).isoformat(),
        "completed_at_utc": completed.isoformat(),
    }
    with pytest.raises(ValueError, match="stale"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_complete_report_without_physical_evidence():
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 2,
        "run_id": "run-expected",
        "status": "complete",
        "error": None,
        "started_at_utc": now,
        "completed_at_utc": now,
    }
    with pytest.raises(ValueError, match="incomplete|missing"):
        validate_readiness(report, run_id="run-expected")


def test_validator_accepts_fresh_dual_panda_physical_evidence():
    report = _complete_report()
    assert validate_readiness(report, run_id="run-expected") is report


def test_validator_rejects_depth_claim_when_camera_geometry_changes():
    report = _complete_report()
    report["camera"]["pose_world"][1] = -4.0
    with pytest.raises(ValueError, match="depth|geometry|camera|sample"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_forged_depth_scalars_for_raw_cube_geometry():
    report = _complete_report()
    report["scene"]["known_geometry"]["center_world_m"][1] = 0.01
    with pytest.raises(ValueError, match="depth|cube|camera"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_claimed_base_drift_that_disagrees_with_poses():
    report = _complete_report()
    report["robots"]["left"]["base_pose_final"][0] += 0.02
    with pytest.raises(ValueError, match="drift"):
        validate_readiness(report, run_id="run-expected")


def test_validator_requires_measured_loaded_fixed_base_state():
    report = _complete_report()
    del report["robots"]["left"]["fixed_base_measured"]
    with pytest.raises(ValueError, match="fixed_base_measured"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_arm_command_without_measured_motion():
    report = _complete_report()
    robot = report["robots"]["left"]
    robot["arm_measured_positions"][0] = robot["arm_initial_positions"][0]
    robot["arm_tracking_error_rad"][0] = -0.03
    robot["final_joint_positions"][0] = 0.0
    with pytest.raises(ValueError, match="measured arm motion"):
        validate_readiness(report, run_id="run-expected")


def test_validator_rejects_finger_measurements_outside_joint_limits():
    report = _complete_report()
    report["robots"]["left"]["finger_closed_positions"] = [-0.01, -0.01]
    with pytest.raises(ValueError, match="joint limits"):
        validate_readiness(report, run_id="run-expected")


def test_isaac_worker_atomically_writes_a_readable_report(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    worker_path = Path(__file__).parents[1] / "scripts" / "isaac_dual_panda_readiness.py"
    assert worker_path.is_file(), "the Isaac dual-Panda readiness worker is missing"
    spec = importlib.util.spec_from_file_location("isaac_dual_panda_readiness", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    report_path = tmp_path / "readiness.json"
    report = {"run_id": "unit-run", "status": "error", "error": "simulated failure"}
    worker.write_report_atomically(report_path, report)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert list(tmp_path.iterdir()) == [report_path]