import numpy as np
import pytest

from openpi.policies.pi05_droid_action import integrate_joint_velocity_action


def test_integrates_15_hz_joint_velocity_and_opens_gripper():
    position = np.zeros(7, dtype=np.float32)
    action = np.array([0.15, -0.3, 0, 0, 0, 0, 0, 0.9], dtype=np.float32)
    limits = np.tile(np.array([-1.0, 1.0], dtype=np.float32), (7, 1))
    target, gripper = integrate_joint_velocity_action(position, action, limits)
    np.testing.assert_allclose(target[:2], [0.01, -0.02], atol=1e-7)
    assert gripper == 0.04


def test_clips_velocity_joint_limits_and_closes_gripper():
    position = np.array([0.99, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action = np.array([5.0, -5.0, 0, 0, 0, 0, 0, 0.1], dtype=np.float32)
    limits = np.tile(np.array([-1.0, 1.0], dtype=np.float32), (7, 1))
    target, gripper = integrate_joint_velocity_action(
        position, action, limits, max_joint_velocity=0.3
    )
    np.testing.assert_allclose(target[:2], [1.0, -0.02], atol=1e-7)
    assert gripper == 0.0


def test_supports_inverted_gripper_convention():
    target, gripper = integrate_joint_velocity_action(
        np.zeros(7), np.zeros(8), np.tile([-1.0, 1.0], (7, 1)), gripper_one_is_open=False
    )
    assert target.shape == (7,)
    assert gripper == 0.04


def test_rejects_invalid_action():
    with pytest.raises(ValueError, match="action must be finite"):
        integrate_joint_velocity_action(
            np.zeros(7), np.full(8, np.nan), np.tile([-1.0, 1.0], (7, 1))
        )
