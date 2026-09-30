"""Utilities for executing unnormalized Pi05-DROID actions safely."""

import numpy as np

DROID_CONTROL_FREQUENCY_HZ = 15.0


def integrate_joint_velocity_action(
    joint_position: np.ndarray,
    action: np.ndarray,
    joint_limits: np.ndarray,
    *,
    max_joint_velocity: float = 1.0,
    gripper_one_is_open: bool = True,
) -> tuple[np.ndarray, float]:
    """Convert one DROID action into bounded Panda position targets."""
    joint_position = np.asarray(joint_position, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    joint_limits = np.asarray(joint_limits, dtype=np.float32)
    if joint_position.shape != (7,):
        raise ValueError(f"joint_position must have shape (7,), got {joint_position.shape}")
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError(f"action must be finite with shape (8,), got {action.shape}")
    if joint_limits.shape != (7, 2):
        raise ValueError(f"joint_limits must have shape (7, 2), got {joint_limits.shape}")
    if not np.isfinite(max_joint_velocity) or max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity must be positive and finite")
    velocity = np.clip(action[:7], -max_joint_velocity, max_joint_velocity)
    target = joint_position + velocity / DROID_CONTROL_FREQUENCY_HZ
    target = np.clip(target, joint_limits[:, 0], joint_limits[:, 1])
    open_command = bool(action[7] > 0.5)
    if not gripper_one_is_open:
        open_command = not open_command
    return target.astype(np.float32), 0.04 if open_command else 0.0
