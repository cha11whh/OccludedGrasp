import json

import numpy as np
import pytest
from PIL import Image

from export_occluded_grasp_expert_episode import validate_episode


def _manifest(tmp_path):
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(tmp_path / "external.png")
    Image.fromarray(np.ones((8, 10, 3), dtype=np.uint8)).save(tmp_path / "wrist.png")
    return {
        "success": True,
        "task": "Clear the blocker and grasp the target.",
        "frames": [
            {"exterior_image_1_left": "external.png", "wrist_image_left": "wrist.png",
             "state": [0.0] * 8, "action": [0.0] * 8}
            for _ in range(2)
        ],
    }


def test_validates_successful_droid_episode(tmp_path):
    frames = validate_episode(_manifest(tmp_path), tmp_path)
    assert len(frames) == 2
    assert frames[0]["joint_position"].shape == (7,)
    assert frames[0]["gripper_position"].shape == (1,)
    assert frames[0]["actions"].shape == (8,)
    assert frames[0]["exterior_image_1_left"].shape == (8, 10, 3)


def test_refuses_unsuccessful_episode(tmp_path):
    episode = _manifest(tmp_path)
    episode["success"] = False
    with pytest.raises(ValueError, match="success is exactly true"):
        validate_episode(episode, tmp_path)


def test_rejects_wrong_action_width(tmp_path):
    episode = _manifest(tmp_path)
    episode["frames"][0]["action"] = [0.0] * 7
    with pytest.raises(ValueError, match=r"action must be finite shape \(8,\)"):
        validate_episode(episode, tmp_path)
