import json

import numpy as np
import pytest

from openpi.policies.occluded_grasp_adapter import build_pi05_observation, load_ranked_actions


def _ranked(task_mode="target"):
    return {
        "planner": "test",
        "task_mode": task_mode,
        "target": {"id": 2, "name": "red cup"},
        "instruction": "move the red cup to the tray",
        "ranked_actions": [
            {"object_id": 7, "policy_score": 0.9, "object_name": "box", "role": "direct_target_blocker"},
            {"object_id": 8, "policy_score": 0.4, "object_name": "bowl"},
        ],
    }


def test_target_prompt_and_droid_observation_contract():
    result = build_pi05_observation(_ranked(), np.zeros((4, 5, 3), dtype=np.uint8), arm="left")
    assert "Use the left arm to grasp the box (object 7)" in result["prompt"]
    assert "remove blocker" in result["prompt"]
    assert "direct target blocker" in result["prompt"]
    assert "Clear access to target red cup" in result["prompt"]
    assert result["observation/exterior_image_1_left"].shape == (4, 5, 3)
    assert result["observation/wrist_image_left"].shape == (4, 5, 3)
    assert result["observation/joint_position"].shape == (7,)
    assert result["observation/gripper_position"].shape == (1,)
    assert result["observation/joint_position"].dtype == np.float32


def test_clear_table_prompt_and_state_split():
    result = build_pi05_observation(_ranked("clear_table"), np.zeros((2, 3, 3), dtype=np.uint8), np.arange(8))
    assert result["prompt"].startswith("Use the right arm to clear the table")
    assert result["observation/joint_position"].shape == (7,)
    assert result["observation/gripper_position"].shape == (1,)
    np.testing.assert_array_equal(result["observation/joint_position"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(result["observation/gripper_position"], np.array([7], dtype=np.float32))


def test_rejects_state_with_wrong_width():
    with pytest.raises(ValueError, match="7 joint positions and 1 gripper"):
        build_pi05_observation(_ranked(), np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(32))


def test_invalid_empty_and_bad_image_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        build_pi05_observation({"ranked_actions": []}, np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="HWC"):
        build_pi05_observation(_ranked(), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        build_pi05_observation(_ranked(), np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="matching height and width"):
        build_pi05_observation(_ranked(), np.zeros((2, 2, 3), dtype=np.uint8), wrist_image=np.zeros((3, 2, 3), dtype=np.uint8))


def test_load_ranked_actions_rejects_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="ranked-actions JSON"):
        load_ranked_actions(path)


def test_load_ranked_actions_reads_policy_score_document(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(_ranked()), encoding="utf-8")
    assert load_ranked_actions(path)["ranked_actions"][0]["object_id"] == 7


def test_accepts_actual_reward_ranker_schema(tmp_path):
    path = tmp_path / "ranked_actions.json"
    payload = {
        "target": {"id": 2, "name": "blue notebook"},
        "ranked_actions": [
            {"object_id": 1, "object_name": "green mug", "reward": 1.0, "role": "direct_target_blocker"},
            {"object_id": 2, "object_name": "blue notebook", "reward": 0.5, "role": "target"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = build_pi05_observation(load_ranked_actions(path), np.zeros((4, 5, 3), dtype=np.uint8))
    assert "green mug" in result["prompt"]
    assert "blue notebook" in result["prompt"]
