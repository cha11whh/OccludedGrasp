import math

import pytest

from arm_conditioned_affordance import OBJECT_SIZES, score_frontal_affordance, segment_intersects_aabb


JOINTS = [0.0] * 7
LIMITS = [(-2.8, 2.8)] * 7


def _score(x, other_objects=()):
    return score_frontal_affordance(
        object_name="BlueBoxProxy", object_position=(x, 0.0, 1.1),
        arm_base_position=(-0.5, -0.72, 0.70), arm_yaw_rad=math.pi / 2,
        hand_position=(-0.5, -0.2, 1.2), joint_positions=JOINTS,
        joint_limits=LIMITS, other_objects=other_objects,
    )


def test_segment_aabb_detects_front_approach_blocker():
    assert segment_intersects_aabb((0, -0.2, 1), (0, -0.05, 1), (0, -0.12, 1), (0.1, 0.05, 0.2))
    assert not segment_intersects_aabb((0, -0.2, 1), (0, -0.05, 1), (0.3, -0.12, 1), (0.1, 0.05, 0.2))


def test_candidate_is_front_only_and_has_explicit_proxy_semantics():
    result = _score(-0.5)
    assert result["front_approach_world"]["pregrasp"][1] < result["front_approach_world"]["contact"][1]
    assert result["score_kind"].startswith("uncalibrated")
    assert set(OBJECT_SIZES) >= {"BlueBoxProxy", "WaterBottleProxy"}


def test_front_corridor_blocker_makes_candidate_infeasible():
    blocker = {"name": "RedBoxProxy", "position": (-0.5, -0.12, 1.1)}
    result = _score(-0.5, [blocker])
    assert result["blocking_objects"] == ["RedBoxProxy"]
    assert result["front_corridor_clear_proxy"] is False
    assert result["score"] == 0.0


def test_lower_supporting_object_is_not_selected_for_sideways_removal():
    top_item = {"name": "RedBoxProxy", "position": (0.0, 0.0, 1.25)}
    result = score_frontal_affordance(
        object_name="BlueBoxProxy", object_position=(0.0, 0.0, 1.0),
        arm_base_position=(0.0, -0.7, 0.7), arm_yaw_rad=math.pi / 2,
        hand_position=(0.0, -0.2, 1.0), joint_positions=JOINTS,
        joint_limits=LIMITS, other_objects=[top_item],
    )
    assert result["supported_objects_above"] == ["RedBoxProxy"]
    assert result["stable_to_remove_proxy"] is False
    assert result["feasible_proxy"] is False


def test_workspace_outside_is_not_mislabeled_feasible():
    result = _score(0.9)
    assert result["workspace_feasible_proxy"] is False
    assert result["score"] == 0.0


def test_invalid_joint_limits_are_rejected():
    with pytest.raises(ValueError, match="joint upper limits"):
        score_frontal_affordance(
            object_name="BlueBoxProxy", object_position=(0, 0, 1),
            arm_base_position=(0, -0.72, 0.7), arm_yaw_rad=math.pi / 2,
            hand_position=(0, -0.2, 1), joint_positions=[0], joint_limits=[(1, 0)],
            other_objects=(),
        )