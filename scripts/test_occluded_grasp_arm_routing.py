"""Arm routing must be stable and must depend on the selected object's pose."""

import pytest

from occluded_grasp_arm_routing import choose_arm, choose_arm_by_fused_scores


BASES = {"left": (-0.68, -0.72), "right": (0.68, -0.72)}


@pytest.mark.parametrize(
    ("position", "expected"),
    [((-0.3, 0.1, 1.0), "left"), ((0.32, 0.1, 1.0), "right"), ((0.0, 0.1, 1.0), "left")],
)
def test_choose_arm_by_distance(position, expected):
    assert choose_arm(position, BASES) == expected


def test_single_arm_remains_supported():
    assert choose_arm((0.3, 0.0, 1.0), {"left": BASES["left"]}) == "left"


def test_missing_arms_rejected():
    with pytest.raises(ValueError):
        choose_arm((0.0, 0.0), {})


def _fusion_model():
    return {
        "model_type": "logistic_two_signal_fusion_v1",
        "model_id": "routing-unit-test",
        "intercept": 0.0,
        "coefficients": {
            "grasp_quality_score": 0.2,
            "motion_feasibility_score": 0.4,
        },
        "calibration": {"output_kind": "uncalibrated_fusion_utility"},
    }


def test_choose_arm_by_fused_scores_hard_filters_motion_infeasible_arm():
    scores = {
        "left": {
            "grasp_quality_score": 0.99,
            "motion_feasibility_score": 0.99,
            "motion_feasible": False,
        },
        "right": {
            "grasp_quality_score": 0.55,
            "motion_feasibility_score": 0.75,
            "motion_feasible": True,
        },
    }
    selected, report = choose_arm_by_fused_scores(scores, _fusion_model())
    assert selected == "right"
    assert list(report["by_arm"]) == ["right"]
    assert report["is_calibrated_operation_success_probability"] is False


def test_choose_arm_by_fused_scores_fails_when_no_motion_feasible_arm():
    scores = {"left": {
        "grasp_quality_score": 0.8,
        "motion_feasibility_score": 0.8,
        "motion_feasible": False,
    }}
    with pytest.raises(ValueError, match="no arm has a feasible motion path"):
        choose_arm_by_fused_scores(scores, _fusion_model())