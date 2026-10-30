"""Arm assignment helpers for a two-Franka occluded-grasp proxy scene."""

from collections.abc import Mapping

from affordance_score_fusion import (
    CALIBRATED_OUTPUT,
    fuse_score_components,
    validate_fusion_model,
)


def choose_arm(object_position: tuple[float, ...], arm_origins: Mapping[str, tuple[float, ...]]) -> str:
    """Choose the closest base in the horizontal plane; ties go to the left arm."""
    if not arm_origins:
        raise ValueError("at least one arm is required")
    if len(object_position) < 2:
        raise ValueError("object position must have x and y coordinates")
    return min(
        arm_origins,
        key=lambda name: (
            (object_position[0] - arm_origins[name][0]) ** 2
            + (object_position[1] - arm_origins[name][1]) ** 2,
            0 if name == "left" else 1,
        ),
    )


def choose_arm_by_fused_scores(
    arm_scores: Mapping[str, Mapping[str, object]],
    fusion_model: Mapping[str, object],
) -> tuple[str, dict[str, dict[str, object]]]:
    """Select a motion-feasible arm by the fused grasp/task/motion score.

    Motion feasibility is a hard gate. Infeasible arms are excluded even when
    their grasp or task score is high. The returned model metadata states
    whether the output is a calibrated probability or only a ranking utility.
    """
    model = validate_fusion_model(fusion_model)
    evaluations: dict[str, dict[str, object]] = {}
    for arm, components in arm_scores.items():
        if not bool(components.get("motion_feasible", False)):
            continue
        fused = fuse_score_components(components, model)
        evaluations[str(arm)] = fused
    if not evaluations:
        raise ValueError("no arm has a feasible motion path for late-fusion selection")
    selected_arm = max(
        evaluations,
        key=lambda arm: (float(evaluations[arm]["fused_score"]), arm),
    )
    return selected_arm, {
        "model_id": model["model_id"],
        "model_type": model["model_type"],
        "score_kind": model["output_kind"],
        "is_calibrated_operation_success_probability": model["output_kind"] == CALIBRATED_OUTPUT,
        "by_arm": evaluations,
    }