"""No-training adapter for RGB-D grasp quality + motion-feasibility ranking."""
from __future__ import annotations

import math
from collections import deque
from typing import Any

from occlusion_affordance_planner import select_joint_action

SCORE_KIND = "uncalibrated_fixed_rgbd_grasp_motion_utility_v1"


def select_joint_action_with_rgbd_quality(
    ranked_actions: dict[str, Any],
    relations: list[dict[str, Any]],
    arm_scores: dict[str, dict[str, dict[str, Any]]],
    grasp_quality_scores: dict[str, float],
    *,
    grasp_quality_weight: float = 0.5,
    affordance_weight: float = 0.60,
    path_weight: float = 0.25,
    ranker_weight: float = 0.15,
    min_relation_confidence: float = 0.0,
) -> dict[str, Any]:
    """Blend normalized geometry and motion proxies; feasible_proxy stays a hard gate."""
    if not math.isfinite(grasp_quality_weight) or not 0.0 <= grasp_quality_weight <= 1.0:
        raise ValueError("grasp_quality_weight must be finite and in [0, 1]")
    target = ranked_actions.get("target")
    actions = ranked_actions.get("ranked_actions", [])
    if not isinstance(target, dict) or not isinstance(actions, list):
        raise ValueError("ranked actions need target and action list")
    target_name = str(target.get("name", ""))
    if not target_name:
        raise ValueError("target must have a name")
    route_names = {target_name}
    for item in actions:
        if item.get("role") in {"direct_target_blocker", "target_path_blocker"}:
            route_names.add(str(item["object_name"]))
    fused_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for object_name, per_arm in arm_scores.items():
        if object_name not in route_names:
            fused_scores[object_name] = dict(per_arm)
            continue
        if object_name not in grasp_quality_scores:
            raise ValueError(f"missing RGB-D grasp quality score for {object_name!r}")
        quality = float(grasp_quality_scores[object_name])
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError(f"grasp quality score for {object_name!r} must be finite and in [0, 1]")
        fused_scores[object_name] = {}
        for arm_name, score in per_arm.items():
            motion = float(score.get("score", 0.0))
            if not math.isfinite(motion) or not 0.0 <= motion <= 1.0:
                raise ValueError(f"motion feasibility proxy for {object_name!r}/{arm_name} must be finite and in [0, 1]")
            utility = grasp_quality_weight * quality + (1.0 - grasp_quality_weight) * motion
            candidate_available = quality > 0.0
            fused_scores[object_name][arm_name] = {
                **score,
                "feasible_proxy": bool(score.get("feasible_proxy")) and candidate_available,
                "motion_feasibility_score": motion,
                "grasp_quality_score": quality,
                "component_fusion_score": utility,
                "score": utility,
                "score_kind": SCORE_KIND,
            }
    plan = select_joint_action(
        ranked_actions, relations, fused_scores,
        min_relation_confidence=min_relation_confidence,
        affordance_weight=affordance_weight, path_weight=path_weight, ranker_weight=ranker_weight,
    )
    plan["score_definition"] = {
        "utility": "fixed weighted uncalibrated RGB-D grasp geometry + geometric/kinematic motion proxy; motion feasibility remains a hard gate",
        "output_kind": SCORE_KIND,
        "is_calibrated_operation_success_probability": False,
        "component_weights": {"grasp_quality": grasp_quality_weight, "motion_feasibility_proxy": 1.0 - grasp_quality_weight},
        "oaprn_edge_direction": "blocker -> blocked",
        "candidate_constraint": "directly feasible target first; otherwise accessible roots in the target's OAPRN ancestor graph",
        "target_direct_frontal_feasible": plan.get("score_definition", {}).get("target_direct_frontal_feasible"),
    }
    for candidate in plan.get("candidate_ranking", []):
        score = fused_scores[str(candidate["object_name"])][str(candidate["arm"])]
        candidate["grasp_quality_score"] = score["grasp_quality_score"]
        candidate["motion_feasibility_score"] = score["motion_feasibility_score"]
        candidate["component_fusion_score"] = score["component_fusion_score"]
    return plan
