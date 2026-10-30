"""Target-path constrained joint object/arm selection."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from affordance_score_fusion import (
    CALIBRATED_OUTPUT,
    fuse_score_components,
    validate_fusion_model,
)


def select_joint_action(
    ranked_actions: dict[str, Any],
    relations: list[dict[str, Any]],
    arm_scores: dict[str, dict[str, dict[str, Any]]],
    *,
    min_relation_confidence: float = 0.0,
    affordance_weight: float = 0.60,
    path_weight: float = 0.25,
    ranker_weight: float = 0.15,
    score_fusion_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick an accessible OAPRN target-path object and its best feasible arm.

    The OAPRN edge convention is ``blocker -> blocked``. Off-target objects are
    never candidates. The score is an uncalibrated planning utility, not a
    grasp-success probability.
    """
    if score_fusion_model is None:
        weights = (affordance_weight, path_weight, ranker_weight)
        if any(weight < 0.0 for weight in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("planner weights must be non-negative and sum to one")
        fusion_definition = None
    else:
        fusion_definition = validate_fusion_model(score_fusion_model)
    actions = ranked_actions.get("ranked_actions")
    target = ranked_actions.get("target")
    if not isinstance(actions, list) or not actions or not isinstance(target, dict):
        raise ValueError("ranked actions and an explicit target are required")
    target_name = str(target.get("name", ""))
    target_id = str(target.get("id", ""))
    if not target_name:
        raise ValueError("target must include its object name")

    action_by_name = {str(action["object_name"]): action for action in actions}
    names_by_id = {str(action.get("object_id")): str(action["object_name"]) for action in actions}
    incoming: dict[str, set[str]] = {}
    outgoing: dict[str, set[str]] = {}
    edge_confidence: dict[tuple[str, str], float] = {}
    for relation in relations:
        confidence = float(relation.get("confidence", relation.get("relation_confidence", 0.0)))
        if not math.isfinite(confidence) or confidence < min_relation_confidence:
            continue
        source = str(relation.get("blocker_name", names_by_id.get(str(relation.get("blocker")), "")))
        destination = str(relation.get("blocked_name", names_by_id.get(str(relation.get("blocked")), "")))
        if not source or not destination or source == destination:
            continue
        outgoing.setdefault(source, set()).add(destination)
        incoming.setdefault(destination, set()).add(source)
        edge_confidence[source, destination] = max(
            confidence, edge_confidence.get((source, destination), 0.0))

    # Reverse traversal finds every object that can contribute to a route to target.
    ancestors = {target_name}
    queue = deque([target_name])
    while queue:
        node = queue.popleft()
        for blocker in incoming.get(node, ()):
            if blocker not in ancestors:
                ancestors.add(blocker)
                queue.append(blocker)

    target_scores = arm_scores.get(target_name, {})
    target_feasible = []
    for arm, score in target_scores.items():
        if not bool(score.get("feasible_proxy")):
            continue
        if fusion_definition is None:
            if float(score.get("score", 0.0)) > 0.0:
                target_feasible.append((arm, score))
        else:
            if "motion_feasible" not in score:
                raise ValueError(f"score fusion requires motion_feasible for {target_name}/{arm}")
            if not bool(score["motion_feasible"]):
                continue
            # Validate component presence/range before allowing direct-target priority.
            fuse_score_components(score, fusion_definition)
            target_feasible.append((arm, score))
    # If at least one arm has a clear frontal grasp corridor for the target,
    # try the target directly even if OAPRN predicts an occlusion edge.
    if target_feasible:
        accessible = {target_name}
    else:
        # Only choose current roots within the target-related subgraph. A blocker
        # that belongs to a disconnected, irrelevant branch must not suppress a root.
        accessible = {
            name for name in ancestors
            if not (incoming.get(name, set()) & ancestors)
        }

    candidates: list[dict[str, Any]] = []
    for object_name in sorted(accessible):
        action = action_by_name.get(object_name)
        per_arm = arm_scores.get(object_name, {})
        if action is None or not per_arm:
            continue
        feasible_arms = []
        for arm_name, score in per_arm.items():
            if not bool(score.get("feasible_proxy")):
                continue
            if fusion_definition is None:
                if float(score.get("score", 0.0)) > 0.0:
                    feasible_arms.append((arm_name, score))
            else:
                if "motion_feasible" not in score:
                    raise ValueError(f"score fusion requires motion_feasible for {object_name}/{arm_name}")
                if bool(score["motion_feasible"]):
                    fused = fuse_score_components(score, fusion_definition)
                    feasible_arms.append((arm_name, {**score, **fused}))
        if not feasible_arms:
            continue
        arm_name, arm_score = max(
            feasible_arms,
            key=lambda pair: (
                float(pair[1]["fused_score"] if fusion_definition is not None else pair[1]["score"]),
                pair[0],
            ),
        )

        # Best path confidence product from this accessible root to target.
        frontier = [(object_name, 1.0, frozenset({object_name}))]
        best_path_confidence = 1.0 if object_name == target_name else 0.0
        while frontier:
            node, product, visited = frontier.pop()
            for child in outgoing.get(node, ()):
                if child not in ancestors or child in visited:
                    continue
                next_product = product * edge_confidence[node, child]
                if child == target_name:
                    best_path_confidence = max(best_path_confidence, next_product)
                else:
                    frontier.append((child, next_product, visited | {child}))

        candidates.append({
            "object_name": object_name,
            "object_id": action.get("object_id"),
            "role": action.get("role"),
            "ranker_reward": float(action.get("reward", action.get("score", 0.0))),
            "arm": arm_name,
            "affordance_score": float(arm_score["score"]),
            "fused_score": float(arm_score["fused_score"]) if fusion_definition is not None else None,
            "fusion_component_scores": arm_score.get("component_scores") if fusion_definition is not None else None,
            "path_confidence_product": best_path_confidence,
        })

    if not candidates:
        return {
            "status": "no_feasible_target_route",
            "target": target_name,
            "target_id": target_id,
            "target_related_objects": sorted(ancestors),
            "accessible_target_related_objects": sorted(accessible),
            "reason": "no accessible object on an OAPRN path to the target has a positive feasible-arm affordance",
            "candidates": [],
        }

    rewards = [item["ranker_reward"] for item in candidates]
    reward_min, reward_max = min(rewards), max(rewards)
    for item in candidates:
        reward_norm = ((item["ranker_reward"] - reward_min) / (reward_max - reward_min)
                       if reward_max > reward_min else 0.5)
        item["normalized_ranker_reward"] = reward_norm
        if fusion_definition is not None:
            # OAPRN target-route membership is the structural constraint; the
            # two-signal stacker ranks feasible object-arm pairs within it.
            item["joint_score"] = float(item["fused_score"])
        else:
            item["joint_score"] = (
                affordance_weight * item["affordance_score"]
                + path_weight * item["path_confidence_product"]
                + ranker_weight * reward_norm
            )
    candidates.sort(key=lambda item: (-item["joint_score"], item["object_name"], item["arm"]))
    chosen = candidates[0]
    selected_action = action_by_name[chosen["object_name"]]
    return {
        "status": "selected",
        "target": target_name,
        "target_id": target_id,
        "selected_object": chosen["object_name"],
        "selected_object_id": chosen["object_id"],
        "selected_arm": chosen["arm"],
        "score_definition": ({
            "utility": "logistic late fusion of grasp_quality_score and motion_feasibility_score over logit-transformed normalized components",
            "output_kind": fusion_definition["output_kind"],
            "fusion_model_id": fusion_definition["model_id"],
            "fusion_model_type": fusion_definition["model_type"],
            "is_calibrated_operation_success_probability": fusion_definition["output_kind"] == CALIBRATED_OUTPUT,
            "oaprn_edge_direction": "blocker -> blocked",
            "candidate_constraint": "directly feasible target first; otherwise accessible roots in the target's OAPRN ancestor graph; arm motion feasibility is a hard gate",
            "target_direct_frontal_feasible": bool(target_feasible),
        } if fusion_definition is not None else {
            "utility": "0.60*arm_affordance + 0.25*target_path_confidence_product + 0.15*normalized_ranker_reward",
            "arm_affordance_is_calibrated_success_probability": False,
            "oaprn_edge_direction": "blocker -> blocked",
            "candidate_constraint": "directly feasible target first; otherwise accessible roots in the target's OAPRN ancestor graph",
            "target_direct_frontal_feasible": bool(target_feasible),
        }),
        "candidate_ranking": candidates,
        "selected_action": selected_action,
    }