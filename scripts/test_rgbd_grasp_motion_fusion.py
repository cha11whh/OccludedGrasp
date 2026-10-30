import pytest
from rgbd_grasp_motion_fusion import SCORE_KIND, select_joint_action_with_rgbd_quality


def _action(name, object_id, reward=1.0):
    return {"object_name": name, "object_id": object_id, "reward": reward, "role": "direct_target_blocker"}


def _arm(score, feasible=True):
    return {"score": score, "feasible_proxy": feasible}


def test_geometry_quality_changes_accessible_route_ranking():
    ranked = {"target": {"id": 9, "name": "target"}, "ranked_actions": [_action("easy_motion", 1), _action("good_grasp", 2)]}
    relations = [{"blocker_name": n, "blocked_name": "target", "confidence": 0.8} for n in ("easy_motion", "good_grasp")]
    scores = {"easy_motion": {"left": _arm(0.9), "right": _arm(0.0, False)},
              "good_grasp": {"left": _arm(0.6), "right": _arm(0.0, False)},
              "target": {"left": _arm(0.0, False), "right": _arm(0.0, False)}}
    plan = select_joint_action_with_rgbd_quality(ranked, relations, scores,
        {"easy_motion": 0.1, "good_grasp": 0.9, "target": 0.0})
    assert plan["selected_object"] == "good_grasp"
    assert plan["score_definition"]["output_kind"] == SCORE_KIND
    assert plan["score_definition"]["is_calibrated_operation_success_probability"] is False
    assert plan["candidate_ranking"][0]["component_fusion_score"] == pytest.approx(0.75)


def test_motion_feasibility_is_a_hard_gate():
    ranked = {"target": {"id": 9, "name": "target"}, "ranked_actions": [_action("blocked", 1)]}
    scores = {"blocked": {"left": _arm(0.0, False)}, "target": {"left": _arm(0.0, False)}}
    plan = select_joint_action_with_rgbd_quality(ranked,
        [{"blocker_name": "blocked", "blocked_name": "target", "confidence": 1.0}],
        scores, {"blocked": 1.0, "target": 0.0})
    assert plan["status"] == "no_feasible_target_route"


def test_invalid_inputs_are_rejected():
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [_action("target", 1)]}
    with pytest.raises(ValueError, match="grasp_quality_weight"):
        select_joint_action_with_rgbd_quality(ranked, [], {"target": {"left": _arm(0.5)}}, {"target": 0.5}, grasp_quality_weight=1.1)
    with pytest.raises(ValueError, match="motion feasibility"):
        select_joint_action_with_rgbd_quality(ranked, [], {"target": {"left": _arm(1.2)}}, {"target": 0.5})


def test_zero_geometric_candidates_are_not_motion_only_fallbacks():
    ranked = {"target": {"id": 9, "name": "target"}, "ranked_actions": [_action("blocker", 1)]}
    scores = {"blocker": {"left": _arm(0.9)}, "target": {"left": _arm(0.0, False)}}
    relations = [{"blocker_name": "blocker", "blocked_name": "target", "confidence": 0.9}]
    plan = select_joint_action_with_rgbd_quality(
        ranked, relations, scores, {"blocker": 0.0, "target": 0.0}
    )
    assert plan["status"] == "no_feasible_target_route"
