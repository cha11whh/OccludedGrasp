import pytest

from occlusion_affordance_planner import select_joint_action


def _action(name, object_id, reward, role):
    return {"object_name": name, "object_id": object_id, "reward": reward, "role": role}


def _arm_scores(left, right):
    return {
        "left": {"score": left, "feasible_proxy": left > 0},
        "right": {"score": right, "feasible_proxy": right > 0},
    }


def test_joint_planner_uses_affordance_among_accessible_target_routes():
    ranked = {"target": {"id": 2, "name": "target"}, "ranked_actions": [
        _action("left_route", 0, 10.0, "direct_target_blocker"),
        _action("right_route", 1, 9.0, "direct_target_blocker"),
        _action("unrelated_easy", 3, 100.0, "scene_object"),
    ]}
    relations = [
        {"blocker_name": "left_route", "blocked_name": "target", "confidence": 0.9},
        {"blocker_name": "right_route", "blocked_name": "target", "confidence": 0.9},
    ]
    scores = {
        "left_route": _arm_scores(0.2, 0.0),
        "right_route": _arm_scores(0.0, 1.0),
        "target": _arm_scores(0.0, 0.0),
        "unrelated_easy": _arm_scores(1.0, 1.0),
    }
    plan = select_joint_action(ranked, relations, scores)
    assert plan["selected_object"] == "right_route"
    assert plan["selected_arm"] == "right"
    assert all(item["object_name"] != "unrelated_easy" for item in plan["candidate_ranking"])


def test_planner_filters_unreachable_target_route_and_uses_next_path():
    ranked = {"target": {"id": 2, "name": "target"}, "ranked_actions": [
        _action("cover", 4, 15.0, "target_path_blocker"),
        _action("blocked_root", 0, 12.0, "target_path_blocker"),
        _action("accessible_root", 1, 5.0, "target_path_blocker"),
        _action("middle", 3, 9.0, "direct_target_blocker"),
    ]}
    relations = [
        {"blocker_name": "cover", "blocked_name": "blocked_root", "confidence": 0.95},
        {"blocker_name": "blocked_root", "blocked_name": "middle", "confidence": 0.9},
        {"blocker_name": "middle", "blocked_name": "target", "confidence": 0.9},
        {"blocker_name": "accessible_root", "blocked_name": "target", "confidence": 0.7},
    ]
    scores = {
        "cover": _arm_scores(0.0, 0.0),
        "blocked_root": _arm_scores(0.9, 0.8),
        "accessible_root": _arm_scores(0.2, 0.6),
        "middle": _arm_scores(0.7, 0.1),
        "target": _arm_scores(0.0, 0.0),
    }
    plan = select_joint_action(ranked, relations, scores)
    assert plan["selected_object"] == "accessible_root"
    assert plan["selected_arm"] == "right"


def test_planner_stops_with_no_feasible_arm_and_does_not_use_unrelated_object():
    ranked = {"target": {"id": 2, "name": "target"}, "ranked_actions": [
        _action("blocker", 0, 10.0, "direct_target_blocker"),
        _action("unrelated", 1, 20.0, "scene_object"),
    ]}
    relations = [{"blocker_name": "blocker", "blocked_name": "target", "confidence": 0.8}]
    scores = {
        "blocker": _arm_scores(0.0, 0.0),
        "target": _arm_scores(0.0, 0.0),
        "unrelated": _arm_scores(1.0, 1.0),
    }
    plan = select_joint_action(ranked, relations, scores)
    assert plan["status"] == "no_feasible_target_route"
    assert plan["candidates"] == []


def test_directly_feasible_target_preempts_predicted_blocker_route():
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [
        _action("blocker", 0, 20.0, "direct_target_blocker"),
        _action("target", 1, 10.0, "target"),
    ]}
    relations = [{"blocker_name": "blocker", "blocked_name": "target", "confidence": 0.95}]
    scores = {"blocker": _arm_scores(0.9, 0.8), "target": _arm_scores(0.2, 0.6)}
    plan = select_joint_action(ranked, relations, scores)
    assert plan["selected_object"] == "target"
    assert plan["selected_arm"] == "right"
    assert len(plan["candidate_ranking"]) == 1
    assert plan["score_definition"]["target_direct_frontal_feasible"] is True


def test_invalid_joint_weights_rejected():
    ranked = {"target": {"id": 0, "name": "target"}, "ranked_actions": [_action("target", 0, 1, "target")]}
    assert select_joint_action(ranked, [], {"target": _arm_scores(1, 0)})["selected_arm"] == "left"
    with pytest.raises(ValueError, match="sum to one"):
        select_joint_action(ranked, [], {"target": _arm_scores(1, 0)}, affordance_weight=1.0)


def _late_fusion_model():
    return {
        "model_type": "logistic_two_signal_fusion_v1",
        "model_id": "unit-test-uncalibrated",
        "intercept": 0.0,
        "coefficients": {
            "grasp_quality_score": 0.2,
            "motion_feasibility_score": 0.4,
        },
        "calibration": {"output_kind": "uncalibrated_fusion_utility"},
    }


def _fusion_arm_score(grasp, motion, *, feasible=True, motion_feasible=True):
    return {
        "score": 0.5,
        "feasible_proxy": feasible,
        "motion_feasible": motion_feasible,
        "grasp_quality_score": grasp,
        "motion_feasibility_score": motion,
    }


def test_late_fusion_uses_two_signals_after_occlusion_frontier_filter():
    ranked = {"target": {"id": 2, "name": "target"}, "ranked_actions": [
        _action("route_a", 0, 100.0, "target_path_blocker"),
        _action("route_b", 1, 1.0, "target_path_blocker"),
        _action("unrelated", 3, 1000.0, "scene_object"),
        _action("target", 2, 0.0, "target"),
    ]}
    relations = [
        {"blocker_name": "route_a", "blocked_name": "target", "confidence": 0.9},
        {"blocker_name": "route_b", "blocked_name": "target", "confidence": 0.8},
    ]
    scores = {
        "route_a": {
            "left": _fusion_arm_score(0.95, 0.70),
            "right": _fusion_arm_score(0.20, 0.60, motion_feasible=False),
        },
        "route_b": {
            "left": _fusion_arm_score(0.60, 0.95),
            "right": _fusion_arm_score(0.70, 0.85),
        },
        "target": {
            "left": _fusion_arm_score(0.0, 0.0, feasible=False, motion_feasible=False),
            "right": _fusion_arm_score(0.0, 0.0, feasible=False, motion_feasible=False),
        },
        "unrelated": {
            "left": _fusion_arm_score(1.0, 1.0),
            "right": _fusion_arm_score(1.0, 1.0),
        },
    }

    plan = select_joint_action(ranked, relations, scores, score_fusion_model=_late_fusion_model())

    assert plan["selected_object"] == "route_b"
    assert plan["selected_arm"] in {"left", "right"}
    assert all(row["object_name"] != "unrelated" for row in plan["candidate_ranking"])
    assert plan["score_definition"]["fusion_model_id"] == "unit-test-uncalibrated"
    assert plan["score_definition"]["output_kind"] == "uncalibrated_fusion_utility"


def test_late_fusion_refuses_missing_component_scores_instead_of_falling_back():
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [
        _action("target", 1, 1.0, "target"),
    ]}
    scores = {"target": {"left": {"score": 0.5, "feasible_proxy": True, "motion_feasible": True}}}

    with pytest.raises(ValueError, match="grasp_quality_score"):
        select_joint_action(ranked, [], scores, score_fusion_model=_late_fusion_model())


def test_late_fusion_requires_explicit_motion_feasibility_gate():
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [
        _action("target", 1, 1.0, "target"),
    ]}
    scores = {"target": {"left": {
        "feasible_proxy": True,
        "grasp_quality_score": 0.8,
        "motion_feasibility_score": 0.8,
    }}}

    with pytest.raises(ValueError, match="motion_feasible"):
        select_joint_action(ranked, [], scores, score_fusion_model=_late_fusion_model())


def test_late_fusion_rejects_nonfeasible_motion_arm():
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [
        _action("target", 1, 1.0, "target"),
    ]}
    scores = {"target": {
        "left": _fusion_arm_score(0.99, 0.99, motion_feasible=False),
        "right": _fusion_arm_score(0.60, 0.70),
    }}

    plan = select_joint_action(ranked, [], scores, score_fusion_model=_late_fusion_model())

    assert plan["selected_arm"] == "right"


def test_late_fusion_marks_probability_only_with_calibration_metadata():
    model = _late_fusion_model()
    model["calibration"] = {
        "output_kind": "calibrated_operation_success_probability",
        "method": "platt",
        "validation_sample_count": 240,
        "validation_positive_count": 71,
        "validation_negative_count": 169,
        "scene_disjoint": True,
        "brier_score": 0.12,
        "platt_slope": 0.8,
        "platt_intercept": -0.15,
        "success_definition": "object lifted >= 0.06m after close and contact IK within 0.04m",
    }
    ranked = {"target": {"id": 1, "name": "target"}, "ranked_actions": [
        _action("target", 1, 1.0, "target"),
    ]}
    scores = {"target": {"left": _fusion_arm_score(0.7, 0.9)}}

    plan = select_joint_action(ranked, [], scores, score_fusion_model=model)

    assert plan["score_definition"]["output_kind"] == "calibrated_operation_success_probability"
