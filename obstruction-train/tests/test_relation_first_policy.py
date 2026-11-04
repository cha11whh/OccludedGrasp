import obstruction_train.relation_first_policy as policy


def test_prediction_candidates_are_clear_required_roots_before_feasibility():
    observation = {
        "observation_id": "obs-1",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [1, 2, 3],
        "objects": [
            {"id": 1, "visibility": "clear"},
            {"id": 2, "visibility": "clear"},
            {"id": 3, "visibility": "occluded"},
            {"id": 4, "visibility": "clear"},
        ],
        "relations": [
            {"blocker": 1, "blocked": 3, "confidence": 1.0},
            {"blocker": 2, "blocked": 3, "confidence": 1.0},
        ],
    }

    candidate_ids = getattr(policy, "prediction_candidate_ids", lambda *_: None)(
        observation, 3
    )

    assert candidate_ids == [1, 2]
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obstruction_train.relation_first_policy import select_next_object


def _inputs():
    observation = {
        "observation_id": "obs-001",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [1, 2],
        "objects": [
            {"id": 1, "visibility": "clear"},
            {"id": 2, "visibility": "unknown"},
        ],
        "relations": [],
    }
    feasibility = {
        "observation_id": "obs-001",
        "coordinate_frame": "camera",
        "distance_unit": "m",
        "objects": [
            {"object_id": 1, "status": "feasible"},
            {"object_id": 2, "status": "feasible"},
        ],
    }
    return observation, feasibility


def _scored_inputs():
    observation = {
        "observation_id": "obs-001",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [1, 2, 3],
        "objects": [
            {"id": 1, "visibility": "clear"},
            {"id": 2, "visibility": "clear"},
            {"id": 3, "visibility": "occluded"},
        ],
        "relations": [
            {"blocker": 1, "blocked": 3, "confidence": 0.9},
            {"blocker": 2, "blocked": 3, "confidence": 0.9},
        ],
    }
    feasibility = {
        "observation_id": "obs-001",
        "coordinate_frame": "camera",
        "distance_unit": "m",
        "objects": [
            {"object_id": 1, "status": "feasible", "candidate_quality": 0.5, "execution_cost": 0.4},
            {"object_id": 2, "status": "feasible", "candidate_quality": 0.9, "execution_cost": 0.1},
            {"object_id": 3, "status": "feasible", "candidate_quality": 0.8, "execution_cost": 0.1},
        ],
    }
    return observation, feasibility


def test_empty_relation_list_does_not_authorize_selection():
    observation, feasibility = _inputs()

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision == {
        "status": "uncertain",
        "next_object_id": None,
        "observation_id": "obs-001",
        "reason": "target_dependency_uncertain",
        "scoring_mode": "fixed",
        "score_formula": "ascending_object_id",
        "legal_candidate_ids": [],
        "diagnostics": [
            {"object_id": 1, "reason": "not_in_target_dependencies"},
            {"object_id": 2, "reason": "visibility_unknown"},
        ],
    }


def test_confirmed_clear_feasible_target_is_selected():
    observation, feasibility = _inputs()
    observation["objects"] = [{"id": 1, "visibility": "clear"}, {"id": 2, "visibility": "clear"}]
    observation["visibility_coverage"] = [1, 2]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "selected"
    assert decision["next_object_id"] == 2
    assert decision["legal_candidate_ids"] == [2]


def test_chain_selects_only_the_clear_frontmost_required_blocker():
    observation, feasibility = _inputs()
    observation["objects"] = [
        {"id": 1, "visibility": "clear"},
        {"id": 2, "visibility": "occluded"},
        {"id": 3, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [1, 2, 3]
    observation["relations"] = [
        {"blocker": 1, "blocked": 2, "confidence": 0.9},
        {"blocker": 2, "blocked": 3, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible"}
        for object_id in (1, 2, 3)
    ]

    decision = select_next_object(observation, 3, feasibility, {})

    assert decision["status"] == "selected"
    assert decision["next_object_id"] == 1
    assert decision["legal_candidate_ids"] == [1]


def test_partial_relation_coverage_does_not_authorize_an_absent_incoming_edge():
    observation, feasibility = _inputs()
    observation["objects"][1]["visibility"] = "clear"
    observation["relation_coverage"] = "partial"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "uncertain"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []
    assert decision["diagnostics"][1]["reason"] == "relation_coverage_incomplete"


def test_unknown_relation_coverage_does_not_authorize_a_clear_target():
    observation, feasibility = _inputs()
    observation["objects"][1]["visibility"] = "clear"
    observation["relation_coverage"] = "unknown"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "uncertain"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []
    assert decision["diagnostics"][1]["reason"] == "relation_coverage_incomplete"


def test_relevant_dependency_cycle_fails_closed_without_breaking_the_cycle():
    observation, feasibility = _inputs()
    observation["objects"] = [
        {"id": 1, "visibility": "occluded"},
        {"id": 2, "visibility": "occluded"},
        {"id": 3, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [1, 2, 3]
    observation["relations"] = [
        {"blocker": 1, "blocked": 2, "confidence": 0.9},
        {"blocker": 2, "blocked": 1, "confidence": 0.9},
        {"blocker": 2, "blocked": 3, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible"}
        for object_id in (1, 2, 3)
    ]

    decision = select_next_object(observation, 3, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []
    assert decision["reason"] == "dependency_cycle"
    assert decision["diagnostics"][2] == {
        "object_id": 3,
        "reason": "blocked_by_dependency_cycle",
    }


def test_two_required_parents_are_both_in_the_and_closure():
    observation, feasibility = _inputs()
    observation["objects"] = [
        {"id": 1, "visibility": "clear"},
        {"id": 2, "visibility": "clear"},
        {"id": 3, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [1, 2, 3]
    observation["relations"] = [
        {"blocker": 1, "blocked": 3, "confidence": 0.9},
        {"blocker": 2, "blocked": 3, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible"}
        for object_id in (1, 2, 3)
    ]

    decision = select_next_object(observation, 3, feasibility, {})

    assert decision["next_object_id"] == 1
    assert decision["legal_candidate_ids"] == [1, 2]
    assert decision["diagnostics"][2] == {
        "object_id": 3,
        "reason": "blocked_by_required_predecessor",
    }


def test_candidate_tie_break_is_stable_when_input_order_is_reversed():
    observation, feasibility = _inputs()
    observation["objects"] = [
        {"id": 9, "visibility": "clear"},
        {"id": 2, "visibility": "clear"},
        {"id": 10, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [9, 2, 10]
    observation["relations"] = [
        {"blocker": 9, "blocked": 10, "confidence": 0.9},
        {"blocker": 2, "blocked": 10, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible"}
        for object_id in (9, 2, 10)
    ]

    first = select_next_object(observation, 10, feasibility, {})
    observation["objects"].reverse()
    observation["relations"].reverse()
    feasibility["objects"].reverse()
    second = select_next_object(observation, 10, feasibility, {})

    assert first["next_object_id"] == second["next_object_id"] == 2
    assert first["legal_candidate_ids"] == second["legal_candidate_ids"] == [2, 9]


def test_diamond_dependencies_share_one_legal_frontier_predecessor():
    observation, feasibility = _inputs()
    observation["objects"] = [
        {"id": 1, "visibility": "clear"},
        {"id": 2, "visibility": "occluded"},
        {"id": 3, "visibility": "occluded"},
        {"id": 4, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [1, 2, 3, 4]
    observation["relations"] = [
        {"blocker": 1, "blocked": 2, "confidence": 0.9},
        {"blocker": 1, "blocked": 3, "confidence": 0.9},
        {"blocker": 2, "blocked": 4, "confidence": 0.9},
        {"blocker": 3, "blocked": 4, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible"}
        for object_id in (1, 2, 3, 4)
    ]

    decision = select_next_object(observation, 4, feasibility, {})

    assert decision["next_object_id"] == 1
    assert decision["legal_candidate_ids"] == [1]
    assert [item["object_id"] for item in decision["diagnostics"]] == [1, 2, 3, 4]


def test_unrelated_unknown_object_does_not_block_confirmed_target():
    observation, feasibility = _inputs()
    observation["objects"][0]["visibility"] = "unknown"
    observation["objects"][1]["visibility"] = "clear"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "selected"
    assert decision["next_object_id"] == 2
    assert decision["legal_candidate_ids"] == [2]


def test_unknown_feasibility_never_authorizes_selection():
    observation, feasibility = _inputs()
    observation["objects"][1]["visibility"] = "clear"
    feasibility["objects"][1]["status"] = "unknown"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "uncertain"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []


def test_infeasible_target_never_authorizes_selection():
    observation, feasibility = _inputs()
    observation["objects"][1]["visibility"] = "clear"
    feasibility["objects"][1]["status"] = "infeasible"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []


def test_visibility_without_coverage_is_reported_as_unknown():
    observation, feasibility = _inputs()
    observation["visibility_coverage"].remove(1)
    observation["objects"][0]["visibility"] = "unknown"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["diagnostics"][0] == {
        "object_id": 1,
        "reason": "visibility_unknown",
    }


def test_uncovered_clear_claim_is_not_treated_as_clear():
    observation, feasibility = _inputs()
    observation["visibility_coverage"].remove(1)

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "uncertain"
    assert decision["diagnostics"][0] == {
        "object_id": 1,
        "reason": "visibility_unknown",
    }


def test_obstruction_relation_conflicts_with_clear_visibility_closed():
    observation, feasibility = _inputs()
    observation["relations"] = [{"blocker": 2, "blocked": 1, "confidence": 0.9}]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["reason"] == "invalid_observation_schema"


def test_complete_relation_coverage_rejects_occluded_without_obstruction():
    observation, feasibility = _inputs()
    observation["objects"][1]["visibility"] = "occluded"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["reason"] == "invalid_observation_schema"


def test_missing_feasibility_observation_id_is_blocked():
    observation, feasibility = _inputs()
    del feasibility["observation_id"]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["reason"] == "invalid_feasibility_schema"


def test_feasibility_from_another_observation_is_not_reused():
    observation, feasibility = _inputs()
    feasibility["observation_id"] = "obs-previous"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["reason"] == "feasibility_observation_mismatch"


def test_stale_observation_is_blocked():
    observation, feasibility = _inputs()
    observation["is_stale"] = True

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "stale_observation"


def test_missing_target_is_blocked():
    observation, feasibility = _inputs()

    decision = select_next_object(observation, 99, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["legal_candidate_ids"] == []
    assert decision["reason"] == "target_missing"


def test_non_metric_feasibility_units_are_blocked():
    observation, feasibility = _inputs()
    feasibility["distance_unit"] = "cm"

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "unsupported_distance_unit"


def test_missing_freshness_attestation_is_blocked():
    observation, feasibility = _inputs()
    del observation["is_stale"]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "invalid_observation_schema"


def test_unhashable_feasibility_status_is_blocked_without_raising():
    observation, feasibility = _inputs()
    feasibility["objects"][0]["status"] = ["feasible"]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "invalid_feasibility_schema"


def test_out_of_range_relation_confidence_is_blocked_without_raising():
    observation, feasibility = _inputs()
    observation["relations"] = [{"blocker": 1, "blocked": 2, "confidence": 10**1000}]

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "invalid_observation_schema"


def test_feasibility_mode_ranks_quality_minus_execution_cost():
    observation, feasibility = _scored_inputs()

    decision = select_next_object(observation, 3, feasibility, {"mode": "feasibility"})

    assert decision["next_object_id"] == 2
    assert [item["object_id"] for item in decision["diagnostics"][:2]] == [1, 2]
    assert [item["total_score"] for item in decision["diagnostics"][:2]] == pytest.approx([0.1, 0.8])


def test_utility_mode_prefers_candidate_with_greater_dependency_reach():
    observation, feasibility = _scored_inputs()
    observation["objects"] = [
        {"id": 1, "visibility": "clear"},
        {"id": 2, "visibility": "clear"},
        {"id": 3, "visibility": "occluded"},
        {"id": 4, "visibility": "occluded"},
    ]
    observation["visibility_coverage"] = [1, 2, 3, 4]
    observation["relations"] = [
        {"blocker": 2, "blocked": 3, "confidence": 0.9},
        {"blocker": 3, "blocked": 4, "confidence": 0.9},
        {"blocker": 1, "blocked": 4, "confidence": 0.9},
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible", "candidate_quality": 0.8, "execution_cost": 0.2}
        for object_id in (1, 2, 3, 4)
    ]

    decision = select_next_object(observation, 4, feasibility, {"mode": "utility"})

    assert decision["next_object_id"] == 2
    by_id = {item["object_id"]: item for item in decision["diagnostics"]}
    assert by_id[2]["dependency_reach"] == pytest.approx(2 / 3)
    assert by_id[1]["dependency_reach"] == pytest.approx(1 / 3)


def test_missing_candidate_scores_are_diagnosed_and_never_defaulted():
    observation, feasibility = _scored_inputs()
    for record in feasibility["objects"]:
        record.pop("candidate_quality")
        record.pop("execution_cost")

    decision = select_next_object(observation, 3, feasibility, {"mode": "feasibility"})

    assert decision["next_object_id"] is None
    by_id = {item["object_id"]: item for item in decision["diagnostics"]}
    assert by_id[1]["reason"] == "candidate_score_missing"
    assert by_id[2]["reason"] == "candidate_score_missing"
    assert by_id[1]["total_score"] is None


def test_missing_score_for_one_legal_candidate_blocks_partial_ranking():
    observation, feasibility = _scored_inputs()
    feasibility["objects"][0].update(candidate_quality=0.99, execution_cost=0.0)
    feasibility["objects"][1].pop("candidate_quality")

    decision = None
    error = None
    try:
        decision = select_next_object(observation, 3, feasibility, {"mode": "feasibility"})
    except KeyError as caught:
        error = caught

    assert error is None, "missing scores must not crash or yield a partial ranking"
    assert decision["status"] == "uncertain"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "candidate_scores_unavailable"


@pytest.mark.parametrize("invalid_quality", [float("nan"), float("inf"), True, -0.01, 1.01, 10**1000])
def test_invalid_candidate_score_values_fail_closed(invalid_quality):
    observation, feasibility = _scored_inputs()
    feasibility["objects"][0]["candidate_quality"] = invalid_quality
    feasibility["objects"][1]["candidate_quality"] = invalid_quality

    decision = select_next_object(observation, 3, feasibility, {"mode": "feasibility"})

    assert decision["next_object_id"] is None
    by_id = {item["object_id"]: item for item in decision["diagnostics"]}
    assert by_id[1]["reason"] == "candidate_score_invalid"
    assert by_id[2]["reason"] == "candidate_score_invalid"


@pytest.mark.parametrize(
    "scoring_config",
    [
        {"mode": "unknown"},
        {"mode": "utility", "unlock_weight": -1},
        {"mode": "feasibility", "quality_weight": 0.5},
        {"mode": "utility", "weights": {"unlock": 999}},
        {"mode": True},
    ],
)
def test_unsupported_scoring_configs_fail_closed(scoring_config):
    observation, feasibility = _scored_inputs()

    decision = select_next_object(observation, 3, feasibility, scoring_config)

    assert decision["status"] == "blocked"
    assert decision["next_object_id"] is None
    assert decision["reason"] == "unsupported_scoring_config"


def test_scored_candidates_tie_break_by_smallest_object_id():
    observation, feasibility = _scored_inputs()
    for record in feasibility["objects"][:2]:
        record["candidate_quality"] = 0.8
        record["execution_cost"] = 0.2
    feasibility["objects"][:2] = reversed(feasibility["objects"][:2])

    decision = select_next_object(observation, 3, feasibility, {"mode": "feasibility"})

    assert decision["next_object_id"] == 1


def test_high_scores_cannot_override_infeasible_or_occluded_hard_gates():
    observation, feasibility = _scored_inputs()
    feasibility["objects"][0].update(candidate_quality=0.1, execution_cost=0.9)
    feasibility["objects"][1].update(
        status="infeasible", candidate_quality=1.0, execution_cost=0.0
    )
    feasibility["objects"][2].update(candidate_quality=1.0, execution_cost=0.0)

    decision = select_next_object(observation, 3, feasibility, {"mode": "utility"})

    assert decision["next_object_id"] == 1
    by_id = {item["object_id"]: item for item in decision["diagnostics"]}
    assert by_id[2]["reason"] == "feasibility_infeasible"
    assert by_id[3]["reason"] == "blocked_by_required_predecessor"


def test_utility_dependency_reach_counts_diamond_successors_once():
    observation, feasibility = _scored_inputs()
    observation["objects"] = [
        {"id": object_id, "visibility": "clear" if object_id in {1, 2} else "occluded"}
        for object_id in range(1, 7)
    ]
    observation["visibility_coverage"] = list(range(1, 7))
    observation["relations"] = [
        {"blocker": blocker, "blocked": blocked, "confidence": 0.9}
        for blocker, blocked in ((1, 3), (1, 4), (3, 5), (4, 5), (5, 6), (2, 6))
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible", "candidate_quality": 0.8, "execution_cost": 0.2}
        for object_id in range(1, 7)
    ]

    decision = select_next_object(observation, 6, feasibility, {"mode": "utility"})

    assert decision["next_object_id"] == 1
    by_id = {item["object_id"]: item for item in decision["diagnostics"]}
    assert by_id[1]["dependency_reach"] == pytest.approx(4 / 5)
    assert by_id[2]["dependency_reach"] == pytest.approx(1 / 5)


def test_utility_reports_structural_successor_reach_without_claiming_and_unlocks():
    observation, feasibility = _scored_inputs()
    observation["objects"] = [
        {"id": object_id, "visibility": "clear" if object_id in {1, 2} else "occluded"}
        for object_id in (1, 2, 3)
    ]
    observation["visibility_coverage"] = [1, 2, 3]
    observation["relations"] = [
        {"blocker": blocker, "blocked": 3, "confidence": 0.9}
        for blocker in (1, 2)
    ]
    feasibility["objects"] = [
        {"object_id": object_id, "status": "feasible", "candidate_quality": 0.8, "execution_cost": 0.2}
        for object_id in (1, 2, 3)
    ]

    decision = select_next_object(observation, 3, feasibility, {"mode": "utility"})

    assert decision["scoring_mode"] == "utility"
    assert decision["score_formula"] == "candidate_quality - execution_cost + dependency_reach"
    assert decision["dependency_reach_semantics"] == (
        "unique_transitive_successors_in_target_dependency_closure; structural_only; "
        "does_not_assert_that_AND_prerequisites_are_unlocked"
    )
    assert all("unlock_gain" not in item for item in decision["diagnostics"])