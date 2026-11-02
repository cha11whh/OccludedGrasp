import sys
from pathlib import Path

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


def test_empty_relation_list_does_not_authorize_selection():
    observation, feasibility = _inputs()

    decision = select_next_object(observation, 2, feasibility, {})

    assert decision == {
        "status": "uncertain",
        "next_object_id": None,
        "observation_id": "obs-001",
        "reason": "selection_deferred_until_t02_t03",
        "diagnostics": [
            {"object_id": 1, "reason": "selection_deferred_until_t02_t03"},
            {"object_id": 2, "reason": "visibility_unknown"},
        ],
    }


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