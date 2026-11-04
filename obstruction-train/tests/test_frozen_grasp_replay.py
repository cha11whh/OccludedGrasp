import json

import numpy as np

from obstruction_train.frozen_grasp_replay import _json_safe, run_mock_replay


def _observation(observation_id, object_ids, visibility, relations):
    return {
        "mode": "mock",
        "observation_id": observation_id,
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": object_ids,
        "objects": [
            {"id": object_id, "visibility": visibility[object_id]}
            for object_id in object_ids
        ],
        "relations": [
            {"blocker": blocker, "blocked": blocked, "confidence": 1.0}
            for blocker, blocked in relations
        ],
    }


def test_mock_replay_predicts_only_gated_ids_and_replans_to_verified_target():
    observations = [
        _observation("obs-0", [1, 2, 3], {1: "clear", 2: "clear", 3: "occluded"}, [(1, 3), (2, 3)]),
        _observation("obs-1", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)]),
        _observation("obs-2", [3], {3: "clear"}, []),
        _observation("obs-3", [], {}, []),
    ]
    predicted_ids = []
    executed = []

    def predictor(points, object_ids):
        predicted_ids.append(object_ids)
        poses, scores, contacts, widths = {}, {}, {}, {}
        for object_id in object_ids:
            score = {1: 0.4, 2: 0.9, 3: 0.8}[object_id]
            poses[object_id] = np.eye(4)[None]
            scores[object_id] = np.array([score])
            contacts[object_id] = np.array([[0.0, 0.0, 1.0]])
            widths[object_id] = np.array([0.04])
        return poses, scores, contacts, widths

    def evaluator(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": observation_id, "status": "feasible",
            "gripper_width_valid": True, "front_approach_valid": True,
        }

    def path(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": observation_id, "status": "feasible",
            "execution_cost": {1: 0.0, 2: 0.1, 3: 0.1}[object_id],
        }

    def executor(object_id, observation_id):
        executed.append((object_id, observation_id))
        return {"mode": "mock", "object_id": object_id, "observation_id": observation_id}

    def verifier(execution, observation_id):
        return {
            "mode": "mock", "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True, "placement_verified": True,
        }

    def observer(previous):
        return {"mode": "mock", "observation": observations[len(executed)]}

    result = run_mock_replay(
        observations[0], 3,
        lambda observation: {
            "mode": "mock", "observation_id": observation["observation_id"],
            "distance_unit": "m", "coordinate_frame": "camera",
            "points": np.array([[0.0, 0.0, 1.0]]),
        },
        predictor=predictor, geometry_evaluator=evaluator, path_evaluator=path,
        executor=executor, verifier=verifier, observer=observer,
        sim_clock=lambda: 0.0,
    )

    assert result["status"] == "target_completed"
    assert [object_id for object_id, _ in executed] == [2, 1, 3]
    assert predicted_ids == [(1, 2), (1,), (3,)]
    assert all(action["observation_id"] == f"obs-{index}" for index, action in enumerate(result["actions"]))
    assert result["mode"] == {"replay": "cpu", "evidence": "mock"}
    assert result["actions"][0]["object_id"] == 2
    assert {entry["event"] for entry in result["audit"]} >= {
        "prediction", "geometry", "path", "execution", "verification",
        "observation",
    }
    assert all(entry["mode"] == "mock" for entry in result["audit"])


def _run_simple_case(
    observation,
    *,
    scores=None,
    geometry_status=None,
    geometry_observation_id=None,
    path_observation_id=None,
    proposals=True,
    next_observations=(),
    confirmed=False,
    point_cloud_observation_id=None,
    omitted_prediction_ids=(),
    execution_error=False,
):
    requested = []
    executed = []
    scores = scores or {object_id: 0.5 for object_id in (1, 2, 3)}
    geometry_status = geometry_status or {}

    def predictor(points, object_ids):
        requested.append(object_ids)
        if not proposals:
            return {}, {}, {}, {}
        poses, qualities, contacts, widths = {}, {}, {}, {}
        for object_id in object_ids:
            if object_id in omitted_prediction_ids:
                continue
            poses[object_id] = np.eye(4)[None]
            qualities[object_id] = np.array([scores[object_id]])
            contacts[object_id] = np.array([[0.0, 0.0, 1.0]])
            widths[object_id] = np.array([0.04])
        return poses, qualities, contacts, widths

    def geometry(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": geometry_observation_id or observation_id,
            "status": geometry_status.get(object_id, "feasible"),
            "gripper_width_valid": True, "front_approach_valid": True,
        }

    def path(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": path_observation_id or observation_id,
            "status": "feasible", "execution_cost": 0.1,
        }

    def executor(object_id, observation_id):
        executed.append((object_id, observation_id))
        if execution_error:
            raise RuntimeError("synthetic mock failure")
        return {"mode": "mock", "object_id": object_id, "observation_id": observation_id}

    def verifier(execution, observation_id):
        return {
            "mode": "mock", "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": confirmed,
            "placement_verified": confirmed,
        }

    next_index = 0

    def observer(previous):
        nonlocal next_index
        if next_index < len(next_observations):
            next_observation = next_observations[next_index]
            next_index += 1
        else:
            next_observation = dict(previous)
            next_observation["observation_id"] += "-stale"
            next_observation["is_stale"] = True
        return {"mode": "mock", "observation": next_observation}

    result = run_mock_replay(
        observation, 3,
        lambda current: {
            "mode": "mock", "observation_id": (
                point_cloud_observation_id or current["observation_id"]
            ),
            "distance_unit": "m", "coordinate_frame": "camera",
            "points": np.array([[0.0, 0.0, 1.0]]),
        },
        predictor=predictor, geometry_evaluator=geometry, path_evaluator=path,
        executor=executor, verifier=verifier, observer=observer,
        sim_clock=lambda: 0.0,
    )
    return result, requested, executed


def test_high_score_candidate_with_collision_is_not_selected():
    observation = _observation(
        "collision-obs", [1, 2, 3],
        {1: "clear", 2: "clear", 3: "occluded"}, [(1, 3), (2, 3)],
    )

    result, requested, executed = _run_simple_case(
        observation, scores={1: 0.2, 2: 0.99, 3: 1.0},
        geometry_status={2: "infeasible"},
    )

    assert requested[0] == (1, 2)
    assert executed[0][0] == 1
    assert all(object_id != 2 for object_id, _ in executed)
    assert result["actions"][0]["object_id"] == 1


def test_rejected_proposal_set_remains_unknown_within_finite_candidate_budget():
    observation = _observation(
        "bounded-rejections", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )

    result, _, executed = _run_simple_case(
        observation, geometry_status={1: "infeasible"},
    )

    assert executed == []
    assert result["decision"]["status"] == "uncertain"
    assert result["decision"]["reason"] == "target_dependency_uncertain"


def test_execution_error_is_audited_and_strictly_json_serializable():
    observation = _observation(
        "error-obs", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )

    result, _, _ = _run_simple_case(observation, execution_error=True)

    serialized = json.dumps(_json_safe(result), allow_nan=False)

    assert json.loads(serialized)["reason"] == "executor_error"
    assert any(
        entry["event"] == "execution" and entry["status"] == "error"
        and entry["reason"] == "RuntimeError"
        for entry in result["audit"]
    )


def test_empty_unknown_and_cross_observation_geometry_or_path_fail_closed():
    observation = _observation(
        "bound-obs", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )
    cases = [
        {"proposals": False},
        {"geometry_status": {1: "unknown"}},
        {"geometry_observation_id": "different-obs"},
        {"path_observation_id": "different-obs"},
    ]

    for options in cases:
        result, requested, executed = _run_simple_case(observation, **options)

        assert requested == [(1,)]
        assert executed == []
        assert result["status"] == "stopped"
        assert result["reason"] == "stale_observation"


def test_point_cloud_from_another_observation_is_never_scored():
    observation = _observation(
        "point-cloud-obs", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )

    result, requested, executed = _run_simple_case(
        observation, point_cloud_observation_id="different-obs",
    )

    assert requested == []
    assert executed == []
    assert result["status"] == "stopped"
    assert result["reason"] == "stale_observation"


def test_missing_candidate_for_one_clear_root_is_audited_as_unknown():
    observation = _observation(
        "partial-candidates", [1, 2, 3],
        {1: "clear", 2: "clear", 3: "occluded"}, [(1, 3), (2, 3)],
    )

    result, _, executed = _run_simple_case(
        observation, omitted_prediction_ids={2},
    )

    prediction = next(
        entry for entry in result["audit"] if entry["event"] == "prediction"
    )
    assert prediction["unknown_object_ids"] == [2]
    assert prediction["mode"] == "mock"
    assert executed == [(1, "partial-candidates")]


def test_unconfirmed_removal_cannot_unlock_target():
    initial = _observation(
        "unconfirmed-0", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )
    apparently_cleared = _observation(
        "unconfirmed-1", [3], {3: "clear"}, [],
    )

    result, _, executed = _run_simple_case(
        initial, next_observations=[apparently_cleared], confirmed=False,
    )

    assert [object_id for object_id, _ in executed] == [1]
    assert result["status"] == "stopped"
    assert result["reason"] == "unconfirmed_removal_observation_mismatch"
    assert result["attempts"] == {1: 1}


def test_attempt_limit_is_preserved_across_fresh_replans():
    initial = _observation(
        "attempt-0", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )
    fresh_one = _observation(
        "attempt-1", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )
    fresh_two = _observation(
        "attempt-2", [1, 3], {1: "clear", 3: "occluded"}, [(1, 3)],
    )

    result, _, executed = _run_simple_case(
        initial, next_observations=[fresh_one, fresh_two], confirmed=False,
    )

    assert len(executed) == 2
    assert result["attempts"] == {1: 2}
    assert result["status"] == "attempt_limit"