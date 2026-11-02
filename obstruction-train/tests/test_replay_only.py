from obstruction_train.replay_only import run_replay


def _input(observation_id="obs-1"):
    observation = {
        "observation_id": observation_id,
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [7],
        "objects": [{"id": 7, "visibility": "clear"}],
        "relations": [],
    }
    feasibility = {
        "observation_id": observation_id,
        "coordinate_frame": "camera",
        "distance_unit": "m",
        "objects": [{"object_id": 7, "status": "feasible"}],
    }
    return observation, feasibility


def test_target_is_completed_only_after_bound_verified_placement_and_reobserve():
    initial, feasibility = _input()
    following, following_feasibility = _input("obs-2")
    events = []

    def execute(object_id, observation_id):
        events.append(("execute", object_id, observation_id))
        return {"object_id": object_id, "succeeded": True}

    def verify(execution, observation_id):
        events.append(("verify", execution["object_id"], observation_id))
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True,
            "placement_verified": True,
        }

    def observe(previous):
        events.append(("observe", previous["observation_id"]))
        return {"observation": following, "feasibility": following_feasibility}

    result = run_replay(
        initial,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: 0.0,
    )

    assert result["status"] == "target_completed"
    assert events == [
        ("execute", 7, "obs-1"),
        ("verify", 7, "obs-1"),
        ("observe", "obs-1"),
    ]


def test_target_removal_does_not_complete_with_empty_feasibility():
    observation, feasibility = _input()
    observation["objects"].append({"id": 8, "visibility": "clear"})
    observation["visibility_coverage"].append(8)
    feasibility["objects"].append({"object_id": 8, "status": "feasible"})
    following, following_feasibility = _input("obs-2")
    following["objects"] = [{"id": 8, "visibility": "clear"}]
    following["visibility_coverage"] = [8]
    following_feasibility["objects"] = []

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: {"object_id": object_id},
        lambda execution, observation_id: {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True,
            "placement_verified": True,
        },
        lambda previous: {
            "observation": following,
            "feasibility": following_feasibility,
        },
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "invalid_feasibility_schema"
def test_failed_action_counts_as_attempt_and_replanning_cannot_exceed_two():
    observation, feasibility = _input()
    seen = []

    def execute(object_id, observation_id):
        seen.append((object_id, observation_id))
        return {"object_id": object_id, "succeeded": False}

    def verify(execution, observation_id):
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": False,
            "placement_verified": False,
        }

    def observe(previous):
        next_observation, next_feasibility = _input(f"obs-{len(seen) + 1}")
        return {"observation": next_observation, "feasibility": next_feasibility}

    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: next(ticks),
    )

    assert result["status"] == "attempt_limit"
    assert result["attempts"] == {7: 2}
    assert len(seen) == 2


def test_invalid_initial_feasibility_fails_closed_without_escaping():
    observation, _ = _input()

    def unexpected(*args):
        raise AssertionError("callbacks must not run for invalid feasibility")

    result = run_replay(
        observation,
        7,
        None,
        unexpected,
        unexpected,
        unexpected,
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "unsupported_distance_unit"
    assert result["decision"]["reason"] == "unsupported_distance_unit"


def test_repeated_observation_stops_before_a_second_execution():
    observation, feasibility = _input()
    executed = []

    def execute(object_id, observation_id):
        executed.append(object_id)
        return {"object_id": object_id}

    def verify(execution, observation_id):
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": False,
            "placement_verified": False,
        }

    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        lambda previous: {"observation": observation, "feasibility": feasibility},
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "repeated_observation"
    assert executed == [7]


def test_unknown_decisions_allow_at_most_three_fresh_reobservations():
    observation, feasibility = _input()
    observation["relation_coverage"] = "unknown"
    observed_ids = []

    def observe(previous):
        next_observation, next_feasibility = _input(f"obs-{len(observed_ids) + 2}")
        next_observation["relation_coverage"] = "unknown"
        observed_ids.append(next_observation["observation_id"])
        return {"observation": next_observation, "feasibility": next_feasibility}

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: pytest.fail("unknown cannot execute"),
        lambda execution, observation_id: pytest.fail("unknown cannot verify"),
        observe,
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "unknown_observation_limit"
    assert result["unknown_reobservations"] == 3
    assert len(observed_ids) == 3


def test_executor_cannot_report_switching_to_an_unselected_object():
    observation, feasibility = _input()
    verifier_calls = []

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: {"object_id": 8},
        lambda execution, observation_id: verifier_calls.append(execution),
        lambda previous: pytest.fail("mismatched execution cannot reobserve"),
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "executor_object_mismatch"
    assert result["attempts"] == {7: 1}
    assert verifier_calls == []


def test_verification_must_match_selected_object_and_observation():
    observation, feasibility = _input()
    following, following_feasibility = _input("obs-2")

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: {"object_id": object_id},
        lambda execution, observation_id: {
            "object_id": execution["object_id"],
            "observation_id": "obs-old",
            "physical_removal_confirmed": True,
            "placement_verified": True,
        },
        lambda previous: {"observation": following, "feasibility": following_feasibility},
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "invalid_verification"
    assert result["attempts"] == {7: 1}


def test_sim_time_is_finite_monotonic_and_stops_at_300_seconds():
    observation, feasibility = _input()
    clock_values = iter([299.0, 300.0])
    executed = []

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: (
            executed.append(object_id) or {"object_id": object_id}
        ),
        lambda execution, observation_id: pytest.fail("time boundary blocks verify"),
        lambda previous: pytest.fail("time boundary blocks observe"),
        lambda: next(clock_values),
    )

    assert result["status"] == "budget_exhausted"
    assert result["reason"] == "sim_time_limit"
    assert executed == [7]


def test_nonmonotonic_sim_time_stops_instead_of_replanning():
    observation, feasibility = _input()
    clock_values = iter([2.0, 1.0])
    executed = []

    result = run_replay(
        observation,
        7,
        feasibility,
        lambda object_id, observation_id: (
            executed.append(object_id) or {"object_id": object_id}
        ),
        lambda execution, observation_id: pytest.fail("bad clock blocks verify"),
        lambda previous: pytest.fail("bad clock blocks observe"),
        lambda: next(clock_values),
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "nonmonotonic_sim_time"
    assert executed == [7]


def test_executor_success_without_verified_placement_never_completes_target():
    observation, feasibility = _input()
    executions = []

    def execute(object_id, observation_id):
        executions.append(object_id)
        return {"object_id": object_id, "succeeded": True}

    def verify(execution, observation_id):
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": False,
            "placement_verified": False,
        }

    def observe(previous):
        next_observation, next_feasibility = _input(f"obs-{len(executions) + 1}")
        return {"observation": next_observation, "feasibility": next_feasibility}

    ticks = iter([0.0] * 20)
    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: next(ticks),
    )

    assert result["status"] == "attempt_limit"
    assert result["attempts"] == {7: 2}
    assert [action["verification"]["placement_verified"] for action in result["actions"]] == [False, False]


def test_unconfirmed_blocker_removal_cannot_clear_its_target_dependency():
    observation = {
        "observation_id": "obs-1",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [2, 7],
        "objects": [
            {"id": 2, "visibility": "clear"},
            {"id": 7, "visibility": "occluded"},
        ],
        "relations": [{"blocker": 2, "blocked": 7, "confidence": 0.9}],
    }
    feasibility = {
        "observation_id": "obs-1",
        "coordinate_frame": "camera",
        "distance_unit": "m",
        "objects": [
            {"object_id": 2, "status": "feasible"},
            {"object_id": 7, "status": "feasible"},
        ],
    }
    cleared_target = {
        "observation_id": "obs-2",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [7],
        "objects": [{"id": 7, "visibility": "clear"}],
        "relations": [],
    }
    cleared_feasibility = {
        **feasibility,
        "observation_id": "obs-2",
        "objects": [{"object_id": 7, "status": "feasible"}],
    }
    executed = []
    observed = []

    def execute(object_id, observation_id):
        executed.append(object_id)
        return {"object_id": object_id}

    def verify(execution, observation_id):
        removed = execution["object_id"] == 7
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": removed,
            "placement_verified": removed,
        }

    def observe(previous):
        observed.append(previous["observation_id"])
        next_id = f"obs-{len(observed) + 1}"
        next_observation = {**cleared_target, "observation_id": next_id}
        next_feasibility = {**cleared_feasibility, "observation_id": next_id}
        return {"observation": next_observation, "feasibility": next_feasibility}

    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "unconfirmed_removal_observation_mismatch"
    assert result["observation_id"] == "obs-1"
    assert result["rejected_observation_id"] == "obs-2"
    assert result["actions"][0]["object_id"] == 2
    assert result["actions"][0]["verification"]["physical_removal_confirmed"] is False
    assert executed == [2]
    assert observed == ["obs-1"]


def test_removal_without_verified_placement_stops_before_another_execution():
    observation, feasibility = _input()
    following, following_feasibility = _input("obs-2")
    executed = []
    observed = []

    def execute(object_id, observation_id):
        executed.append(object_id)
        return {"object_id": object_id}

    def verify(execution, observation_id):
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True,
            "placement_verified": False,
        }

    def observe(previous):
        observed.append(previous["observation_id"])
        return {"observation": following, "feasibility": following_feasibility}

    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "placement_unverified"
    assert executed == [7]
    assert observed == []


def test_new_object_identity_stops_replay_even_when_sim_time_is_frozen():
    observation = {
        "observation_id": "obs-1",
        "is_stale": False,
        "relation_coverage": "complete",
        "visibility_coverage": [2, 7],
        "objects": [
            {"id": 2, "visibility": "clear"},
            {"id": 7, "visibility": "occluded"},
        ],
        "relations": [{"blocker": 2, "blocked": 7, "confidence": 0.9}],
    }
    feasibility = {
        "observation_id": "obs-1",
        "coordinate_frame": "camera",
        "distance_unit": "m",
        "objects": [
            {"object_id": 2, "status": "feasible"},
            {"object_id": 7, "status": "feasible"},
        ],
    }
    executed = []
    observation_count = []

    def execute(object_id, observation_id):
        executed.append(object_id)
        return {"object_id": object_id}

    def verify(execution, observation_id):
        return {
            "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True,
            "placement_verified": True,
        }

    def observe(previous):
        observation_count.append(previous["observation_id"])
        object_id = 3
        next_id = "obs-2"
        next_observation = {
            "observation_id": next_id,
            "is_stale": False,
            "relation_coverage": "complete",
            "visibility_coverage": [object_id, 7],
            "objects": [
                {"id": object_id, "visibility": "clear"},
                {"id": 7, "visibility": "occluded"},
            ],
            "relations": [
                {"blocker": object_id, "blocked": 7, "confidence": 0.9}
            ],
        }
        next_feasibility = {
            "observation_id": next_id,
            "coordinate_frame": "camera",
            "distance_unit": "m",
            "objects": [
                {"object_id": object_id, "status": "feasible"},
                {"object_id": 7, "status": "feasible"},
            ],
        }
        return {"observation": next_observation, "feasibility": next_feasibility}

    result = run_replay(
        observation,
        7,
        feasibility,
        execute,
        verify,
        observe,
        lambda: 0.0,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "unknown_object_identity"
    assert executed == [2]
    assert observation_count == ["obs-1"]