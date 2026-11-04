import importlib.util
from copy import deepcopy

import pytest


def _valid_payload():
    binding = {
        "robot_model": "robot-a",
        "joint_names": ["joint-a", "joint-b"],
        "coordinate_frame": "base",
        "calibration_id": "cal-1",
    }
    request = {
        "mode": "simulation",
        "command_id": "cmd-1",
        **binding,
        "position_unit": "rad",
        "time_unit": "s",
        "points": [
            {"positions": [0.25, -0.5], "time_from_start": 0.1},
            {"positions": [0.5, 0.0], "time_from_start": 0.3},
        ],
    }
    return request, binding


def test_real_mode_is_rejected_before_simulator():
    spec = importlib.util.find_spec("obstruction_train.simulation_execution")
    assert spec is not None, "the public simulation dispatch seam must exist"

    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    calls = []
    binding = {
        "robot_model": "robot-a",
        "joint_names": ["joint-a"],
        "coordinate_frame": "base",
        "calibration_id": "cal-1",
    }
    with pytest.raises(ValueError):
        dispatch_simulation_trajectory(
            {"mode": "real"},
            expected_binding=binding,
            simulator=calls.append,
        )

    assert calls == []


def test_valid_si_trajectory_reaches_injected_simulator():
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    calls = []
    result = object()

    actual = dispatch_simulation_trajectory(
        request,
        expected_binding=binding,
        simulator=lambda payload: (calls.append(payload), result)[1],
    )

    assert actual is result
    assert calls == [request]


@pytest.mark.parametrize("mode", [None, "unknown", "real"])
def test_missing_or_non_simulation_mode_never_dispatches(mode):
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    if mode is None:
        del request["mode"]
    else:
        request["mode"] = mode
    calls = []

    with pytest.raises(ValueError):
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=calls.append
        )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("robot_model", "robot-b"),
        ("joint_names", ["joint-b", "joint-a"]),
        ("coordinate_frame", "world"),
        ("calibration_id", "cal-2"),
    ],
)
def test_binding_mismatch_never_dispatches(field, replacement):
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    request[field] = replacement
    calls = []

    with pytest.raises(ValueError):
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=calls.append
        )

    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update(points=[]),
        lambda request: request["points"][0].update(positions=[0.1]),
        lambda request: request["points"][0].update(positions=[0.1, float("nan")]),
        lambda request: request["points"][0].update(positions=[0.1, float("inf")]),
        lambda request: request["points"][0].update(positions=[0.1, True]),
        lambda request: request["points"][0].update(time_from_start=0),
        lambda request: request["points"][1].update(time_from_start=0.1),
        lambda request: request["points"][1].update(time_from_start=0.05),
        lambda request: request.update(position_unit="degrees"),
        lambda request: request.update(extra="unsupported"),
    ],
)
def test_malformed_trajectory_never_dispatches(mutation):
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    mutation(request)
    calls = []

    with pytest.raises(ValueError):
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=calls.append
        )

    assert calls == []


@pytest.mark.parametrize(
    "binding_mutation",
    [
        lambda binding: binding.update(joint_names=[]),
        lambda binding: binding.update(joint_names=["joint-a", "joint-a"]),
        lambda binding: binding.update(joint_names=["joint-a", ""]),
        lambda binding: binding.update(robot_model=""),
        lambda binding: binding.pop("calibration_id"),
    ],
)
def test_invalid_expected_binding_never_dispatches(binding_mutation):
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    binding_mutation(binding)
    calls = []

    with pytest.raises(ValueError):
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=calls.append
        )

    assert calls == []


def test_simulator_receives_validated_copy_and_cannot_mutate_inputs():
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()
    original_request = deepcopy(request)
    original_binding = deepcopy(binding)
    executed_payloads = []
    result = object()

    def simulator(payload):
        executed_payloads.append(payload)
        payload["joint_names"][0] = "simulator-mutation"
        payload["points"][0]["positions"][0] = 999
        return result

    assert (
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=simulator
        )
        is result
    )
    assert executed_payloads[0] is not request
    assert request == original_request
    assert binding == original_binding


def test_simulator_exception_propagates():
    from obstruction_train.simulation_execution import dispatch_simulation_trajectory

    request, binding = _valid_payload()

    def simulator(_payload):
        raise RuntimeError("simulator failure")

    with pytest.raises(RuntimeError, match="simulator failure"):
        dispatch_simulation_trajectory(
            request, expected_binding=binding, simulator=simulator
        )