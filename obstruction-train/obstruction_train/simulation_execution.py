from collections.abc import Callable
from copy import deepcopy
from math import isfinite
from numbers import Real


_BINDING_FIELDS = {
    "robot_model",
    "joint_names",
    "coordinate_frame",
    "calibration_id",
}
_REQUEST_FIELDS = _BINDING_FIELDS | {
    "mode",
    "command_id",
    "position_unit",
    "time_unit",
    "points",
}


def _validate_binding(binding: object) -> dict[str, object]:
    if not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS:
        raise ValueError("binding must contain exactly the required binding fields")

    for field in _BINDING_FIELDS - {"joint_names"}:
        value = binding[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a nonempty string")

    joint_names = binding["joint_names"]
    if (
        not isinstance(joint_names, list)
        or not joint_names
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
    ):
        raise ValueError("joint_names must be a nonempty list of unique nonempty strings")

    return binding


def _finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def dispatch_simulation_trajectory(
    request: dict[str, object],
    *,
    expected_binding: dict[str, object],
    simulator: Callable[[dict[str, object]], object],
) -> object:
    payload = deepcopy(request)
    binding = _validate_binding(expected_binding)

    if not isinstance(payload, dict) or set(payload) != _REQUEST_FIELDS:
        raise ValueError("request must contain exactly the supported trajectory fields")
    if payload.get("mode") != "simulation":
        raise ValueError("request mode must be explicitly set to 'simulation'")
    if not callable(simulator):
        raise ValueError("simulator must be callable")

    command_id = payload["command_id"]
    if not isinstance(command_id, str) or not command_id:
        raise ValueError("command_id must be a nonempty string")

    for field, expected in binding.items():
        if payload[field] != expected:
            raise ValueError(f"request {field} does not match expected binding")

    if payload["position_unit"] != "rad" or payload["time_unit"] != "s":
        raise ValueError("trajectory units must be radians and seconds")

    joint_count = len(binding["joint_names"])
    points = payload["points"]
    if not isinstance(points, list) or not points:
        raise ValueError("points must be a nonempty list")

    previous_time = 0
    for point in points:
        if not isinstance(point, dict) or set(point) != {"positions", "time_from_start"}:
            raise ValueError("each point must contain positions and time_from_start only")
        positions = point["positions"]
        if (
            not isinstance(positions, list)
            or len(positions) != joint_count
            or any(not _finite_real(value) for value in positions)
        ):
            raise ValueError("positions must contain one finite real value per joint")

        time_from_start = point["time_from_start"]
        if not _finite_real(time_from_start) or time_from_start <= previous_time:
            raise ValueError("time_from_start must be positive, finite, and strictly increasing")
        previous_time = time_from_start

    return simulator(payload)