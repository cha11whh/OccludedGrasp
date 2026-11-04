"""CPU replay adapter for injected, explicitly mock-only grasp evidence."""

import json
import math
from collections.abc import Mapping

from obstruction_train.frozen_grasp_backend import predict_candidates
from obstruction_train.relation_first_policy import prediction_candidate_ids
from obstruction_train.replay_only import run_replay


def _unknown_feasibility(observation):
    observation_id = (
        observation.get("observation_id")
        if isinstance(observation, Mapping) else None
    )
    objects = observation.get("objects", []) if isinstance(observation, Mapping) else []
    object_ids = [
        record["id"] for record in objects
        if isinstance(record, Mapping) and type(record.get("id")) is int
    ] if isinstance(objects, list) else []
    return {
        "observation_id": observation_id,
        "distance_unit": "m",
        "coordinate_frame": "camera",
        "objects": [
            {"object_id": object_id, "status": "unknown"}
            for object_id in object_ids
        ],
    }


def _bound_mock_result(result, object_id, observation_id, required):
    if (
        not isinstance(result, Mapping)
        or result.get("mode") != "mock"
        or result.get("object_id") != object_id
        or result.get("observation_id") != observation_id
        or result.get("status") not in {"feasible", "infeasible", "unknown"}
    ):
        return None
    if any(key not in result for key in required):
        return None
    return result


def _score_observation(
    observation,
    target_id,
    point_cloud_provider,
    predictor,
    geometry_evaluator,
    path_evaluator,
    audit,
):
    feasibility = _unknown_feasibility(observation)
    observation_id = feasibility["observation_id"]
    if not isinstance(observation, Mapping) or observation.get("mode") != "mock":
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "blocked", "reason": "observation_not_mock",
        })
        return feasibility

    candidate_ids = prediction_candidate_ids(observation, target_id)
    if not candidate_ids:
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "prediction_skipped", "reason": "no_evidenced_clear_root",
            "requested_object_ids": [] if candidate_ids == [] else None,
        })
        return feasibility
    try:
        point_request = point_cloud_provider(observation)
    except Exception:
        point_request = None
    if (
        not isinstance(point_request, Mapping)
        or point_request.get("mode") != "mock"
        or point_request.get("observation_id") != observation_id
        or point_request.get("distance_unit") != "m"
        or point_request.get("coordinate_frame") != "camera"
    ):
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "prediction_skipped", "reason": "invalid_point_cloud_binding",
            "requested_object_ids": candidate_ids,
        })
        return feasibility

    prediction = predict_candidates(
        point_request.get("points"), candidate_ids,
        observation_id=observation_id, distance_unit="m",
        coordinate_frame="camera", model_version="mock-injected",
        predictor=predictor,
    )
    candidate_groups = {object_id: [] for object_id in candidate_ids}
    if prediction["status"] == "candidates_generated":
        for candidate in prediction["candidates"]:
            candidate_groups[candidate["object_id"]].append(candidate)
    unknown_object_ids = [
        object_id for object_id in candidate_ids
        if not candidate_groups[object_id]
    ] if prediction["status"] == "candidates_generated" else list(candidate_ids)
    audit.append({
        "mode": "mock", "observation_id": observation_id,
        "event": "prediction", "requested_object_ids": candidate_ids,
        "status": prediction["status"], "reason": prediction["reason"],
        "candidate_count": len(prediction["candidates"]),
        "unknown_object_ids": unknown_object_ids,
    })
    if prediction["observation_id"] != observation_id or prediction["status"] != "candidates_generated":
        return feasibility

    records = {record["object_id"]: record for record in feasibility["objects"]}
    for object_id, candidates in candidate_groups.items():
        accepted = []
        for candidate_index, candidate in enumerate(candidates):
            mock_candidate = {
                **candidate, "mode": "mock", "observation_id": observation_id,
                "distance_unit": "m", "coordinate_frame": "camera",
            }
            geometry_error = None
            try:
                geometry = geometry_evaluator(mock_candidate, object_id, observation_id)
            except Exception as error:
                geometry = None
                geometry_error = type(error).__name__
            geometry = _bound_mock_result(
                geometry, object_id, observation_id,
                ("gripper_width_valid", "front_approach_valid"),
            )
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "geometry", "object_id": object_id,
                "candidate_index": candidate_index,
                "status": geometry["status"] if geometry else "unknown",
                "gripper_width_valid": (
                    geometry["gripper_width_valid"] if geometry else None
                ),
                "front_approach_valid": (
                    geometry["front_approach_valid"] if geometry else None
                ),
                "reason": geometry_error or (
                    "invalid_or_unbound_result" if geometry is None else None
                ),
            })
            if geometry is None:
                continue
            width_ok = geometry["gripper_width_valid"]
            approach_ok = geometry["front_approach_valid"]
            if type(width_ok) is not bool or type(approach_ok) is not bool:
                continue
            if geometry["status"] == "unknown":
                continue
            if geometry["status"] == "infeasible" or not width_ok or not approach_ok:
                continue
            path_error = None
            try:
                path_result = path_evaluator(mock_candidate, object_id, observation_id)
            except Exception as error:
                path_result = None
                path_error = type(error).__name__
            path_result = _bound_mock_result(
                path_result, object_id, observation_id, ("execution_cost",)
            )
            cost = path_result["execution_cost"] if path_result else None
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "path", "object_id": object_id,
                "candidate_index": candidate_index,
                "status": path_result["status"] if path_result else "unknown",
                "execution_cost": (
                    float(cost) if isinstance(cost, (int, float))
                    and not isinstance(cost, bool) and math.isfinite(cost)
                    else None
                ),
                "reason": path_error or (
                    "invalid_or_unbound_result" if path_result is None else None
                ),
            })
            if path_result is None or path_result["status"] == "unknown":
                continue
            if path_result["status"] == "infeasible":
                continue
            cost = path_result["execution_cost"]
            if (
                isinstance(cost, bool) or not isinstance(cost, (int, float))
                or not math.isfinite(cost) or not 0 <= cost <= 1
            ):
                continue
            accepted.append((candidate["score"], float(cost), candidate_index))

        if accepted:
            quality, cost, _ = min(accepted, key=lambda item: (-item[0], item[1], item[2]))
            records[object_id].update({
                "status": "feasible", "candidate_quality": quality,
                "execution_cost": cost,
            })
    return feasibility


def run_mock_replay(
    initial_observation,
    target_id,
    point_cloud_provider,
    *,
    predictor,
    geometry_evaluator,
    path_evaluator,
    executor,
    verifier,
    observer,
    sim_clock,
):
    """Use injected mock evidence while delegating every choice to run_replay."""
    audit = []

    def score(observation):
        return _score_observation(
            observation, target_id, point_cloud_provider, predictor,
            geometry_evaluator, path_evaluator, audit,
        )

    def execute(object_id, observation_id):
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "selected", "object_id": object_id,
        })
        try:
            result = executor(object_id, observation_id)
        except Exception as error:
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "execution", "object_id": object_id,
                "status": "error", "reason": type(error).__name__,
            })
            raise
        if (
            not isinstance(result, Mapping)
            or result.get("mode") != "mock"
            or result.get("object_id") != object_id
            or result.get("observation_id") != observation_id
        ):
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "execution", "object_id": object_id,
                "status": "error", "reason": "identity_mismatch",
            })
            raise ValueError("mock execution identity mismatch")
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "execution", "object_id": object_id,
            "status": "completed",
        })
        return result

    def verify(execution, observation_id):
        try:
            result = verifier(execution, observation_id)
        except Exception as error:
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "verification", "object_id": execution["object_id"],
                "status": "error", "reason": type(error).__name__,
            })
            raise
        if (
            not isinstance(result, Mapping)
            or result.get("mode") != "mock"
            or result.get("object_id") != execution.get("object_id")
            or result.get("observation_id") != observation_id
        ):
            audit.append({
                "mode": "mock", "observation_id": observation_id,
                "event": "verification", "object_id": execution["object_id"],
                "status": "error", "reason": "identity_mismatch",
            })
            raise ValueError("mock verification identity mismatch")
        audit.append({
            "mode": "mock", "observation_id": observation_id,
            "event": "verification", "object_id": execution["object_id"],
            "status": "completed",
            "physical_removal_confirmed": result.get("physical_removal_confirmed"),
            "placement_verified": result.get("placement_verified"),
        })
        return result

    def observe(previous):
        previous_id = (
            previous.get("observation_id")
            if isinstance(previous, Mapping) else None
        )
        try:
            result = observer(previous)
        except Exception as error:
            audit.append({
                "mode": "mock", "observation_id": previous_id,
                "event": "observation", "status": "error",
                "reason": type(error).__name__,
            })
            raise
        if not isinstance(result, Mapping) or result.get("mode") != "mock":
            audit.append({
                "mode": "mock", "observation_id": previous_id,
                "event": "observation", "status": "error",
                "reason": "mock_observation_required",
            })
            raise ValueError("mock observation required")
        observation = result.get("observation")
        if not isinstance(observation, Mapping):
            audit.append({
                "mode": "mock", "observation_id": previous_id,
                "event": "observation", "status": "error",
                "reason": "observation_payload_required",
            })
            raise ValueError("mock observation payload required")
        audit.append({
            "mode": "mock", "observation_id": previous_id,
            "event": "observation", "status": "received",
            "next_observation_id": (
                observation.get("observation_id")
                if isinstance(observation.get("observation_id"), str) else None
            ),
        })
        return {"observation": observation, "feasibility": score(observation)}

    initial_feasibility = score(initial_observation)
    replay = run_replay(
        initial_observation, target_id, initial_feasibility,
        execute, verify, observe, sim_clock,
        scoring_config={"mode": "feasibility"},
    )
    return {
        **replay,
        "mode": {"replay": "cpu", "evidence": "mock"},
        "audit": audit,
    }


def _demo_mock_replay():
    """Run a small dependency-clearance example with synthetic mock inputs."""
    import numpy as np

    observations = [
        {
            "mode": "mock", "observation_id": "demo-0", "is_stale": False,
            "relation_coverage": "complete", "visibility_coverage": [1, 2],
            "objects": [
                {"id": 1, "visibility": "clear"},
                {"id": 2, "visibility": "occluded"},
            ],
            "relations": [{"blocker": 1, "blocked": 2, "confidence": 1.0}],
        },
        {
            "mode": "mock", "observation_id": "demo-1", "is_stale": False,
            "relation_coverage": "complete", "visibility_coverage": [2],
            "objects": [{"id": 2, "visibility": "clear"}], "relations": [],
        },
        {
            "mode": "mock", "observation_id": "demo-2", "is_stale": False,
            "relation_coverage": "complete", "visibility_coverage": [],
            "objects": [], "relations": [],
        },
    ]
    next_observation = {"demo-0": observations[1], "demo-1": observations[2]}

    def predictor(points, object_ids):
        poses, scores, contacts, widths = {}, {}, {}, {}
        for object_id in object_ids:
            poses[object_id] = np.eye(4)[None]
            scores[object_id] = np.array([0.8])
            contacts[object_id] = np.array([[0.0, 0.0, 1.0]])
            widths[object_id] = np.array([0.04])
        return poses, scores, contacts, widths

    def geometry(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": observation_id, "status": "feasible",
            "gripper_width_valid": True, "front_approach_valid": True,
        }

    def path(candidate, object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": observation_id, "status": "feasible",
            "execution_cost": 0.1,
        }

    def executor(object_id, observation_id):
        return {
            "mode": "mock", "object_id": object_id,
            "observation_id": observation_id,
        }

    def verifier(execution, observation_id):
        return {
            "mode": "mock", "object_id": execution["object_id"],
            "observation_id": observation_id,
            "physical_removal_confirmed": True, "placement_verified": True,
        }

    return run_mock_replay(
        observations[0], 2,
        lambda observation: {
            "mode": "mock", "observation_id": observation["observation_id"],
            "distance_unit": "m", "coordinate_frame": "camera",
            "points": np.array([[0.0, 0.0, 1.0]]),
        },
        predictor=predictor, geometry_evaluator=geometry, path_evaluator=path,
        executor=executor, verifier=verifier,
        observer=lambda previous: {
            "mode": "mock",
            "observation": next_observation[previous["observation_id"]],
        },
        sim_clock=lambda: 0.0,
    )


def main():
    print(json.dumps(_json_safe(_demo_mock_replay()), indent=2, allow_nan=False))
    return 0


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())