import math
from collections.abc import Mapping


def _is_object_id(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_observation(observation):
    if not isinstance(observation, Mapping):
        return False, [], {}
    objects = observation.get("objects")
    visibility_coverage = observation.get("visibility_coverage", [])
    if (
        not isinstance(observation.get("observation_id"), str)
        or not observation["observation_id"].strip()
        or not isinstance(observation.get("is_stale"), bool)
        or not isinstance(observation.get("relation_coverage"), str)
        or observation["relation_coverage"] not in {"complete", "partial", "unknown"}
        or not isinstance(objects, list)
        or not isinstance(observation.get("relations"), list)
        or not isinstance(visibility_coverage, list)
    ):
        return False, [], {}
    object_ids = []
    visibility_by_id = {}
    for item in objects:
        if not isinstance(item, Mapping) or not _is_object_id(item.get("id")):
            return False, object_ids, visibility_by_id
        visibility = item.get("visibility", "unknown")
        if not isinstance(visibility, str) or visibility not in {"clear", "occluded", "unknown"}:
            return False, object_ids, visibility_by_id
        object_ids.append(item["id"])
        visibility_by_id[item["id"]] = visibility
    if len(set(object_ids)) != len(object_ids):
        return False, object_ids, visibility_by_id
    known_ids = set(object_ids)
    covered_ids = set()
    for object_id in visibility_coverage:
        if not _is_object_id(object_id) or object_id not in known_ids or object_id in covered_ids:
            return False, object_ids, visibility_by_id
        covered_ids.add(object_id)
    obstructed_ids = set()
    for relation in observation["relations"]:
        if not isinstance(relation, Mapping):
            return False, object_ids, visibility_by_id
        blocker, blocked = relation.get("blocker"), relation.get("blocked")
        confidence = relation.get("confidence")
        if (
            not _is_object_id(blocker)
            or not _is_object_id(blocked)
            or blocker == blocked
            or blocker not in known_ids
            or blocked not in known_ids
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
            or not math.isfinite(confidence)
        ):
            return False, object_ids, visibility_by_id
        if visibility_by_id[blocked] == "clear":
            return False, object_ids, visibility_by_id
        obstructed_ids.add(blocked)
    if observation["relation_coverage"] == "complete" and any(
        visibility == "occluded" and object_id not in obstructed_ids
        for object_id, visibility in visibility_by_id.items()
    ):
        return False, object_ids, visibility_by_id
    evidenced_visibility = {
        object_id: visibility if visibility == "unknown" or object_id in covered_ids else "unknown"
        for object_id, visibility in visibility_by_id.items()
    }
    return True, object_ids, evidenced_visibility


def _valid_feasibility(feasibility, object_ids):
    if not isinstance(feasibility, Mapping):
        return False
    if not isinstance(feasibility.get("observation_id"), str) or not feasibility["observation_id"].strip():
        return False
    if not isinstance(feasibility.get("coordinate_frame"), str) or not feasibility["coordinate_frame"].strip():
        return False
    records = feasibility.get("objects")
    if not isinstance(records, list):
        return False
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            return False
        object_id, status = record.get("object_id"), record.get("status")
        if (
            not _is_object_id(object_id)
            or object_id not in object_ids
            or object_id in seen
            or not isinstance(status, str)
            or status not in {"feasible", "infeasible", "unknown"}
        ):
            return False
        seen.add(object_id)
    return seen == set(object_ids)


def _decision(status, observation_id, reason, object_ids=()):
    return {
        "status": status,
        "next_object_id": None,
        "observation_id": observation_id,
        "reason": reason,
        "diagnostics": [{"object_id": object_id, "reason": reason} for object_id in object_ids],
    }


def select_next_object(observation, target_id, feasibility, scoring_config):
    """Return a fail-closed decision until relation gating and scoring exist."""
    observation_id = (
        observation.get("observation_id")
        if isinstance(observation, Mapping) and isinstance(observation.get("observation_id"), str)
        else None
    )
    valid, object_ids, visibility_by_id = _valid_observation(observation)
    if not valid:
        return _decision("blocked", observation_id, "invalid_observation_schema", object_ids)
    if observation.get("is_stale") is True:
        return _decision("blocked", observation_id, "stale_observation", object_ids)
    if not _is_object_id(target_id):
        return _decision("blocked", observation_id, "invalid_target_id", object_ids)
    if target_id not in object_ids:
        return _decision("blocked", observation_id, "target_missing", object_ids)
    if not isinstance(feasibility, Mapping) or feasibility.get("distance_unit") != "m":
        return _decision("blocked", observation_id, "unsupported_distance_unit", object_ids)
    if not _valid_feasibility(feasibility, object_ids):
        return _decision("blocked", observation_id, "invalid_feasibility_schema", object_ids)
    if feasibility["observation_id"] != observation_id:
        return _decision("blocked", observation_id, "feasibility_observation_mismatch", object_ids)
    if not isinstance(scoring_config, Mapping) or scoring_config:
        return _decision("blocked", observation_id, "unsupported_scoring_config", object_ids)
    decision = _decision("uncertain", observation_id, "selection_deferred_until_t02_t03", object_ids)
    for diagnostic in decision["diagnostics"]:
        if visibility_by_id[diagnostic["object_id"]] == "unknown":
            diagnostic["reason"] = "visibility_unknown"
    return decision