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


def validate_snapshot(observation, feasibility):
    """Validate a fresh snapshot without requiring a target to remain visible."""
    valid, object_ids, _ = _valid_observation(observation)
    if not valid:
        return "invalid_observation_schema"
    if observation["is_stale"]:
        return "stale_observation"
    if not isinstance(feasibility, Mapping) or feasibility.get("distance_unit") != "m":
        return "unsupported_distance_unit"
    if not _valid_feasibility(feasibility, object_ids):
        return "invalid_feasibility_schema"
    if feasibility["observation_id"] != observation["observation_id"]:
        return "feasibility_observation_mismatch"
    return None


def _decision(status, observation_id, reason, object_ids=()):
    return {
        "status": status,
        "next_object_id": None,
        "observation_id": observation_id,
        "reason": reason,
        "legal_candidate_ids": [],
        "diagnostics": [{"object_id": object_id, "reason": reason} for object_id in object_ids],
    }


def _bounded_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return 0 <= value <= 1 and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def select_next_object(observation, target_id, feasibility, scoring_config):
    """Apply shared relation/feasibility gates, then a frozen candidate ordering."""
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
    if not isinstance(scoring_config, Mapping):
        return _decision("blocked", observation_id, "unsupported_scoring_config", object_ids)
    if not scoring_config:
        scoring_mode = "fixed"
    elif (
        set(scoring_config) == {"mode"}
        and isinstance(scoring_config.get("mode"), str)
        and scoring_config["mode"] in {"fixed", "feasibility", "utility"}
    ):
        scoring_mode = scoring_config["mode"]
    else:
        return _decision("blocked", observation_id, "unsupported_scoring_config", object_ids)
    score_formula = {
        "fixed": "ascending_object_id",
        "feasibility": "candidate_quality - execution_cost",
        "utility": "candidate_quality - execution_cost + dependency_reach",
    }[scoring_mode]
    blockers_by_id = {object_id: set() for object_id in object_ids}
    successors_by_id = {object_id: set() for object_id in object_ids}
    incoming = set()
    for relation in observation["relations"]:
        blockers_by_id[relation["blocked"]].add(relation["blocker"])
        successors_by_id[relation["blocker"]].add(relation["blocked"])
        incoming.add(relation["blocked"])
    feasibility_by_id = {
        record["object_id"]: record["status"] for record in feasibility["objects"]
    }
    required_ids = set()
    pending = [target_id]
    while pending:
        object_id = pending.pop()
        if object_id in required_ids:
            continue
        required_ids.add(object_id)
        pending.extend(blockers_by_id[object_id] - required_ids)

    visiting = set()
    visited = set()

    def has_cycle(object_id):
        if object_id in visiting:
            return True
        if object_id in visited:
            return False
        visiting.add(object_id)
        if any(has_cycle(blocker_id) for blocker_id in blockers_by_id[object_id]):
            return True
        visiting.remove(object_id)
        visited.add(object_id)
        return False

    if any(has_cycle(object_id) for object_id in required_ids):
        decision = _decision("blocked", observation_id, "dependency_cycle")
        decision["legal_candidate_ids"] = []
        decision["diagnostics"] = [
            {
                "object_id": object_id,
                "reason": "blocked_by_dependency_cycle" if object_id in required_ids else "not_in_target_dependencies",
            }
            for object_id in object_ids
        ]
        return decision

    legal_candidate_ids = sorted(
        object_id
        for object_id in required_ids
        if object_id not in incoming
        and visibility_by_id[object_id] == "clear"
        and feasibility_by_id[object_id] == "feasible"
    )
    diagnostics = []
    for object_id in object_ids:
        if visibility_by_id[object_id] == "unknown":
            reason = "visibility_unknown"
        elif object_id not in required_ids:
            reason = "not_in_target_dependencies"
        elif object_id in incoming:
            reason = "blocked_by_required_predecessor"
        elif feasibility_by_id[object_id] != "feasible":
            reason = f"feasibility_{feasibility_by_id[object_id]}"
        elif observation["relation_coverage"] != "complete":
            reason = "relation_coverage_incomplete"
        else:
            reason = "legal_candidate"
        diagnostics.append({"object_id": object_id, "reason": reason})

    if observation["relation_coverage"] != "complete":
        legal_candidate_ids = []
    scoring_records = {
        record["object_id"]: record for record in feasibility["objects"]
    }
    score_by_id = {}
    dependency_reach_by_id = {}
    if scoring_mode == "utility":
        denominator = max(1, len(required_ids) - 1)
        for object_id in legal_candidate_ids:
            descendants = set()
            pending = list(successors_by_id[object_id] & required_ids)
            while pending:
                descendant_id = pending.pop()
                if descendant_id in descendants:
                    continue
                descendants.add(descendant_id)
                pending.extend(successors_by_id[descendant_id] & required_ids - descendants)
            dependency_reach_by_id[object_id] = len(descendants) / denominator
    if scoring_mode in {"feasibility", "utility"}:
        for object_id in legal_candidate_ids:
            record = scoring_records[object_id]
            quality = record.get("candidate_quality")
            cost = record.get("execution_cost")
            if _bounded_score(quality) and _bounded_score(cost):
                score_by_id[object_id] = (
                    quality - cost + dependency_reach_by_id.get(object_id, 0.0)
                )
    scores_unavailable = scoring_mode in {"feasibility", "utility"} and any(
        object_id not in score_by_id for object_id in legal_candidate_ids
    )
    selected_id = (
        min(legal_candidate_ids, key=lambda object_id: (-score_by_id[object_id], object_id))
        if scoring_mode in {"feasibility", "utility"} and score_by_id and not scores_unavailable
        else legal_candidate_ids[0] if scoring_mode == "fixed" and legal_candidate_ids
        else None
    )
    uncertain = any(
        item["reason"] in {"visibility_unknown", "feasibility_unknown", "relation_coverage_incomplete"}
        for item in diagnostics
        if item["object_id"] in required_ids
    ) or scores_unavailable
    status = "selected" if selected_id is not None else "uncertain" if uncertain else "blocked"
    reason = (
        "target_clear" if selected_id == target_id
        else "dependency_clearance_required" if selected_id is not None
        else "candidate_scores_unavailable" if scores_unavailable
        else "target_dependency_uncertain" if uncertain
        else "no_legal_candidate"
    )
    decision = _decision(status, observation_id, reason)
    decision["next_object_id"] = selected_id
    decision["legal_candidate_ids"] = legal_candidate_ids
    decision["diagnostics"] = diagnostics
    decision["scoring_mode"] = scoring_mode
    decision["score_formula"] = score_formula
    if scoring_mode == "utility":
        decision["dependency_reach_semantics"] = (
            "unique_transitive_successors_in_target_dependency_closure; structural_only; "
            "does_not_assert_that_AND_prerequisites_are_unlocked"
        )
    if scoring_mode in {"feasibility", "utility"}:
        for item in diagnostics:
            object_id = item["object_id"]
            record = scoring_records[object_id]
            quality = record.get("candidate_quality")
            cost = record.get("execution_cost")
            item["candidate_quality"] = quality if _bounded_score(quality) else None
            item["execution_cost"] = cost if _bounded_score(cost) else None
            item["dependency_reach"] = dependency_reach_by_id.get(object_id)
            item["total_score"] = score_by_id.get(object_id)
            if object_id in legal_candidate_ids and object_id not in score_by_id:
                item["reason"] = (
                    "candidate_score_missing"
                    if quality is None or cost is None
                    else "candidate_score_invalid"
                )
    return decision