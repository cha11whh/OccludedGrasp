import copy
import math
from collections.abc import Mapping
from numbers import Real

from obstruction_train.relation_first_policy import select_next_object, validate_snapshot


_MAX_ATTEMPTS_PER_OBJECT = 2
_MAX_UNKNOWN_REOBSERVATIONS = 3
_MAX_SIM_SECONDS = 300.0


def _result(
    status,
    reason,
    observation,
    decision,
    attempts,
    actions,
    unknowns,
    sim_time,
    rejected_observation_id=None,
):
    observation_id = observation.get("observation_id") if isinstance(observation, Mapping) else None
    result = {
        "status": status,
        "reason": reason,
        "observation_id": observation_id,
        "decision": decision,
        "attempts": dict(attempts),
        "unknown_reobservations": unknowns,
        "sim_time": sim_time,
        "actions": actions,
    }
    if rejected_observation_id is not None:
        result["rejected_observation_id"] = rejected_observation_id
    return result


def _read_sim_time(clock, previous):
    try:
        value = clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            return None, "invalid_sim_time"
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return None, "invalid_sim_time"
        if previous is not None and value < previous:
            return None, "nonmonotonic_sim_time"
        return value, None
    except Exception:
        return None, "invalid_sim_time"


def _limited_feasibility(feasibility, attempts):
    if not isinstance(feasibility, Mapping) or not isinstance(feasibility.get("objects"), list):
        return feasibility
    if any(
        not isinstance(record, Mapping)
        or "object_id" not in record
        or "status" not in record
        for record in feasibility["objects"]
    ):
        return feasibility
    limited = dict(feasibility)
    records = []
    for record in feasibility["objects"]:
        adjusted = dict(record)
        if (
            adjusted["status"] == "feasible"
            and attempts.get(adjusted["object_id"], 0) >= _MAX_ATTEMPTS_PER_OBJECT
        ):
            adjusted["status"] = "infeasible"
        records.append(adjusted)
    limited["objects"] = records
    return limited


def _replay_graph_state(observation):
    if not isinstance(observation, Mapping):
        return None
    objects = observation.get("objects")
    relations = observation.get("relations", [])
    if not isinstance(objects, list) or not isinstance(relations, list):
        return None
    try:
        object_ids = {record["id"] for record in objects if isinstance(record, Mapping)}
        if len(object_ids) != len(objects):
            return None
        edges = {
            (relation["blocker"], relation["blocked"])
            for relation in relations
            if isinstance(relation, Mapping)
        }
        if len(edges) != len(relations):
            return None
    except (KeyError, TypeError):
        return None
    return object_ids, edges


def _observation_transition_error(graph_state, observation, known_ids, removed_ids):
    next_state = _replay_graph_state(observation)
    if next_state is None:
        return "invalid_observation_identity", graph_state
    next_ids, next_edges = next_state
    if not next_ids <= known_ids:
        return "unknown_object_identity", next_state
    if not known_ids.difference(next_ids) <= removed_ids:
        return "unconfirmed_removal_observation_mismatch", next_state
    previous_edges = graph_state[1]
    retained_edges = {
        edge
        for edge in previous_edges
        if edge[0] not in removed_ids and edge[1] not in removed_ids
    }
    if not retained_edges <= next_edges:
        return "unconfirmed_removal_observation_mismatch", next_state
    return None, next_state


def run_replay(
    initial_observation,
    target_id,
    initial_feasibility,
    executor,
    verifier,
    observer,
    sim_clock,
    scoring_config=None,
):
    """Run a CPU-only replay; callbacks supply execution, verification, observation, and sim time."""
    attempts = {}
    actions = []
    unknown_reobservations = 0
    decision = None
    observation = initial_observation
    feasibility = initial_feasibility
    sim_time = None
    if not all(callable(callback) for callback in (executor, verifier, observer, sim_clock)):
        return _result("stopped", "invalid_callback", observation, decision, attempts, actions, 0, sim_time)

    seen_observations = set()
    graph_state = _replay_graph_state(observation)
    known_object_ids = set() if graph_state is None else set(graph_state[0])
    removed_object_ids = set()
    if isinstance(observation, Mapping):
        initial_id = observation.get("observation_id")
        if isinstance(initial_id, str) and initial_id:
            seen_observations.add(initial_id)

    while True:
        sim_time, time_error = _read_sim_time(sim_clock, sim_time)
        if time_error:
            return _result("stopped", time_error, observation, decision, attempts, actions, unknown_reobservations, sim_time)
        if sim_time >= _MAX_SIM_SECONDS:
            return _result("budget_exhausted", "sim_time_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)

        try:
            decision = select_next_object(
                observation,
                target_id,
                _limited_feasibility(feasibility, attempts),
                {} if scoring_config is None else scoring_config,
            )
        except Exception:
            reason = (
                "invalid_feasibility_schema"
                if not isinstance(feasibility, Mapping)
                or not isinstance(feasibility.get("objects"), list)
                else "selector_error"
            )
            decision = {
                "status": "blocked",
                "next_object_id": None,
                "observation_id": observation.get("observation_id")
                if isinstance(observation, Mapping)
                else None,
                "reason": reason,
            }
            return _result("stopped", reason, observation, decision, attempts, actions, unknown_reobservations, sim_time)

        if decision.get("status") == "selected":
            unknown_reobservations = 0
            object_id = decision.get("next_object_id")
            observation_id = decision.get("observation_id")
            if attempts.get(object_id, 0) >= _MAX_ATTEMPTS_PER_OBJECT:
                return _result("stopped", "attempt_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            attempts[object_id] = attempts.get(object_id, 0) + 1
            try:
                execution = executor(object_id, observation_id)
            except Exception:
                return _result("stopped", "executor_error", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if not isinstance(execution, Mapping) or execution.get("object_id") != object_id:
                return _result("stopped", "executor_object_mismatch", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            action_record = {
                "object_id": object_id,
                "observation_id": observation_id,
                "execution": dict(execution),
                "verification": None,
            }
            actions.append(action_record)

            sim_time, time_error = _read_sim_time(sim_clock, sim_time)
            if time_error:
                return _result("stopped", time_error, observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if sim_time >= _MAX_SIM_SECONDS:
                return _result("budget_exhausted", "sim_time_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)

            try:
                feedback = verifier(execution, observation_id)
            except Exception:
                return _result("stopped", "verifier_error", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if (
                not isinstance(feedback, Mapping)
                or feedback.get("object_id") != object_id
                or feedback.get("observation_id") != observation_id
                or type(feedback.get("physical_removal_confirmed")) is not bool
                or type(feedback.get("placement_verified")) is not bool
                or (feedback["placement_verified"] and not feedback["physical_removal_confirmed"])
            ):
                return _result("stopped", "invalid_verification", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            action_record["verification"] = dict(feedback)

            if (
                feedback["physical_removal_confirmed"]
                and not feedback["placement_verified"]
            ):
                return _result("stopped", "placement_unverified", observation, decision, attempts, actions, unknown_reobservations, sim_time)

            sim_time, time_error = _read_sim_time(sim_clock, sim_time)
            if time_error:
                return _result("stopped", time_error, observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if sim_time >= _MAX_SIM_SECONDS:
                return _result("budget_exhausted", "sim_time_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)

            try:
                observed = observer(copy.deepcopy(observation))
            except Exception:
                return _result("stopped", "observer_error", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            sim_time, time_error = _read_sim_time(sim_clock, sim_time)
            if time_error:
                return _result("stopped", time_error, observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if sim_time >= _MAX_SIM_SECONDS:
                return _result("budget_exhausted", "sim_time_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if (
                not isinstance(observed, Mapping)
                or not isinstance(observed.get("observation"), Mapping)
                or "feasibility" not in observed
            ):
                return _result("stopped", "invalid_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            next_observation = observed["observation"]
            next_id = next_observation.get("observation_id")
            if next_observation.get("is_stale") is True:
                return _result("stopped", "stale_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if not isinstance(next_id, str) or not next_id or next_id in seen_observations:
                return _result("stopped", "repeated_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if graph_state is not None:
                if feedback["physical_removal_confirmed"]:
                    removed_object_ids.add(object_id)
                transition_error, next_graph_state = _observation_transition_error(
                    graph_state, next_observation, known_object_ids, removed_object_ids
                )
                if transition_error:
                    return _result(
                        "stopped",
                        transition_error,
                        observation,
                        decision,
                        attempts,
                        actions,
                        unknown_reobservations,
                        sim_time,
                        rejected_observation_id=next_id,
                    )
                graph_state = next_graph_state
            seen_observations.add(next_id)
            placement_verified = feedback["placement_verified"] and feedback["physical_removal_confirmed"]
            observation, feasibility = next_observation, observed["feasibility"]
            if object_id == target_id and placement_verified:
                validation_error = validate_snapshot(observation, feasibility)
                if validation_error:
                    return _result(
                        "stopped", validation_error, observation, decision,
                        attempts, actions, unknown_reobservations, sim_time,
                    )
                return _result("target_completed", "verified_target_placement", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            continue

        if decision.get("status") == "uncertain":
            if unknown_reobservations >= _MAX_UNKNOWN_REOBSERVATIONS:
                return _result("stopped", "unknown_observation_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            unknown_reobservations += 1
            try:
                observed = observer(copy.deepcopy(observation))
            except Exception:
                return _result("stopped", "observer_error", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            sim_time, time_error = _read_sim_time(sim_clock, sim_time)
            if time_error:
                return _result("stopped", time_error, observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if sim_time >= _MAX_SIM_SECONDS:
                return _result("budget_exhausted", "sim_time_limit", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if (
                not isinstance(observed, Mapping)
                or not isinstance(observed.get("observation"), Mapping)
                or "feasibility" not in observed
            ):
                return _result("stopped", "invalid_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            next_observation = observed["observation"]
            next_id = next_observation.get("observation_id")
            if next_observation.get("is_stale") is True:
                return _result("stopped", "stale_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if not isinstance(next_id, str) or not next_id or next_id in seen_observations:
                return _result("stopped", "repeated_observation", observation, decision, attempts, actions, unknown_reobservations, sim_time)
            if graph_state is not None:
                transition_error, next_graph_state = _observation_transition_error(
                    graph_state, next_observation, known_object_ids, removed_object_ids
                )
                if transition_error:
                    return _result(
                        "stopped",
                        transition_error,
                        observation,
                        decision,
                        attempts,
                        actions,
                        unknown_reobservations,
                        sim_time,
                        rejected_observation_id=next_id,
                    )
                graph_state = next_graph_state
            seen_observations.add(next_id)
            observation, feasibility = next_observation, observed["feasibility"]
            continue

        reason = decision.get("reason") or decision.get("status") or "selection_stopped"
        status = "attempt_limit" if any(count >= _MAX_ATTEMPTS_PER_OBJECT for count in attempts.values()) else "stopped"
        return _result(status, reason, observation, decision, attempts, actions, unknown_reobservations, sim_time)