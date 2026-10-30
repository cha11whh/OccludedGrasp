"""Uncalibrated frontal-approach feasibility features for paired-arm experiments.

This is a geometric/kinematic proxy, not a grasp-success probability or a
replacement for collision-aware trajectory planning. All distances are metres.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

OBJECT_SIZES: dict[str, tuple[float, float, float]] = {
    "BlueBoxProxy": (0.25, 0.18, 0.22),
    "RedBoxProxy": (0.19, 0.18, 0.28),
    "YellowBoxProxy": (0.24, 0.18, 0.23),
    "MilkCartonProxy": (0.105, 0.105, 0.19),
    "TissueBoxProxy": (0.16, 0.12, 0.095),
    "GreenMugProxy": (0.11, 0.11, 0.14),
    "WaterBottleProxy": (0.084, 0.084, 0.28),
    "SoupCanProxy": (0.086, 0.086, 0.13),
    "BowlProxy": (0.17, 0.17, 0.055),
    "DetergentBottleProxy": (0.116, 0.116, 0.22),
}

# Conservative Panda workspace screening bounds calibrated from the current
# Isaac paired IK grid (left-side and centre positions for fitting, right-side
# mirrored positions held out). These are not manufacturer reach limits.
WORKSPACE_LIMITS = {
    "forward_m": (0.08, 0.95),
    "lateral_abs_m": 0.48,
    "height_relative_to_shoulder_m": (-0.48, 0.15),
    "radial_m": (0.20, 1.02),
}


def segment_intersects_aabb(start: Sequence[float], end: Sequence[float],
                             center: Sequence[float], size: Sequence[float],
                             padding: float = 0.0) -> bool:
    """Return whether a 3-D line segment intersects a padded axis-aligned box."""
    lower = [float(c) - float(s) / 2.0 - padding for c, s in zip(center, size, strict=True)]
    upper = [float(c) + float(s) / 2.0 + padding for c, s in zip(center, size, strict=True)]
    t_min, t_max = 0.0, 1.0
    for origin, destination, lo, hi in zip(start, end, lower, upper, strict=True):
        delta = float(destination) - float(origin)
        if abs(delta) < 1e-9:
            if float(origin) < lo or float(origin) > hi:
                return False
            continue
        t1, t2 = (lo - float(origin)) / delta, (hi - float(origin)) / delta
        t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True


def score_frontal_affordance(*, object_name: str, object_position: Sequence[float],
                             arm_base_position: Sequence[float], arm_yaw_rad: float,
                             hand_position: Sequence[float], joint_positions: Sequence[float],
                             joint_limits: Sequence[Sequence[float]],
                             other_objects: Sequence[Mapping[str, Any]],
                             approach_distance: float = 0.12, corridor_padding: float = 0.035,
                             shoulder_height: float = 0.333) -> dict[str, Any]:
    """Score a centred, cabinet-front approach for one object/arm pair.

    Screens gross workspace and straight finger-corridor feasibility only; it
    does not check IK convergence, swept-arm collision, contact quality or policy success.
    """
    if object_name not in OBJECT_SIZES:
        raise KeyError(f"unknown affordance proxy object: {object_name}")
    if any(len(value) != 3 for value in (object_position, arm_base_position, hand_position)):
        raise ValueError("object, arm base and hand positions must each be 3-D")
    if len(joint_positions) != len(joint_limits):
        raise ValueError("joint position and joint limit counts must match")
    sx, sy, sz = OBJECT_SIZES[object_name]
    center = tuple(float(value) for value in object_position)
    # The cabinet opens toward world -Y; the permitted approach is inward along +Y.
    contact = (center[0], center[1] - sy / 2.0, center[2])
    pregrasp = (contact[0], contact[1] - approach_distance, contact[2])
    blocking_objects = []
    supported_objects_above = []
    for item in other_objects:
        name = str(item["name"])
        if name == object_name:
            continue
        size = item.get("size", OBJECT_SIZES.get(name))
        if size is None:
            raise KeyError(f"missing proxy bounds for obstacle {name}")
        if segment_intersects_aabb(pregrasp, contact, item["position"], size,
                                   padding=corridor_padding):
            blocking_objects.append(name)
        # Do not extract a lower item sideways while it physically supports
        # another object; this proxy flags the unstable-removal dependency.
        item_center = tuple(float(value) for value in item["position"])
        item_size = tuple(float(value) for value in size)
        vertical_gap = item_center[2] - item_size[2] / 2.0 - (center[2] + sz / 2.0)
        overlap_x = max(0.0, min(center[0] + sx / 2.0, item_center[0] + item_size[0] / 2.0)
                        - max(center[0] - sx / 2.0, item_center[0] - item_size[0] / 2.0))
        overlap_y = max(0.0, min(center[1] + sy / 2.0, item_center[1] + item_size[1] / 2.0)
                        - max(center[1] - sy / 2.0, item_center[1] - item_size[1] / 2.0))
        footprint_overlap = overlap_x * overlap_y / max(1e-9, min(sx * sy, item_size[0] * item_size[1]))
        if -0.015 <= vertical_gap <= 0.025 and footprint_overlap >= 0.15:
            supported_objects_above.append(name)

    dx, dy, dz = (pregrasp[i] - float(arm_base_position[i]) for i in range(3))
    cosine, sine = math.cos(arm_yaw_rad), math.sin(arm_yaw_rad)
    local_forward = cosine * dx + sine * dy
    local_lateral = -sine * dx + cosine * dy
    local_height = dz - shoulder_height
    radial_reach = math.sqrt(local_forward**2 + local_lateral**2 + local_height**2)
    # Conservative screened envelope; report these features separately from IK
    # reach and from corridor/support feasibility.
    workspace_feasible = (
        WORKSPACE_LIMITS["forward_m"][0] <= local_forward <= WORKSPACE_LIMITS["forward_m"][1]
        and abs(local_lateral) <= WORKSPACE_LIMITS["lateral_abs_m"]
        and WORKSPACE_LIMITS["height_relative_to_shoulder_m"][0] <= local_height
        <= WORKSPACE_LIMITS["height_relative_to_shoulder_m"][1]
        and WORKSPACE_LIMITS["radial_m"][0] <= radial_reach <= WORKSPACE_LIMITS["radial_m"][1]
    )
    hand_distance = math.sqrt(sum((pregrasp[i] - float(hand_position[i])) ** 2 for i in range(3)))
    margins = []
    for position, limits in zip(joint_positions, joint_limits, strict=True):
        lo, hi = float(limits[0]), float(limits[1])
        if hi <= lo:
            raise ValueError("joint upper limits must exceed lower limits")
        margins.append(min(float(position) - lo, hi - float(position)) / (hi - lo))
    joint_margin = max(0.0, min(1.0, 2.0 * min(margins))) if margins else 0.0
    corridor_clear = not blocking_objects
    removal_stable = not supported_objects_above
    feasible = workspace_feasible and corridor_clear and removal_stable
    score = math.exp(-hand_distance / 0.65) * (0.5 + 0.5 * joint_margin) if feasible else 0.0
    return {
        "object_name": object_name,
        "front_approach_world": {"pregrasp": list(pregrasp), "contact": list(contact)},
        "arm_base_position": list(map(float, arm_base_position)),
        "arm_local_pregrasp": [local_forward, local_lateral, local_height],
        "workspace_limits_used": WORKSPACE_LIMITS,
        "radial_reach_m": radial_reach,
        "current_hand_to_pregrasp_m": hand_distance,
        "joint_limit_margin": joint_margin,
        "workspace_feasible_proxy": workspace_feasible,
        "front_corridor_clear_proxy": corridor_clear,
        "blocking_objects": blocking_objects,
        "supported_objects_above": supported_objects_above,
        "stable_to_remove_proxy": removal_stable,
        "feasible_proxy": feasible,
        "score": score,
        "score_kind": "uncalibrated_geometric_kinematic_proxy_not_success_probability",
    }