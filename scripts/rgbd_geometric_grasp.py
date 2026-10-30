"""Fast, uncalibrated RGB-D planar grasp candidate heuristic (no training)."""
from __future__ import annotations

from math import cos, exp, pi, sin
from typing import Any

import numpy as np

SCORE_KIND = "uncalibrated_rgbd_antipodal_geometry_heuristic_v1"


def rank_rgbd_grasp_candidates(
    depth_m: np.ndarray,
    instance_mask: np.ndarray,
    camera_intrinsics: np.ndarray,
    *,
    gripper_min_width_m: float = 0.015,
    gripper_max_width_m: float = 0.080,
    z_range_m: tuple[float, float] = (0.2, 2.2),
    candidate_limit: int = 96,
    jaw_bandwidth_m: float = 0.008,
    depth_tolerance_m: float = 0.015,
    min_contact_points: int = 2,
    top_k: int = 20,
) -> dict[str, Any]:
    """Rank image-plane antipodal candidates; scores are utilities, not probabilities."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(instance_mask, dtype=bool)
    K = np.asarray(camera_intrinsics, dtype=np.float64)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and mask must be same-sized 2D arrays")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("camera_intrinsics must be a finite 3x3 matrix with positive focal lengths")
    if not (0 < gripper_min_width_m < gripper_max_width_m):
        raise ValueError("gripper widths must satisfy 0 < min < max")
    if not (0 < z_range_m[0] < z_range_m[1]):
        raise ValueError("z_range_m must satisfy 0 < near < far")
    if candidate_limit <= 0 or top_k <= 0 or min_contact_points <= 0:
        raise ValueError("candidate_limit, top_k, and min_contact_points must be positive")
    if jaw_bandwidth_m <= 0 or depth_tolerance_m <= 0:
        raise ValueError("jaw_bandwidth_m and depth_tolerance_m must be positive")

    rows, cols = np.nonzero(mask & np.isfinite(depth) & (depth >= z_range_m[0]) & (depth <= z_range_m[1]))
    if rows.size == 0:
        raise ValueError("instance mask contains no valid depth points in z_range_m")
    z = depth[rows, cols].astype(np.float64)
    points = np.column_stack(((cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z))
    sampled = np.linspace(0, len(rows) - 1, min(len(rows), candidate_limit), dtype=np.int64)
    centroid = int(np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2))
    centers = np.unique(np.append(sampled, centroid))
    min_width, max_width = float(gripper_min_width_m), float(gripper_max_width_m)
    preferred_width = (min_width + max_width) * 0.5
    width_half_range = (max_width - min_width) * 0.5
    candidates: list[dict[str, Any]] = []

    for center_index in centers:
        center = points[center_index]
        for angle in (0.0, pi / 4.0, pi / 2.0, 3.0 * pi / 4.0):
            axis = np.array((cos(angle), sin(angle)), dtype=np.float64)
            across_axis = np.array((-axis[1], axis[0]), dtype=np.float64)
            delta = points[:, :2] - center[:2]
            along, across = delta @ axis, delta @ across_axis
            strip = np.abs(across) <= jaw_bandwidth_m
            negative, positive = strip & (along < 0), strip & (along > 0)
            neg_ids, pos_ids = np.flatnonzero(negative), np.flatnonzero(positive)
            if len(neg_ids) < min_contact_points or len(pos_ids) < min_contact_points:
                continue
            left = neg_ids[int(np.argmin(along[neg_ids]))]
            right = pos_ids[int(np.argmax(along[pos_ids]))]
            left_extent, right_extent = -float(along[left]), float(along[right])
            width = left_extent + right_extent
            if not min_width <= width <= max_width:
                continue
            width_score = max(0.0, 1.0 - abs(width - preferred_width) / width_half_range)
            symmetry = max(0.0, 1.0 - abs(left_extent - right_extent) / max(width, 1e-9))
            depth_score = exp(-abs(float(points[left, 2] - points[right, 2])) / depth_tolerance_m)
            support = min(1.0, min(int(negative.sum()), int(positive.sum())) / float(min_contact_points))
            quality = float(np.clip(width_score * symmetry * depth_score * support, 0.0, 1.0))
            if quality <= 0:
                continue
            candidates.append({
                "grasp_quality_score": quality,
                "score_kind": SCORE_KIND,
                "center_pixel_xy": [int(cols[center_index]), int(rows[center_index])],
                "center_camera_xyz_m": [float(v) for v in center],
                "jaw_axis_camera_xy": [float(v) for v in axis],
                "estimated_jaw_span_m": float(width),
                "components": {"width_fit": float(width_score), "opposed_contact_symmetry": float(symmetry),
                               "contact_depth_consistency": float(depth_score), "contact_support": float(support)},
            })
    candidates.sort(key=lambda item: item["grasp_quality_score"], reverse=True)
    candidates = candidates[:top_k]
    return {
        "schema": "rgbd_geometric_grasp_candidates.v1",
        "score_kind": SCORE_KIND,
        "is_calibrated_probability": False,
        "valid_object_points": int(len(points)),
        "candidate_count": len(candidates),
        "grasp_quality_score": candidates[0]["grasp_quality_score"] if candidates else 0.0,
        "candidates": candidates,
    }
