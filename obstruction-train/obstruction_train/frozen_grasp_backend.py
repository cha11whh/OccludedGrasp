"""CPU contract for object-assigned frozen grasp proposals, not robot feasibility.

The injected predictor accepts (camera_points_m, object_ids) and returns the
four object-indexed dictionaries produced by Contact-GraspNet's estimator.
A future model-owning worker must bind its masks/segments to these same IDs.
This module neither loads weights nor enforces a subprocess deadline; production
workers require the bounded runner. Test predictors are not model evidence.
"""

from collections.abc import Mapping

import numpy as np


def _real_array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("real numeric arrays required; no implicit coercion")
    return array.astype(np.float64, copy=False)


def predict_candidates(points, object_ids, *, observation_id, distance_unit,
                       coordinate_frame, model_version, predictor):
    result = {
        "status": "unknown",
        "reason": None,
        "observation_id": observation_id,
        "model_version": model_version,
        "distance_unit": distance_unit,
        "coordinate_frame": coordinate_frame,
        "candidates": [],
    }
    if (
        distance_unit != "m" or coordinate_frame != "camera"
        or not isinstance(observation_id, str) or not observation_id.strip()
        or not isinstance(model_version, str) or not model_version.strip()
        or not isinstance(object_ids, (list, tuple)) or not object_ids
        or any(type(value) is not int or value <= 0 for value in object_ids)
        or len(set(object_ids)) != len(object_ids) or not callable(predictor)
    ):
        result["reason"] = "invalid_request"
        return result
    try:
        cloud = _real_array(points)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
            raise ValueError("invalid camera point cloud")
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "invalid_point_cloud"
        return result
    try:
        output = predictor(cloud.copy(), tuple(object_ids))
    except TimeoutError:
        result["reason"] = "backend_timeout"
        return result
    except Exception:
        result["reason"] = "backend_error"
        return result
    try:
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise ValueError("expected four dictionaries")
        poses, scores, contacts, widths = output
        if any(not isinstance(item, Mapping) for item in output):
            raise ValueError("expected object-indexed output")
        keys = set(poses)
        if any(set(item) != keys for item in output):
            raise ValueError("unaligned output keys")
        if any(type(key) is not int or key not in object_ids for key in keys):
            raise ValueError("unassigned or unexpected object")
        candidates = []
        for object_id in sorted(keys):
            object_poses = _real_array(poses[object_id])
            object_scores = _real_array(scores[object_id])
            object_contacts = _real_array(contacts[object_id])
            object_widths = _real_array(widths[object_id])
            count = len(object_scores)
            if (object_poses.shape != (count, 4, 4)
                or object_scores.shape != (count,)
                or object_contacts.shape != (count, 3)
                or object_widths.shape != (count,)):
                raise ValueError("unaligned candidate arrays")
            if any(not np.isfinite(array).all() for array in (
                object_poses, object_scores, object_contacts, object_widths
            )):
                raise ValueError("nonfinite candidate")
            if np.any((object_scores < 0) | (object_scores > 1)) or np.any(object_widths <= 0):
                raise ValueError("invalid score or width")
            for index in range(count):
                pose = object_poses[index]
                rotation = pose[:3, :3]
                if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0)):
                    raise ValueError("invalid rigid pose")
                candidates.append({
                    "object_id": object_id, "pose_camera": pose.tolist(),
                    "score": float(object_scores[index]),
                    "width": float(object_widths[index]),
                    "contact_camera": object_contacts[index].tolist(),
                })
    except (TypeError, ValueError, OverflowError, KeyError):
        result["reason"] = "invalid_backend_output"
        return result
    result["candidates"] = candidates
    result["status"] = "candidates_generated" if candidates else "unknown"
    result["reason"] = "proposals_only_not_feasibility" if candidates else "no_candidates_found"
    return result