import numpy as np
import pytest
from rgbd_geometric_grasp import SCORE_KIND, rank_rgbd_grasp_candidates


def _scene(width_m=0.04, height_m=0.10):
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    fx = fy = 500.0
    half_x = round(width_m * fx / 2)
    half_y = round(height_m * fy / 2)
    depth[50-half_y:51+half_y, 50-half_x:51+half_x] = 1.0
    mask[50-half_y:51+half_y, 50-half_x:51+half_x] = True
    K = np.array([[fx, 0, 50], [0, fy, 50], [0, 0, 1]], dtype=np.float32)
    return depth, mask, K


def test_returns_uncalibrated_candidates_with_valid_jaw_span():
    depth, mask, K = _scene()
    result = rank_rgbd_grasp_candidates(depth, mask, K)
    assert result["score_kind"] == SCORE_KIND
    assert result["is_calibrated_probability"] is False
    assert result["candidate_count"] > 0
    assert 0.015 <= result["candidates"][0]["estimated_jaw_span_m"] <= 0.080


def test_rejects_empty_valid_depth_mask():
    depth, mask, K = _scene()
    depth[:] = np.inf
    with pytest.raises(ValueError, match="no valid depth points"):
        rank_rgbd_grasp_candidates(depth, mask, K)
