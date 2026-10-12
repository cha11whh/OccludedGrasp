import numpy as np

from obstruction_train.export_isaac_proxy_pairs import _pair_rows


def _two_disjoint_visible_masks(front_depth=1.0, back_depth=1.2):
    instances = np.zeros((12, 16), dtype=np.int32)
    instances[3:9, 2:7] = 10
    instances[3:9, 8:14] = 11
    depth = np.zeros_like(instances, dtype=np.float32)
    depth[instances == 10] = front_depth
    depth[instances == 11] = back_depth
    return depth, instances


def test_designated_pair_preserves_occlusion_label_when_modal_boxes_do_not_overlap():
    depth, instances = _two_disjoint_visible_masks()
    rows = _pair_rows(
        0, "images/image_000000.png", "annotations_meta/ALL_NPZ/scene0_view0.npz",
        depth, instances, [10, 11], designated_front_back=(10, 11),
        overlap_threshold=0.03, depth_margin_m=0.015,
    )
    assert [row["label"] for row in rows] == [0, 1]
    assert [(row["obj_a"], row["obj_b"]) for row in rows] == [(10, 11), (11, 10)]
    assert rows[0]["point"] == rows[1]["point"]
    assert set(rows[0]["point"]) == {"x", "y"}


def test_designated_layout_is_rejected_when_camera_depth_disagrees():
    depth, instances = _two_disjoint_visible_masks(front_depth=1.3, back_depth=1.0)
    rows = _pair_rows(
        0, "images/image_000000.png", "annotations_meta/ALL_NPZ/scene0_view0.npz",
        depth, instances, [10, 11], designated_front_back=(10, 11),
        overlap_threshold=0.03, depth_margin_m=0.015,
    )
    assert [row["label"] for row in rows] == [2, 2]
