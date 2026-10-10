import json

import numpy as np
import pytest

from validate_occluded_grasp_proxy_run import validate_run


def _write_run(tmp_path, *, false_success=False):
    cycles = []
    for index, selected in enumerate(("RedBoxProxy", "BlueBoxProxy")):
        cycle = tmp_path / f"cycle_{index:03d}"
        (cycle / "relations").mkdir(parents=True)
        (cycle / "ranked").mkdir(parents=True)
        for name in ("rgb.png", "wrist_rgb.png", "depth.npy", "instance_segmentation.npy"):
            (cycle / name).touch()
        (cycle / "relations" / "occ_detail.json").write_text("[]", encoding="utf-8")
        (cycle / "ranked" / "ranked_actions.json").write_text(
            json.dumps({"ranked_actions": [{"object_name": selected}]}), encoding="utf-8"
        )
        (cycle / "pi05_summary.json").write_text("{}", encoding="utf-8")
        np.save(cycle / "pi05_actions.npy", np.zeros((15, 8), dtype=np.float32))
        result = {
            "selected": {"object_name": selected},
            "pi05_execution": {
                "execution_mode": "kinematic_proxy",
                "proxy_relocated_selected_object": True,
                "proxy_transition_is_not_grasp": True,
                "actions_executed": 1,
            },
            "grasp_success": bool(false_success),
        }
        (cycle / "cycle_result.json").write_text(json.dumps(result), encoding="utf-8")
        cycles.append(result)
    (tmp_path / "feedback_summary.json").write_text(
        json.dumps({"evaluation": {"cycles": len(cycles)}, "cycles": cycles}), encoding="utf-8"
    )


def test_accepts_complete_proxy_loop_but_not_physical_claim(tmp_path):
    _write_run(tmp_path)
    result = validate_run(tmp_path, expected_cycles=2)
    assert result["valid"] is True
    assert result["selected_objects"] == ["RedBoxProxy", "BlueBoxProxy"]
    assert result["physical_grasp_verified"] is False


def test_rejects_proxy_transition_mislabeled_as_grasp(tmp_path):
    _write_run(tmp_path, false_success=True)
    with pytest.raises(ValueError, match="proxy transition"):
        validate_run(tmp_path)
