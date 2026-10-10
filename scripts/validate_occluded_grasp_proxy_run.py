#!/usr/bin/env python3
"""Validate a completed, low-fidelity occluded-grasp proxy-loop run.

This checks software-chain artifacts only. Oracle proxy relocation is explicitly
not accepted as a physical grasp or successful expert demonstration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_run(run_dir: Path, expected_cycles: int | None = None) -> dict[str, Any]:
    summary_path = run_dir / "feedback_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"feedback summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cycle_count = int(summary.get("evaluation", {}).get("cycles", -1))
    if cycle_count <= 0 or (expected_cycles is not None and cycle_count != expected_cycles):
        raise ValueError(f"unexpected completed cycle count: {cycle_count}")

    selected_objects: list[str] = []
    cycle_summaries = summary.get("cycles")
    if not isinstance(cycle_summaries, list) or len(cycle_summaries) != cycle_count:
        raise ValueError("feedback summary must contain one record per completed cycle")

    for index in range(cycle_count):
        cycle_dir = run_dir / f"cycle_{index:03d}"
        result_path = cycle_dir / "cycle_result.json"
        required_files = (
            "rgb.png", "wrist_rgb.png", "depth.npy", "instance_segmentation.npy",
            "relations/occ_detail.json", "ranked/ranked_actions.json",
            "pi05_summary.json", "pi05_actions.npy",
        )
        missing = [name for name in required_files if not (cycle_dir / name).is_file()]
        if missing or not result_path.is_file():
            raise ValueError(f"cycle {index} is incomplete; missing {missing or [result_path.name]}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        ranked = json.loads((cycle_dir / "ranked/ranked_actions.json").read_text(encoding="utf-8"))
        actions = np.load(cycle_dir / "pi05_actions.npy")
        if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
            raise ValueError(f"cycle {index} Pi05 actions must be finite [T,8], got {actions.shape}")
        if not ranked.get("ranked_actions"):
            raise ValueError(f"cycle {index} has no obstruction-ranked action")
        selected = result.get("selected", {}).get("object_name")
        ranked_first = ranked["ranked_actions"][0].get("object_name")
        if selected != ranked_first:
            raise ValueError(f"cycle {index} selection {selected!r} differs from ranker top choice {ranked_first!r}")
        execution = result.get("pi05_execution")
        if not isinstance(execution, dict) or execution.get("execution_mode") != "kinematic_proxy":
            raise ValueError(f"cycle {index} did not use the kinematic proxy action executor")
        if execution.get("proxy_relocated_selected_object") is not True:
            raise ValueError(f"cycle {index} did not apply the configured proxy scene transition")
        if execution.get("proxy_transition_is_not_grasp") is not True or result.get("grasp_success") is not False:
            raise ValueError(f"cycle {index} incorrectly represents a proxy transition as a grasp success")
        if execution.get("actions_executed", 0) < 1:
            raise ValueError(f"cycle {index} did not execute a Pi05 action")
        selected_objects.append(str(selected))

    if cycle_count > 1 and len(set(selected_objects)) < 2:
        raise ValueError("scene changed but the selected action did not change across cycles")
    return {
        "valid": True,
        "run_dir": str(run_dir),
        "cycles": cycle_count,
        "selected_objects": selected_objects,
        "chain": ["simulated capture", "obstruction inference", "ranked selection", "Pi05 inference",
                  "kinematic action integration", "proxy-only scene transition", "reobserve and rerank"],
        "physical_grasp_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run_dir, args.expected_cycles), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
