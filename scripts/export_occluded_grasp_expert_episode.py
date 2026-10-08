#!/usr/bin/env python3
"""Import a verified successful OccludedGrasp rollout into a Pi05-DROID LeRobot dataset.

Manifest schema:
{
  "success": true,
  "task": "Remove the blocker and grasp the target.",
  "frames": [{
    "exterior_image_1_left": "frames/0000_external.png",
    "wrist_image_left": "frames/0000_wrist.png",
    "state": [joint1..joint7, gripper_position],
    "action": [velocity1..velocity7, gripper_position]
  }]
}
Actions are the unnormalized 15-Hz DROID contract: seven Panda joint velocities
and one gripper position. The exporter deliberately refuses unsuccessful episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")


def validate_episode(episode: dict[str, Any], manifest_dir: Path) -> list[dict[str, Any]]:
    if episode.get("success") is not True:
        raise ValueError("refusing to export an episode unless success is exactly true")
    task = episode.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("episode task must be a non-empty string")
    frames = episode.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("a successful episode must contain at least two frames")

    validated = []
    expected_shape = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        state = np.asarray(frame.get("state"), dtype=np.float32)
        action = np.asarray(frame.get("action"), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"frames[{index}].state must be finite shape (8,) (7 joints + gripper)")
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"frames[{index}].action must be finite shape (8,) (7 velocities + gripper)")
        output = {"joint_position": state[:7], "gripper_position": state[7:8], "actions": action, "task": task}
        for key in IMAGE_KEYS:
            value = frame.get(key)
            if not isinstance(value, str):
                raise ValueError(f"frames[{index}].{key} must be an image path")
            image_path = (manifest_dir / value).resolve()
            if not image_path.is_file():
                raise ValueError(f"frames[{index}].{key} does not exist: {image_path}")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if expected_shape is None:
                expected_shape = rgb.shape
            elif rgb.shape != expected_shape:
                raise ValueError(f"frames[{index}].{key} has shape {rgb.shape}, expected {expected_shape}")
            output[key] = rgb
        # The stock pi05_droid repack transform requires a second exterior view.
        # The current simulator provides one external camera, so duplicate it
        # explicitly for schema compatibility; this is not a true second view.
        output["exterior_image_2_left"] = output["exterior_image_1_left"].copy()
        validated.append(output)
    return validated


def export_episode(manifest_path: Path, repo_id: str, output_root: Path) -> int:
    with manifest_path.open(encoding="utf-8") as stream:
        episode = json.load(stream)
    frames = validate_episode(episode, manifest_path.parent)
    image_shape = frames[0]["exterior_image_1_left"].shape
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    for frame in frames:
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSON successful-episode manifest")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset id, e.g. user/occluded-grasp-expert")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    count = export_episode(args.manifest, args.repo_id, args.output_root)
    print(json.dumps({"success": True, "frames_exported": count, "repo_id": args.repo_id, "output_root": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
