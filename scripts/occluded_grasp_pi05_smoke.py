#!/usr/bin/env python3
"""Smoke-test the OccludedGrasp to official Pi05 observation bridge."""

import argparse
import json
from pathlib import Path
import sys
REPO_SRC = (Path(__file__).resolve().parents[1] / "src").resolve()
sys.path.insert(0, str(REPO_SRC))

import numpy as np

from openpi.policies.occluded_grasp_adapter import build_pi05_observation, load_ranked_actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-actions", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--wrist-image", type=Path)
    parser.add_argument("--state-dim", type=int, default=8, help="DROID-compatible state width: 7 joints + 1 gripper.")
    parser.add_argument("--state", type=Path, help="Optional .npy file containing 7 joint positions and normalized gripper position.")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--actions-out", type=Path, help="Optional .npy path to save inferred [T, 8] actions for a downstream simulator adapter.")
    args = parser.parse_args()
    if args.state_dim != 8:
        parser.error("--state-dim must be 8 (7 joint positions + 1 gripper position) for pi05_droid")
    ranked_actions = load_ranked_actions(args.ranked_actions)
    image = args.image if args.image else np.zeros((224, 224, 3), dtype=np.uint8)
    robot_state = np.load(args.state).astype(np.float32) if args.state else np.zeros(args.state_dim, dtype=np.float32)
    if robot_state.shape != (8,) or not np.isfinite(robot_state).all():
        parser.error(f"--state must contain a finite float array with shape (8,), got {robot_state.shape}")
    observation = build_pi05_observation(
        ranked_actions, image, robot_state=robot_state, arm=args.arm, wrist_image=args.wrist_image
    )
    selected = ranked_actions["ranked_actions"][0]
    image_array = observation["observation/exterior_image_1_left"]
    wrist_array = observation["observation/wrist_image_left"]
    state_array = np.concatenate((observation["observation/joint_position"], observation["observation/gripper_position"]))
    print(f"prompt: {observation['prompt']}")
    print(f"selected object: {selected['object_id']}")
    print(f"image shape: {image_array.shape}, dtype: {image_array.dtype}")
    print(f"state shape: {state_array.shape}, dtype: {state_array.dtype}")
    summary = {"prompt": observation["prompt"], "selected_object_id": selected["object_id"], "image_shape": list(image_array.shape), "image_dtype": str(image_array.dtype),
               "wrist_image_shape": list(wrist_array.shape), "wrist_image_nonzero_fraction": float(np.mean(wrist_array != 0)), "state_shape": list(state_array.shape), "state_dtype": str(state_array.dtype), "state": state_array.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.actions_out and not args.checkpoint:
        parser.error("--actions-out requires --checkpoint")
    if args.checkpoint:
        actions = _run_checkpoint(args.checkpoint, observation)
        if args.actions_out:
            args.actions_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.actions_out, actions)
            print(f"saved actions: {args.actions_out} shape={actions.shape}")
    return 0


def _run_checkpoint(checkpoint: Path, bridge_observation: dict) -> np.ndarray:
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        policy = policy_config.create_trained_policy(training_config.get_config("pi05_droid"), checkpoint)
        result = policy.infer(bridge_observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
            raise ValueError(f"expected finite Pi05 DROID actions with shape [T, 8], got {actions.shape}")
        print(f"checkpoint inference actions shape: {actions.shape}")
        return actions
    except Exception as exc:
        raise RuntimeError(f"checkpoint inference failed for {checkpoint}; provide a local compatible Pi05 checkpoint (no checkpoint is downloaded by the default smoke): {exc}") from exc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
