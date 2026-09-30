import json
from pathlib import Path
import numpy as np
from typing import Any

def load_ranked_actions(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load ranked-actions JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("ranked-actions JSON must contain an object")
    _validate_ranked_actions(data)
    return data

def select_ranked_action(ranked_actions: dict[str, Any]) -> dict[str, Any]:
    _validate_ranked_actions(ranked_actions)
    return ranked_actions["ranked_actions"][0]


from typing import Any
def build_pi05_observation(ranked_actions: dict[str, Any], image: str | Path | np.ndarray,
                           robot_state: np.ndarray | list[float] | None = None,
                           arm: str = "right",
                           wrist_image: str | Path | np.ndarray | None = None) -> dict[str, Any]:
    """Build the raw observation expected by the official Pi05 DROID policy.

    ``robot_state`` uses the DROID layout: seven joint positions followed by
    one normalized gripper position, matching the Pi05-DROID policy contract.
    """
    _validate_ranked_actions(ranked_actions)
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("arm must be a non-empty string")
    selected = ranked_actions["ranked_actions"][0]
    task_mode = str(ranked_actions.get("task_mode", "target")).strip().lower()
    target = ranked_actions.get("target")
    target_id = target.get("id") if isinstance(target, dict) else ranked_actions.get("target_id")
    instruction = str(ranked_actions.get("instruction", "")).strip()
    object_id = selected["object_id"]
    object_name = str(selected.get("object_name", object_id)).strip()
    role = str(selected.get("role", "")).strip().replace("_", " ")
    if task_mode == "clear_table":
        prompt = f"Use the {arm} arm to clear the table by removing the selected objects."
    else:
        action_type = str(ranked_actions.get("action_type", "remove_blocker")).strip()
        prompt = f"Use the {arm} arm to grasp the {object_name} (object {object_id}) and {action_type.replace('_', ' ')}."
        if role:
            prompt = f"{prompt} Its scene role is {role}."
    if target_id is not None and str(target_id) != str(object_id):
        target_name = target.get("name", target_id) if isinstance(target, dict) else target_id
        prompt = f"{prompt} Clear access to target {target_name}."
    if instruction:
        prompt = f"{prompt} Task: {instruction}"
    if robot_state is None:
        state = np.zeros(8, dtype=np.float32)
    else:
        state = np.asarray(robot_state, dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"robot_state must contain 7 joint positions and 1 gripper position, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("robot_state must contain only finite values")
    base_image = _load_rgb_image(image)
    wrist_rgb = np.zeros_like(base_image) if wrist_image is None else _load_rgb_image(wrist_image)
    if wrist_rgb.shape[:2] != base_image.shape[:2]:
        raise ValueError("base and wrist images must have matching height and width")
    return {
        "observation/exterior_image_1_left": base_image,
        "observation/wrist_image_left": wrist_rgb,
        "observation/joint_position": state[:7],
        "observation/gripper_position": state[7:8],
        "prompt": prompt,
    }

def _validate_ranked_actions(data: dict[str, Any]) -> None:
    ranked = data.get("ranked_actions")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("ranked_actions must be a non-empty list")
    for index, action in enumerate(ranked):
        if not isinstance(action, dict):
            raise ValueError(f"ranked_actions[{index}] must be an object")
        if "object_id" not in action:
            raise ValueError(f"ranked_actions[{index}] requires object_id")
        if not any(key in action for key in ("policy_score", "reward", "score")):
            raise ValueError(f"ranked_actions[{index}] requires policy_score, reward, or score")


def _load_rgb_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"image path does not exist: {path}")
        if path.suffix.lower() == ".npy":
            try:
                image = np.load(path)
            except (OSError, ValueError) as exc:
                raise ValueError(f"could not load image array from {path}: {exc}") from exc
        else:
            try:
                from PIL import Image
                with Image.open(path) as pil_image:
                    image = np.asarray(pil_image.convert("RGB"))
            except ImportError as exc:
                raise ValueError("loading image paths requires Pillow, or provide a .npy image") from exc
            except (OSError, ValueError) as exc:
                raise ValueError(f"could not load RGB image from {path}: {exc}") from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must have HWC shape (height, width, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"image must have dtype uint8, got {array.dtype}")
    return np.ascontiguousarray(array)
