#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate occlusion-aware VQA dataset with different ablation versions:
  --style ratio_only   : include occlusion ratio only
  --style degree_only  : include natural language occlusion severity
  --style point_only   : include contact point only
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

# =====================================================
# 1. Human question templates (basemodel version)
# =====================================================
QUESTION_TEMPLATES = [
    "Which object must be removed first to grasp object {target}?",
    "I want to grasp object {target}. What should I remove first?",
    "To pick up object {target}, which object is blocking it?",
    "What is the first object I need to grasp to reach object {target}?",
    "To grasp object {target}, which object is on top of it?",
    "What is the top object that occludes object {target}?",
    "What object should be removed before grasping object {target}?",
    "Tell me the top-most object that blocks access to object {target}.",
    "Before I grasp object {target}, what must be removed?",
    "Identify the object occluding object {target}."
]

# =====================================================
# 2. System prompts for each ablation version
# =====================================================
BASE_SYSTEM_PROMPT = (
    "You are an assistant for robotic grasp planning. When asked which object must be removed first "
    "to grasp a specific object:\n\n"
    "- If the target object is not occluded, return the target object's ID itself.\n"
    "- If the object has one occlusion path, reason step-by-step along that path and end with the "
    "top-most object that must be removed first.\n"
    "- If the object has multiple occlusion paths, reason step-by-step for each path separately. "
    "In the <answer>...</answer> tag, output ALL distinct top-most objects as a JSON list.\n\n"
    "Use <think>...</think> tags for your reasoning, and put ONLY the final object IDs in <answer>...</answer>."
)

SYSTEM_PROMPTS = {
    "ratio_only": BASE_SYSTEM_PROMPT.replace(
        "reason step-by-step along that path and end with the ",
        "reason step-by-step along that path, include the occlusion ratio for each occlusion relation when available, and end with the "
    ),
    "degree_only": BASE_SYSTEM_PROMPT.replace(
        "reason step-by-step along that path and end with the ",
        "reason step-by-step along that path, describe each occlusion using natural-language severity terms such as slightly, partially, mostly, or heavily, and end with the "
    ),
    "point_only": BASE_SYSTEM_PROMPT.replace(
        "reason step-by-step along that path and end with the ",
        "reason step-by-step along that path, include the contact point using (x,y) coordinates for each occlusion relation when available, and end with the "
    ),
    "ratio_point": BASE_SYSTEM_PROMPT.replace(
    "reason step-by-step along that path and end with the ",
    "reason step-by-step along that path, include the occlusion ratio and contact point using (x,y) coordinates for each occlusion relation when available, and end with the "
    ),
    "ratio_point_short": BASE_SYSTEM_PROMPT.replace(
    "reason step-by-step along that path and end with the ",
    "reason step-by-step along that path, include the occlusion ratio and contact point using (x,y) coordinates for each occlusion relation when available, and end with the "
    ),
}

# =====================================================
# 3. Helper: build image path
# =====================================================
def make_image_path(scene_id: str, view_id: str, image_root: Optional[str]) -> str:
    scene_name = scene_id.replace("/", "_")
    filename = f"{scene_name}_{view_id}_labeled.png"
    return os.path.join(image_root, filename) if image_root else filename

# =====================================================
# 4. Load occlusion detail dictionary
# =====================================================
def load_occ_detail(path: str) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
    with open(path, "r") as f:
        occ_data = json.load(f)
    occ_map = {}
    for item in occ_data:
        key = (item["scene_id"], str(item["view_id"]), item["obj1"], item["obj2"])
        occ_map[key] = {
            "mask_ratio": item.get("mask_ratio"),
            "point": (
                item["point"]["x"],
                item["point"]["y"]
            ) if item.get("point") else (None, None),
            "mask_path": item.get("mask_path"),
        }
    return occ_map

# =====================================================
# 5. Occlusion severity mapping
# =====================================================
def get_severity(ratio: Optional[float]) -> Optional[str]:
    if ratio is None:
        return None
    if ratio < 0.10:
        return "slightly"
    elif ratio < 0.40:
        return "partially"
    elif ratio < 0.70:
        return "mostly"
    else:
        return "heavily"

# =====================================================
# 6. Build one occlusion description (style-controlled)
# =====================================================
def format_occ_sentence(
    blocked: int,
    blocker: int,
    info: Optional[Dict[str, Any]],
    style: str
) -> str:
    if info is None:
        return f"Object {blocked} is occluded by object {blocker}."

    ratio = info.get("mask_ratio", None)
    px, py = info.get("point", (None, None))
    sev = get_severity(ratio)

    if style == "ratio_only" and ratio is not None:
        return f"Object {blocked} is occluded by object {blocker} with the occlusion ratio of {ratio*100:.0f}%."
    elif style == "degree_only" and sev is not None:
        return f"Object {blocked} is {sev} occluded by object {blocker}."
    elif style == "point_only" and px is not None and py is not None:
        return f"Object {blocked} is occluded by object {blocker} at the contact point ({px}, {py})."
    elif style == "ratio_point" and ratio is not None and px is not None and py is not None:
        return f"Object {blocked} is occluded by object {blocker} at the contact point ({px}, {py}) with the occlusion ratio of {ratio*100:.0f}%."
    elif style == "ratio_point_short" and ratio is not None and px is not None and py is not None:
        return f"Object {blocked} is occluded by object {blocker} with {ratio*100:.0f}% occlusion at ({px}, {py})."
    else:
        return f"Object {blocked} is occluded by object {blocker}."

# =====================================================
# 7. Build reasoning (with style)
# =====================================================
def build_reasoning(
    target_object: int,
    paths: List[List[int]],
    scene_id: str,
    view_id: str,
    occ_map: Dict[Tuple[str, str, int, int], Dict[str, Any]],
    style: str
) -> (str, List[int], List[Dict[str, Any]]):
    if not paths:
        return f"<think>Object {target_object} is not occluded.</think>", [target_object], []

    all_steps, top_objects, occ_infos = [], set(), []
    multi_path = len(paths) > 1

    for idx, path in enumerate(paths, start=1):
        rev_path = list(reversed(path))
        steps = []

        for i in range(len(rev_path) - 1):
            blocker, blocked = rev_path[i + 1], rev_path[i]
            key = (scene_id, str(view_id), blocker, blocked)

            info = occ_map.get(key)
            if info is not None:
                ratio = info.get("mask_ratio")
                px, py = info.get("point", (None, None))
                sev = get_severity(ratio)
                occ_infos.append({
                    "obj1": blocker,
                    "obj2": blocked,
                    "ratio": None if ratio is None else round(ratio, 4),
                    "point": [px, py],
                    "mask": info.get("mask_path"),
                    "severity": sev
                })

            desc = format_occ_sentence(blocked, blocker, info, style)
            steps.append(desc)

        steps.append(f"Object {rev_path[-1]} is not occluded.")
        step_str = " ".join(steps)
        if multi_path:
            step_str = f"Path{idx}: " + step_str
        all_steps.append(step_str)
        top_objects.add(path[0])

    reasoning = f"<think>{' '.join(all_steps)}</think>"
    return reasoning, sorted(list(top_objects)), occ_infos

# =====================================================
# 8. Main conversion function
# =====================================================
def convert_to_vqa_with_occinfo(
    occlusion_json_path,
    occ_detail_path,
    output_json_path,
    image_root=None,
    style="ratio_only",
    seed=1337
):

    assert style in {"ratio_only", "degree_only", "point_only", "ratio_point", "ratio_point_short"}, \
        "style must be one of {'ratio_only', 'degree_only', 'point_only', 'ratio_point', 'ratio_point_short'}"

    if seed is not None:
        random.seed(seed)

    print(f"Loading occlusion paths from {occlusion_json_path}")
    with open(occlusion_json_path, "r") as f:
        occ_data = json.load(f)

    occ_detail = load_occ_detail(occ_detail_path)
    vqa_entries = []
    system_prompt = SYSTEM_PROMPTS[style]

    for entry in occ_data:
        scene_id = entry["scene_id"]
        view_id = str(entry["view_id"])
        target = int(entry["target_object"])
        paths = entry.get("occlusion_paths", [])

        reasoning, top_objs, occ_infos = build_reasoning(
            target, paths, scene_id, view_id, occ_detail, style
        )
        question = random.choice(QUESTION_TEMPLATES).format(target=target)
        image_path = make_image_path(scene_id, view_id, image_root)

        vqa_entries.append({
            "image": [image_path],
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": reasoning + "\n" + f"<answer>{json.dumps(top_objs)}</answer>"}
            ],
            "scene_id": scene_id,
            "occlusion_info": occ_infos
        })

    with open(output_json_path, "w") as f_out:
        json.dump(vqa_entries, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Saved ablation VQA dataset: {output_json_path}")
    print(f"📦 Total samples: {len(vqa_entries)} | Style: {style} | Image root: {image_root or '(none)'}")

# =====================================================
# 9. CLI entry
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Generate occlusion-aware VQA dataset for ablation study")
    parser.add_argument("--occlusion_json", required=True)
    parser.add_argument("--occ_detail_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--style", choices=["ratio_only", "degree_only", "point_only", "ratio_point", "ratio_point_short"], required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    convert_to_vqa_with_occinfo(
        args.occlusion_json,
        args.occ_detail_json,
        args.output,
        args.image_root,
        style=args.style,
        seed=args.seed
    )

if __name__ == "__main__":
    main()