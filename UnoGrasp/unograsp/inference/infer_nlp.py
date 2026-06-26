import os
import re
import json
import argparse
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "You are an assistant specialized in robotic grasp planning based on occlusion reasoning. "
    "When asked which object must be removed first to grasp a specific target object in a single image:\n\n"
    "- If the target object is not occluded, return the target object's name/description and its coordinates.\n"
    "- If the target object has one occlusion path, reason step-by-step along that path, include the occlusion ratio for each occlusion relation when available, and end with the top-most object that must be removed first. "
    "Each reasoning step must reference objects with explicit (x,y) coordinates.\n"
    "- If the target object has multiple occlusion paths, reason step by step for each path separately, and include all distinct top-most occluding objects in the final answer.\n"
    "- All reasoning must be enclosed within a single pair of <think>...</think> tags.\n"
    "- The final answer must be enclosed in <answer>...</answer> tags and formatted strictly as:\n"
    "  <answer>[<points x y>object name</points>, ...]</answer>\n"
)

IMG_NAME_RE = re.compile(r"image_(\d+)", re.IGNORECASE)
TARGET_RE = re.compile(r"object\s+(\d+)", re.IGNORECASE)


def parse_image_id_from_image_path(p: str):
    """Parse image_id from paths like image_000003.png."""
    name = os.path.basename(p)
    m = IMG_NAME_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def get_image_id(sample: dict, image_path: str):
    image_id = sample.get("image_id")
    if image_id is not None:
        try:
            return int(image_id)
        except (TypeError, ValueError):
            return image_id
    return parse_image_id_from_image_path(image_path)


def extract_query_from_human(text: str):
    """Extract query_object integer from human prompt by taking the last occurrence of object N."""
    matches = list(TARGET_RE.finditer(text or ""))
    if not matches:
        return None
    return int(matches[-1].group(1))


def ensure_image_path(dataset_root: Path, rel_or_abs: str):
    """
    Return the absolute path for an image.
    Supports:
    - absolute paths
    - paths relative to dataset_root
    """
    p = Path(rel_or_abs)

    if not p.is_absolute():
        p = dataset_root / p

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    return str(p)


def build_messages(image_path: str, human_text: str):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": human_text},
            ],
        },
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1500)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "predictions.jsonl"

    # Load completed indices
    done_indices = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_indices.add(int(obj["index"]))
                except Exception:
                    # ignore malformed lines, missing index, or invalid int conversion
                    pass

    print(f"[INFO] Found {len(done_indices)} completed results, continuing inference on remaining samples.")

    # Open result file in append mode (creates file if missing)
    fout = open(results_path, "a", encoding="utf-8")

    # ======================================
    # Load model
    # ======================================
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.chat_template = processor.tokenizer.chat_template

    # ======================================
    # Load data
    # ======================================
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            items = json.load(f)
        else:
            items = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded dataset: {len(items)} samples.")

    # ======================================
    # Inference loop (resume support)
    # ======================================
    for idx, sample in enumerate(items):

        if idx in done_indices:
                continue  # skip already completed sample
        conv = sample.get("conversations", [])
        human_text = None
        for c in conv:
            if c.get("from") in ["human", "user"]:
                human_text = c.get("value", "").replace("<image>\n", "").strip()
                break
        if not human_text:
            continue

        imgs = sample.get("image", [])
        if not imgs:
            continue
        img_path = ensure_image_path(Path(args.dataset_root), imgs[0])

        image_id = get_image_id(sample, img_path)
        query_object = sample.get("query_object")

        messages = build_messages(img_path, human_text)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
            output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        print("=" * 80)
        print(f"#{idx} | image_id={image_id} | image={os.path.basename(img_path)} | query={query_object}")
        print(f"Q: {human_text}")
        print(output)

        rec = {
            "index": idx,
            "image": img_path,
            "image_id": image_id,
            "query_object": query_object,
            "human": human_text,
            "model_output": output,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"[INFO] Inference completed, output: {results_path}")


if __name__ == "__main__":
    main()
