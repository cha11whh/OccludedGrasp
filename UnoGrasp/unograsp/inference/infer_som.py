import os
import re
import json
import argparse
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
      "You are an assistant for robotic grasp planning. When asked which object must be removed first "
    "to grasp a specific object:\n\n"
    "- If the target object is not occluded, return the target object's ID itself.\n"
    "- If the object has one occlusion path, reason step-by-step along that path, include the occlusion ratio for each occlusion relation when available, and end with the "
    "top-most object that must be removed first.\n"
    "- If the object has multiple occlusion paths, reason step-by-step for each path separately. "
    "In the <answer>...</answer> tag, output ALL distinct top-most objects as a JSON list.\n\n"
    "Use <think>...</think> tags for your reasoning, and put ONLY the final object IDs in <answer>...</answer>."
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
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Qwen2.5-VL model or LoRA output")
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="Path to the dataset JSON for VQA inference")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory used to resolve relative image paths")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # ==================== Load data ====================
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # standard JSON
            items = json.load(f)
        else:
            # JSONL
            items = [json.loads(line) for line in f if line.strip()]

    dataset_root = Path(args.dataset_root)
    print(f"[INFO] Loaded dataset: {len(items)} samples from {args.dataset_json}")

    # ==================== Prepare output paths ====================
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "predictions.jsonl"

    # Load completed indices
    done_indices = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_indices.add(rec["index"])
                except Exception:
                    continue

    mode = "a" if done_indices else "w"
    max_done = max(done_indices) if done_indices else -1
    print(f"[INFO] Found {len(done_indices)} existing results, last index={max_done}, resuming from {max_done+1}.")

    # ======================================
    # Load model
    # ======================================
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0} if args.device.startswith("cuda") else "auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.chat_template = processor.tokenizer.chat_template

    # ==================== Inference loop ====================
    with open(results_path, mode, encoding="utf-8") as fout:
        for idx, sample in enumerate(items):
            if idx in done_indices:
                continue  # skip already completed sample

            # 1) extract human instruction
            conv = sample.get("conversations", [])
            human_text = None
            for c in conv:
                if c.get("from") in ["human", "user"]:
                    v = c.get("value", "")
                    human_text = v.replace("<image>\n", "").strip()
                    break
            if not human_text:
                continue

            # 2) get image path
            imgs = sample.get("image", [])
            if not imgs:
                continue
            img_path = ensure_image_path(dataset_root, imgs[0])

            # 3) parse metadata
            image_id = get_image_id(sample, img_path)
            query_object = sample.get("query_object")
            if query_object is None:
                query_object = extract_query_from_human(human_text)

            # 4) build input messages
            messages = build_messages(img_path, human_text)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # 5) model inference
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            gen_text = output_texts[0] if output_texts else ""

            # === print output ===
            print("=" * 80)
            print(f"#{idx} | image_id={image_id} | image={os.path.basename(img_path)} | query={query_object}")
            print(f"Q: {human_text}")
            print("MODEL OUTPUT:")
            print(gen_text)

            # 6) write result
            record = {
                "index": idx,
                "image": img_path,
                "image_id": image_id,
                "query_object": query_object,
                "human": human_text,
                "model_output": gen_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # --- ADD THIS LINE FOR REAL-TIME SAFETY ---
            fout.flush() 
            # ------------------------------------------

    print(f"Inference completed: {len(items)} samples processed")
    print(f"Saved predictions to: {results_path.resolve()}")



if __name__ == "__main__":
    main()
