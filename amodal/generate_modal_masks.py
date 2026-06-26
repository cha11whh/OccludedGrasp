import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.append("Grounded-Segment-Anything")
sys.path.append("Grounded-Segment-Anything/GroundingDINO")

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, build_sam


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_gdino(config_path, ckpt_path, device):
    gdino_args = SLConfig.fromfile(config_path)
    gdino_args.device = device
    model = build_model(gdino_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def transform_image(img_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor, _ = transform(img_pil, None)
    return img_tensor


def run_gdino(gdino_model, img, caption, box_thresh=0.35, text_thresh=0.35, with_logits=True, device="cuda"):
    gdino_model = gdino_model.to(device)
    img = img.to(device)

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.inference_mode():
        outputs = gdino_model(img[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_thresh
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    pred_phrases = []
    tokenized = gdino_model.tokenizer(caption)
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_thresh, tokenized, gdino_model.tokenizer)
        if with_logits:
            pred_phrase += f"({str(logit.max().item())[:4]})"
        pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def run_sam(img_pil, sam_ckpt, boxes_filt, device="cuda"):
    img = np.array(img_pil)
    predictor = SamPredictor(build_sam(checkpoint=sam_ckpt).to(device))
    predictor.set_image(img)

    size = img_pil.size
    h, w = size[1], size[0]
    boxes_xyxy = boxes_filt.clone()
    for i in range(boxes_xyxy.size(0)):
        boxes_xyxy[i] = boxes_xyxy[i] * torch.Tensor([w, h, w, h])
        boxes_xyxy[i][:2] -= boxes_xyxy[i][2:] / 2
        boxes_xyxy[i][2:] += boxes_xyxy[i][:2]

    try:
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy.cpu(), img.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    except Exception as exc:
        print(f"SAM failed: {exc}")
        return img, None

    return img, masks.cpu().numpy().squeeze(1)


def parse_score(pred_phrase):
    match = re.search(r"\(([^()]*)\)$", pred_phrase)
    if not match:
        return 1.0
    return float(match.group(1))


def phrase_without_score(text):
    return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip().lower()


def token_set(text):
    text = phrase_without_score(text)
    return set(re.findall(r"[a-z0-9]+", text))


def match_prompt(phrase, prompts):
    phrase_tokens = token_set(phrase)
    best_prompt = None
    best_score = 0.0
    for prompt in prompts:
        prompt_tokens = token_set(prompt)
        if not prompt_tokens:
            continue
        inter = len(phrase_tokens & prompt_tokens)
        score = inter / float(len(prompt_tokens))
        if score > best_score:
            best_score = score
            best_prompt = prompt
    if best_score <= 0.0:
        return None
    return best_prompt


def main_cli():
    parser = argparse.ArgumentParser(description="Generate targeted modal masks with GroundingDINO + SAM")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--img_filenames_txt", required=True)
    parser.add_argument("--target_prompts_txt", required=True)
    parser.add_argument("--output_dir", default="targeted_modal_masks")
    parser.add_argument("--gdino_config", default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino_ckpt", default="Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", default="Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--box_thresh", type=float, default=0.25)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt_mode", choices=["combined", "per_prompt"], default="per_prompt")
    args = parser.parse_args()

    prompts = read_txt(args.target_prompts_txt)
    img_filenames = read_txt(args.img_filenames_txt)
    gdino_model = load_gdino(args.gdino_config, args.gdino_ckpt, args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    for img_filename in img_filenames:
        img_basename = os.path.splitext(img_filename)[0]
        img_path = os.path.join(args.input_dir, img_filename)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform_image(img_pil)
        output_img_dir = os.path.join(args.output_dir, img_basename)
        os.makedirs(output_img_dir, exist_ok=True)

        metadata = []
        all_boxes = []
        all_records = []
        if args.prompt_mode == "combined":
            caption = ". ".join(prompt.strip() for prompt in prompts if prompt.strip())
            boxes, phrases = run_gdino(
                gdino_model,
                img_tensor,
                caption,
                box_thresh=args.box_thresh,
                text_thresh=args.text_thresh,
                device=args.device,
            )
            if boxes is None or boxes.size(0) == 0:
                print(f"{img_filename}: no boxes for combined prompt")
            else:
                for box, phrase in zip(boxes, phrases):
                    prompt = match_prompt(phrase, prompts)
                    if prompt is None:
                        print(f"{img_filename}: skipped unmatched phrase {phrase}")
                        continue
                    all_boxes.append(box)
                    all_records.append(
                        {
                            "prompt": prompt,
                            "class_name": prompt.strip().lower(),
                            "phrase": phrase,
                            "score": parse_score(phrase),
                        }
                    )
        else:
            for prompt in prompts:
                boxes, phrases = run_gdino(
                    gdino_model,
                    img_tensor,
                    prompt,
                    box_thresh=args.box_thresh,
                    text_thresh=args.text_thresh,
                    device=args.device,
                )
                if boxes is None or boxes.size(0) == 0:
                    print(f"{img_filename}: no box for {prompt}")
                    continue
                for box, phrase in zip(boxes, phrases):
                    all_boxes.append(box)
                    all_records.append(
                        {
                            "prompt": prompt,
                            "class_name": prompt.strip().lower(),
                            "phrase": phrase,
                            "score": parse_score(phrase),
                        }
                    )

        if all_boxes:
            all_boxes_tensor = torch.stack(all_boxes, dim=0)
            img, masks = run_sam(img_pil, args.sam_ckpt, all_boxes_tensor, device=args.device)
            if masks is None or len(masks) == 0:
                print(f"{img_filename}: no masks from SAM")
            else:
                by_prompt = defaultdict(list)
                for idx, record in enumerate(all_records):
                    by_prompt[record["prompt"]].append((idx, record))

                for prompt in prompts:
                    entries = by_prompt.get(prompt, [])
                    if not entries:
                        continue
                    best_idx, best_record = max(entries, key=lambda item: item[1]["score"])
                    mask = masks[best_idx].astype(np.uint8) * 255
                    class_name = best_record["class_name"]
                    mask_name = f"{len(metadata):02d}_{slugify(class_name)}.png"
                    Image.fromarray(mask).save(os.path.join(output_img_dir, mask_name))
                    metadata.append(
                        {
                            "mask": mask_name,
                            "class_name": class_name,
                            "score": best_record["score"],
                            "source_phrase": best_record["phrase"],
                            "prompt": prompt,
                        }
                    )
                    print(f"{img_filename}: saved {mask_name} from {best_record['phrase']}")

        with open(os.path.join(output_img_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main_cli()
