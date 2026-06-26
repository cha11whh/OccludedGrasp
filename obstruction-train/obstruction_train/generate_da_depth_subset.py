import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

import sys


def unique_images_from_jsonl(path: Path, max_images: int):
    images = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = row["image_path"]
            if image_path in seen:
                continue
            seen.add(image_path)
            images.append(image_path)
            if max_images > 0 and len(images) >= max_images:
                break
    return images


def read_rgb_from_zip(images_zip, member):
    with images_zip.open(member) as f:
        data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(member)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate Depth Anything V2 depth maps for a UnoBench JSONL subset.")
    parser.add_argument("--depth-anything-root", required=True)
    parser.add_argument("--unobench-root", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    depth_root = Path(args.depth_anything_root)
    sys.path.insert(0, str(depth_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = depth_root / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = unique_images_from_jsonl(Path(args.jsonl), args.max_images)
    zip_path = Path(args.unobench_root) / "images.zip"

    with zipfile.ZipFile(zip_path) as images_zip:
        for idx, image_path in enumerate(images, start=1):
            stem = Path(image_path).stem
            out_path = out_dir / f"{stem}_raw.npy"
            if out_path.exists():
                print(f"[{idx}/{len(images)}] skip {out_path}", flush=True)
                continue
            bgr = read_rgb_from_zip(images_zip, image_path)
            with torch.inference_mode():
                depth = model.infer_image(bgr, args.input_size)
            np.save(out_path, depth.astype(np.float32))
            print(f"[{idx}/{len(images)}] wrote {out_path} shape={depth.shape}", flush=True)


if __name__ == "__main__":
    main()
