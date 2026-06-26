import argparse
import os

import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def normalize_depth(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--outdir", default="outputs/single")
    parser.add_argument("--encoder", default="vits", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    ckpt_path = f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    image = cv2.imread(args.img_path)
    if image is None:
        raise FileNotFoundError(args.img_path)

    with torch.inference_mode():
        depth = model.infer_image(image, args.input_size)

    stem = os.path.splitext(os.path.basename(args.img_path))[0]
    depth_norm = normalize_depth(depth)

    np.save(os.path.join(args.outdir, f"{stem}_raw.npy"), depth.astype(np.float32))
    cv2.imwrite(os.path.join(args.outdir, f"{stem}_depth16.png"), (depth_norm * 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(args.outdir, f"{stem}_gray.png"), (depth_norm * 255).astype(np.uint8))

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    color = (cmap((depth_norm * 255).astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, f"{stem}_color.png"), color)

    print(f"saved depth outputs to {args.outdir}")
    print(f"raw depth shape={depth.shape}, min={depth.min():.6f}, max={depth.max():.6f}")


if __name__ == "__main__":
    main()
