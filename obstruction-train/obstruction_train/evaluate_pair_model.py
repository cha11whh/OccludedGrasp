import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import UnoBenchPairDataset
from .model import build_model
from .train_pair_transformer import move_batch, relation_metrics


def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["image"])
            all_logits.append(outputs["relation_logits"].detach().cpu())
            all_targets.append(batch["relation"].detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = relation_metrics(logits, targets)
    pred = logits.argmax(dim=1)
    metrics["confusion"] = [
        [int(((targets == t) & (pred == p)).sum().item()) for p in range(3)]
        for t in range(3)
    ]
    metrics["num_samples"] = int(targets.numel())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PairRelationTransformer checkpoint.")
    parser.add_argument("--unobench-root", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="pair_transformer_base", choices=["pair_transformer_small", "pair_transformer_base", "pair_transformer_large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--external-depth-dir", default=None)
    parser.add_argument("--depth-source-name", default="unobench_gt_depth")
    parser.add_argument("--depth-transform", choices=["identity", "inverse"], default="identity")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ds = UnoBenchPairDataset(
        args.unobench_root,
        args.jsonl,
        image_size=args.image_size,
        external_depth_dir=args.external_depth_dir,
        depth_transform=args.depth_transform,
    )
    if args.limit > 0:
        ds.rows = ds.rows[: args.limit]
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, image_size=args.image_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, loader, device)
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["checkpoint_metrics"] = ckpt.get("metrics", {})
    metrics["jsonl"] = args.jsonl
    metrics["depth_source"] = args.depth_source_name
    metrics["external_depth_dir"] = args.external_depth_dir
    metrics["depth_transform"] = args.depth_transform

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
