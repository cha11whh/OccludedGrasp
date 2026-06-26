import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .dataset import UnoBenchPairDataset
from .model import build_model


def compute_class_weights(rows, num_classes=3):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for row in rows:
        label = int(row["label"])
        if 0 <= label < num_classes:
            counts[label] += 1
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def relation_metrics(logits, target, num_classes=3):
    pred = logits.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    stats = {"acc": acc}
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        stats[f"class_{c}_p"] = precision
        stats[f"class_{c}_r"] = recall
        stats[f"class_{c}_f1"] = f1
        f1s.append(f1)
    stats["macro_f1"] = sum(f1s) / len(f1s)
    return stats


def compute_loss(outputs, batch, weights, class_weights=None):
    relation_loss = F.cross_entropy(outputs["relation_logits"], batch["relation"], weight=class_weights)
    depth_loss = F.cross_entropy(outputs["depth_order_logits"], batch["depth_order"], weight=class_weights)
    ratio_loss = F.smooth_l1_loss(outputs["ratio"], batch["ratio"])

    # Contact heatmap is only meaningful for positive relation samples.
    positive = batch["relation"] != 2
    if positive.any():
        contact_loss = F.binary_cross_entropy_with_logits(
            outputs["contact_heatmap_logits"][positive],
            batch["contact_heatmap"][positive],
        )
    else:
        contact_loss = outputs["contact_heatmap_logits"].sum() * 0.0

    total = (
        weights["relation"] * relation_loss
        + weights["depth"] * depth_loss
        + weights["ratio"] * ratio_loss
        + weights["contact"] * contact_loss
    )
    return total, {
        "loss": float(total.detach().item()),
        "relation_loss": float(relation_loss.detach().item()),
        "depth_loss": float(depth_loss.detach().item()),
        "ratio_loss": float(ratio_loss.detach().item()),
        "contact_loss": float(contact_loss.detach().item()),
    }


def run_epoch(model, loader, optimizer, scaler, device, weights, class_weights=None, train=True, log_every=50):
    model.train(train)
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(batch["image"])
                loss, parts = compute_loss(outputs, batch, weights, class_weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        bs = batch["image"].shape[0]
        total_loss += parts["loss"] * bs
        n += bs
        all_logits.append(outputs["relation_logits"].detach().cpu())
        all_targets.append(batch["relation"].detach().cpu())

        if train and step % log_every == 0:
            print(f"step {step}/{len(loader)} loss={parts['loss']:.4f} rel={parts['relation_loss']:.4f}", flush=True)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = relation_metrics(logits, targets)
    metrics["loss"] = total_loss / max(1, n)
    return metrics


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unobench-root", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="pair_transformer_base", choices=["pair_transformer_small", "pair_transformer_base", "pair_transformer_large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument("--ratio-weight", type=float, default=0.1)
    parser.add_argument("--contact-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Load model weights but start a fresh optimizer.")
    parser.add_argument("--reset-best", action="store_true", help="When resuming, pick the best checkpoint only from this run.")
    parser.add_argument("--train-external-depth-dir", default=None)
    parser.add_argument("--val-external-depth-dir", default=None)
    parser.add_argument("--depth-transform", choices=["identity", "inverse"], default="identity")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = UnoBenchPairDataset(
        args.unobench_root,
        args.train_jsonl,
        image_size=args.image_size,
        external_depth_dir=args.train_external_depth_dir,
        depth_transform=args.depth_transform,
    )
    val_ds = UnoBenchPairDataset(
        args.unobench_root,
        args.val_jsonl,
        image_size=args.image_size,
        external_depth_dir=args.val_external_depth_dir,
        depth_transform=args.depth_transform,
    )
    if args.limit_train > 0:
        train_ds.rows = train_ds.rows[: args.limit_train]
    if args.limit_val > 0:
        val_ds.rows = val_ds.rows[: args.limit_val]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, image_size=args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    class_weights = compute_class_weights(train_ds.rows).to(device)
    print(f"relation/depth class weights: {class_weights.detach().cpu().tolist()}")
    weights = {
        "relation": args.relation_weight,
        "depth": args.depth_weight,
        "ratio": args.ratio_weight,
        "contact": args.contact_weight,
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    start_epoch = 1
    best_f1 = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, None if args.reset_optimizer else optimizer, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if args.reset_best:
            best_f1 = -1.0
        else:
            best_f1 = float((ckpt.get("metrics") or {}).get("val", {}).get("macro_f1", -1.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}, previous best_f1={best_f1:.4f}", flush=True)

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n=== epoch {epoch}/{end_epoch} ===")
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, weights, class_weights=class_weights, train=True, log_every=args.log_every)
        print("train:", json.dumps(train_metrics, indent=2), flush=True)

        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, weights, class_weights=class_weights, train=False)
        print("val:", json.dumps(val_metrics, indent=2), flush=True)

        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, metrics, args)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics, args)
            print(f"saved best checkpoint, macro_f1={best_f1:.4f}", flush=True)

    print(f"done. best macro_f1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
