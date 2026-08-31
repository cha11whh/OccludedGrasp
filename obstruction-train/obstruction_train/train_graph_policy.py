import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .graph_policy import TaskConditionedGraphPolicy
from .plan_graph_policy import EDGE_DIM, NODE_DIM, build_graph
from .task_text_encoder import HashTaskEncoder


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_epoch(records, model, encoder, optimizer, device, graph_features):
    model.train()
    if encoder:
        encoder.train()
    total_loss = 0.0
    correct = 0
    for record in records:
        object_ids, nodes, edges = build_graph(record["objects"], record.get("relations", []), record.get("target_id"), features=graph_features)
        action_id = int(record["next_action_id"])
        if action_id not in object_ids:
            raise ValueError(f"next_action_id {action_id} is not in objects")
        task_mode = record.get("task_mode", "target")
        task = model.TASK_TARGET if task_mode == "target" else model.TASK_CLEAR_TABLE
        target_mask = torch.tensor([item == record.get("target_id") for item in object_ids], device=device)[None]
        logits = model(nodes[None].to(device), edges[None].to(device), torch.ones(1, len(object_ids), dtype=torch.bool, device=device), torch.tensor([task], device=device), target_mask, encoder([record.get("instruction", "")]) if encoder else None)
        target = torch.tensor([object_ids.index(action_id)], device=device)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += int(logits.argmax(dim=1).item() == target.item())
    return total_loss / max(1, len(records)), correct / max(1, len(records))


def main():
    parser = argparse.ArgumentParser(description="Train a Graph Transformer next-grasp policy from demonstration JSONL.")
    parser.add_argument("--train-jsonl", required=True, help="Each row needs objects, relations, task_mode, instruction, and next_action_id.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--disable-language", action="store_true")
    parser.add_argument("--disable-obstruction", action="store_true")
    parser.add_argument("--disable-support", action="store_true")
    parser.add_argument("--disable-nearby", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    records = load_records(args.train_jsonl)
    graph_features = {"obstruction": not args.disable_obstruction, "support": not args.disable_support, "nearby": not args.disable_nearby}
    if not records:
        raise ValueError("training JSONL contains no demonstrations")
    config = {"node_dim": NODE_DIM, "edge_dim": EDGE_DIM, "task_dim": args.text_dim, "hidden_dim": args.hidden_dim, "num_heads": args.num_heads, "num_layers": args.num_layers}
    model = TaskConditionedGraphPolicy(**config).to(args.device)
    encoder = HashTaskEncoder(embedding_dim=args.text_dim).to(args.device) if not args.disable_language else None
    optimizer = torch.optim.AdamW(list(model.parameters()) + (list(encoder.parameters()) if encoder else []), lr=args.lr)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_epoch(records, model, encoder, optimizer, args.device, graph_features)
        print(f"epoch={epoch} loss={loss:.4f} action_accuracy={accuracy:.4f}", flush=True)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save({"model": model.state_dict(), "text_encoder": None if encoder is None else encoder.state_dict(), "config": config, "text_encoder_config": None if encoder is None else {"embedding_dim": args.text_dim, "num_buckets": encoder.num_buckets}, "graph_features": graph_features, "epoch": epoch, "action_accuracy": accuracy}, output / "best.pt")


if __name__ == "__main__":
    main()
