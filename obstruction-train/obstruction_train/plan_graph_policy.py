import argparse
import json
from pathlib import Path

import torch

from .graph_policy import TaskConditionedGraphPolicy


NODE_DIM = 10
EDGE_DIM = 4


def _records(payload, field):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get(field), list):
        return payload[field]
    raise ValueError(f"expected a list or a {field} list")


def build_graph(objects, relations, target_id=None):
    object_ids = [int(obj["id"]) for obj in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("object IDs must be unique")
    if target_id is not None and target_id not in object_ids:
        raise ValueError("target ID is absent from objects")
    index = {object_id: offset for offset, object_id in enumerate(object_ids)}
    nodes = torch.zeros(len(objects), NODE_DIM)
    edges = torch.zeros(len(objects), len(objects), EDGE_DIM)
    for offset, obj in enumerate(objects):
        x1, y1, x2, y2 = (float(value) for value in (obj.get("bbox") or [0, 0, 0, 0]))
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        nodes[offset, :6] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2, width, height, width * height, float(obj.get("score") or 0.0)])
    for relation in relations:
        blocker, blocked = int(relation["blocker"]), int(relation["blocked"])
        if blocker not in index or blocked not in index:
            continue
        source, destination = index[blocker], index[blocked]
        confidence = float(relation.get("confidence", relation.get("relation_confidence", 0.0)))
        ratio = float(relation.get("mask_ratio", 0.0))
        depth = float(relation.get("edge_features", {}).get("depth_order_score", 0.0))
        edges[source, destination] = torch.tensor([1.0, confidence, ratio, depth])
        nodes[source, 7] += 1
        nodes[destination, 6] += 1
    for offset, object_id in enumerate(object_ids):
        nodes[offset, 8] = float(object_id == target_id)
        nodes[offset, 9] = float(any(edges[offset, other, 0] and object_ids[other] == target_id for other in range(len(objects))))
    return object_ids, nodes, edges


def main():
    parser = argparse.ArgumentParser(description="Rank grasp actions with a task-conditioned Graph Transformer.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-mode", choices=["target", "clear_table"], required=True)
    parser.add_argument("--target-id", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.task_mode == "target" and args.target_id is None:
        parser.error("--target-id is required for target tasks")
    with open(args.objects, encoding="utf-8") as handle:
        objects = _records(json.load(handle), "objects")
    with open(args.relations, encoding="utf-8") as handle:
        relations = _records(json.load(handle), "relations")
    object_ids, nodes, edges = build_graph(objects, relations, args.target_id)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = TaskConditionedGraphPolicy(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    task = model.TASK_TARGET if args.task_mode == "target" else model.TASK_CLEAR_TABLE
    target_mask = torch.tensor([object_id == args.target_id for object_id in object_ids], device=args.device)[None]
    with torch.no_grad():
        logits = model(nodes[None].to(args.device), edges[None].to(args.device), torch.ones(1, len(object_ids), dtype=torch.bool, device=args.device), torch.tensor([task], device=args.device), target_mask)[0]
    ranking = sorted(zip(object_ids, torch.softmax(logits, dim=0).cpu().tolist()), key=lambda item: item[1], reverse=True)
    result = {"planner": "task_conditioned_graph_transformer", "task_mode": args.task_mode, "target_id": args.target_id, "ranked_actions": [{"object_id": object_id, "policy_score": score} for object_id, score in ranking]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
