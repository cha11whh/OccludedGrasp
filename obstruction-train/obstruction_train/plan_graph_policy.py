import argparse
import json
from pathlib import Path

import torch

from .graph_policy import TaskConditionedGraphPolicy
from .task_text_encoder import HashTaskEncoder


NODE_DIM = 10
EDGE_DIM = 7


def _records(payload, field):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get(field), list):
        return payload[field]
    raise ValueError(f"expected a list or a {field} list")


def _overlap_or_proximity(first, second):
    ax1, ay1, ax2, ay2 = (float(value) for value in (first.get("bbox") or [0, 0, 0, 0]))
    bx1, by1, bx2, by2 = (float(value) for value in (second.get("bbox") or [0, 0, 0, 0]))
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)
    return intersection / union


def build_graph(objects, relations, target_id=None, near_iou=0.02, features=None):
    features = {"obstruction": True, "support": True, "nearby": True} if features is None else features
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
        source_id, destination_id = int(relation.get("blocker", relation.get("source", -1))), int(relation.get("blocked", relation.get("target", -1)))
        if source_id not in index or destination_id not in index:
            continue
        source, destination = index[source_id], index[destination_id]
        relation_type = str(relation.get("relation_type", relation.get("type", "obstruction"))).lower()
        confidence = float(relation.get("confidence", relation.get("relation_confidence", 0.0)))
        if relation_type in {"support", "supporting", "supported_by"}:
            if features.get("support", True):
                edges[source, destination, 4] = max(edges[source, destination, 4], confidence or 1.0)
        elif features.get("obstruction", True):
            ratio = float(relation.get("mask_ratio", 0.0))
            depth = float(relation.get("edge_features", {}).get("depth_order_score", 0.0))
            edges[source, destination, :4] = torch.tensor([1.0, confidence, ratio, depth])
            nodes[source, 7] += 1
            nodes[destination, 6] += 1
    for source, first in enumerate(objects):
        for destination, second in enumerate(objects):
            if source == destination:
                continue
            iou = _overlap_or_proximity(first, second)
            if features.get("nearby", True) and iou >= near_iou:
                edges[source, destination, 5:] = torch.tensor([1.0, iou])
    for offset, object_id in enumerate(object_ids):
        nodes[offset, 8] = float(object_id == target_id)
        nodes[offset, 9] = float(edges[offset, :, 0][torch.tensor([object_ids[item] == target_id for item in range(len(objects))])].any()) if target_id is not None else 0.0
    return object_ids, nodes, edges


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TaskConditionedGraphPolicy(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    encoder_config = checkpoint.get("text_encoder_config")
    encoder = None
    if encoder_config:
        encoder = HashTaskEncoder(**encoder_config).to(device)
        encoder.load_state_dict(checkpoint["text_encoder"])
        encoder.eval()
    model.eval()
    return model, encoder, checkpoint.get("graph_features", {"obstruction": True, "support": True, "nearby": True})


def main():
    parser = argparse.ArgumentParser(description="Rank grasp actions with a task-conditioned Graph Transformer.")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-mode", choices=["target", "clear_table"], required=True)
    parser.add_argument("--target-id", type=int)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.task_mode == "target" and args.target_id is None:
        parser.error("--target-id is required for target tasks")
    with open(args.objects, encoding="utf-8") as handle:
        objects = _records(json.load(handle), "objects")
    with open(args.relations, encoding="utf-8") as handle:
        relations = _records(json.load(handle), "relations")
    model, encoder, graph_features = load_policy(args.checkpoint, args.device)
    object_ids, nodes, edges = build_graph(objects, relations, args.target_id, features=graph_features)
    task = model.TASK_TARGET if args.task_mode == "target" else model.TASK_CLEAR_TABLE
    targets = torch.tensor([object_id == args.target_id for object_id in object_ids], device=args.device)[None]
    task_features = encoder([args.instruction]) if encoder else None
    with torch.no_grad():
        logits = model(nodes[None].to(args.device), edges[None].to(args.device), torch.ones(1, len(object_ids), dtype=torch.bool, device=args.device), torch.tensor([task], device=args.device), targets, task_features)[0]
    ranking = sorted(zip(object_ids, torch.softmax(logits, dim=0).cpu().tolist()), key=lambda item: item[1], reverse=True)
    result = {"planner": "task_conditioned_graph_transformer", "task_mode": args.task_mode, "target_id": args.target_id, "instruction": args.instruction, "ranked_actions": [{"object_id": object_id, "policy_score": score} for object_id, score in ranking]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
