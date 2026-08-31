import argparse
import json
from pathlib import Path

import torch

from .online_replan import plan_snapshot
from .plan_graph_policy import load_policy


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(records, model, encoder, graph_features, device):
    top1 = 0
    reciprocal_rank = 0.0
    equivalent_hit = 0
    success_count = 0
    success_total = 0
    for record in records:
        ranking = plan_snapshot(record, model, encoder, graph_features, device)
        expected = int(record["next_action_id"])
        positions = {row["object_id"]: index + 1 for index, row in enumerate(ranking)}
        rank = positions.get(expected)
        top1 += int(rank == 1)
        reciprocal_rank += 0.0 if rank is None else 1.0 / rank
        valid_actions = set(record.get("equivalent_action_ids", [expected]))
        equivalent_hit += int(bool(ranking) and ranking[0]["object_id"] in valid_actions)
        if "grasp_succeeded" in record:
            success_total += 1
            success_count += int(bool(record["grasp_succeeded"]))
    count = max(1, len(records))
    return {"num_samples": len(records), "top1_accuracy": top1 / count, "mean_reciprocal_rank": reciprocal_rank / count, "equivalent_action_top1": equivalent_hit / count, "recorded_grasp_success_rate": None if success_total == 0 else success_count / success_total, "graph_features": graph_features, "uses_language": encoder is not None}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Graph Transformer grasp policy on held-out demonstrations.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, encoder, graph_features = load_policy(args.checkpoint, args.device)
    metrics = evaluate(load_records(args.jsonl), model, encoder, graph_features, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
