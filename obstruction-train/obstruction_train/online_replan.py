import argparse
import json
from pathlib import Path

import torch

from .plan_graph_policy import build_graph, load_policy


def load_snapshots(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plan_snapshot(snapshot, model, encoder, device):
    task_mode = snapshot["task_mode"]
    target_id = snapshot.get("target_id")
    object_ids, nodes, edges = build_graph(snapshot["objects"], snapshot.get("relations", []), target_id)
    task = model.TASK_TARGET if task_mode == "target" else model.TASK_CLEAR_TABLE
    target_mask = torch.tensor([object_id == target_id for object_id in object_ids], device=device)[None]
    task_features = encoder([snapshot.get("instruction", "")]) if encoder else None
    with torch.no_grad():
        logits = model(nodes[None].to(device), edges[None].to(device), torch.ones(1, len(object_ids), dtype=torch.bool, device=device), torch.tensor([task], device=device), target_mask, task_features)[0]
    scores = torch.softmax(logits, dim=0).cpu().tolist()
    ranking = sorted(zip(object_ids, scores), key=lambda item: item[1], reverse=True)
    return [{"object_id": object_id, "policy_score": score} for object_id, score in ranking]


def main():
    parser = argparse.ArgumentParser(description="Replan after each RGB-D observation and grasp outcome.")
    parser.add_argument("--observations-jsonl", required=True, help="One state per line; later rows are post-execution observations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, encoder = load_policy(args.checkpoint, args.device)
    history = []
    for step, snapshot in enumerate(load_snapshots(args.observations_jsonl)):
        ranked_actions = plan_snapshot(snapshot, model, encoder, args.device)
        history.append({"step": step, "observation_id": snapshot.get("observation_id", step), "previous_grasp_succeeded": snapshot.get("grasp_succeeded"), "ranked_actions": ranked_actions, "selected_action": ranked_actions[0] if ranked_actions else None})
    result = {"controller": "online_graph_policy_replanner", "steps": history}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
