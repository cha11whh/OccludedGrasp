#!/usr/bin/env python3

import argparse
import ast
import json
import os
from collections import defaultdict


DIFFICULTIES = ("No-Occ", "Easy", "Medium", "Hard")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.values()
    else:
        items = data

    gt_by_index = {}
    for item in items:
        test_index = int(item["test_index"])
        answer = [int(obj["obj_id"]) for obj in item.get("target_objects", [])]
        difficulty = item.get("difficulty", item.get("new_difficulty", "Easy"))
        if not answer:
            difficulty = "No-Occ"
        gt_by_index[test_index] = {
            "answer": set(answer),
            "difficulty": difficulty,
        }
    return gt_by_index


def parse_id_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, int):
        return [value]
    return [int(x) for x in value]


def load_predictions(path):
    preds = {}
    for record in load_jsonl(path):
        test_index = int(record["test_index"])
        output = record.get("output", record.get("prediction", record.get("answer", [])))
        preds[test_index] = set(parse_id_list(output))
    return preds


def prf(pred, gt):
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = 1.0 if pred == gt else 0.0
    return precision, recall, f1, exact


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_balanced_sr_f1(results):
    group_scores = {diff: mean(results[diff]["F1"]) for diff in DIFFICULTIES}
    balanced_sr_f1 = sum(group_scores.values()) / len(DIFFICULTIES)

    print("=== Overall (Group-weighted) ===")
    for diff in DIFFICULTIES:
        count = len(results[diff]["F1"])
        print(f"{diff}: SR-F1={group_scores[diff]:.4f} ({count} samples)")
    print(f"Balanced SR-F1 (Group-weighted) = {balanced_sr_f1:.4f}")


def evaluate(pred_path, gt_path):
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    gt_by_index = load_gt(gt_path)
    preds = load_predictions(pred_path)
    results = defaultdict(lambda: {"P": [], "R": [], "F1": [], "Exact": []})
    missing = 0

    for test_index, gt in gt_by_index.items():
        if test_index not in preds:
            missing += 1
            continue
        p, r, f1, exact = prf(preds[test_index], gt["answer"])
        bucket = results[gt["difficulty"]]
        bucket["P"].append(p)
        bucket["R"].append(r)
        bucket["F1"].append(f1)
        bucket["Exact"].append(exact)

    print("========== SOM SR Evaluation ==========")
    print(f"GT samples: {len(gt_by_index)}")
    print(f"Predicted samples: {len(preds)}")
    print(f"Matched samples: {len(gt_by_index) - missing}")
    print(f"Missing predictions: {missing}")
    print()

    for diff in DIFFICULTIES:
        count = len(results[diff]["F1"])
        if count == 0:
            continue
        print(f"=== {diff} ({count} samples) ===")
        print(
            f"SR-P={mean(results[diff]['P']):.4f}, "
            f"SR-R={mean(results[diff]['R']):.4f}, "
            f"SR-F1={mean(results[diff]['F1']):.4f}, "
            f"Exact={mean(results[diff]['Exact']):.4f}"
        )
        print()

    print_balanced_sr_f1(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SOM predictions with answer-level SR only.")
    parser.add_argument("--pred_path", default="outputs/som.jsonl", help="Path to SOM prediction JSONL.")
    parser.add_argument("--gt_path", default="Synthetic_test.json", help="Path to ground-truth JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred_path, args.gt_path)
