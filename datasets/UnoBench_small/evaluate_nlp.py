#!/usr/bin/env python3

import argparse
import ast
import json
import os
from collections import defaultdict

import numpy as np


DIFFICULTIES = ("No-Occ", "Easy", "Medium", "Hard")


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
            "image_id": int(item["image_id"]),
        }
    return gt_by_index


def parse_prediction_line(line, path, line_no):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid prediction at {path}:{line_no}") from exc


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(parse_prediction_line(line, path, line_no))
    return records


def parse_points(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    points = []
    for item in value:
        if len(item) < 2:
            continue
        points.append((float(item[0]), float(item[1])))
    return points


def mask_path(npz_root, image_id):
    return os.path.join(npz_root, f"image_{int(image_id):06d}.npy")


def points_to_object_ids(points, image_id, npz_root):
    path = mask_path(npz_root, image_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation mask not found: {path}")

    mask = np.load(path).astype(int)
    height, width = mask.shape
    object_ids = []
    hits = 0
    misses = 0

    for x, y in points:
        x_i = int(round(x))
        y_i = int(round(y))
        if 0 <= x_i < width and 0 <= y_i < height:
            obj_id = int(mask[y_i, x_i])
            if obj_id > 0:
                object_ids.append(obj_id)
                hits += 1
            else:
                misses += 1
        else:
            misses += 1

    return object_ids, hits, misses


def load_predictions(path, gt_by_index, npz_root):
    preds = {}
    total_points = 0
    total_hits = 0
    total_misses = 0
    for record in load_jsonl(path):
        test_index = int(record["test_index"])
        if test_index not in gt_by_index:
            continue
        output = record.get("output", record.get("prediction", record.get("answer", [])))
        points = parse_points(output)
        image_id = gt_by_index[test_index]["image_id"]
        object_ids, hits, misses = points_to_object_ids(points, image_id, npz_root)
        preds[test_index] = set(object_ids)
        total_points += len(points)
        total_hits += hits
        total_misses += misses
    return preds, {
        "total_points": total_points,
        "total_hits": total_hits,
        "total_misses": total_misses,
    }


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


def evaluate(pred_path, gt_path, npz_root):
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    if not os.path.isdir(npz_root):
        raise FileNotFoundError(f"Annotation directory not found: {npz_root}")

    gt_by_index = load_gt(gt_path)
    preds, point_stats = load_predictions(pred_path, gt_by_index, npz_root)
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


    print("========== Point parse ==========")
    print(f"Parsed coordinates: {point_stats['total_points']}")
    print(f"Mask hits: {point_stats['total_hits']}")
    print(f"Mask misses: {point_stats['total_misses']}")
    hit_ratio = point_stats["total_hits"] / point_stats["total_points"] if point_stats["total_points"] else 0.0
    print(f"Mask hit ratio: {hit_ratio:.4f}")
    print("========== NLP SR Evaluation ==========")
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
    parser = argparse.ArgumentParser(description="Evaluate NLP predictions with answer-level SR only.")
    parser.add_argument("--pred_path", default="outputs/nlp.jsonl", help="Path to NLP prediction JSONL.")
    parser.add_argument("--gt_path", default="Synthetic_test.json", help="Path to ground-truth JSON.")
    parser.add_argument("--npz_root", default="annotations", help="Directory containing instance mask .npy files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred_path, args.gt_path, args.npz_root)
