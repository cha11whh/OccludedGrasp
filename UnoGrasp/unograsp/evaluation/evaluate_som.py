import argparse
import json
import os
import re
import ast
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

ANSWER_LIST_RE = re.compile(r"<answer>\s*(\[.*?\])\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
IMAGE_ID_RE = re.compile(r"image_(\d+)", re.IGNORECASE)


def parse_image_id_from_path(img_path):
    m = IMAGE_ID_RE.search(img_path or "")
    if not m:
        return None
    return int(m.group(1))


def get_image_id(record):
    image_id = record.get("image_id")
    if image_id is None:
        image_id = parse_image_id_from_path(record.get("image", ""))
    if image_id is None:
        return None
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return image_id


def get_record_key(record):
    image_id = get_image_id(record)
    if image_id is not None:
        return ("image", image_id)

    scene_id = record.get("scene_id")
    view_id = record.get("view_id")
    if scene_id is not None and view_id is not None:
        return ("scene_view", str(scene_id).split("/")[-1], str(view_id))

    return None


def extract_answer_list(gen_text: str):
    m = ANSWER_LIST_RE.search(gen_text or "")
    if not m:
        return []
    try:
        ids = ast.literal_eval(m.group(1))
        if isinstance(ids, int):
            return [ids]
        return [int(x) for x in ids]
    except Exception:
        return []

def parse_think_paths(gen_text: str, tgt: int):
    """Parse reasoning paths inside <think>...</think> tags."""
    m = THINK_RE.search(gen_text or "")
    if not m:
        return [[tgt]]
    think = m.group(1).strip()
    parts = re.split(r"Path\d*:", think)
    paths = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        rels = re.findall(r"Object\s+(\d+)\s+is occluded by object\s+(\d+)", p)
        if rels:
            # Correct direction: occluder -> occluded
            tmp_list = list(rels[0])
            tmp_list += [x[1] for x in rels[1:]]
            tmp_list.reverse()
            tmp_list = [int(x) for x in tmp_list]
            paths.append(tmp_list)
        else:
            mobj = re.search(r"Object\s+(\d+)", p)
            if mobj:
                oid = int(mobj.group(1))
                paths.append([oid])
    if not paths:
        # no paths found -> return empty list
        paths = []
    # deduplicate while preserving order
    paths = [list(dict.fromkeys(p)) for p in paths]
    return paths

def ned(seq1, seq2):
    import editdistance
    return editdistance.eval(seq1, seq2) / max(len(seq1), len(seq2))

def mp_ned(pred_paths, gt_paths, alpha=1.0, beta=1.0):
    m, n = len(pred_paths), len(gt_paths)
    size = max(m, n)
    if size == 0:
        return 0.0
    C = np.zeros((size, size))
    for i in range(m):
        for j in range(n):
            C[i, j] = ned(pred_paths[i], gt_paths[j])
    if m < n:
        for i in range(m, size):
            C[i, :n] = alpha
    elif m > n:
        for j in range(n, size):
            C[:m, j] = beta
    row_ind, col_ind = linear_sum_assignment(C)
    return C[row_ind, col_ind].sum() / size

def paths_to_triplets(paths):
    triplets = set()
    for path in paths:
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            triplets.add((path[i], path[i+1]))
    return triplets

def compute_precision_recall_f1(pred_set, gt_set):
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

def main(pred_path, gt_path):
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: JSON decode failed at line {i}, skipping. Snippet: {line[:80]}...")
                continue

            record_key = get_record_key(it)
            if record_key is None:
                print(f"Warning: line {i} missing image_id or scene_id/view_id and couldn't extract from path, skipping.")
                continue

            # ---- Check if query_object exists ----
            if "query_object" not in it:
                print(f"Warning: line {i} missing query_object, skipping.")
                continue

            tgt = int(it["query_object"])
            key = (*record_key, tgt)

            ans = extract_answer_list(it.get("model_output", ""))
            think_paths = parse_think_paths(it.get("model_output", ""), tgt)
            preds[key] = {"answer": ans, "paths": think_paths}


    gts = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
        for it in arr:
            record_key = get_record_key(it)
            if record_key is None:
                continue
            tgt = int(it["query_object"])
            diff = it.get("new_difficulty", "Easy")
            gt_paths = it.get("occlusion_paths", [])
            if not gt_paths:
                gt_paths = [[tgt]]
            gts[(*record_key, tgt)] = {
                "paths": gt_paths,
                "tops": it.get("top_objects", [p[0] for p in gt_paths]),
                "difficulty": diff
            }

    results = defaultdict(lambda: {"SR-P": [], "SR-R": [], "SR-F1": [],
                                   "OP": [], "OR": [], "F1": [], 
                                   "MP_NED": []})

    for key, gt in gts.items():
        if key not in preds:
            continue
        pred = preds[key]
        pred_paths = pred["paths"]
        gt_paths = gt["paths"]
        diff = gt["difficulty"]

        gt_top = set(gt["tops"])
        pred_ans = pred["answer"]

        # --- SR precision/recall/F1 ---
        prec, rec, f1 = compute_precision_recall_f1(set(pred_ans), gt_top)
        results[diff]["SR-P"].append(prec)
        results[diff]["SR-R"].append(rec)
        results[diff]["SR-F1"].append(f1)

        # --- Occlusion reasoning precision/recall/F1 ---
        gt_triplets = paths_to_triplets(gt_paths)
        pred_triplets = paths_to_triplets(pred_paths)
        op, orc, f1_trip = compute_precision_recall_f1(pred_triplets, gt_triplets)
        results[diff]["OP"].append(op)
        results[diff]["OR"].append(orc)
        results[diff]["F1"].append(f1_trip)

        # --- MP_NED ---
        results[diff]["MP_NED"].append(mp_ned(pred_paths, gt_paths))

    # Output
    for diff in ["No-Occ", "Easy", "Medium", "Hard"]:
        if not results[diff]["MP_NED"]:
            continue
        print(f"=== {diff} ===")
        print(f"SR: P={np.mean(results[diff]['SR-P']):.4f}, R={np.mean(results[diff]['SR-R']):.4f}, F1={np.mean(results[diff]['SR-F1']):.4f} ({len(results[diff]['SR-P'])} samples)")
        if diff != "No-Occ":
            print(f"Occlusion reasoning: P={np.mean(results[diff]['OP']):.4f}, R={np.mean(results[diff]['OR']):.4f}, F1={np.mean(results[diff]['F1']):.4f}")
        print(f"MP_NED: {np.mean(results[diff]['MP_NED']):.4f} ({len(results[diff]['MP_NED'])} samples)")
        print()

    # Overall (group-weighted): equal weight for No-Occ, Easy, Medium, and Hard.
    def mean_or_zero(lst):
        return float(np.mean(lst)) if lst else 0.0

    sr_no_occ_val = mean_or_zero(results["No-Occ"]["SR-F1"])
    srf1_easy_val = mean_or_zero(results["Easy"]["SR-F1"])
    srf1_medium_val = mean_or_zero(results["Medium"]["SR-F1"])
    srf1_hard_val = mean_or_zero(results["Hard"]["SR-F1"])
    overall_group_srf1 = 0.25 * (sr_no_occ_val + srf1_easy_val + srf1_medium_val + srf1_hard_val)

    print("=== Overall (Group-weighted) ===")
    print(f"Balanced SR-F1 (Group-weighted) = {overall_group_srf1:.3f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SOM predictions against GT occlusion paths.")
    parser.add_argument("--pred_path", required=True,
                        help="Path to predictions jsonl file.")
    parser.add_argument("--gt_path", required=True,
                        help="Path to GT json file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pred_path, args.gt_path)

