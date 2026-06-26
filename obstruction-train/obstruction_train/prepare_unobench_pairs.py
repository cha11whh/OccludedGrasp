import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LABEL_A_OBSTRUCTS_B = 0
LABEL_B_OBSTRUCTS_A = 1
LABEL_NONE = 2


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scene_key(scene_id, view_id):
    return int(scene_id), int(view_id)


def parse_occ_scene(scene_id: str):
    # obs_information.json uses values like "data_ifl_0/scene123".
    tail = str(scene_id).split("/")[-1]
    if tail.startswith("scene"):
        return int(tail[len("scene"):])
    return int(tail)


def load_mapping(root: Path):
    data = read_json(root / "meta_data" / "image_id_scene_view_id_mapping.json")
    mapping = {}
    for item in data["mapping"]:
        mapping[scene_key(item["scene_id"], item["view_id"])] = {
            "image_id": int(item["image_id"]),
            "image_path": f'images/image_{int(item["image_id"]):06d}.png',
            "som_image_path": f'images_som/image_{int(item["image_id"]):06d}.png',
            "npz_path": f'annotations_meta/ALL_NPZ/scene{int(item["scene_id"])}_view{int(item["view_id"])}.npz',
            "scene_id": int(item["scene_id"]),
            "view_id": int(item["view_id"]),
        }
    return mapping


def load_objects_by_scene(root: Path):
    names = read_json(root / "meta_data" / "name_for_all.json")
    by_scene = {}
    for key, obj_map in names.items():
        # Example: scene30/0_rgb.png
        scene_part, view_part = key.split("/", 1)
        sid = int(scene_part.replace("scene", ""))
        vid = int(view_part.split("_", 1)[0])
        by_scene[scene_key(sid, vid)] = sorted(int(k) for k in obj_map.keys())
    return by_scene


def build_positive_edges(root: Path):
    obs = read_json(root / "meta_data" / "occ_info" / "obs_information.json")
    edges = defaultdict(dict)
    for item in obs:
        sid = parse_occ_scene(item["scene_id"])
        vid = int(item["view_id"])
        a = int(item["obj1"])
        b = int(item["obj2"])
        edges[scene_key(sid, vid)][(a, b)] = {
            "mask_ratio": float(item.get("mask_ratio", 0.0)),
            "mask_area": int(item.get("mask_area", 0)),
            "point": item.get("point"),
            "mask_path": item.get("mask_path"),
        }
    return edges


def make_row(meta, obj_a, obj_b, label, edge=None):
    edge = edge or {}
    return {
        "image_id": meta["image_id"],
        "image_path": meta["image_path"],
        "npz_path": meta["npz_path"],
        "scene_id": meta["scene_id"],
        "view_id": meta["view_id"],
        "obj_a": int(obj_a),
        "obj_b": int(obj_b),
        "label": int(label),
        "mask_ratio": float(edge.get("mask_ratio", 0.0)),
        "mask_area": int(edge.get("mask_area", 0)),
        "point": edge.get("point"),
        "occ_mask_path": edge.get("mask_path"),
    }


def build_pairs(root: Path, neg_per_positive: int, seed: int):
    rng = random.Random(seed)
    mapping = load_mapping(root)
    objects_by_scene = load_objects_by_scene(root)
    positives = build_positive_edges(root)

    rows = []
    skipped = 0
    for key, edge_map in positives.items():
        meta = mapping.get(key)
        objects = objects_by_scene.get(key)
        if meta is None or not objects:
            skipped += 1
            continue

        positive_pairs = set(edge_map.keys())
        all_ordered = [(a, b) for a in objects for b in objects if a != b]
        negative_candidates = [(a, b) for a, b in all_ordered if (a, b) not in positive_pairs and (b, a) not in positive_pairs]

        for (a, b), edge in edge_map.items():
            rows.append(make_row(meta, a, b, LABEL_A_OBSTRUCTS_B, edge))
            rows.append(make_row(meta, b, a, LABEL_B_OBSTRUCTS_A, edge))

            for na, nb in rng.sample(negative_candidates, min(neg_per_positive, len(negative_candidates))):
                rows.append(make_row(meta, na, nb, LABEL_NONE))

    rng.shuffle(rows)
    return rows, skipped


def split_rows(rows, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image_id"]].append(r)

    image_ids = list(by_image.keys())
    rng.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:n_val])

    train, val = [], []
    for image_id, group in by_image.items():
        (val if image_id in val_ids else train).extend(group)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unobench-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--neg-per-positive", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    root = Path(args.unobench_root)
    out_dir = Path(args.out_dir)
    rows, skipped = build_pairs(root, args.neg_per_positive, args.seed)
    train, val = split_rows(rows, args.val_ratio, args.seed)

    write_jsonl(out_dir / "train_pairs.jsonl", train)
    write_jsonl(out_dir / "val_pairs.jsonl", val)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_pairs": len(rows),
                "train_pairs": len(train),
                "val_pairs": len(val),
                "skipped_scene_views": skipped,
                "neg_per_positive": args.neg_per_positive,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"total pairs: {len(rows)}")
    print(f"train pairs: {len(train)}")
    print(f"val pairs: {len(val)}")
    print(f"skipped scene/views without mapping or object names: {skipped}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
