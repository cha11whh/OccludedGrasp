# Obstruction Train

Train a high-accuracy pairwise obstruction relation model on UnoBench without VLM fine-tuning.

The training design follows the current direction:

```text
RGB-D + object masks
  -> pairwise Transformer relation model
  -> optional graph refinement with GNN
  -> target-centric path decoding
```

## Model Scope

This repository trains the relation model that predicts, for an object pair `(A, B)`:

```text
0: A obstructs B
1: B obstructs A
2: no direct obstruction relation
```

It also predicts auxiliary outputs:

- contact heatmap
- depth-order class
- obstruction-ratio regression

The obstruction ratio is an auxiliary feature/supervision target, not a graph-edge criterion.

## Prepare Pair Index

```bash
cd /home/ubuntu22-zy/occlusion/obstruction-train

python -m obstruction_train.prepare_unobench_pairs \
  --unobench-root /home/ubuntu22-zy/UnoBench/UnoBenchSyn \
  --out-dir data/pairs \
  --neg-per-positive 3 \
  --val-ratio 0.05
```

## Train V1 Pair Transformer

Run inside the existing `gsa:amodal` Docker image:

```bash
cd /home/ubuntu22-zy/occlusion

docker run --rm --gpus all \
  -v /home/ubuntu22-zy:/home/ubuntu22-zy \
  --ipc=host gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-train &&
python -m obstruction_train.train_pair_transformer \
  --unobench-root /home/ubuntu22-zy/UnoBench/UnoBenchSyn \
  --train-jsonl data/pairs/train_pairs.jsonl \
  --val-jsonl data/pairs/val_pairs.jsonl \
  --out-dir outputs/pair_transformer_v1 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4
'
```

## Architecture

V1 uses a pair crop only, not a separate global-context branch:

```text
Input crop channels:
  RGB                         3
  depth                       1
  modal mask A                1
  modal mask B                1
  boundary mask A             1
  boundary mask B             1
--------------------------------
  total                       8

Patch embedding
Transformer encoder
CLS token
Heads:
  relation class              3
  depth order                 3
  obstruction ratio           1
  contact heatmap             1 x H x W
```

V2 will use `graph_refiner.py` to refine pair logits across an object graph.

## Depth Anything Validation

The current best checkpoint is:

```text
outputs/pair_transformer_v1_base_e5_resume/best.pt
```

Full UnoBench val with dataset GT/sim depth:

```text
Acc      0.9455
Macro-F1 0.9382
```

Depth Anything V2 Large must be normalized and inverted before feeding the relation model. Raw Depth Anything depth uses the opposite direction from the training depth convention.

Full UnoBench val with Depth Anything V2 Large, `--depth-transform inverse`:

```text
Acc      0.9327
Macro-F1 0.9186
```

After one epoch of DA-ViT-L inverse-depth fine-tuning:

```text
checkpoint outputs/pair_transformer_v1_base_da_vitl_ft_e1/best.pt
Acc        0.9404
Macro-F1   0.9293
```

A second fine-tuning epoch with a lower LR reached `0.9276` Macro-F1, so the e1 checkpoint is the current best DA-depth relation model.

The fine-tuned result recovers about 1.07 Macro-F1 points over zero-shot DA depth, but remains about 0.88 points below GT/sim depth.

```bash
cd /home/ubuntu22-zy/occlusion

docker run --rm --gpus all \
  -v /home/ubuntu22-zy:/home/ubuntu22-zy \
  --ipc=host gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-train &&
python -m obstruction_train.evaluate_pair_model \
  --unobench-root /home/ubuntu22-zy/UnoBench/UnoBenchSyn \
  --jsonl data/pairs/val_pairs.jsonl \
  --checkpoint outputs/pair_transformer_v1_base_e5_resume/best.pt \
  --out evals/best_val_full_da_vitl_depth_inverse.json \
  --model pair_transformer_base \
  --external-depth-dir data/depth_anything_v2_vitl_val \
  --depth-source-name depth_anything_v2_vitl_val_inverse \
  --depth-transform inverse
'
```

## Graph Policy Training and Replanning

The Graph Transformer policy now accepts obstruction, support, and nearby-object
edge features. It also uses a trainable hashed task-text encoder, so target
instructions and `clear_table` tasks condition the policy rather than being used
only by a caller. Train from demonstration JSONL, one state/action per row:

```json
{"objects": [{"id": 1, "bbox": [0, 0, 10, 10]}], "relations": [], "task_mode": "target", "target_id": 1, "instruction": "grasp the red cup", "next_action_id": 1}
```

```bash
python -m obstruction_train.train_graph_policy --train-jsonl demos.jsonl --out-dir outputs/graph_policy
python -m obstruction_train.online_replan --observations-jsonl observations.jsonl --checkpoint outputs/graph_policy/best.pt --out outputs/replan_history.json
```

`online_replan` writes a new ranking for every observation and preserves the
previous grasp outcome. A robot bridge should collect calibrated RGB-D, run
perception, execute its safety-checked grasp controller, then append the next
observation; it must not execute the policy score directly as a robot trajectory.
