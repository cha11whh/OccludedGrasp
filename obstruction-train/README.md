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

## Relation-First Decision Contract (T01)

## Relation-First Decision Contract (T02)
computes the target's required predecessor closure, and selects only a currently
confirmed legal candidate. Utility scoring remains deferred to T03.

- `observation` is a mapping with a non-empty `observation_id`, boolean
  `is_stale`, `relation_coverage` (`complete`, `partial`, or `unknown`), an
  `objects` list, an optional `visibility_coverage` list (omitted means empty),
  and a `relations` list. Object
  IDs are unique positive integers. Each object has `visibility` (`clear`,
  `occluded`, or `unknown`); omitted visibility is treated as `unknown`. The
  coverage list contains unique object IDs for which visibility was assessed;
  any non-`unknown` state without its ID in this list is downgraded to
  `unknown`. Missing coverage evidence never authorizes a clear state.
- Each obstruction relation has `blocker`, `blocked`, and finite `confidence`
  in `[0, 1]`; both IDs must exist in the same observation. A positive incoming
  obstruction relation contradicting `visibility: "clear"` invalidates the
  observation and fails closed. When `relation_coverage` is `complete`, an
  `occluded` object must have an incoming obstruction relation; otherwise the
  contradictory observation fails closed.
- `relations` contains positive obstruction edges only. An absent edge means
  unknown, never clear. `relation_coverage: "complete"` attests that the
  positive relation list is exhaustive, but does not by itself prove an object
  clear; explicit `visibility: "clear"` plus its coverage entry is required.
- `is_stale` is a required freshness attestation supplied by the observation
  adapter. Missing attestation or `is_stale: true` fails closed. This seam does
  not calculate age from timestamps.
- `feasibility` must include an `observation_id` exactly matching the current
  observation, a non-empty `coordinate_frame`, `distance_unit: "m"`, and exactly
  one record per observed object in `objects`; each record has `object_id` and
  `status` (`feasible`, `infeasible`, or `unknown`). Missing, stale, or malformed
  feasibility cannot authorize a selection.
- `scoring_config` must be an empty mapping until T03 defines supported scoring
  inputs. Non-empty configurations fail closed rather than being ignored.
- The relation direction is `blocker -> blocked`. Starting at `target_id`, the
  required set is the target plus every transitive incoming blocker; unrelated
  objects are never legal candidates. This is an AND dependency: all parent
  branches stay in the closure, while one confirmed frontmost object may be
  selected per observation.
- A legal candidate must be in that closure, have no incoming edge, have
  explicitly covered `visibility: "clear"`, and have matching
  `feasibility: "feasible"`. `relation_coverage` must be `complete` before an
  absent edge can support selection; `partial` or `unknown` coverage yields no
  legal candidates and an `uncertain` decision. Missing/unknown visibility or
  feasibility never authorizes selection. With the T03 scoring surface still
  empty, legal IDs are sorted numerically and the smallest is selected; scoring
  cannot override these gates.
- A relevant cycle returns `blocked` with `reason: "dependency_cycle"` and an
  empty legal set; the selector never breaks a cycle. Unknown evidence outside
  the target closure is diagnosed but does not veto an otherwise confirmed
  target. Feasibility `unknown` is uncertain; `infeasible` is rejected. No
  execution-completion state is accepted or inferred by this seam.
- The decision includes `status`, nullable `next_object_id`, `observation_id`,
  `reason`, sorted `legal_candidate_ids`, and per-object rejection diagnostics.
  Invalid/stale input and absent targets remain `blocked`; uncertainty never
  fabricates a selection. An empty relation list, even with complete relation
  coverage, does **not** make an object clear: explicit covered visibility
  evidence is required. The existing `plan_graph_policy` and `online_replan`
  interfaces still emit policy rankings and are not this safety-gated seam.
## Controlled Ablations

Run all variants with the same JSONL split, seed, model shape, and training
budget. Each variant saves `metrics.json` beside its checkpoint:

```bash
python -m obstruction_train.run_graph_policy_ablations \
  --train-jsonl demos/train.jsonl \
  --val-jsonl demos/val.jsonl \
  --out-dir outputs/graph_policy_ablations \
  --epochs 20 --seed 0 --device cuda
```

The matrix reports full context, no language, no obstruction edges, no support
edges, and no nearby edges. Compare `top1_accuracy`,
`mean_reciprocal_rank`, and `equivalent_action_top1`. The optional
`equivalent_action_ids` field records multiple equally valid next actions; an
optional `grasp_succeeded` field gives the observed execution success rate.
Repeat each variant over at least three seeds before reporting a result.
