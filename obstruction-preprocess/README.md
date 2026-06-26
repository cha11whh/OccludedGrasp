# Obstruction Preprocess

Estimate pairwise obstruction information from:

- RGB image
- relative depth map
- modal masks
- amodal/final masks

The output is designed to be compatible with the `occ_detail_json` style consumed by UnoGrasp's `data_gen/data_gen_som_occ.py`.

## What It Estimates

For every ordered pair `(blocker, blocked)`, the script estimates:

- whether the blocker obstructs the blocked object
- obstruction ratio
- contact point
- severity degree: `slightly`, `partially`, `mostly`, `heavily`
- optional depth statistics around the contact region

The main cue is:

```text
hidden_region(blocked) = amodal_mask(blocked) - modal_mask(blocked)
```

If another object's modal mask touches or is close to that hidden region, it is treated as a candidate blocker. The graph edge is decided by contact evidence plus optional depth-order support:

```text
edge(blocker -> blocked) =
  contact near hidden region is strong enough
  AND relation confidence passes threshold
```

`obstruction_ratio`, `hidden_area`, and `degree` are saved as edge features for downstream grasp policy or RL. They are not used as hard criteria for graph construction.

## Quick Run On Current Image

From `/home/ubuntu22-zy/occlusion`, run:

```bash
docker run --rm --gpus all \
  -v /home/ubuntu22-zy/occlusion:/home/ubuntu22-zy/occlusion \
  --ipc=host gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-preprocess &&
python -m obstruction_preprocess.estimate_obstruction \
  --rgb /home/ubuntu22-zy/occlusion/amodal/experiment_input/1.jpg \
  --depth /home/ubuntu22-zy/occlusion/depth-anything-v2/outputs/vits_single/1_raw.npy \
  --modal-dir /home/ubuntu22-zy/occlusion/amodal/targeted_modal_masks_1/1 \
  --modal-metadata /home/ubuntu22-zy/occlusion/amodal/targeted_modal_masks_1/1/metadata.json \
  --amodal-dir /home/ubuntu22-zy/occlusion/amodal/experiment_output_fast_modal_first_allowlist/1 \
  --amodal-metadata /home/ubuntu22-zy/occlusion/amodal/experiment_output_fast_modal_first_allowlist/1/mask_metadata.json \
  --scene-id 1 \
  --view-id 0 \
  --dedupe-iou 0.98 \
  --depth-transform inverse \
  --relation-mode model \
  --relation-model-checkpoint /home/ubuntu22-zy/occlusion/obstruction-train/outputs/pair_transformer_v1_base_da_vitl_ft_e1/best.pt \
  --relation-model-min-confidence 0.50 \
  --out-dir /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1
'
```

## Parallel RGB Pipeline

To run Depth Anything and Grounded-SAM/SAM in parallel, then build the obstruction graph after both finish:

```bash
python3 obstruction-preprocess/run_parallel_rgb_pipeline.py \
  --rgb /home/ubuntu22-zy/occlusion/amodal/experiment_input/1.jpg \
  --input-dir /home/ubuntu22-zy/occlusion/amodal/experiment_input \
  --target-prompts-txt /home/ubuntu22-zy/occlusion/amodal/target_prompts_1.txt \
  --amodal-classes-txt /home/ubuntu22-zy/occlusion/amodal/amodal_classes_1.txt
```

The script starts two Docker branches at the same time:

```text
RGB -> Depth Anything -> depth .npy
RGB -> GroundingDINO/SAM -> modal masks -> optional amodal masks
```

After both branches finish, it runs `estimate_obstruction` to produce `occ_detail.json`.
If `--rl-target` or `--run-rl-affordance true` is set, it also runs `rl_affordance_preprocess` to produce reward-ranked grasp actions.

Useful flags:

- `--relation-mode model`: use the fine-tuned PairRelationTransformer to predict pairwise obstruction edges. This is the default graph builder.
- `--relation-mode heuristic`: use the previous amodal-hidden-region/contact heuristic.
- `--relation-model-checkpoint`: relation checkpoint path. The current default is `obstruction-train/outputs/pair_transformer_v1_base_da_vitl_ft_e1/best.pt`.
- `--relation-model-min-confidence 0.50`: minimum model confidence for writing an obstruction edge.
- `--rl-target "white thermos bottle"`: after graph construction, rank next grasp actions for a target object.
- `--run-rl-affordance true`: rank grasp actions for the whole scene without a specific target.
- `--amodal-completion-mode geometric`: mask-only amodal completion. This is the default fast path and avoids Stable Diffusion/LaMA/InstaOrder.
- `--amodal-completion-mode iterative`: original heavy inpainting-based amodal completion.
- `--prompt-mode combined`: run one combined GroundingDINO prompt and one SAM pass. This is the default fast path.
- `--prompt-mode per_prompt`: more conservative prompt-by-prompt GroundingDINO, still with one SAM pass.
- `--depth-encoder vitl`: use Depth Anything V2 Large. This is the default accuracy-oriented depth path.
- `--depth-transform inverse`: normalize Depth Anything output and flip its direction before graph/model use. This is the default because UnoBench-trained relation models expect the opposite convention from raw Depth Anything maps.
- `--run-amodal false`: fast modal-only check. This validates the parallel scheduling, but hidden regions are empty, so obstruction edges may be empty.
- `--process-max-side 1600`: run obstruction graph estimation on a downscaled image and map points back to original coordinates. This is the default fast path.
- `--save-visualizations true`: write large overlay PNGs for debugging. The default fast path leaves these off.
- `--dry-run`: print the Docker commands without running them.
- `--skip-existing true`: reuse existing depth/mask outputs when present.

Current timing on `1.jpg`:

- old modal-only parallel validation: total `99.95s`, depth `11.29s`, mask `93.55s`, graph `6.39s`
- geometric amodal + combined prompt + single SAM: total `42.93s`, depth `7.43s`, mask `31.94s`, graph `10.99s`
- light modal script + no visualization + graph downscale: total `32.13s`, depth `7.72s`, mask `29.02s`, graph `3.11s`
- Depth Anything V2 Large (`vitl`) + fast mask/graph: total `32.30s`, depth `12.23s`, mask `29.17s`, graph `3.13s`
- model relation graph only, reusing existing depth/masks: about `6.15s` for 5 objects / 10 pairs
- geometric amodal completion alone, reusing existing modal masks: about `9s`
- graph estimation alone at `--process-max-side 1600`: about `2s`

A serial run would be roughly depth + mask + graph; the parallel script overlaps depth with the mask branch.

## Outputs

```text
outputs/1/
|-- objects.json
|-- occ_detail.json
|-- occlusion_paths.json
|-- duplicate_objects.json
|-- rl_affordance_<target>/ranked_actions.json
|-- overlay_objects.png
|-- overlay_obstructions.png
`-- hidden_regions/
```

`occ_detail.json` entries use:

```json
{
  "scene_id": "1",
  "view_id": "0",
  "obj1": 2,
  "obj2": 0,
  "blocker": 2,
  "blocked": 0,
  "subject": 2,
  "object": 0,
  "relation": "obstructs",
  "mask_ratio": 0.12,
  "degree": "partially",
  "point": {"x": 123, "y": 456},
  "contact": {"score": 0.21},
  "relation_confidence": 0.83,
  "edge_features": {
    "obstruction_ratio": 0.12,
    "degree": "partially",
    "contact_score": 0.21,
    "depth_order_score": 0.64
  },
  "graph_decision": {
    "used_obstruction_ratio_for_edge": false
  },
  "mask_path": "hidden_regions/blocked_0_by_2.png"
}
```

## RL Affordance Reward Ranking

The first RL-facing module is a reward reranker, not a learned policy yet. It turns the obstruction graph into interpretable action candidates:

```bash
docker run --rm \
  -v /home/ubuntu22-zy:/home/ubuntu22-zy \
  gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-preprocess &&
python -m obstruction_preprocess.rl_affordance_preprocess \
  --objects outputs/1_vitl_model_pipeline_test/objects.json \
  --relations outputs/1_vitl_model_pipeline_test/occ_detail.json \
  --target "white thermos bottle" \
  --depth /home/ubuntu22-zy/occlusion/depth-anything-v2/outputs/vitl_parallel_1600_test/1_raw.npy \
  --out-dir outputs/1_vitl_model_pipeline_test/rl_affordance_white_thermos
'
```

It writes:

```text
ranked_actions.json
candidates.json
reward_terms.json
```

The reward is:

```text
success_prior
+ graph_progress
+ affordance
+ collision_penalty
+ efficiency
+ target_priority
+ target_release_gain
```

`blocked_grasp_region_ratio` is kept as an explicit term: it penalizes trying to grasp an object whose candidate grasp region is occupied by incoming blockers, and rewards removing blockers that release target graspability.

## Grasp Handoff

The grasp order can be handed off to a lower-level grasp network with:

```bash
docker run --rm \
  -v /home/ubuntu22-zy:/home/ubuntu22-zy \
  gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-preprocess &&
python -m obstruction_preprocess.grasp_handoff \
  --ranked-actions outputs/1_vitl_model_rl_pipeline_test/rl_affordance_white_thermos_bottle/ranked_actions.json \
  --objects outputs/1_vitl_model_rl_pipeline_test/objects.json \
  --backend anygrasp \
  --rgb /home/ubuntu22-zy/occlusion/amodal/experiment_input/1.jpg \
  --metric-depth /path/to/real_metric_depth.npy \
  --camera-intrinsics /path/to/camera_intrinsics.json \
  --out outputs/1_vitl_model_rl_pipeline_test/grasp_handoff_white_thermos.json
'
```

Recommended backend:

- primary: AnyGrasp, for real 6/7-DoF grasp candidates in clutter.
- fallback: Contact-GraspNet, for object-centric 6-DoF candidates from point cloud contacts.

`grasp_handoff` selects the top reward-ranked object, passes its object mask/bbox to the grasp backend, and can rerank backend candidates if `--grasp-candidates` is provided.

Important: real execution needs calibrated metric RGB-D depth. Depth Anything relative depth is useful for obstruction reasoning, but it should not be used as the metric depth input for a physical 6-DoF grasp pose.

In UnoGrasp's prompt-generation code, `obj1` is the blocker and `obj2` is the blocked object.

## Simulated Grasp Experiment

Before connecting to a real robot, you can validate the planned grasp order in a lightweight PyBullet scene:

```bash
docker run --rm \
  -v /home/ubuntu22-zy:/home/ubuntu22-zy \
  gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-preprocess &&
PYTHONPATH=/home/ubuntu22-zy/occlusion/.deps/pybullet:$PYTHONPATH \
python -m obstruction_preprocess.sim_grasp_experiment \
  --task-mode reach_target \
  --objects outputs/1_vitl_model_rl_pipeline_test/objects.json \
  --relations outputs/1_vitl_model_rl_pipeline_test/occ_detail.json \
  --ranked-actions outputs/1_vitl_model_rl_pipeline_test/rl_affordance_white_thermos_bottle/ranked_actions.json \
  --out outputs/1_vitl_model_rl_pipeline_test/sim_reach_white_thermos_tabletop.json \
  --visualize \
  --frames-dir outputs/1_vitl_model_rl_pipeline_test/sim_frames_reach_white_thermos_tabletop \
  --gif outputs/1_vitl_model_rl_pipeline_test/sim_reach_white_thermos_tabletop.gif
'
```

Task modes:

- `--task-mode reach_target`: remove blockers above the target, then grasp the target itself.
- `--task-mode clear_table`: repeatedly grasp accessible objects until the tabletop is empty.
- `--camera-preset front_occlusion --layout-mode front_occlusion --add-demo-props`: use a low front-facing camera with many tabletop objects placed in front/back rows, creating strong visual occlusion without vertical stacking.

This approximates each segmented object as a tabletop item, stacks objects according to the obstruction graph, lifts/carries away each selected object, and checks task completion.

The current white-thermos `reach_target` test executes `cream fabric pouch -> white thermos bottle`.
The current `clear_table` test executes all five objects and leaves the table empty.

Visual outputs:

```text
sim_reach_white_thermos_tabletop.gif
sim_clear_table_tabletop.gif
sim_frames_<task>/frame_*.png
```

## Isaac Lab Simulation Backend

For higher-fidelity rendering, sensors, PhysX contact, and future RL/robot integration, use the Isaac Lab backend:

```bash
export ISAACLAB_ROOT=/home/ubuntu22-zy/occlusion/IsaacLab

/home/ubuntu22-zy/occlusion/obstruction-preprocess/run_isaac_lab_tabletop.sh \
  --headless \
  --task-mode reach_target \
  --objects /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1_vitl_model_rl_pipeline_test/objects.json \
  --relations /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1_vitl_model_rl_pipeline_test/occ_detail.json \
  --ranked-actions /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1_vitl_model_rl_pipeline_test/rl_affordance_white_thermos_bottle/ranked_actions.json \
  --add-demo-props \
  --out /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1_vitl_model_rl_pipeline_test/isaac_reach_white_thermos.json \
  --save-usd /home/ubuntu22-zy/occlusion/obstruction-preprocess/outputs/1_vitl_model_rl_pipeline_test/isaac_reach_white_thermos.usd
```

Use `--task-mode clear_table` to clear every object from the table. The script uses the same planner inputs as the PyBullet backend, but spawns the scene in Isaac Lab/Isaac Sim with a front-facing tabletop layout. The first version uses scripted kinematic pick/carry motions; the next step is to replace that motion primitive with a Franka/UR10 gripper controller and AnyGrasp/VGN grasp poses.

If Isaac Lab is not installed, `run_isaac_lab_tabletop.sh` prints the expected `ISAACLAB_ROOT` setup. Official references:

- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/main/index.html
- Isaac Lab GitHub: https://github.com/isaac-sim/IsaacLab

## Query One Target

After estimating relations, query a target object by ID or name:

```bash
docker run --rm \
  -v /home/ubuntu22-zy/occlusion:/home/ubuntu22-zy/occlusion \
  gsa:amodal bash -lc '
cd /home/ubuntu22-zy/occlusion/obstruction-preprocess &&
python -m obstruction_preprocess.query_target \
  --objects outputs/1_dedup/objects.json \
  --relations outputs/1_dedup/occ_detail.json \
  --rgb /home/ubuntu22-zy/occlusion/amodal/experiment_input/1.jpg \
  --target "white thermos bottle" \
  --out outputs/1_dedup/query_white_thermos.json \
  --vis outputs/1_dedup/query_white_thermos.png
'
```

The query result follows the target-centric obstruction-graph idea in UNOGrasp: start from the target, repeatedly follow incoming blockers, and stop when an object has no blocker above it.

Example output shape:

```json
{
  "target": {"id": 0, "name": "white thermos bottle"},
  "paths": [
    {
      "object_ids": [0, 2],
      "top_object": {"id": 2, "name": "cream fabric pouch"},
      "steps": [
        {
          "blocked": {"id": 0, "name": "white thermos bottle"},
          "blocker": {"id": 2, "name": "cream fabric pouch"},
          "ratio": 0.446,
          "degree": "mostly",
          "contact_point": {"x": 2082, "y": 4205}
        }
      ]
    }
  ],
  "answer": [
    {"id": 2, "name": "cream fabric pouch"}
  ]
}
```

## Notes

This module can run either the trained PairRelationTransformer relation model or the earlier heuristic obstruction estimator. The default pipeline uses the trained model. The most important input quality factors are still object masks and depth convention; poor masks or wrong depth direction will directly hurt relation accuracy.

If two detections have almost identical modal masks, the script can merge duplicates with `--dedupe-iou`. Set `--dedupe-iou 1.1` to disable this behavior.
