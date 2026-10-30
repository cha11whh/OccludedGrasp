# RGB-D Geometric Grasp Scoring (No Training)

This optional path ranks planar parallel-jaw grasp hypotheses from simulator depth, instance masks, and camera intrinsics. It requires no learned checkpoint and performs no fine-tuning. It is a deterministic geometry heuristic, not a pretrained neural grasp model.

## Score semantics

`scripts/rgbd_geometric_grasp.py` projects valid object-mask depth pixels to camera-space XYZ and samples candidate centers along four image-plane jaw axes. Candidates are filtered by gripper width and ranked using jaw-width fit, opposed-contact symmetry/support, and contact-depth consistency. It returns a `grasp_quality_score` in `[0, 1]` and up to `top_k` candidates.

The score is an **uncalibrated ranking utility**, not a success probability. Candidates are planar hypotheses only; they are not complete 6-DoF poses and do not include IK or swept-volume collision checks.

## Optional occlusion-route integration

The runtime integration is in `scripts/isaac_occluded_grasp_feedback_loop.py` and is opt-in. Enable both `--arm-routing joint-affordance` and `--joint-rgbd-grasp-quality`. The feedback loop computes scores only for the target and its OAPRN blocking ancestors, then blends object-level geometry quality with the existing geometric/kinematic motion proxy using `--joint-grasp-quality-weight` (default `0.5`; motion receives the remaining weight). Motion infeasibility remains a hard gate. OAPRN still determines target-route membership and relation direction; the geometric scorer does not alter occlusion edges. Off-route objects are excluded from this scoring decision.

Per-object scores, errors, and top candidates are saved to `rgbd_grasp_quality.json` in the cycle output folder. The resulting score can change which feasible target-path object/arm pair is selected. It does **not** replace the later Pi05/visual-IK grasp target with the returned geometric candidate pose.

When the flag is omitted, the legacy routing behavior is unchanged.

## Example invocation

Run from the Isaac Lab environment used by this project, with the usual Pi05 and obstruction-model checkpoints configured. For one simulator cycle:

```bash
./isaaclab.sh -p /home/media/zy/pi_work/openpi_official/scripts/isaac_occluded_grasp_feedback_loop.py \
  --headless --enable_cameras \
  --out-dir /tmp/rgbd_joint_affordance \
  --target RedBoxProxy --cycles 1 \
  --arm-routing joint-affordance \
  --joint-rgbd-grasp-quality \
  --joint-grasp-quality-weight 0.5
```

This starts Isaac and uses GPU resources. Do not run while another user's training job occupies the GPU. It is a simulator command, not hardware-control guidance.

## CPU-side tests

From the repository root:

```bash
PYTHONPATH=scripts /home/media/miniforge3/bin/pytest -q \
  scripts/test_rgbd_geometric_grasp.py \
  scripts/test_rgbd_grasp_motion_fusion.py \
  scripts/test_occlusion_affordance_planner.py
```

These tests cover deterministic software behavior with synthetic inputs. They do not establish physical grasp success, calibration, generalization, or full Isaac runtime validity.

## Limitations

- Instance masks are assumed to identify the intended visible object correctly.
- The heuristic tests four in-plane jaw axes, not arbitrary 3-D rotations.
- The score and fixed fusion weight are neither learned nor calibrated.
- Contact-GraspNet's optional pretrained checkpoint path is separate; its weights are not bundled or required here.
