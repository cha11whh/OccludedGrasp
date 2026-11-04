# R01 weightless replay

## Run

From the `frozen-grasp-scoring` worktree, run the CPU-only test suite with the
configured Python environment:

```sh
cd /home/media/zy/pi_work/.scratch/worktrees/frozen-grasp-scoring
timeout --signal=TERM --kill-after=5s 120s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests
```

Observed in the worktree based on HEAD `c1270dc77589b0ef2f325a1d8b20ca871588a2e1`:
**110 passed in 2.38s**.

Run the multi-step synthetic replay and print its JSON result:

```sh
cd /home/media/zy/pi_work/.scratch/worktrees/frozen-grasp-scoring
timeout --signal=TERM --kill-after=5s 30s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m obstruction_train.frozen_grasp_replay
```

Observed CLI result: valid JSON, `status: target_completed`, actions for object
IDs `[1, 2]`, and mode `{"replay": "cpu", "evidence": "mock"}`. The audit
contains prediction, geometry, path, execution, verification, and observation
events. Error outcomes use the same JSON output path; unrepresentable values are
normalized before serialization.

## Evidence limits

The CLI uses synthetic observations and injected callbacks defined in the
module. It does not load or run grasp-model weights, and the output is **not
model evidence**. Its geometry and path callbacks are illustrative mocks, not
robot collision checking, calibrated feasibility, or execution guarantees.
Rejecting every member of a finite generated candidate set leaves feasibility
`unknown`; it does not prove that the object or task is globally infeasible.
No GPU or robot hardware validation is covered by these results.