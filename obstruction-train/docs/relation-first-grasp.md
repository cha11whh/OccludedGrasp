# Relation-First Grasp: T00/T01 Status

## T00 — Baseline and seam

- Branch: `feature/relation-first-grasp`; starting baseline: `bf2eb5f62e1e5ae0632ff0af21bb6b9e178e72e9`.
- Public seam: `select_next_object(observation, target_id, feasibility, scoring_config)`.
- CPU baseline: `timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_relation_first_policy.py` — **7 passed**.

## T01 — Contract and evidence

- Visibility is explicitly `clear`, `occluded`, or `unknown` per object. Only IDs in `visibility_coverage` have visibility coverage evidence; omitted evidence/state stays unknown. `relation_coverage` alone cannot establish clear. Contradictions fail closed, including clear plus a positive incoming edge and occluded without an incoming edge when relation coverage is complete.
- Feasibility requires a non-empty `observation_id` equal to the current observation ID. Missing IDs and mismatches are blocked; old feasibility is not reused.
- The seam still returns no object. T02 relation gating and T03 scoring were not implemented.

Developer-reported public-test red/green runs, not independently retained as raw transcripts by the coordinator (each command used `timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q <test-node>`):

| Behavior | Red observed | Green observed |
| --- | --- | --- |
| Uncovered visibility is unknown | 1 failed: expected `visibility_unknown`, got deferred reason | 1 passed |
| Complete relation coverage conflicts with occluded state lacking an edge | 1 failed: expected blocked, got uncertain | 1 passed |
| Missing feasibility observation ID | 1 failed: expected blocked, got uncertain | 1 passed |
| Feasibility ID belongs to an earlier observation | 1 failed: expected blocked, got uncertain | 1 passed |

The positive-edge-versus-clear contradiction test passed on its first run; no red was observed for that case. No GPU, package installation, or environment repair was performed.