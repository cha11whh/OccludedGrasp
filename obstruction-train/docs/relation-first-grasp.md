# Relation-First Grasp: T00/T01/T02/T03 Status

## T00 — Baseline and seam

- Branch: `feature/relation-first-grasp`; starting baseline: `bf2eb5f62e1e5ae0632ff0af21bb6b9e178e72e9`.
- Public seam: `select_next_object(observation, target_id, feasibility, scoring_config)`.
- CPU baseline: `timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_relation_first_policy.py` — **7 passed**.

## T01 — Contract and evidence

- Visibility is explicitly `clear`, `occluded`, or `unknown` per object. Only IDs in `visibility_coverage` have visibility coverage evidence; omitted evidence/state stays unknown. `relation_coverage` alone cannot establish clear. Contradictions fail closed, including clear plus a positive incoming edge and occluded without an incoming edge when relation coverage is complete.
- Feasibility requires a non-empty `observation_id` equal to the current observation ID. Missing IDs and mismatches are blocked; old feasibility is not reused.
- At the T01 checkpoint the seam returned no object; T02 relation gating and T03 scoring are recorded below as subsequent checkpoints.

Developer-reported public-test red/green runs, not independently retained as raw transcripts by the coordinator (each command used `timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q <test-node>`):

| Behavior | Red observed | Green observed |
| --- | --- | --- |
| Uncovered visibility is unknown | 1 failed: expected `visibility_unknown`, got deferred reason | 1 passed |
| Complete relation coverage conflicts with occluded state lacking an edge | 1 failed: expected blocked, got uncertain | 1 passed |
| Missing feasibility observation ID | 1 failed: expected blocked, got uncertain | 1 passed |
| Feasibility ID belongs to an earlier observation | 1 failed: expected blocked, got uncertain | 1 passed |

The positive-edge-versus-clear contradiction test passed on its first run; no red was observed for that case.

## T02 — Relation gating and evidence

- At the T02 checkpoint, `select_next_object` computes the target's transitive incoming-blocker closure, gates candidates on that closure, no incoming edge, explicitly covered `clear` visibility, matching `feasible` status, and complete relation coverage. `partial` or `unknown` relation coverage cannot authorize selection. T02 required an empty scoring configuration; T03 adds explicit frozen scoring modes.
- Legal candidate IDs are sorted numerically, so ties do not depend on object, relation, or feasibility input order. Shared predecessors in a diamond are included once in the closure. Relevant dependency cycles remain blocked.
- All decision paths include `legal_candidate_ids`; early blocked decisions expose an empty list. The T02 docstring described relation gating with scoring deferred to T03.

Observed T02 test evidence (CPU, existing `.venv_perception`, no installation):

| Check | Exact test node / result |
| --- | --- |
| Coordinator recovery note for the clear-target contract | 1 failed, 13 passed (reported by the coordinator; raw transcript not retained here) |
| Missing target, before fix | `test_missing_target_is_blocked`: 1 failed in 0.04s, `KeyError: 'legal_candidate_ids'` |
| Missing target, after fix | `test_missing_target_is_blocked`: 1 passed in 0.01s |
| Reverse-input stable-ID tie-break | `test_candidate_tie_break_is_stable_when_input_order_is_reversed`: 1 passed in 0.03s on first run; no RED observed |
| Shared-predecessor diamond | `test_diamond_dependencies_share_one_legal_frontier_predecessor`: 1 passed in 0.03s on first run; no RED observed |
| Unknown relation coverage | `test_unknown_relation_coverage_does_not_authorize_a_clear_target`: 1 passed in 0.03s on first run; no RED observed |
| Full `obstruction-train/tests` CPU regression | `python -m pytest -q tests`: 29 passed in 1.21s |

Each targeted run used
`timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q tests/test_relation_first_policy.py::<test-name>`;
the full regression used
`timeout --signal=TERM --kill-after=5s 120s /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q tests`.
No GPU, package installation, or environment repair was performed.

## T03 — Frozen candidate scoring

- The seam accepts only frozen `fixed`, `feasibility`, and `utility` modes; fixed mode orders legal IDs, feasibility uses candidate quality minus execution cost, and utility adds normalized unique transitive successor reach. No weights are configurable.
- Every mode uses the existing relation, confirmed-visibility, target-closure, and feasibility gates. Utility diagnostics call the successor term structural dependency reach and explicitly do not claim AND prerequisites are unlocked.
- Scored modes fail closed with `uncertain` when any legal candidate lacks a finite normalized score; partial ranking does not choose among an incomplete legal set.
- Environment: `/home/media/zy/pi_work/.venv_perception/bin/python`, Python 3.11.15, pytest 8.3.5; no install/upgrade. Initial developer environment-configuration tools were unavailable; the coordinator subsequently configured and retrieved this exact interpreter successfully before independent validation.
- Developer-recorded evidence: existing interrupted policy tests ran before new behavior tests: 41 passed. New REDs were recorded for missing score semantics (`KeyError: 'scoring_mode'`) and mixed missing scores (`KeyError(2)` escaped instead of returning uncertain); each was followed by targeted GREEN. The coordinator did not independently retain those RED transcripts.
- Coordinator-observed full CPU regression: `PYTHONPATH=obstruction-train CUDA_VISIBLE_DEVICES='' /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` — **48 passed in 1.21 seconds**. `git diff --check` passed.
- Independent read-only `spec-reviewer`: PASS, no blocking findings. GPU, simulation and training: zero; this is candidate-selection logic, not evidence of physical grasp success.