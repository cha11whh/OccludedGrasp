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

## T06 — CPU-Only Replay Loop

`obstruction_train.replay_only.run_replay(initial_observation, target_id,
initial_feasibility, executor, verifier, observer, sim_clock,
scoring_config=None)` drives the existing `select_next_object` seam through
select → execute → verify → observe → replan. The callbacks are supplied by the
caller; this implementation does not perform perception, robot control, or
simulation. Tests use synthetic fixtures only. Passing these tests is not
physical-removal, grasp, placement, or simulator evidence.

- The executor receives exactly `(selected_object_id, source_observation_id)`;
	an execution receipt for another object stops the loop. Each attempted action
	increments the persistent object count before calling the executor, including
	exceptions and failed receipts. At two attempts, that object is made
	unselectable for later replans; the loop stops with `attempt_limit` when no
	candidate remains.
- The verifier receives the executor receipt and source observation ID. Its
	required feedback contains matching `object_id` and `observation_id`, boolean
	`physical_removal_confirmed`, and boolean `placement_verified`. A placement
	cannot be verified unless physical removal is confirmed. Completion requires
	both flags for the selected target, then a valid fresh observation; executor
	success alone or an old observation never completes the target.
- Every action is followed by a new observation before another selection.
	Observation IDs must be non-empty, non-stale, and never repeated in the replay;
	duplicate or stale observations stop before another execution. Uncertain
	decisions trigger at most three consecutive fresh reobservations and never
	execute. A selection resets that consecutive-unknown count, but never resets
	per-object attempts.
- `sim_clock` is an injected caller-owned source of simulation-time seconds. Each
	sample must be finite, nonnegative, and monotonic. At 300 seconds the replay
	stops; no wall-clock sleeps are used. This is CPU-side state accounting only,
	not a GPU watchdog, process watchdog, simulator timeout enforcement, or
	resource budget ledger. Exceptions in callbacks and malformed feedback fail
	closed with a stop reason. Execution receipts remain in `actions` even if a
	later verifier or time check stops the replay.
- The selector continues to own relation, feasibility, and scoring decisions;
	the replay does not see or inject hidden simulation truth and does not reorder
	candidates. The existing selector's `target_completed` status is not inferred
	from selection.

TDD and CPU verification (all commands run from the task worktree with
`CUDA_VISIBLE_DEVICES=''` and
`PYTHONPATH=obstruction-train`):

| Check | Command/result |
| --- | --- |
| Baseline before T06 changes | `timeout --signal=TERM --kill-after=5s 240s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` — **48 passed in 1.18s** |
| Observed RED, first public-seam target completion behavior | `... -m pytest -q obstruction-train/tests/test_replay_only.py::test_target_is_completed_only_after_bound_verified_placement_and_reobserve` — **1 failed**, expected result lacked `status` (`KeyError`); then implemented the first vertical slice and observed **1 passed in 0.01s**. |
| Observed RED, malformed feasibility fail-closed reason | `... -m pytest -q obstruction-train/tests/test_replay_only.py::test_invalid_initial_feasibility_fails_closed_without_escaping` — **1 failed**, replay returned generic `selector_error` instead of a stable selector reason; corrected the behavior, then aligned the assertion with the pre-existing selector's actual `unsupported_distance_unit` schema response. |
| Full T06 public-seam tests | `timeout --signal=TERM --kill-after=5s 60s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_replay_only.py` — **13 passed in 0.01s**. |
| Full `obstruction-train/tests` regression | `timeout --signal=TERM --kill-after=5s 180s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` — **61 passed in 1.19s**. |

The literal `...` above abbreviates only the repeated interpreter prefix:
`timeout --signal=TERM --kill-after=5s 60s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python`;
focused tests had a 60-second hard deadline (baseline/full regression: 240
seconds). No dependency installation, GPU use, physical operation, or simulation
was performed. Formal T04 perception/evaluation data adapters and T05 physical
single-object grasp Gate remain unimplemented; T06's injected callbacks do not
replace either gate.

### T06 independent spec-review follow-up

The independent review's blocker-removal bypass and frozen-clock identity loop
are addressed at the public `run_replay` seam:

- The identity set is established from the initial valid observation. A later
	observation introducing any new object ID stops with
	`unknown_object_identity`; IDs are never silently reinterpreted or used to
	refresh an object's attempt budget. This is a conservative stop requiring
	caller-side identity reconstruction and reauthorization, not an iteration
	cap.
- A known object may disappear from a later graph only when its removal was
	confirmed by the verifier. Any prior blocker edge whose endpoints remain
	unremoved must also remain present. A contradictory snapshot stops with
	`unconfirmed_removal_observation_mismatch`; the replay does not synthesize or
	repair graph edges and does not replan from that snapshot.
- A mismatch result preserves `observation_id` as the last accepted snapshot
	and includes `rejected_observation_id` for the contradictory fresh snapshot.
	The action record includes the selected object, source observation, and
	verifier feedback (including `physical_removal_confirmed: false` for this
	report). This is the interruption evidence; the rejected graph itself is not
	committed as current replay state.
- `physical_removal_confirmed: true` with
	`placement_verified: false` stops immediately with `placement_unverified`,
	before calling the observer or selecting another action. This prevents a
	confirmed removal without a verified destination from continuing simulated
	handling.

TDD evidence for this follow-up (all commands CPU-only, bounded, from the task
worktree):

| Check | Observed result |
| --- | --- |
| Pre-follow-up baseline: `timeout --signal=TERM --kill-after=5s 180s env CUDA_VISIBLE_DEVICES='' /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` | **58 passed in 1.19s**. This included the original ten replay-only T06 tests already green; they are not represented as having an original failing run. |
| Blocker-removal regression RED before transition guard | **1 failed**: `run_replay` returned `target_completed` when object 2 had `physical_removal_confirmed: false` but the fresh snapshot removed object 2 and its edge and marked target 7 clear. |
| Placement regression RED before placement stop | **1 failed**: returned `repeated_observation` after observing instead of stopping as `placement_unverified` before another action. |
| Stable-identity regression RED with the new-ID guard temporarily disabled and the fixture bounded by a repeated observation ID | **1 failed**: returned `repeated_observation`, not `unknown_object_identity`, despite a frozen `sim_clock`; the regression terminates deterministically without relying on a timeout. |
| Precise interruption-evidence assertion RED | **1 failed** with `KeyError: 'rejected_observation_id'`; adding this field made the rejected fresh snapshot identifiable in the stop result. |
| Three new follow-up safety tests after fixes | **3 passed in 0.03s**. |
| Replay-only module, including original tests | **13 passed in 0.01s**. |
| Full regression after all changes | **61 passed in 1.19s**. |

No test or command timed out; the one initial focused invocation without the
repository import path failed during collection with `ModuleNotFoundError` and
was rerun with `PYTHONPATH=obstruction-train`. No dependencies were installed,
and no GPU, simulator, or physical system was used. The existing original T06
RED claims above document the earlier implementation's history; they are not
newly reproduced as part of this follow-up.