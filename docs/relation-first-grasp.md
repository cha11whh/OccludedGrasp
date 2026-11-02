# Relation-first grasp development evidence

## T00: isolation and dependency boundary

- Approved task branch: `feature/relation-first-grasp`, isolated in the original repository's ignored `.scratch/worktrees/relation-first-grasp` directory.
- Integration baseline: `bf2eb5f62e1e5ae0632ff0af21bb6b9e178e72e9` (`feature/graph-transformer-policy`). The original checkout was not switched.
- Before implementation, original unstaged diff SHA256: `a548aa806cbc6c590903d5c0551de00ac53c3f38223b408aefc9fdb4030134fa`; staged diff was empty (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- Independent OpenPI HEAD: `ec28d3ddbe87b9544701fa8b1260030b5a902240`; unstaged diff SHA256: `b92c273e879107183edde4318c754c9e3195dd8f3639c0880befea0ae975b582`; staged diff empty. No OpenPI changes or untracked scripts were imported.
- The existing modified `.gitignore` already ignored `.scratch/`; it was not edited to create isolation. Existing original modified files and untracked directories remain outside this task.
- CPU runtime fallback: `/home/media/zy/pi_work/.venv_perception/bin/python`. Editor environment configuration and executable lookup both returned `Language model unavailable`; no claim is made that the editor interpreter was configured. No packages installed or upgraded.
- This checkout includes the committed graph policy, graph builder and replanning modules. Their ranked outputs lack explicit visibility evidence, freshness attestation and observation-bound feasibility required by the new seam. They are not yet adapted or wired to physical execution.
- Checkpoints, calibrated live perception, official JAKA assets, a gripper/path executor and OpenPI are external dependencies, not validated or imported task implementations. Any reuse of dirty files requires separate authorization.

## T01: decision contract

The public seam is `obstruction_train.relation_first_policy.select_next_object`.
See the training README for the schema. T01 validates inputs and returns no selected object: relation gating and ranking remain T02/T03 work. Visibility assessment is an adapter attestation, not a proof supplied by the existing pair checkpoint. Feasibility records must refer to the current observation. No absent relation or successful command implies physical success.

Observed coordinator validation (2026-11-02):

- `CUDA_VISIBLE_DEVICES='' PYTHONPATH=<worktree>/obstruction-train:<worktree>/obstruction-preprocess /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests`: **18 passed in 1.16 seconds** (13 new contract tests and 5 existing graph tests).
- `git diff --check`: passed.
- Independent initial spec review rejected missing visibility evidence and missing feasibility observation binding; these were corrected before re-review.
- Developer invocations returned no execution narrative. The coordinator observed GREEN regression but has no retained RED transcript, so T01 is **not claimed as verified red-before-green**. Later tickets must record actual red/green evidence explicitly.

GPU occupation for this task: **0 seconds**. No simulation, training or physical grasp success is claimed.