# S00 CPU bounded-runner evidence

## Coordinator recovery and durability review

An independent read-only review found that atomic ledger replacement did not
persist the parent directory. The coordinator added the public-seam test
`test_directory_sync_failure_prevents_worker_start`. Its first actual run failed
with `DID NOT RAISE OSError` (**1 failed in 0.05s**). After adding directory
`fsync` after `os.replace` and before worker startup, the complete CPU regression
passed: **74 passed in 2.25s**. Command from the task-worktree root:

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests`.

`git diff --check` passed. A directory durability failure propagates before any
worker starts; an on-disk pending marker remains conservative. This closes the
reviewed rename durability window on supported local POSIX storage, not a
guarantee against arbitrary storage failure or deletion of the ledger.

No GPU/model execution, training, dependency installation or protected-worktree
mutation occurred. The sections below retain the earlier developer evidence;
their “final” counts refer to that earlier checkpoint, not this later regression.

## Scope and protection

- Resumed task branch `feature/frozen-grasp-scoring` at base/current HEAD `5f4aa1afc88e70954b8b4528e2375d039caeb586`. The three untracked files already present on resumption were the task artifacts: `obstruction_train/budget_runner.py`, `tests/test_budget_runner.py`, and this note. No reset, cleanup, staging, or commit was performed.
- Original root `/home/media/zy/pi_work`: branch `feature/graph-transformer-policy`, HEAD `bf2eb5f62e1e5ae0632ff0af21bb6b9e178e72e9`; tracked-diff SHA256 remained the requested `a548aa806cbc6c590903d5c0551de00ac53c3f38223b408aefc9fdb4030134fa`.
- Prior relation-first worktree: branch `feature/relation-first-grasp`, HEAD `5f4aa1afc88e70954b8b4528e2375d039caeb586`, clean. OpenPI: branch `feature/pi05-sim-integration`, HEAD `ec28d3ddbe87b9544701fa8b1260030b5a902240`; its tracked-diff SHA256 remained `b92c273e879107183edde4318c754c9e3195dd8f3639c0880befea0ae975b582`. Contact-GraspNet remained at `721706067cc7f7889c32afe7b4f7955eb90bd866` with pre-existing compiled-artifact modifications. None of these protected checkouts was changed.
- CPU-only verification used `/home/media/zy/pi_work/.venv_perception/bin/python` (Python 3.11.15, pytest 8.3.5) with `CUDA_VISIBLE_DEVICES=''`. No GPU work, training, dependency installation, or environment mutation occurred.

## Public behavior

`obstruction_train.budget_runner.run_bounded(command, *, ledger_path, timeout_seconds, budget_seconds=7200)` serializes ledger check, reservation, and execution under an OS lock. It defaults to a cumulative 7200-second allowance, rejects exhausted or invalid state before execution, and fails closed on a crash-left `pending` marker. Success, nonzero process exits, timeouts, and OS-level worker-start failures are charged. A worker-start failure clears `pending` only after recording the failed attempt; interrupted/uncertain runs remain fail-closed. Timeout cleanup terminates the entire POSIX process group (TERM, then KILL after a bounded grace period). Per-run deadlines are capped by remaining budget with a cleanup reserve.

The result reports process execution status only: `returncode == 0` is not evidence that a model passed validation or produced a correct result. Callers must validate model outputs separately. The caller supplies a durable ledger path; do not put it in disposable storage or reset it to obtain a new allowance.

## Test evidence

Commands below include the configured interpreter and environment explicitly. Each test command used an outer `timeout` that terminates the process tree.

- Interrupted runner tests, before edits; working directory `/home/media/zy/pi_work/.scratch/worktrees/frozen-grasp-scoring/obstruction-train`:
	`timeout --foreground 60s env CUDA_VISIBLE_DEVICES='' /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q tests/test_budget_runner.py` — **10 passed in 1.09s**. No prior RED was recreated or claimed.
- Full baseline, before edits; working directory `/home/media/zy/pi_work/.scratch/worktrees/frozen-grasp-scoring`:
	`timeout --signal=TERM --kill-after=5s 120s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` — **72 passed in 2.33s**.
- TDD RED, same task-worktree working directory and environment:
	`timeout --signal=TERM --kill-after=3s 30s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_budget_runner.py::test_worker_start_failure_is_charged_without_poisoning_ledger` — **1 failed in 0.03s**, actual assertion `charged_seconds > 0` observed `0.0` after a missing executable raised `FileNotFoundError`.
- TDD GREEN, same command — **1 passed in 0.01s** after accounting for the spawn attempt and clearing `pending` when no worker was created.
- Final targeted suite, same task-worktree working directory and environment:
	`timeout --signal=TERM --kill-after=5s 45s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_budget_runner.py` — **11 passed in 1.09s**.
- Final full CPU regression:
	`timeout --signal=TERM --kill-after=5s 120s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests` — **73 passed in 2.31s**.

Process-group timeout and TERM-resistant descendant cleanup tests passed; they check that child processes are gone or zombies after cleanup. No test command timed out or failed for infrastructure reasons. The only failure was the intentional assertion-level TDD RED above; all final targeted and regression runs passed within their stated budgets.