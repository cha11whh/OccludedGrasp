# K00 budget runner evidence

`run_bounded(command, *, ledger_path, timeout_seconds, budget_seconds=7200)` keeps
the existing 7,200-second default. Callers may explicitly select a positive,
finite cap up to 14,400 seconds. Boolean and non-finite values are not accepted.

The selected cap is persisted in the ledger. An existing ledger can only be
reused with the same cap; an attempted escalation or downgrade raises
`BudgetError` before the worker starts. Use a separate ledger for a separately
budgeted stage. The runner's pending recovery, time charging, process-group
cleanup, exhaustion check, and durable ledger writes are unchanged.

## Verification

The full public-seam suite passed with the approved interpreter, CPU-only:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train /home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q obstruction-train/tests/test_budget_runner.py
16 passed in 1.18s
```

The suite includes default-cap persistence, explicit 14,400-second persistence
and reuse, both directions of persisted-cap mismatch with worker-start markers,
invalid-cap rejection with worker-start markers, and prior failure-accounting,
timeout cleanup, pending-ledger, fsync, and exhausted-budget checks. No GPU or
robot worker was run for this verification.