"""Bounded CPU-process runner with a persistent cumulative time budget."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Sequence


DEFAULT_BUDGET_SECONDS = 7200.0
_CLEANUP_RESERVE_SECONDS = 0.5


class BudgetError(RuntimeError):
    """Raised when the persisted budget cannot safely authorize a run."""


@dataclass(frozen=True)
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    remaining_seconds: float
    timed_out: bool = False


def run_bounded(
    command: Sequence[str],
    *,
    ledger_path: str | Path,
    timeout_seconds: float,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
) -> RunResult:
    """Run one command within a per-run deadline and persistent cumulative cap."""
    if not command or isinstance(command, (str, bytes)):
        raise ValueError("command must be a non-empty argument sequence")
    if not _finite_number(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not _finite_number(budget_seconds) or not 0 < budget_seconds <= DEFAULT_BUDGET_SECONDS:
        raise ValueError("budget_seconds must be in (0, 7200]")

    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(ledger.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger.exists():
            try:
                state = json.loads(ledger.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BudgetError("budget ledger is unreadable") from error
            if (
                not isinstance(state, dict)
                or set(state) - {"budget_seconds", "charged_seconds", "pending"}
                or not _finite_number(state.get("budget_seconds"))
                or not 0 < state["budget_seconds"] <= DEFAULT_BUDGET_SECONDS
                or not _finite_number(state.get("charged_seconds"))
                or state["charged_seconds"] < 0
                or ("pending" in state and type(state["pending"]) is not bool)
            ):
                raise BudgetError("budget ledger has invalid schema or values")
            if state.get("pending"):
                raise BudgetError("previous run is pending; refusing unsafe reuse")
            if state.get("budget_seconds") != budget_seconds:
                raise BudgetError("budget_seconds differs from the persisted ledger")
        else:
            state = {"budget_seconds": budget_seconds, "charged_seconds": 0.0}

        remaining = budget_seconds - state["charged_seconds"]
        if remaining <= _CLEANUP_RESERVE_SECONDS:
            raise BudgetError("persistent time budget is exhausted")

        state["pending"] = True
        _write_ledger(ledger, state)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            state["charged_seconds"] += time.monotonic() - started
            state.pop("pending")
            _write_ledger(ledger, state)
            raise
        timed_out = False
        try:
            run_deadline = min(
                timeout_seconds, remaining - _CLEANUP_RESERVE_SECONDS
            )
            stdout, stderr = process.communicate(timeout=run_deadline)
            returncode = process.returncode
            _terminate_process_group(process)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=0.1)
            except subprocess.TimeoutExpired as error:
                raise BudgetError("cleanup deadline exceeded; ledger remains pending") from error
            returncode = None
        elapsed = time.monotonic() - started
        state["charged_seconds"] += elapsed
        state.pop("pending")
        _write_ledger(ledger, state)
        return RunResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
            remaining_seconds=budget_seconds - state["charged_seconds"],
            timed_out=timed_out,
        )


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired as error:
        raise BudgetError("cleanup deadline exceeded; ledger remains pending") from error


def _write_ledger(path: Path, state: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)