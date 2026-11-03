import sys
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import time

from obstruction_train.budget_runner import run_bounded


def test_directory_sync_failure_prevents_worker_start(tmp_path, monkeypatch):
    import stat
    import pytest

    real_fsync = os.fsync
    started = tmp_path / "must-not-start"

    def fail_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory durability unavailable")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="directory durability unavailable"):
        run_bounded(
            [sys.executable, "-c", f"open({str(started)!r}, 'w').close()"],
            ledger_path=tmp_path / "budget.json",
            timeout_seconds=1,
        )
    assert not started.exists()


def test_corrupt_charged_seconds_cannot_authorize_execution(tmp_path):
    import pytest
    from obstruction_train.budget_runner import BudgetError

    for charged in (-1, float("nan"), float("inf"), True, "0"):
        ledger = tmp_path / "invalid-budget.json"
        ledger.write_text(json.dumps({"budget_seconds": 10, "charged_seconds": charged}))
        with pytest.raises(BudgetError):
            run_bounded([sys.executable, "-c", "pass"], ledger_path=ledger,
                        timeout_seconds=1, budget_seconds=10)


def test_nonfinite_timeout_is_rejected_before_start(tmp_path):
    import pytest

    for timeout in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            run_bounded([sys.executable, "-c", "pass"],
                        ledger_path=tmp_path / "invalid-timeout.json",
                        timeout_seconds=timeout)


def test_successful_command_is_charged_to_persistent_budget(tmp_path):
    ledger_path = tmp_path / "gpu-budget.json"

    result = run_bounded(
        [sys.executable, "-c", "print('finished')"],
        ledger_path=ledger_path,
        timeout_seconds=5,
        budget_seconds=10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "finished"
    assert result.elapsed_seconds > 0
    assert 0 < result.remaining_seconds < 10


def test_default_persistent_budget_is_7200_seconds(tmp_path):
    ledger_path = tmp_path / "gpu-budget.json"

    result = run_bounded(
        [sys.executable, "-c", "pass"],
        ledger_path=ledger_path,
        timeout_seconds=5,
    )

    assert result.returncode == 0
    assert json.loads(ledger_path.read_text())["budget_seconds"] == 7200


def test_failed_command_is_also_charged_to_persistent_budget(tmp_path):
    result = run_bounded(
        [sys.executable, "-c", "import sys; print('failed'); sys.exit(7)"],
        ledger_path=tmp_path / "gpu-budget.json",
        timeout_seconds=5,
        budget_seconds=10,
    )

    assert result.returncode == 7
    assert result.stdout.strip() == "failed"
    assert 0 < result.remaining_seconds < 10


def test_worker_start_failure_is_charged_without_poisoning_ledger(tmp_path):
    import pytest

    ledger_path = tmp_path / "gpu-budget.json"
    missing_worker = tmp_path / "missing-worker"

    with pytest.raises(FileNotFoundError):
        run_bounded(
            [str(missing_worker)],
            ledger_path=ledger_path,
            timeout_seconds=5,
            budget_seconds=10,
        )

    state = json.loads(ledger_path.read_text())
    assert state["budget_seconds"] == 10
    assert state["charged_seconds"] > 0
    assert "pending" not in state


def test_timed_out_process_group_is_killed_and_time_is_charged(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
        "time.sleep(30)"
    )
    result = run_bounded(
        [sys.executable, "-c", code],
        ledger_path=tmp_path / "gpu-budget.json",
        timeout_seconds=0.2,
        budget_seconds=10,
    )

    assert result.timed_out is True
    assert result.returncode is None
    assert result.elapsed_seconds < 3
    assert result.remaining_seconds < 10
    child_pid = int(pid_file.read_text())
    status_file = Path(f"/proc/{child_pid}/status")
    if status_file.exists():
        assert "State:\tZ" in status_file.read_text()


def test_timeout_kills_descendant_that_ignores_terminate(tmp_path):
    pid_file = tmp_path / "stubborn-child.pid"
    child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    parent_code = (
        "import subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
        "time.sleep(30)"
    )
    run_bounded(
        [sys.executable, "-c", parent_code],
        ledger_path=tmp_path / "gpu-budget.json",
        timeout_seconds=0.2,
        budget_seconds=10,
    )
    child_pid = int(pid_file.read_text())
    status_file = Path(f"/proc/{child_pid}/status")
    try:
        assert not status_file.exists() or "State:\tZ" in status_file.read_text()
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_exhausted_budget_rejects_work_without_starting_it(tmp_path):
    ledger_path = tmp_path / "gpu-budget.json"
    pid_file = tmp_path / "started"

    result = run_bounded(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        ledger_path=ledger_path,
        timeout_seconds=1,
        budget_seconds=0.65,
    )

    import pytest
    from obstruction_train.budget_runner import BudgetError

    with pytest.raises(BudgetError, match="exhausted"):
        run_bounded(
            [sys.executable, "-c", f"open({str(pid_file)!r}, 'w').write('started')"],
            ledger_path=ledger_path,
            timeout_seconds=0.1,
            budget_seconds=0.65,
        )
    assert not pid_file.exists()


def test_crash_left_pending_ledger_fails_closed(tmp_path):
    ledger_path = tmp_path / "gpu-budget.json"
    ledger_path.write_text(
        json.dumps({"budget_seconds": 10, "charged_seconds": 0, "pending": True})
    )
    started_file = tmp_path / "must-not-start"

    import pytest
    from obstruction_train.budget_runner import BudgetError

    with pytest.raises(BudgetError, match="pending"):
        run_bounded(
            [sys.executable, "-c", f"open({str(started_file)!r}, 'w').write('started')"],
            ledger_path=ledger_path,
            timeout_seconds=1,
            budget_seconds=10,
        )
    assert not started_file.exists()


def test_concurrent_calls_are_serialized_by_the_persistent_ledger_lock(tmp_path):
    ledger_path = tmp_path / "gpu-budget.json"
    event_path = tmp_path / "events.txt"

    def command(label):
        script = (
            "import time; "
            f"f = open({str(event_path)!r}, 'a'); f.write('{label} start\\n'); f.flush(); "
            "time.sleep(0.2); "
            f"f.write('{label} end\\n'); f.close()"
        )
        return [sys.executable, "-c", script]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            run_bounded,
            command("first"),
            ledger_path=ledger_path,
            timeout_seconds=3,
            budget_seconds=10,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not event_path.exists():
            time.sleep(0.01)
        assert event_path.exists(), "first child did not start within the test deadline"
        second = pool.submit(
            run_bounded,
            command("second"),
            ledger_path=ledger_path,
            timeout_seconds=3,
            budget_seconds=10,
        )
        assert first.result(timeout=4).returncode == 0
        assert second.result(timeout=4).returncode == 0

    events = event_path.read_text().splitlines()
    assert events == ["first start", "first end", "second start", "second end"]