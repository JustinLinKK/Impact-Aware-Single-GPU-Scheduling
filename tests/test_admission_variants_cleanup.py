"""Timeout cleanup must include workers which created their own sessions."""
import json
import os
import subprocess
import sys

import psutil
import pytest

from localml_scheduler.examples import verify_admission_variants as driver


@pytest.mark.skipif(os.name != "posix", reason="A10 preflight runs on POSIX")
def test_timeout_reaps_parent_and_detached_worker_and_never_starts_second_variant(tmp_path, monkeypatch):
    child = tmp_path / "detached.py"
    child.write_text("""import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getsid(0)))
time.sleep(60)
""")
    parent = tmp_path / "parent.py"
    parent.write_text("""import json, os, signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[3]], start_new_session=True)
def terminate(*args):
    child.wait(timeout=5)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
Path(sys.argv[2]).write_text(json.dumps({'parent': os.getpid(), 'child': child.pid, 'parent_session': os.getsid(0)}))
while True: time.sleep(.1)
""")
    pids_path, session_path = tmp_path / "pids.json", tmp_path / "child-session"
    launched = []
    run = driver._run_isolated
    def timed_variant(command, log):
        launched.append(command)
        return run([sys.executable, str(parent), str(child), str(pids_path), str(session_path)], log,
                   timeout_seconds=.75, grace_seconds=.25)
    monkeypatch.setattr(driver, "_run_isolated", timed_variant)
    output = tmp_path / "checks"
    monkeypatch.setattr(sys, "argv", ["verify_admission_variants", "--jobs", "unused", "--dataset", "unused",
        "--release-root", "unused", "--model-dir", "unused", "--scheduler-commit", "cpu-test",
        "--output", str(output)])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            driver.main()
        pids = json.loads(pids_path.read_text())
        assert pids["parent_session"] == pids["parent"]
        assert int(session_path.read_text()) == pids["child"]
        assert not psutil.pid_exists(pids["parent"])
        assert not psutil.pid_exists(pids["child"])
        assert len(launched) == 1
        assert launched[0][-1] == "0.1"
        assert not (output / "proposal50.log").exists()
        assert not (output / "verification.json").exists()
    finally:
        if pids_path.exists():
            for pid in json.loads(pids_path.read_text()).values():
                try:
                    process = psutil.Process(pid)
                    if str(parent) in process.cmdline() or str(child) in process.cmdline():
                        process.kill()
                except psutil.NoSuchProcess:
                    pass


@pytest.mark.parametrize("code", [0, 7])
def test_isolated_runner_preserves_success_and_nonzero_exit_semantics(tmp_path, code):
    with (tmp_path / "run.log").open("w") as log:
        if code:
            with pytest.raises(subprocess.CalledProcessError) as error:
                driver._run_isolated([sys.executable, "-c", f"raise SystemExit({code})"], log,
                                     timeout_seconds=2, grace_seconds=.25)
            assert error.value.returncode == code
        else:
            driver._run_isolated([sys.executable, "-c", "pass"], log, timeout_seconds=2, grace_seconds=.25)
