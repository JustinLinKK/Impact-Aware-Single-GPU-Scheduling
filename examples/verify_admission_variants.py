"""Sequential actual-GPU integration checks for both frozen admission policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def _live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _capture_descendants(parent, owned):
    """Capture detached sessions too, before their parent can disappear."""
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children:
        try:
            owned[(process.pid, process.create_time())] = process
        except psutil.NoSuchProcess:
            continue  # One exiting child must not hide its surviving siblings.


def _cleanup_owned(process, parent, owned, *, grace_seconds):
    # Stop scheduler respawns before taking the final ancestry snapshot. A
    # worker's new session is unrelated to its ancestry and is included here.
    try:
        parent.suspend()
    except psutil.NoSuchProcess:
        pass
    _capture_descendants(parent, owned)
    # Freeze known descendants as well, then recapture children they may have
    # launched between enumeration and suspension. This includes grandchildren
    # whose own process groups differ from both scheduler and worker groups.
    for _ in range(8):
        before = len(owned)
        for child in list(owned.values()):
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
        for ancestor in [parent, *list(owned.values())]:
            _capture_descendants(ancestor, owned)
        if len(owned) == before:
            break
    descendants = list(owned.values())
    for child in descendants:
        try:
            child.terminate()
            child.resume()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    # Terminate then resume the suspended parent, allowing its signal handler
    # to reap children if it has one. Popen.wait always reaps our direct child.
    try:
        parent.terminate()
        parent.resume()
    except psutil.NoSuchProcess:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)
    _, alive = psutil.wait_procs(descendants, timeout=grace_seconds)
    remaining = [child.pid for child in alive if _live(child)]
    if remaining:
        raise RuntimeError(f"Preflight cleanup could not stop owned descendants: {remaining}")


def _run_isolated(command, log, *, timeout_seconds=840, grace_seconds=2):
    """Run one variant; timeout/failure cannot leave known workers running."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    parent = psutil.Process(process.pid)
    owned = {}
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            _capture_descendants(parent, owned)
            returncode = process.poll()
            if returncode is not None:
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                if any(_live(child) for child in owned.values()):
                    raise RuntimeError("Preflight parent exited with owned descendants still running")
                return
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
    except BaseException:
        _cleanup_owned(process, parent, owned, grace_seconds=grace_seconds)
        raise


def main():
    parser = argparse.ArgumentParser()
    for flag in ('jobs', 'dataset', 'release-root', 'model-dir', 'output', 'scheduler-commit'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--cancer-jobs')
    parser.add_argument('--cancer-dataset')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks = {}
    for name, threshold in (('proposal10', .10), ('proposal50', .50)):
        output = root / name
        command = [sys.executable, '-m', 'localml_scheduler.examples.verify_admission_rerun',
            '--jobs', args.jobs, '--dataset', args.dataset,
            '--release-root', args.release_root, '--model-dir', args.model_dir,
            '--scheduler-commit', args.scheduler_commit, '--output', str(output),
            '--max-incumbent-slowdown-fraction', str(threshold)]
        # The same frozen cancer architectures and CPU model are checked once;
        # both policies separately exercise real pause/resume and overlap.
        if name == 'proposal10' and args.cancer_jobs:
            assert args.cancer_dataset, 'cancer data required'
            command += ['--cancer-jobs', args.cancer_jobs, '--cancer-dataset', args.cancer_dataset]
        with (root / (name + '.log')).open('w') as log:
            _run_isolated(command, log)
        receipt_path = output / 'verification.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['complete'] is True and receipt['scheduler_commit'] == args.scheduler_commit
        assert receipt['actual_gpu']['name'] == 'NVIDIA A10' and not receipt['test_only_target_override']
        assert receipt['max_incumbent_slowdown_fraction'] == threshold
        assert receipt['scored_overlap_decisions'] > 0
        assert receipt['incumbent_yields'] == receipt['incumbent_admission_pauses'] == 0
        checks[name] = {'receipt': str(receipt_path.relative_to(root)),
            'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'complete': True, 'max_incumbent_slowdown_fraction': threshold}
    result = {'complete': True, 'scheduler_commit': args.scheduler_commit,
        'scope': 'actual A10 integration only; no performance or prediction-accuracy claim',
        'policy_thresholds': {'proposal10': .10, 'proposal50': .50},
        'policies_complete': {name: row['complete'] for name, row in checks.items()},
        'checks': checks, 'elapsed_seconds': time.monotonic() - started}
    target = root / 'verification.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
