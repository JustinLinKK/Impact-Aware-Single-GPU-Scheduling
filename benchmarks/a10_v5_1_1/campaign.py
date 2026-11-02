"""Finite sequential traces with cold namespaces and genuine calibration refresh."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .protocol import validate_plan, calibration_plan, row_manifest
from .audit import audit_episode
from .episode import source_snapshot
from ..a10_v5_1_ablation.protocol import calibration_manifest, validate_table, validate_neural_table
from ..a10_v5_1_ablation.fresh_calibration import freshness, observation_range
from ..a10_v5_1_ablation.process_ownership import owned_descendants, cleanup_owned
from ..a10_feedback_latency.run import atomic_json, file_hash
from ..a10_resource_mechanisms.profiles import SOURCE


def load(path): return json.loads(Path(path).read_text())


def coverage(records):
    completed = [r for r in records if r['status'] == 'completed' and r.get('audit', {}).get('valid')]
    return {'planned_cohorts': len(records), 'planned_jobs': sum(r['expected_jobs'] for r in records),
        'completed_valid_cohorts': len(completed), 'usable_receipts': sum(r['expected_jobs'] for r in completed),
        'failed_cohorts': sum(r['status'] == 'failed' for r in records),
        'failed_planned_jobs': sum(r['expected_jobs'] for r in records if r['status'] == 'failed'),
        'running_cohorts': sum(r['status'] == 'running' for r in records),
        'unattempted_cohorts': sum(r['status'] == 'unattempted' for r in records),
        'unattempted_jobs': sum(r['expected_jobs'] for r in records if r['status'] == 'unattempted')}


class Runner:
    def __init__(self, plan_path, settings, root):
        self.plan_path, self.settings, self.root = map(lambda p: Path(p).resolve(), (plan_path, settings, root))
        self.plan = validate_plan(load(self.plan_path))
        self.started = time.time()
        self.cutoff = min(self.started + self.plan['bounds']['campaign_seconds'],
            datetime.fromisoformat(self.plan['deadline_utc'].replace('Z', '+00:00')).timestamp())
        self.pins = {str(p): file_hash(p) for p in (self.plan_path, self.settings)}
        self.active = None
        self.generations, self.tables = [], {}
        self.records = [{**r, 'output': f'{i:02d}-{r["id"]}', 'status': 'unattempted',
            'expected_jobs': len(r['jobs']) if r['jobs'] is not None else r['expected_jobs'],
            'usable_receipts': 0} for i, r in enumerate(self.plan['prepared']['rows'])]

    def guard(self, headroom=0.):
        if time.time() + headroom >= self.cutoff:
            raise TimeoutError('full phase/campaign/recovery headroom exhausted')
        if any(file_hash(p) != sha for p, sha in self.pins.items()):
            raise ValueError('frozen campaign inputs changed')

    def save(self, state, error=None):
        atomic_json(self.root / 'campaign-results.json', self.records)
        atomic_json(self.root / 'coverage.json', coverage(self.records))
        atomic_json(self.root / 'status.json', {'status': state, 'error': error,
            'active_output': self.active['output'] if self.active else None,
            'coverage': coverage(self.records), 'elapsed_seconds': time.time() - self.started})
        atomic_json(self.root / 'calibration-generations.json', self.generations)

    def child(self, name, module, argv, *, seconds, cpu=False, directory=None):
        self.guard(seconds)
        directory = directory or self.root
        command = [sys.executable, '-m', module, *map(str, argv)]
        env = {**os.environ, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})}
        process, owned = None, {}
        start = time.time()
        intent = directory / (name + '-dispatch.json')
        if intent.exists(): raise FileExistsError('exclusive child dispatch intent exists')
        atomic_json(intent, {'command': command, 'started_unix': start, 'seconds': seconds,
                             'source_commit': self.plan['source_commit']})
        with (directory / (name + '.log')).open('x') as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env=env)
                atomic_json(self.root / 'active-child.json', {'pid': process.pid, 'command': command,
                    'started_unix': start, 'status': 'running', 'output': name})
                while process.poll() is None:
                    owned.update(owned_descendants(process.pid))
                    self.guard()
                    if time.time() - start >= seconds: raise TimeoutError('bounded child deadline: ' + name)
                    time.sleep(.25)
            finally:
                if process is not None: cleanup_owned(process, owned)
        proof = {'pid': process.pid, 'command': command, 'started_unix': start,
                 'finished_unix': time.time(), 'status': 'exited', 'exit_code': process.returncode}
        atomic_json(self.root / 'active-child.json', proof)
        atomic_json(directory / (name + '-completion.json'), proof)
        if process.returncode != 0: raise RuntimeError('child failed: ' + name)
        self.guard()

    def install_table(self, measured_path, neural_path, *, variant, started=None, manifest_path=None):
        measured_path, neural_path = Path(measured_path).resolve(), Path(neural_path).resolve()
        measured, neural = load(measured_path), load(neural_path)
        validate_table(measured); validate_neural_table(neural, measured)
        oldest, newest = observation_range(measured, neural)
        if newest > time.time() or (started is not None and oldest < started):
            raise ValueError('calibration observations must be real and newly measured')
        if self.generations and any(g['gpu_uuid'] != measured['gpu_uuid'] for g in self.generations):
            raise ValueError('calibration GPU identity changed')
        value = {'generation': f'g{len(self.generations):02d}', 'variant': variant,
            'measured_path': str(measured_path), 'neural_path': str(neural_path),
            'measured_sha256': file_hash(measured_path), 'neural_sha256': file_hash(neural_path),
            'manifest_path': str(manifest_path) if manifest_path else None,
            'manifest_sha256': file_hash(manifest_path) if manifest_path else None,
            'gpu_uuid': measured['gpu_uuid'], 'oldest_observation_unix': oldest,
            'newest_observation_unix': newest, 'refresh_started_unix': started,
            'registered_unix': time.time()}
        self.generations.append(value)
        self.tables[variant] = (value, measured, neural)
        self.pins.update({str(measured_path): file_hash(measured_path), str(neural_path): file_hash(neural_path)})

    def refresh(self, variant):
        # 150 qualification +3700 real calibration +1800 CPU conversion,
        # then a full episode and margin. No optimistic shortening.
        self.guard(150 + 3700 + 1800 + 3600 + 300)
        if len(self.generations) >= 64: raise ValueError('generation bound exhausted')
        directory = self.root / 'calibration-refresh' / f'g{len(self.generations):02d}'
        directory.mkdir(parents=True, exist_ok=False)
        started = time.time()
        manifest = calibration_manifest(calibration_plan(self.plan, variant=variant, generation=len(self.generations)))
        path = directory / 'calibration-manifest.json'; atomic_json(path, manifest)
        gpu = self.generations[0]['gpu_uuid']
        prefix = 'localml_scheduler.benchmarks.'
        self.child('qualification', prefix + 'a10_resource_mechanisms.visibility_qualification',
            ['--gpu-uuid', gpu, '--output', directory / 'qualification', '--max-seconds', 120], seconds=150, directory=directory)
        self.child('calibration', prefix + 'a10_resource_mechanisms.calibrate',
            ['--manifest', path, '--settings', self.settings, '--output', directory / 'calibration',
             '--max-seconds', 3600, '--visibility-qualification', directory / 'qualification/qualification-proof.json'],
            seconds=3700, directory=directory)
        self.child('conversion', prefix + 'a10_v5_ablation.calibrate_predictor',
            ['--manifest', path, '--settings', self.settings, '--measured-calibration', directory / 'calibration',
             '--output', directory / 'predictor-calibration', '--source-commit', self.plan['source_commit']],
            seconds=1800, cpu=True, directory=directory)
        self.install_table(directory / 'calibration/calibration.json', directory / 'predictor-calibration/calibration.json',
                           variant=variant, started=started, manifest_path=path)

    def run(self, measured, neural, *, variant_measured=None, variant_neural=None):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with (self.root.parent / '.v511-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.root.mkdir(exist_ok=False)
            history = self.root / 'history'
            history.mkdir()
            self.install_table(measured, neural, variant=False,
                manifest_path=Path(measured).resolve().parents[2] / 'calibration-manifest.json')
            if self.plan['stage'] == 'development':
                if variant_measured is None or variant_neural is None:
                    raise ValueError('real variant calibration required before development')
                self.install_table(variant_measured, variant_neural, variant=True,
                    manifest_path=Path(variant_measured).resolve().parents[2] / 'calibration-variant-manifest.json')
            atomic_json(self.root / 'campaign-protocol-before.json', {'plan': self.plan,
                'source_commit': self.plan['source_commit'], 'input_sha256': self.pins,
                'campaign_started_unix': self.started, 'campaign_expires_unix': self.cutoff,
                'denominators': self.plan['prepared']['denominators']})
            def signal_error(signum, frame): raise InterruptedError('owned campaign interrupted ' + str(signum))
            previous = {s: signal.signal(s, signal_error) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
            self.save('running')
            try:
                for row, record in zip(self.plan['prepared']['rows'], self.records):
                    variant = row.get('shape_variant_development_fixture', False)
                    value, table, neural_table = self.tables[variant]
                    check = freshness(table, neural_table, now=time.time(), remaining_episodes=1, episode_seconds=3600)
                    if not check['fresh']:
                        self.refresh(variant); value, table, neural_table = self.tables[variant]
                        check = freshness(table, neural_table, now=time.time(), remaining_episodes=1, episode_seconds=3600)
                    if not check['fresh']: raise TimeoutError('real refresh cannot cover one full screening episode')
                    self.guard(3900)
                    record.update(calibration_generation=value['generation'], freshness=check)
                    manifest = row_manifest(self.plan, row, table, neural_table, history_root=history)
                    path = self.root / 'manifests' / (row['id'] + '.json')
                    path.parent.mkdir(exist_ok=True); atomic_json(path, manifest)
                    snapshot = source_snapshot(manifest, path, self.plan['source_commit'])
                    if snapshot['scheduler_python_sources'] != table['source_identity']['scheduler_python_sources']:
                        raise ValueError('candidate/source calibration closure changed')
                    selected = value['neural_path'] if manifest['profile_source'] == 'ml_predictor' else value['measured_path']
                    module = 'a10_v5_1_1.episode' if row['arm'] == 'v5_1_1' else 'a10_v5_ablation.episode'
                    output = self.root / record['output']
                    record.update(status='running', manifest=str(path), manifest_sha256=file_hash(path),
                                  source_snapshot=snapshot, calibration=selected)
                    self.active = record; self.save('running')
                    self.child(record['output'], 'localml_scheduler.benchmarks.' + module,
                        ['--manifest', path, '--settings', self.settings, '--calibration', selected,
                         '--output', output, '--arm', row['arm'], '--mode', row['mode'],
                         '--source-commit', self.plan['source_commit'], '--max-seconds', 3600,
                         '--deadline-utc', self.plan['deadline_utc']], seconds=3600)
                    audit = audit_episode(output, row, manifest, snapshot,
                        neural_table if manifest['profile_source'] == 'ml_predictor' else table,
                        development=self.plan['stage'] == 'development')
                    record.update(audit=audit, usable_receipts=audit['usable_receipts'])
                    if not audit['valid']: raise ValueError('raw episode audit failed: ' + str(audit['errors']))
                    record['status'] = 'completed'; self.active = None
                    self.save('running'); time.sleep(2.)
                self.guard()
                atomic_json(self.root / 'campaign-complete.json', {'complete': True,
                    'coverage': coverage(self.records), 'elapsed_seconds': time.time() - self.started})
                self.save('complete')
            except BaseException as exc:
                if self.active: self.active.update(status='failed', error=repr(exc))
                self.save('failed', repr(exc)); raise
            finally:
                for s, handler in previous.items(): signal.signal(s, handler)


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    for name in ('plan', 'settings', 'output', 'calibration', 'neural-calibration'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--variant-calibration', type=Path)
    p.add_argument('--variant-neural-calibration', type=Path)
    a = p.parse_args(argv)
    Runner(a.plan, a.settings, a.output).run(a.calibration, a.neural_calibration,
        variant_measured=a.variant_calibration, variant_neural=a.variant_neural_calibration)


if __name__ == '__main__': main()
