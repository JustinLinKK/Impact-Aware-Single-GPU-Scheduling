"""Real, disjoint calibration refresh at matched-block boundaries only.

No scheduler policy, FIFO ordering, model weights or TTL is changed. A refresh
creates a new immutable generation, never renews a timestamp on old evidence.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..a10_feedback_latency.run import atomic_json, file_hash, FEEDBACK_PHASE_MODEL_TTL_SECONDS
from .process_ownership import cleanup_owned, owned_descendants

CONTRACT = dict(schema='a10_v51_fresh_calibration_v1', ttl_seconds=14400,
                block_arms=3, margin_seconds=300, maximum_generations=21,
                refresh_only_between_blocks=True, evaluation_feedback_used=False,
                qualification_seconds=150, calibration_seconds=3700,
                conversion_seconds=1800, sealing_seconds=90)


def observed_seconds(value):
    if not isinstance(value, str):
        raise ValueError('actual timezone-aware observation timestamp required')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('calibration observation must have a timezone')
    result = stamp.timestamp()
    if not math.isfinite(result):
        raise ValueError('finite observation timestamp required')
    return result


def observation_range(table, neural):
    stamps = [observed_seconds(entry['observed_at']) for entry in table['entries'].values()]
    stamps.append(observed_seconds(neural['observed_at']))
    if len(stamps) < 2:
        raise ValueError('complete measured and neural calibration required')
    return min(stamps), max(stamps)


def freshness(table, neural, *, now, remaining_episodes, episode_seconds):
    if (not math.isfinite(now) or type(remaining_episodes) is not int
            or not 0 <= remaining_episodes <= CONTRACT['block_arms']
            or type(episode_seconds) is not int or not 0 < episode_seconds <= 3600):
        raise ValueError('bounded matched-block freshness input required')
    oldest, newest = observation_range(table, neural)
    required = remaining_episodes * episode_seconds + CONTRACT['margin_seconds']
    expires = oldest + CONTRACT['ttl_seconds']
    return dict(oldest_observation_unix=oldest, newest_observation_unix=newest,
                checked_unix=now, remaining_episodes=remaining_episodes,
                required_remaining_seconds=required, expires_unix=expires,
                fresh=0 <= now - newest and now + required < expires)


def matched_block(order, index):
    if index % 3 or index + 3 > len(order):
        raise ValueError('complete block boundary required')
    block = order[index:index+3]
    keys = ('host_id', 'condition', 'mode', 'repeat')
    if (len({tuple(row[key] for key in keys) for row in block}) != 1
            or {row['arm'] for row in block} != {'fixed_three', 'combined', 'v5_1'}):
        raise ValueError('three exact arms sharing condition/mode/repeat required')
    return {key: block[0][key] for key in keys}


class FreshCalibration:
    def __init__(self, bundle, root, campaign_deadline, campaign_expires_unix):
        if bundle['plan'].get('calibration_refresh') != CONTRACT:
            raise ValueError('exact predeclared calibration lifecycle required')
        # make_settings installs this common override; raw package settings do
        # not contain a feedback_latency section. Never invent a second TTL.
        if FEEDBACK_PHASE_MODEL_TTL_SECONDS != CONTRACT['ttl_seconds']:
            raise ValueError('four-hour policy TTL must remain unchanged')
        self.original = bundle
        self.bundle = bundle
        self.root = Path(root)
        self.host = bundle['directory'].parent
        self.campaign_deadline = campaign_deadline
        self.campaign_expires_unix = campaign_expires_unix
        self.generations, self.blocks = [], []
        self.active = None
        self.host_input_sha256 = self._verify_inputs()
        self._publish_generation(bundle, 'initial', None)

    def cutoff(self):
        return min(self.campaign_expires_unix,
                   datetime.fromisoformat(self.original['plan']['deadline_utc']).timestamp())

    def _verify_inputs(self):
        from .campaign import verify_bundle
        from .protocol import calibration_manifest
        verify_bundle(self.original)
        verify_bundle(self.bundle)
        plan = self.host/'plan.json'
        manifest = self.host/'calibration-manifest.json'
        if plan.read_bytes() != (self.original['directory']/'plan.json').read_bytes():
            raise ValueError('root refresh plan differs from original sealed bytes')
        if json.loads(manifest.read_text()) != calibration_manifest(self.original['plan']):
            raise ValueError('refresh calibration manifest differs from fixed disjoint workload plan')
        pins={str(path):file_hash(path) for path in (plan,manifest)}
        if hasattr(self,'host_input_sha256') and pins != self.host_input_sha256:
            raise ValueError('refresh inputs changed since campaign admission')
        return pins

    def _publish_generation(self, bundle, generation, started):
        neural = json.loads(bundle['neural_calibration'].read_text())
        oldest, newest = observation_range(bundle['table'], neural)
        if newest > time.time() or (started is not None and oldest < started):
            raise ValueError('refresh must contain new real observations, never re-dated old data')
        if (bundle['plan'] != self.original['plan'] or bundle['order'] != self.original['order']
                or bundle['table']['gpu_uuid'] != self.original['table']['gpu_uuid']
                or bundle['table']['hardware_key'] != self.original['table']['hardware_key']
                or bundle['table']['source_identity'] != self.original['table']['source_identity']):
            raise ValueError('refresh changed workload plan, host, source or runtime')
        for name, item in bundle['manifests'].items():
            if item['manifest']['jobs'] != self.original['manifests'][name]['manifest']['jobs']:
                raise ValueError('refresh changed workload recipes, seeds or budgets')
        self.generations.append(dict(generation=generation, started_unix=started,
            finished_unix=time.time(), oldest_observation_unix=oldest, newest_observation_unix=newest,
            manifests=str(bundle['directory']), calibration=str(bundle['calibration']),
            neural_calibration=str(bundle['neural_calibration']), input_sha256=bundle['hashes'],
            source_pins=bundle['snapshots'],host_input_sha256=self.host_input_sha256))
        atomic_json(self.root/'calibration-generations.json', self.generations)

    def _run(self, generation_root, name, module, args, seconds, *, cpu=False):
        self.campaign_deadline()
        self._verify_inputs()
        command = [sys.executable, '-m', module, *map(str, args)]
        child, owned = None, {}
        started = time.monotonic()
        env = {**os.environ, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})}
        with (generation_root/(name+'.log')).open('x') as log:
            try:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=env)
                atomic_json(generation_root/(name+'-dispatch.json'), dict(command=command,
                    pid=child.pid, started_unix=time.time(), max_seconds=seconds, cpu_only=cpu))
                while child.poll() is None:
                    owned.update(owned_descendants(child.pid))
                    self.campaign_deadline()
                    if time.monotonic()-started >= seconds:
                        raise TimeoutError('calibration refresh phase deadline: '+name)
                    time.sleep(.25)
            finally:
                if child is not None: cleanup_owned(child, owned)
        if child.returncode != 0:
            raise RuntimeError('fresh calibration phase failed: '+name)
        self._verify_inputs()
        atomic_json(generation_root/(name+'-complete.json'), dict(exit_code=0,
                    elapsed_seconds=time.monotonic()-started, finished_unix=time.time()))

    def _refresh(self):
        from .campaign import sealed_bundle
        if len(self.generations) >= CONTRACT['maximum_generations']:
            raise ValueError('finite calibration-generation limit reached')
        started = time.time()
        root = self.root/'calibration-refresh'/f'g{len(self.generations):02d}'
        root.mkdir(parents=True, exist_ok=False)
        prefix = 'localml_scheduler.benchmarks.'
        # Qualification proves current GPU isolation before any fresh calibration.
        self._run(root, 'qualification', prefix+'a10_resource_mechanisms.visibility_qualification',
            ['--gpu-uuid', self.original['table']['gpu_uuid'], '--output', root/'qualification',
             '--max-seconds', 120], CONTRACT['qualification_seconds'])
        self._run(root, 'calibration', prefix+'a10_resource_mechanisms.calibrate',
            ['--manifest', self.host/'calibration-manifest.json', '--settings', self.original['settings'],
             '--output', root/'calibration', '--max-seconds', 3600,
             '--visibility-qualification', root/'qualification/qualification-proof.json'], CONTRACT['calibration_seconds'])
        self._run(root, 'conversion', prefix+'a10_v5_ablation.calibrate_predictor',
            ['--manifest', self.host/'calibration-manifest.json', '--settings', self.original['settings'],
             '--measured-calibration', root/'calibration', '--output', root/'predictor-calibration',
             '--source-commit', self.original['plan']['source_commit']], CONTRACT['conversion_seconds'], cpu=True)
        self._run(root, 'seal', prefix+'a10_v5_1_ablation.protocol',
            ['evaluation-manifests', '--plan', self.host/'plan.json', '--settings', self.original['settings'],
             '--calibration', root/'calibration/calibration.json',
             '--neural-calibration', root/'predictor-calibration/calibration.json',
             '--output', root/'sealed-manifests'], CONTRACT['sealing_seconds'], cpu=True)
        bundle = sealed_bundle(root/'sealed-manifests', self.original['settings'],
            root/'calibration/calibration.json', root/'predictor-calibration/calibration.json')
        self._publish_generation(bundle, root.name, started)
        self.bundle = bundle

    def begin_block(self, index):
        self.active = None
        identity = matched_block(self.original['order'], index)
        check = self.check(3)
        if not check['fresh']:
            required = sum(CONTRACT[key] for key in ('qualification_seconds', 'calibration_seconds',
                'conversion_seconds', 'sealing_seconds')) + check['required_remaining_seconds']
            if time.time()+required >= self.cutoff():
                raise TimeoutError('refresh plus complete matched block cannot fit campaign/host cutoff')
            self._refresh()
            check = self.check(3)
        if not check['fresh']:
            raise TimeoutError('fresh calibration cannot cover complete matched block')
        if time.time()+check['required_remaining_seconds'] >= self.cutoff():
            raise TimeoutError('complete matched block cannot fit remaining campaign/host bound')
        self.active = dict(block_index=index//3, first_record=index, **identity,
            generation=self.generations[-1]['generation'], freshness_at_start=check,
            episode_checks=[], status='running')
        self.blocks.append(self.active)
        self.save()
        return self.bundle

    def check(self, remaining):
        return freshness(self.bundle['table'], json.loads(self.bundle['neural_calibration'].read_text()),
            now=time.time(), remaining_episodes=remaining,
            episode_seconds=self.original['plan']['episode_max_seconds'])

    def before_episode(self, index):
        check = self.check(3-index%3)
        if not self.active or not check['fresh']:
            raise TimeoutError('same calibration no longer covers remaining block; no mid-block refresh')
        if time.time()+check['required_remaining_seconds'] >= self.cutoff():
            raise TimeoutError('remaining matched block cannot fit campaign/host bound')
        self.active['episode_checks'].append(dict(record_index=index, **check))
        self.save()

    def guard(self):
        if self.active and time.time()+CONTRACT['margin_seconds'] >= self.active['freshness_at_start']['expires_unix']:
            raise TimeoutError('calibration freshness envelope exhausted; stop incomplete block')

    def end_episode(self, index):
        self.guard()
        if index%3 == 2:
            self.active.update(status='completed', completed_unix=time.time())
        self.save()

    def save(self):
        atomic_json(self.root/'block-calibration-bindings.json', self.blocks)
