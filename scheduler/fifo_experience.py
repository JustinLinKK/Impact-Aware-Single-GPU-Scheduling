"""Trace-local append-only evidence with episode-balanced, causal retrieval.

An observation is never an independent vote per polling tick. The index folds
all valid windows of the same episode/workload/context into one aggregate.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

from .workload_fingerprint import digest


class ExperienceStore:
    def __init__(self, namespace, *, path=None, maximum=4096):
        if not namespace or type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError('bounded independent trace namespace required')
        self.namespace, self.maximum = namespace, maximum
        self.path = Path(path) if path is not None else None
        self.rows, self.events, self.control = {}, [], {}
        self.parent = '0' * 64
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            log = self.path / 'experience.jsonl'
            if log.exists():
                for line in log.read_text().splitlines():
                    event = json.loads(line)
                    proof = event.pop('sha256')
                    if (event['namespace'] != namespace or event['sequence'] != len(self.events) or event['parent_sha256'] != self.parent
                            or digest(event) != proof):
                        raise ValueError('history namespace/hash chain changed')
                    self._apply(event['payload'])
                    self.parent = proof
                    self.events.append({**event, 'sha256': proof})
            index = self.path / 'index.json'
            if index.exists():
                saved = json.loads(index.read_text())
                # A crash can leave the atomic index behind the durable log.
                known = {'0' * 64, *(e['sha256'] for e in self.events)}
                if saved['namespace'] != namespace or saved['parent_sha256'] not in known:
                    raise ValueError('history index is not an ancestor of its log')

    def _apply(self, payload):
        kind = payload['kind']
        if kind == 'observation':
            row = payload['row']
            key = digest({k: row[k] for k in ('episode', 'fingerprint', 'context', 'phase')})
            old = self.rows.get(key)
            if old is None:
                value = deepcopy(row)
                value.update(first_observed_unix=row['observed_unix'], windows=1)
            else:
                # Rates may have different solo references across calibration
                # generations. Keep raw updates/time; re-express explicitly.
                newest = row if row['observed_unix'] >= old['observed_unix'] else old
                value = {**old, **deepcopy(newest),
                         'first_observed_unix': min(old['first_observed_unix'], row['observed_unix']),
                         'updates': old['updates'] + row['updates'],
                         'seconds': old['seconds'] + row['seconds'], 'windows': old['windows'] + 1}
            value['raw_rate'] = value['updates'] / value['seconds']
            value['aggregate_key'] = key
            self.rows[key] = value
            if len(self.rows) > self.maximum:
                del self.rows[min(self.rows, key=lambda k: (self.rows[k]['observed_unix'], k))]
        elif kind == 'control':
            self.control = deepcopy(payload['state'])
        elif kind not in {'decision', 'history_query', 'episode_closed', 'context_demoted'}:
            raise ValueError('unknown history event')

    def append(self, payload):
        # Validate before writing; then persist log before publishing the index.
        old_rows, old_control = dict(self.rows), self.control
        try:
            self._apply(payload)
            event = {'namespace': self.namespace, 'parent_sha256': self.parent,
                     'sequence': len(self.events), 'payload': deepcopy(payload)}
            proof = digest(event)
            event['sha256'] = proof
            if self.path:
                with (self.path / 'experience.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            self.events.append(event)
            self.parent = proof
        except BaseException:
            self.rows, self.control = old_rows, old_control
            raise

    def checkpoint(self):
        if not self.path:
            return
        value = {'schema': 'fifo_experience_index_v1', 'namespace': self.namespace,
                 'parent_sha256': self.parent, 'rows': self.rows, 'control': self.control}
        temporary = self.path / 'index.json.tmp'
        with temporary.open('w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(self.path / 'index.json')

    def observe(self, *, episode, fingerprint, context, updates, seconds,
                observed_unix, reference_rate, reference_memory, reference_generation, job_id,
                accepted_unix, window_start_unix, phase='training', counter_proof=None):
        values = (updates, seconds, observed_unix, reference_rate, reference_memory, accepted_unix, window_start_unix)
        if (not episode or not job_id or not reference_generation or phase != 'training'
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or updates < 4 or seconds < 1 or min(reference_rate, reference_memory) <= 0
                or not accepted_unix <= window_start_unix < observed_unix
                or abs(observed_unix - window_start_unix - seconds) > .01):
            raise ValueError('invalid causal successful-update observation')
        row = dict(episode=episode, fingerprint=deepcopy(fingerprint), context=deepcopy(context),
            updates=updates, seconds=seconds, observed_unix=observed_unix,
            reference_rate=reference_rate, reference_memory=reference_memory, reference_generation=reference_generation,
            job_id=job_id, accepted_unix=accepted_unix, window_start_unix=window_start_unix,
            phase=phase, raw_rate=updates / seconds, counter_proof=deepcopy(counter_proof))
        self.append({'kind': 'observation', 'row': row})

    def candidates(self, now, *, deadline):
        values = []
        for row in self.rows.values():
            if time.perf_counter() >= deadline:
                raise TimeoutError('history retrieval elapsed budget')
            age = now - row['observed_unix']
            if 0 <= age <= 86400:
                values.append(row)
        return values
