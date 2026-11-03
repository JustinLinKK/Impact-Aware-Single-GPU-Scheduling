"""Nonpreemptive FIFO concurrency control; forecasts are count-model heuristics.

The caller owns execution and the unchanged memory safety gate. This controller
never pauses, launches, or declares a receipt. Times at the API are monotonic.
The legacy-named cpu_budget_ms option bounds forecast elapsed wall time using
perf_counter, including descheduling; it is not a process CPU-time allowance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
import time

from ..domain import BatchResolution
from .completion_objective import remaining_optimizer_updates
from .feedback_objective import timestamp


def _number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError('boolean numeric value')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('invalid nonnegative finite value')
    return value


def _clock_identity():
    """Linux monotonic clocks survive process restart, not machine reboot."""
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


@dataclass(frozen=True)
class AdaptiveFIFOOptions:
    min_half_seconds: float = 1.
    min_half_updates: int = 4
    dwell_seconds: float = 10.
    evidence_ttl_seconds: float = 60.
    min_gain_fraction: float = .05
    min_gain_seconds: float = 1.
    max_events: int = 256
    cpu_budget_ms: float = 10.

    def __post_init__(self):
        for name in ('min_half_seconds', 'dwell_seconds', 'evidence_ttl_seconds',
                     'min_gain_seconds', 'cpu_budget_ms'):
            _number(getattr(self, name), positive=True)
        if (type(self.min_half_updates) is not int or self.min_half_updates < 4
                or type(self.max_events) is not int or not 1 <= self.max_events <= 256
                or self.min_half_seconds < 1 or self.cpu_budget_ms > 10
                or not 0 < self.min_gain_fraction < 1):
            raise ValueError('invalid bounded adaptive FIFO options')


class AdaptiveFIFOController:
    def __init__(self, mode='individual', *, options=None):
        if mode not in {'individual', 'group', 'bracket'}:
            raise ValueError('unknown feedback mode')
        self.mode = 'group' if mode == 'bracket' else mode
        self.options = options or AdaptiveFIFOOptions()
        self.admission_limit = 3
        self.events = []
        self._jobs, self._accepted, self._order, self._launched = {}, {}, [], {}
        self._profiles, self._profile_ids = {}, {}
        self._counts, self._explored = {}, set()
        self._anchor, self._halves, self._identity = None, [], None
        self._last_change = None
        self._last_now = None
        self._queue_order = []
        self._pending_exploration = None
        self._last_hold = None
        self._generations = {}
        self._accepted_metadata = {}

    @staticmethod
    def _acceptance_metadata(job):
        """Only durable ordering/request fields, never mutable training progress."""
        return {'queue_sequence': job.queue_sequence, 'workflow_id': job.workflow_id,
                'submitted_at': job.submitted_at,
                'feedback_request': deepcopy(job.metadata.get('feedback_request')),
                'request_accepted_at': job.metadata.get('v51_request_accepted_at')}

    def _event(self, now, kind, **values):
        self.events.append({'at': now, 'kind': kind, 'alpha': self.admission_limit, **values})

    def export_state(self):
        """Durable dispatch state; rate evidence is deliberately never restored."""
        return {'schema': 'adaptive_fifo_v1', 'mode': self.mode,
                'clock_identity': _clock_identity(),
                'options': asdict(self.options), 'admission_limit': self.admission_limit,
                'last_change': self._last_change, 'order': list(self._order),
                'accepted': dict(self._accepted), 'launched': dict(self._launched),
                'accepted_metadata': {j: self._acceptance_metadata(self._jobs[j])
                    if j in self._jobs else deepcopy(self._accepted_metadata[j]) for j in self._order},
                'explored_counts': sorted(self._explored)}

    @classmethod
    def restore_state(cls, state):
        if state.get('schema') != 'adaptive_fifo_v1':
            raise ValueError('unknown adaptive FIFO state')
        if state.get('clock_identity') != _clock_identity():
            raise ValueError('cannot restore monotonic dispatch clocks across reboot')
        result = cls(state['mode'], options=AdaptiveFIFOOptions(**state['options']))
        alpha = state['admission_limit']
        if type(alpha) is not int or alpha < 1 or len(set(state['order'])) != len(state['order']):
            raise ValueError('invalid dispatch state')
        result.admission_limit = alpha
        result._order = list(state['order'])
        result._accepted = {j: None if v is None else _number(v) for j, v in state['accepted'].items()}
        result._launched = {j: _number(v) for j, v in state['launched'].items()}
        if set(result._accepted) != set(result._order) or not set(result._launched) <= set(result._order):
            raise ValueError('inconsistent durable ordering')
        result._accepted_metadata = deepcopy(state.get('accepted_metadata', {}))
        if set(result._accepted_metadata) != set(result._order):
            raise ValueError('missing durable accepted request metadata')
        for metadata in result._accepted_metadata.values():
            if (type(metadata.get('queue_sequence')) is not int or metadata['queue_sequence'] < 0
                    or not isinstance(metadata.get('submitted_at'), str)):
                raise ValueError('invalid accepted request metadata')
        result._explored = set(state['explored_counts'])
        if any(type(k) is not int or k < 1 for k in result._explored):
            raise ValueError('invalid exploration state')
        result._last_change = state['last_change']
        if result._last_change is not None:
            result._last_change = _number(result._last_change)
        return result

    def on_launch(self, job_id, now):
        now = _number(now)
        if job_id in self._launched or not self._queue_order or self._queue_order[0] != job_id:
            raise ValueError('launch must be the accepted FIFO head, exactly once')
        self._queue_order.pop(0)
        self._launched[job_id] = now
        self._anchor, self._halves, self._identity = None, [], None
        self._event(now, 'launch', job_id=job_id, dispatch_ordinal=self._order.index(job_id))

    @staticmethod
    def _remaining(job, *, queued=False):
        m = job.metadata
        step = m.get('runtime_global_step')
        if step is None and queued and not job.started_at and not job.resume_from_checkpoint and not job.latest_checkpoint_path:
            step = 0
        epoch_steps = m.get('runtime_steps_per_epoch')
        if epoch_steps is None:
            from ..prediction.released_student import steps_for_job
            try:
                epoch_steps = steps_for_job(job, BatchResolution.resolved_batch_size(job))
            except (KeyError, ValueError, TypeError):
                pass
        left = remaining_optimizer_updates(job, {'global_step': step,
            'steps_per_epoch': epoch_steps,
            'total_optimizer_updates': m.get('runtime_total_optimizer_updates'),
            'planned_epochs': m.get('runtime_planned_epochs')})
        if left is None:
            raise ValueError('unknown exact update budget: ' + job.job_id)
        return left

    @staticmethod
    def _profile_identity(job, p):
        if (p['job_signature'] != (job.packing.signature or job.job_id)
                or p['batch_size'] != BatchResolution.resolved_batch_size(job)
                or not p['artifact_sha256'] or not p['hardware']):
            raise ValueError('incompatible profile identity')
        _number(p['step_cost'], positive=True)
        _number(p['peak_vram_mib'], positive=True)
        return json.dumps({'profile': p, 'config': asdict(job.config),
            'max_steps': job.max_steps, 'max_epochs': job.max_epochs,
            'total': job.metadata.get('runtime_total_optimizer_updates'),
            'epochs': job.metadata.get('runtime_planned_epochs'),
            'steps': job.metadata.get('runtime_steps_per_epoch')}, sort_keys=True, allow_nan=False)

    def _snapshot(self, jobs, profiles, now):
        ids, steps = [], {}
        for job in sorted(jobs, key=lambda j: j.job_id):
            m = job.metadata
            phase = m.get('v51_phase', 'training' if m.get('trial_runner_ready') else 'starting')
            counter = m.get('runtime_nonupdate_phase_counter')
            identity = m.get('runtime_completion_identity')
            step = m.get('runtime_global_step')
            epoch_steps = m.get('runtime_steps_per_epoch')
            if (phase != 'training' or self._remaining(job) == 0
                    or not identity or type(counter) is not int or counter < 0
                    or type(step) is not int or step <= 0
                    or type(epoch_steps) is not int or epoch_steps <= 0 or step % epoch_steps == 0
                    or job.job_id not in self._launched
                    or not m.get('trial_runner_ready')):
                raise ValueError('not a clean training snapshot')
            ids.append((job.job_id, identity, counter, self._launched[job.job_id],
                        self._profile_identity(job, profiles[job.job_id]),
                        m.get('placement_backend'), json.dumps(m.get('placement_backend_config'), sort_keys=True),
                        m.get('precision_mode'), step // epoch_steps,
                        job.max_steps, job.max_epochs, m.get('runtime_total_optimizer_updates'),
                        m.get('runtime_planned_epochs'), json.dumps(asdict(job.config), sort_keys=True)))
            steps[job.job_id] = step
        if not ids:
            raise ValueError('no resident training group')
        return tuple(ids), (now, steps)

    def _measure(self, running, profiles, now):
        try:
            identity, current = self._snapshot(running, profiles, now)
        except (ValueError, KeyError, TypeError):
            self._anchor, self._halves, self._identity = None, [], None
            return None
        if identity != self._identity:
            self._identity, self._anchor, self._halves = identity, current, []
            return None
        start, old = self._anchor
        elapsed = now - start
        deltas = {j: n - old[j] for j, n in current[1].items()}
        if elapsed <= 0 or any(v < 0 for v in deltas.values()):
            self._anchor, self._halves = current, []
            return None
        if elapsed < self.options.min_half_seconds or min(deltas.values()) < self.options.min_half_updates:
            return None
        half = {'start': start, 'end': now, 'rates': {j: v / elapsed for j, v in deltas.items()}}
        self._halves.append(half)
        self._anchor = current
        if len(self._halves) < 2:
            return None
        halves, self._halves = self._halves, []
        count = len(running)
        factors = [median(h['rates'][j.job_id] / (1000. / profiles[j.job_id]['step_cost']) for j in running) for h in halves]
        self._counts[count] = {'at': now, 'factors': factors}
        if self._pending_exploration == count:
            self._pending_exploration = None
        self._event(now, 'valid_windows', members=[j.job_id for j in running],
                    halves=halves, count_factors=factors, basis='successful-update deltas; two disjoint windows, not independent replicates')
        return halves

    def _costs(self, job, profile, now):
        m, p = job.metadata, job.metadata.get('feedback_phase_model') or {}
        if (p.get('source') != 'measured_phase_model_v1' or p.get('covers_required_phases') is not True
                or p.get('job_signature') != (job.packing.signature or job.job_id)
                or not m.get('v51_hardware_key') or p.get('hardware_key') != m['v51_hardware_key']
                or p.get('backend') != m.get('v51_backend') or not m.get('v51_backend')
                or p.get('batch_size') != profile['batch_size']):
            raise ValueError('missing compatible completion phases')
        # Wall timestamps belong only to model provenance. API clocks are monotonic.
        wall_now = time.time() + now - time.monotonic()
        ttl = _number(m.get('v51_phase_model_ttl_seconds'), positive=True)
        if not 0 <= wall_now - timestamp(p.get('observed_at')) <= ttl:
            raise ValueError('stale completion phase model')
        values = {k: _number(p[k + '_seconds']) for k in ('startup', 'finish', 'delivery')}
        if job.resume_from_checkpoint or job.latest_checkpoint_path:
            values['startup'] = _number(p['resume_seconds'])
        return values

    def _rates(self, members, occupancy, half, profiles, halves, now):
        if set(members) == set(halves[half]['rates']) and occupancy == len(members):
            return {j: halves[half]['rates'][j] for j in members}
        valid = {k: v for k, v in self._counts.items() if 0 <= now - v['at'] <= self.options.evidence_ttl_seconds}
        if not valid:
            raise ValueError('no fresh count model')
        source = min(valid, key=lambda k: (abs(k - occupancy), k))
        # Constant aggregate throughput extrapolation from the nearest observed
        # count. Unseen lower counts are capped at the observed solo factor (or
        # nominal 1); the solo prior is not a physical performance upper bound.
        factor = valid[source]['factors'][half] * source / occupancy
        if occupancy < source:
            solo_anchor = valid[1]['factors'][half] if 1 in valid else 1.
            factor = min(solo_anchor, factor)
        return {j: (1000. / profiles[j]['step_cost']) * factor for j in members}

    def _rates_for_state(self, active, half, profiles, halves, now):
        # Override point for controllers that have explicit resident-phase history.
        training = [j for j, row in active.items() if row[0] == 'training' and row[1] > 1e-9]
        return self._rates(training, len(active), half, profiles, halves, now) if training else {}

    def _forecast(self, cap, half, queued, running, profiles, completed, sample, halves, now, deadline, budget=None,
                  *, trace=None, first_release_only=False):
        """Fixed-alpha FIFO drain/refill. Actual residents survive a decrease."""
        queue, active, receipts = list(queued), {}, dict(completed)
        trace = [] if trace is None else trace
        if any(v is None for v in self._accepted.values()):
            raise ValueError('missing accepted monotonic clock')
        budget = {'remaining': self.options.max_events} if budget is None else budget
        used = _number(sample.get('memory_used_mb', sample.get('used_mib')))
        capacity = _number(sample.get('memory_budget_mib', .9 * sample.get('total_mib', 0)), positive=True)
        reservations = {j: max(_number(p['peak_vram_mib'], positive=True),
            _number(self._jobs[j].metadata.get('trial_reserved_mb') or 0.)) for j, p in profiles.items() if j in self._jobs}
        external = max(0., used - sum(reservations[j] for j in running))
        costs = {}
        for j in self._order:
            if j in receipts:
                _number(receipts[j])
                if not self._accepted[j] <= receipts[j] <= now:
                    raise ValueError('receipt outside accepted causal interval')
                continue
            if j not in self._jobs:
                raise ValueError('outstanding accepted member requires fresh job snapshot')
            job = self._jobs[j]
            if j not in profiles or job.hold:
                raise ValueError('unprepared or held accepted member')
            costs[j] = self._costs(job, profiles[j], now)
            remaining = self._remaining(job, queued=j in queue)
            phase = job.metadata.get('v51_phase')
            if j in running:
                phase = 'finishing' if remaining == 0 else (phase or ('training' if job.metadata.get('trial_runner_ready') else 'starting'))
                if phase not in {'starting', 'training', 'finishing'}:
                    raise ValueError('resident phase mismatch')
                phase_start = job.metadata.get('v51_phase_started_at')
                if phase_start is None:
                    phase_start = self._launched.get(j, now) if phase == 'starting' else now
                elapsed = max(0., now - _number(phase_start))
                left = max(0., costs[j]['startup' if phase == 'starting' else 'finish'] - elapsed) if phase != 'training' else 0.
                active[j] = [phase, float(remaining), left]
            elif j not in queue:
                release = job.metadata.get('v51_resource_released_at')
                if remaining != 0 or phase != 'delivering' or release is None or _number(release) > now:
                    raise ValueError('unknown resource release/delivery lifecycle')
                receipts[j] = now + max(0., costs[j]['delivery'] - (now - release))
        t = now
        for _ in range(self.options.max_events):
            budget['remaining'] -= 1
            if budget['remaining'] < 0:
                raise TimeoutError('forecast aggregate event bound')
            if time.perf_counter() >= deadline:
                raise TimeoutError('forecast CPU deadline')
            for j, row in list(active.items()):
                phase, work, left = row
                if phase == 'starting' and left <= 1e-9:
                    row[0] = 'training'
                if row[0] == 'training' and work <= 1e-9:
                    row[0], row[2] = 'finishing', costs[j]['finish']
                if row[0] == 'finishing' and row[2] <= 1e-9:
                    receipts[j] = t + costs[j]['delivery']
                    del active[j]
                    trace.append({'at': t, 'kind': 'release', 'job_id': j})
                    if first_release_only:
                        return  # The caller consumes the physical release from the trace.
            while queue and len(active) < cap:
                j = queue[0]
                if external + sum(reservations[k] for k in active) + reservations[j] > capacity:
                    break  # Strict prefix head blocking; never skip younger work.
                queue.pop(0)
                active[j] = ['starting', float(self._remaining(self._jobs[j], queued=True)), costs[j]['startup']]
                trace.append({'at': t, 'kind': 'launch', 'job_id': j})
            if not active:
                if queue:
                    raise ValueError('FIFO head memory infeasible')
                break
            training = [j for j, r in active.items() if r[0] == 'training' and r[1] > 1e-9]
            rates = self._rates_for_state(active, half, profiles, halves, now) if training else {}
            durations = [row[2] for row in active.values() if row[0] != 'training']
            for j in training:
                rate = rates[j]
                if rate is None:  # Explicit zero-work scenario, never a measured rate.
                    durations.append(0.)
                elif rate > 0:
                    durations.append(active[j][1] / rate)
                elif rate != 0:
                    raise ValueError('invalid training rate')
            if not durations:
                raise ValueError('no timed event for suspended training')
            dt = min(durations)
            if not math.isfinite(dt) or dt < 0:
                raise ValueError('invalid forecast time')
            t += dt
            for j, row in active.items():
                if row[0] == 'training':
                    row[1] = 0. if rates[j] is None else max(0., row[1] - dt * rates[j])
                else:
                    row[2] = max(0., row[2] - dt)
        else:
            raise TimeoutError('forecast event bound')
        if set(receipts) != set(self._order):
            raise ValueError('forecast missing sealed accepted members')
        score = self._score(receipts)
        return {'score': score, 'receipts': receipts, 'trace': trace,
                'rate_model': 'actual current-group rates; median normalized count factors elsewhere; nearest count inverse-count extrapolation with observed/nominal solo anchor for unseen lower counts (not a physical bound)',
                'phase_model': 'serialized phase proxy; full future finish reserve may overcount phases already completed; first-observed current-phase elapsed is conservative'}

    def _score(self, receipts):
        if any(v is None for v in self._accepted.values()):
            raise ValueError('missing accepted monotonic clock')
        if self.mode == 'individual':
            return sum(receipts[j] - self._accepted[j] for j in self._order) / len(self._order)
        requests, owners = {}, {}
        for j in self._order:
            metadata = self._acceptance_metadata(self._jobs[j]) if j in self._jobs else self._accepted_metadata[j]
            raw = metadata.get('feedback_request') or {}
            members = tuple(raw.get('required_job_ids', [j]))
            if raw and (raw.get('completion_policy') != 'group' or raw.get('membership_finalized') is not True):
                raise ValueError('unsealed bracket')
            if j not in members or len(set(members)) != len(members) or not set(members) <= set(self._order):
                raise ValueError('missing sealed bracket member')
            submitted = metadata.get('request_accepted_at')
            if submitted is None:
                submitted = self._accepted[j] - (timestamp(metadata['submitted_at']) - timestamp(raw.get('submitted_at', metadata['submitted_at'])))
            submitted = _number(submitted)
            key = (raw.get('workflow_id', metadata.get('workflow_id')), raw.get('request_id', j))
            value = (tuple(sorted(members)), submitted)
            if key in requests and requests[key] != value:
                raise ValueError('inconsistent sealed bracket')
            requests[key] = value
            for member in members:
                if member in owners and owners[member] != key:
                    raise ValueError('overlapping or inconsistently owned bracket')
                owners[member] = key
                if submitted > self._accepted[member]:
                    raise ValueError('request follows member acceptance')
        return sum(max(receipts[j] for j in members) - start for members, start in requests.values()) / len(requests)

    def observe(self, *, now, queued_jobs, running_jobs, profiles, gpu_sample, completed_job_ids):
        now = _number(now)
        if self._last_now is not None and now <= self._last_now:
            self._anchor, self._halves, self._identity = None, [], None
            return
        self._last_now = now
        if self._last_change is None:
            self._last_change = now
        seen = list(queued_jobs) + list(running_jobs)
        if len({j.job_id for j in seen}) != len(seen):
            raise ValueError('duplicate observed job identity')
        for job in seen:
            j = job.job_id
            raw_accepted = job.metadata.get('v51_accepted_at')
            accepted = None if raw_accepted is None else _number(raw_accepted)
            if ((accepted is not None and accepted > now)
                    or (j in self._accepted and self._accepted[j] != accepted)):
                raise ValueError('acceptance clock changed or in future')
            if j not in self._accepted:
                self._order.append(j)
                self._accepted[j] = accepted
            self._jobs[j] = deepcopy(job)
            self._accepted_metadata[j] = self._acceptance_metadata(job)
        # Input is already accepted FIFO order. Durable acceptance sequence is
        # preferred over which list (queued/running) happened to be iterated first.
        rank = {j: i for i, j in enumerate(self._order)}
        self._order.sort(key=lambda j: (self._accepted_metadata[j]['queue_sequence'] if self._accepted_metadata[j]['queue_sequence'] > 0 else math.inf,
            self._accepted[j] if self._accepted[j] is not None else math.inf, rank[j]))
        queued = [j.job_id for j in queued_jobs if not j.status.is_terminal and j.job_id not in self._launched]
        if queued != [j for j in self._order if j in set(queued)]:
            raise ValueError('queued jobs violate durable FIFO order')
        self._queue_order = queued
        running = [j.job_id for j in running_jobs]
        if not running:
            self._anchor, self._halves, self._identity = None, [], None
            if not queued and set(self._order) <= set(completed_job_ids):
                self._pending_exploration = None
                self._explored.clear()
                self._counts.clear()
                if self.admission_limit != 3:
                    self.admission_limit = 3
                    self._last_change = now
                    self._event(now, 'busy_period_empty_reset')
            return
        try:
            changed = False
            for job in running_jobs:
                generation = job.metadata.get('runtime_completion_identity')
                if generation:
                    changed |= job.job_id in self._generations and self._generations[job.job_id] != generation
                    self._generations[job.job_id] = generation
            for j, p in profiles.items():
                if j not in self._jobs:
                    continue
                identity = self._profile_identity(self._jobs[j], p)
                changed |= j in self._profile_ids and self._profile_ids[j] != identity
                self._profile_ids[j], self._profiles[j] = identity, deepcopy(p)
            if changed:
                self._counts.clear()
                self._anchor, self._halves, self._identity = None, [], None
            self._counts = {k: v for k, v in self._counts.items()
                            if 0 <= now - v['at'] <= self.options.evidence_ttl_seconds}
            if len({(p['artifact_sha256'], p['hardware']) for p in profiles.values()}) != 1:
                raise ValueError('mixed predictor/hardware identities')
            halves = self._measure(running_jobs, profiles, now)
            if halves is None or now - self._last_change < self.options.dwell_seconds:
                return
            deadline = time.perf_counter() + self.options.cpu_budget_ms / 1000.
            budget = {'remaining': self.options.max_events}
            results = {cap: [self._forecast(cap, h, queued, running, profiles, completed_job_ids, gpu_sample, halves, now, deadline, budget)
                            for h in range(2)] for cap in (max(1, self.admission_limit - 1), self.admission_limit, self.admission_limit + 1)
                       if cap <= max(len(self._order), self.admission_limit)}
            base = results[self.admission_limit]
            eligible = []
            for cap, pair in results.items():
                if cap == self.admission_limit or (cap > self.admission_limit and cap not in self._counts):
                    continue
                if all(b['score'] - v['score'] >= max(self.options.min_gain_seconds, self.options.min_gain_fraction * b['score']) for b, v in zip(base, pair)):
                    eligible.append(cap)
            old = self.admission_limit
            reason = 'hold'
            if eligible:
                self.admission_limit = min(eligible, key=lambda cap: (max(v['score'] for v in results[cap]), cap))
                reason = 'two_half_predicted_gain'
            elif queued and self._pending_exploration is None:
                target, head = old + 1, queued[0]
                duration = halves[-1]['end'] - halves[0]['start']
                next_completion = min(self._remaining(j) / max(halves[h]['rates'][j.job_id] for h in range(2)) for j in running_jobs)
                occupied = sum(max(profiles[j]['peak_vram_mib'], self._jobs[j].metadata.get('trial_reserved_mb') or 0.) for j in running)
                used = gpu_sample.get('memory_used_mb', gpu_sample.get('used_mib'))
                budget = gpu_sample.get('memory_budget_mib', .9 * gpu_sample.get('total_mib', 0))
                if (len(running) == old and target not in self._counts and target not in self._explored
                        and not self._jobs[head].hold
                        and max(occupied, used) + profiles[head]['peak_vram_mib'] <= budget
                        and next_completion >= duration + self._costs(self._jobs[head], profiles[head], now)['startup'] + 5):
                    self.admission_limit = target
                    self._explored.add(target)
                    self._pending_exploration = target
                    reason = 'productive_head_exploration_no_latency_guarantee'
            if self.admission_limit != old:
                self._last_change = now
            self._event(now, 'decision', previous_alpha=old, reason=reason, occupancy=len(running),
                        predictions={str(k): v for k, v in results.items()})
            self._last_hold = None
        except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError, TimeoutError) as exc:
            # Scoring failure never changes the ordinary executable FIFO cap.
            if str(exc) != self._last_hold:
                self._event(now, 'forecast_hold', reason=str(exc))
                self._last_hold = str(exc)
