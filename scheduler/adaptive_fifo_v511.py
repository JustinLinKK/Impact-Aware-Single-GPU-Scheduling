"""Strict FIFO, nonpreemptive alpha control with trace-local workload experience.

Execution/exit/receipt ownership stays with the harness. Unknown rate evidence
abstains; a separately bounded productive-head exploration may still proceed.
"""
from copy import deepcopy
import math
import time

from .adaptive_fifo import AdaptiveFIFOController, _number
from .fifo_experience import ExperienceStore
from .contextual_rates import InsufficientHistory, estimate, reference_generation
from .workload_fingerprint import context, digest


class AdaptiveFIFO511Controller(AdaptiveFIFOController):
    def __init__(self, mode='individual', *, options=None, store=None, episode='standalone',
                 hardware_prior=3, wall_offset=None):
        super().__init__(mode, options=options)
        if type(hardware_prior) is not int or not 1 <= hardware_prior <= 6:
            raise ValueError('hardware prior outside experimental bounds')
        self.store = store or ExperienceStore('standalone-' + mode)
        self.episode = episode
        self.admission_limit = hardware_prior
        self.wall_offset = time.time() - time.monotonic() if wall_offset is None else wall_offset
        self._last_observation = self._last_score = -math.inf
        self._rate_bound, self._query_deadline = 'rate', 0.
        self._queries, self._estimates = [], {}
        self._support = None
        self._support_log = []
        self._instant_tail = False
        self._demoted = set(self.store.control.get('demoted', []))
        self._errors = {}
        self._tried = set(self.store.control.get('tried', []))
        self._trace_pending = self.store.control.get('pending')
        self._context_priors = dict(self.store.control.get('context_priors', {}))
        self._initial_context = None
        self._latest_halves = None
        self._pending_exploration = self._trace_pending['target'] if self._trace_pending else None
        self._last_overhead_seconds = .01

    def _save_control(self):
        self.store.append({'kind': 'control', 'state': {
            'tried': sorted(self._tried), 'pending': deepcopy(self._trace_pending),
            'context_priors': dict(self._context_priors), 'demoted': sorted(self._demoted)}})

    def export_state(self):
        return {**super().export_state(), 'schema': 'adaptive_fifo_v511_v1',
            'namespace': self.store.namespace, 'episode': self.episode,
            'history_parent_sha256': self.store.parent, 'trace_control': deepcopy(self.store.control),
            'wall_offset': self.wall_offset}

    @classmethod
    def restore_state(cls, state, *, store):
        if state.get('schema') != 'adaptive_fifo_v511_v1' or state.get('namespace') != store.namespace:
            raise ValueError('incompatible V5.1.1 restore namespace')
        if state['history_parent_sha256'] != store.parent or state['trace_control'] != store.control:
            raise ValueError('dispatch state and history are not the same checkpoint')
        legacy = {**state, 'schema': 'adaptive_fifo_v1'}
        base = AdaptiveFIFOController.restore_state(legacy)
        result = cls(state['mode'], options=base.options, store=store, episode=state['episode'],
                     hardware_prior=state['admission_limit'], wall_offset=state['wall_offset'])
        for key in ('_order', '_accepted', '_launched', '_accepted_metadata', '_last_change', '_explored'):
            setattr(result, key, deepcopy(getattr(base, key)))
        # Raw windows are intentionally discarded after restart; pending debt is
        # durable. The caller must reconcile exact worker births before dispatch.
        return result

    @staticmethod
    def _fingerprint(profile):
        fp = profile.get('v511_fingerprint')
        if not fp or fp.get('schema') != 'fifo_training_fingerprint_v1':
            raise ValueError('missing prepared training-graph fingerprint')
        return fp

    def _peer_context(self, members, target, occupancy, profiles, phases=None):
        phases = phases or {j: 'training' for j in members}
        if occupancy != len(phases) or phases.get(target) != 'training':
            raise ValueError('resident phase membership not proven')
        if any(p not in {'starting', 'training', 'finishing'} for p in phases.values()):
            raise ValueError('unknown resident phase')
        transient = {p: [self._fingerprint(profiles[j]) for j in phases if phases[j] == p]
                     for p in ('starting', 'finishing')}
        return context([self._fingerprint(profiles[j]) for j in members if j != target],
            resident_count=occupancy, starting=len(transient['starting']), finishing=len(transient['finishing']),
            phase_peers=transient if any(transient.values()) else None)

    def _snapshot(self, jobs, profiles, now):
        # Every resident remains in both the identity and memory accounting.
        # Only jobs doing successful optimizer updates contribute training rates.
        phases, training, residents = {}, [], []
        for job in sorted(jobs, key=lambda j: j.job_id):
            m = job.metadata
            phase = m.get('v51_phase', 'training' if m.get('trial_runner_ready') else 'starting')
            if phase == 'training' and self._remaining(job) == 0: phase = 'finishing'
            if phase not in {'starting', 'training', 'finishing'} or job.job_id not in self._launched:
                raise ValueError('resident phase identity not proven')
            phases[job.job_id] = phase
            residents.append((job.job_id, phase, self._launched[job.job_id],
                m.get('runtime_completion_identity'), m.get('runtime_nonupdate_phase_counter'),
                digest(self._profile_identity(job, profiles[job.job_id]))))
            if phase == 'training': training.append(job)
        identity, counters = super()._snapshot(training, profiles, now)
        self._window_phases = phases
        identity = tuple(tuple(digest(v) if i == 4 else v for i, v in enumerate(row)) for row in identity)
        full_identity = (identity, tuple(residents))
        self._event(now, 'phase_snapshot', identity=full_identity, counters=counters[1],
                    resident_phases=dict(phases), identity_sha256=digest(full_identity))
        return full_identity, counters

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
        duration = now - start
        deltas = {j: value - old[j] for j, value in current[1].items()}
        if duration <= 0 or any(v < 0 for v in deltas.values()):
            self._anchor, self._halves = current, []
            return None
        if duration < self.options.min_half_seconds or min(deltas.values()) < self.options.min_half_updates:
            return None
        self._halves.append({'start': start, 'end': now,
            'rates': {j: n / duration for j, n in deltas.items()},
            'start_steps': dict(old), 'end_steps': dict(current[1]),
            'stable_identity_sha256': digest(identity), 'resident_phases': dict(self._window_phases)})
        self._anchor = current
        required = 2 if all(p == 'training' for p in self._window_phases.values()) else 1
        if len(self._halves) < required: return None
        halves, self._halves = self._halves, []
        self._event(now, 'valid_windows', members=[j.job_id for j in running], halves=halves,
                    basis='exact successful-update counters; stable membership and phase')
        return halves

    def _record_windows(self, running, profiles, halves, now):
        phases = halves[0]['resident_phases']
        members = [j for j, phase in phases.items() if phase == 'training']
        # Query previous history before adding this observation, so error checks
        # cannot grade a row against itself.
        old_deadline = time.perf_counter() + .010
        for job in running:
            if job.job_id not in members: continue
            j, p = job.job_id, profiles[job.job_id]
            fp = self._fingerprint(p)
            peers = self._peer_context(members, j, len(phases), profiles, phases)
            key = digest([fp['exact'], peers['key']])
            try:
                prior = estimate(self.store, fp, peers, p, now=halves[0]['start'] + self.wall_offset,
                                 deadline=old_deadline)
            except (ValueError, TimeoutError):
                prior = None
            for half in halves:
                if prior:
                    # A predeclared 50% multiplicative miss on two consecutive
                    # clean windows demotes this context, independent of JCT.
                    error = abs(math.log(half['rates'][j] / prior['rate'])) > math.log(1.5)
                    self._errors[key] = self._errors.get(key, 0) + 1 if error else 0
                    if self._errors[key] >= 2 and key not in self._demoted:
                        self._demoted.add(key)
                        self.store.append({'kind': 'context_demoted', 'context_key': key,
                                           'at_unix': now + self.wall_offset, 'rule': 'two_windows_factor_1.5'})
                duration = half['end'] - half['start']
                self.store.observe(episode=self.episode, fingerprint=fp, context=peers,
                    updates=round(half['rates'][j] * duration), seconds=duration,
                    observed_unix=half['end'] + self.wall_offset,
                    reference_rate=1000. / p['step_cost'], reference_memory=p['peak_vram_mib'],
                    reference_generation=reference_generation(p), job_id=j,
                    accepted_unix=self._accepted[j] + self.wall_offset,
                    window_start_unix=half['start'] + self.wall_offset,
                    counter_proof={'start_step': half['start_steps'][j], 'end_step': half['end_steps'][j],
                                   'stable_identity_sha256': half['stable_identity_sha256'],
                                   'resident_phases': dict(phases), 'peer_context': deepcopy(peers)})
        if len(members) != len(running) or len(halves) != 2:
            return  # Mixed-phase evidence never proves a stable alpha prior.
        if self._trace_pending and len(running) == self._trace_pending['target']:
            self._event(now, 'exploration_observed', pending=self._trace_pending,
                        note='observed occupancy, not causal counterfactual proof')
            self._trace_pending = None
            self._pending_exploration = None
        if self._initial_context and len(running) == self.admission_limit:
            self._context_priors[self._initial_context] = {'alpha': self.admission_limit,
                'observed_unix': now + self.wall_offset, 'episode': self.episode,
                'basis': 'two_clean_windows_at_actual_occupancy'}
        self._save_control()

    def _forecast(self, cap, half, queued, running, profiles, completed, sample, halves, now, deadline, budget=None):
        evidence = {'cap': cap, 'half': half, 'direction': self._rate_bound,
                    'suspended_targets': 0, 'optimistic_tail': False}
        self._support = evidence
        try:
            return super()._forecast(cap, half, queued, running, profiles, completed, sample,
                                     halves, now, deadline, budget)
        finally:
            self._support = None
            self._instant_tail = False
            self._support_log.append(evidence)

    def _rates_for_state(self, active, half, profiles, halves, now):
        phases = {j: row[0] for j, row in active.items()}
        members = [j for j in phases if phases[j] == 'training']
        return self._phase_rates(members, len(active), half, profiles, halves, now, phases)

    def _rates(self, members, occupancy, half, profiles, halves, now):
        return self._phase_rates(members, occupancy, half, profiles, halves, now)

    def _phase_rates(self, members, occupancy, half, profiles, halves, now, phases=None):
        if self._instant_tail:
            return dict.fromkeys(members, None)
        if (len(halves) > half and set(members) == set(halves[half]['rates']) and occupancy == len(members)
                and now - halves[half]['end'] <= self.options.evidence_ttl_seconds):
            factor = .8 if self._rate_bound == 'lower' else 1.2 if self._rate_bound == 'upper' else 1.
            return {j: halves[half]['rates'][j] * factor for j in members}
        result = {}
        for j in members:
            if time.perf_counter() >= self._query_deadline:
                raise TimeoutError('retrieval and scoring elapsed budget')
            fp = self._fingerprint(profiles[j])
            peers = self._peer_context(members, j, occupancy, profiles, phases)
            if digest([fp['exact'], peers['key']]) in self._demoted:
                raise ValueError('context demoted after consecutive rate misses')
            key = (j, peers['key'])
            if key not in self._estimates:
                try:
                    value = estimate(self.store, fp, peers, profiles[j], now=now + self.wall_offset,
                                     deadline=self._query_deadline)
                except InsufficientHistory as exc:
                    if self._support is None:
                        raise
                    miss = {'target': j, 'fingerprint': fp['exact'], 'context': peers['key'],
                            'phases': peers['phases'], **exc.evidence}
                    if occupancy == len(members):
                        self._support['unsupported_steady'] = miss
                        raise
                    self._support.setdefault('first_transition_gap', miss)
                    if self._rate_bound == 'upper':
                        # Favor keeping the limit. Subsequent training can finish
                        # instantly; FIFO, reservations and lifecycle timers still apply.
                        self._instant_tail = self._support['optimistic_tail'] = True
                        return dict.fromkeys(members, None)
                    if self._rate_bound != 'lower':
                        raise
                    self._support['suspended_targets'] += 1
                    result[j] = 0.
                    continue
                self._estimates[key] = value
                self._queries.append({'target_job': j, 'context': peers, **value})
            result[j] = self._estimates[key][self._rate_bound]
        return result

    def _register(self, now, queued_jobs, running_jobs, profiles):
        seen = list(queued_jobs) + list(running_jobs)
        if len({j.job_id for j in seen}) != len(seen):
            raise ValueError('duplicate observed job')
        for job in seen:
            j = job.job_id
            accepted = _number(job.metadata['v51_accepted_at'])
            if accepted > now or (j in self._accepted and self._accepted[j] != accepted):
                raise ValueError('acceptance clock changed or in future')
            if j not in self._accepted:
                self._order.append(j); self._accepted[j] = accepted
            old = self._accepted_metadata.get(j)
            metadata = self._acceptance_metadata(job)
            if old is not None and old != metadata:
                raise ValueError('immutable FIFO/request metadata changed')
            self._jobs[j], self._accepted_metadata[j] = job, metadata
        rank = {j: i for i, j in enumerate(self._order)}
        self._order.sort(key=lambda j: (self._accepted_metadata[j]['queue_sequence'] or math.inf,
                                        self._accepted[j], rank[j]))
        queued = [j.job_id for j in queued_jobs if not j.status.is_terminal and j.job_id not in self._launched]
        if queued != [j for j in self._order if j in set(queued)]:
            raise ValueError('queued jobs violate durable FIFO order')
        self._queue_order = queued
        return queued

    def _explore(self, now, queued, running, profiles, sample, halves):
        old = self.admission_limit
        if not queued or self._trace_pending is not None or len(self._tried) >= 6 or old >= 6:
            return False
        head = queued[0]
        if head not in profiles or self._jobs[head].hold or len(running) != old:
            return False
        fingerprints = [self._fingerprint(profiles[j])['exact'] for j in running + [head]]
        trial = digest({'target': old + 1, 'mix': sorted(fingerprints)})
        if trial in self._tried:
            return False
        used = _number(sample.get('used_mib', sample.get('memory_used_mb')))
        capacity = _number(sample.get('memory_budget_mib', .9 * sample.get('total_mib', 0)), positive=True)
        def reserve(j):
            return max(profiles[j]['peak_vram_mib'], self._jobs[j].metadata.get('trial_reserved_mb') or 0.)
        occupied = max(used, sum(reserve(j) for j in running))
        startup = self._costs(self._jobs[head], profiles[head], now)['startup']
        # Validate all current phase inputs as well; exploration cannot bypass TTL.
        for j in running: self._costs(self._jobs[j], profiles[j], now)
        remaining = min(self._remaining(self._jobs[j]) / max(h['rates'][j] for h in halves) for j in running)
        observation = halves[-1]['end'] - halves[0]['start']
        if occupied + reserve(head) > capacity or remaining < startup + observation + 5:
            return False
        if time.perf_counter() >= self._query_deadline:
            raise TimeoutError('exploration scoring elapsed budget')
        self.admission_limit += 1
        self._tried.add(trial)
        self._trace_pending = {'context': trial, 'target': self.admission_limit,
                               'episode': self.episode, 'at_unix': now + self.wall_offset}
        self._pending_exploration = self.admission_limit
        return True

    def observe(self, *, now, queued_jobs, running_jobs, profiles, gpu_sample, completed_job_ids):
        now = _number(now)
        queued = self._register(now, queued_jobs, running_jobs, profiles)
        # Registration is cheap and mandatory before on_launch; rate observation
        # and scoring have their own cadence, independent of the safety gate.
        if now - self._last_observation < 1.:
            return
        self._last_observation = now
        self._queries, self._support_log = [], []
        if self._last_change is None: self._last_change = now
        running = [j.job_id for j in running_jobs]
        for j, p in profiles.items():
            if j in self._jobs:
                self._profile_identity(self._jobs[j], p)
        # Bind a learned initial prior to a prepared accepted prefix. A partial
        # preparation queue is not evidence that alpha should be reduced.
        prefix = self._order[:3]
        if self._initial_context is None and len(prefix) == 3 and all(j in profiles and profiles[j].get('v511_fingerprint') for j in prefix):
            self._initial_context = digest([self._fingerprint(profiles[j])['exact'] for j in prefix])
            prior = self._context_priors.get(self._initial_context)
            if (prior and self._trace_pending is None and
                    prior.get('basis') == 'two_clean_windows_at_actual_occupancy' and
                    0 <= now + self.wall_offset - prior['observed_unix'] <= 14400):
                old = self.admission_limit
                # Asynchronous preparation can finish after the first dispatch.
                # Reconsider the initial target once the same accepted prefix is
                # known; never wait for all captures before ordinary FIFO starts.
                self.admission_limit = max(old - 1, min(old + 1, prior['alpha']))
                if self.admission_limit != old:
                    self._last_change = now
                    payload = {'at': now, 'kind': 'decision', 'episode': self.episode,
                        'alpha': self.admission_limit, 'previous_alpha': old,
                        'reason': 'compatible_initial_prefix_prior', 'prior': deepcopy(prior),
                        'context_key': self._initial_context, 'accepted_prefix': prefix,
                        'history_queries': [], 'history_parent_sha256': self.store.parent,
                        'note': 'initial-target heuristic; no forecast-gain or counterfactual claim'}
                    self.events.append(payload); self.store.append(payload)
        if not running:
            self._anchor, self._halves, self._identity = None, [], None
            # Preserve evidence, alpha and unresolved exploration across idle time.
            return
        try:
            if len(self._order) > 64: raise ValueError('accepted queue exceeds bounded simulator')
            if any(j not in profiles for j in running): raise ValueError('resident profile missing')
            for j in running: self._fingerprint(profiles[j])
            halves = self._measure(running_jobs, profiles, now)
            if halves is None: return
            self._record_windows(running_jobs, profiles, halves, now)
            if len(halves) != 2 or any(p != 'training' for p in halves[0]['resident_phases'].values()):
                return
            self._latest_halves = halves
            # No queued, prepared FIFO head means there is no useful admission
            # decision. Learning continues, but expensive tail forecasts do not.
            if not queued or queued[0] not in profiles or self._jobs[queued[0]].hold:
                return
            # The FIFO head already has a slot. Its ordinary admission path
            # can perform the memory check without a speculative score first.
            if len(running) < self.admission_limit:
                return
            if not profiles[queued[0]].get('v511_fingerprint'):
                return
            if now - self._last_change < self.options.dwell_seconds or now - self._last_score < 2.:
                return
            self._last_score = now
            started = time.perf_counter()
            self._query_deadline = started + self.options.cpu_budget_ms / 1000.
            self._queries, self._estimates = [], {}
            old, results, failures = self.admission_limit, {}, {}
            budget = {'remaining': self.options.max_events}
            # Optimistic incumbent vs pessimistic alternatives prevents a wide
            # empirical envelope from manufacturing an improvement.
            for cap in (old, max(1, old - 1), min(6, old + 1)):
                if cap in results or str(cap) in failures: continue
                self._rate_bound = 'upper' if cap == old else 'lower'
                try:
                    results[cap] = [self._forecast(cap, h, queued, running, profiles,
                        completed_job_ids, gpu_sample, halves, now, self._query_deadline, budget) for h in range(2)]
                    for value in results[cap]:
                        value['rate_model'] = 'current group or trace-local empirical envelope, with explicit mixed-phase scenarios'
                except (ValueError, KeyError, TimeoutError) as exc:
                    failures[str(cap)] = str(exc)
            eligible = []
            if old in results:
                for cap, pair in results.items():
                    if cap != old and self._benefits_every_half(results[old], pair):
                        eligible.append(cap)
            reason = 'insufficient_or_no_gain'
            if time.perf_counter() >= self._query_deadline:
                raise TimeoutError('retrieval and scoring elapsed budget')
            if eligible:
                self.admission_limit = min(eligible, key=lambda c: (max(v['score'] for v in results[c]), c))
                reason = 'two_half_pessimistic_history_gain'
            elif self._explore(now, queued, running, profiles, gpu_sample, halves):
                reason = 'productive_head_exploration_no_latency_guarantee'
            elapsed = time.perf_counter() - started
            if self.admission_limit != old:
                self._last_change = now
            previous_overhead = self._last_overhead_seconds
            self._last_overhead_seconds = elapsed
            payload = {'at': now, 'kind': 'decision', 'previous_alpha': old,
                'alpha': self.admission_limit, 'reason': reason, 'occupancy': len(running),
                'predictions': {str(k): v for k, v in results.items()}, 'failures': failures,
                'retrieval_and_scoring_seconds': elapsed, 'history_queries': self._queries,
                'forecast_support': self._support_log, 'previous_scoring_seconds': previous_overhead,
                'history_parent_sha256': self.store.parent}
            self.events.append(payload)
            self.store.append({'kind': 'decision', 'episode': self.episode, **payload})
            self._save_control()
            self._last_hold = None
        except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError, TimeoutError) as exc:
            if self._queries or self._support_log:
                # Retain every estimate actually consulted, including scoring
                # attempts that abstained or exhausted their elapsed budget.
                self.store.append({'kind': 'history_query', 'episode': self.episode,
                    'at': now, 'reason': str(exc), 'history_queries': self._queries,
                    'forecast_support': self._support_log,
                    'history_parent_sha256': self.store.parent})
            if str(exc) != self._last_hold:
                self._event(now, 'forecast_hold', reason=str(exc))
                self._last_hold = str(exc)

    def _benefits_every_half(self, incumbent, challenger):
        for base, proposed in zip(incumbent, challenger, strict=True):
            required = max(self.options.min_gain_seconds, self.options.min_gain_fraction * base['score'])
            if base['score'] - proposed['score'] < required + self._last_overhead_seconds:
                return False
            if any(proposed['receipts'][j] > when for j, when in base['receipts'].items()):
                return False
        return True

    def close_episode(self):
        self.store.append({'kind': 'episode_closed', 'episode': self.episode,
                           'alpha': self.admission_limit, 'pending': self._trace_pending})
        self._save_control()
        self.store.checkpoint()
