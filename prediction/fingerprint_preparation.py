"""Charged CPU preparation of the V5.1.1 graph cache; no scheduler-path capture."""
from copy import deepcopy
import multiprocessing
import queue
from pathlib import Path
import threading
import time

from .fingerprint_worker import capture_fingerprint_worker
from .ml_predictor import released_source_request
from .released_student import prediction_metadata, steps_for_job, worker_resident_mib
from .source_contract import fingerprint_source
from ..scheduler.workload_fingerprint import digest, make_fingerprint


class FingerprintPredictor:
    def __init__(self, predictor, released, *, gpu_uuid, runtime_identity, backend, profile_source, source_closure, optional_async=False):
        self.predictor, self.released = predictor, released
        if not source_closure or any(not isinstance(v, str) or len(v) != 64 for v in source_closure.values()):
            raise ValueError('frozen execution-closure SHA map required')
        self.gpu_uuid, self.runtime_identity, self.backend = gpu_uuid, {
            'runtime': runtime_identity, 'execution_closure_sha256': digest(source_closure)}, backend
        self.profile_source = profile_source
        self._cache, self._lock = {}, threading.Lock()
        self.capture_events = []
        self.optional_async = optional_async
        self._results, self._requested, self.failures = {}, set(), []
        self._result_lock = threading.Lock()
        self._cancel = threading.Event()
        self._queue = queue.Queue(maxsize=64)
        self._worker = None
        self._cleanup_complete = True
        if optional_async:
            self._worker = threading.Thread(target=self._run_optional, name='v511-optional-features', daemon=True)
            self._worker.start()

    def _run_optional(self):
        while not self._cancel.is_set():
            try: job, batch = self._queue.get(timeout=.05)
            except queue.Empty: continue
            try:
                fp = self._capture(job, batch)
                with self._result_lock:
                    self._results[(job.job_id, batch)] = deepcopy(fp)
            except Exception as exc:
                # Optional features can abstain. Numeric memory preparation and
                # normal FIFO service are never failed or held by this worker.
                self.failures.append({'job_id': job.job_id, 'batch': batch, 'reason': str(exc)})
            finally:
                self._queue.task_done()

    def augment(self, job, batch, profile):
        result = deepcopy(profile)
        with self._result_lock:
            fp = self._results.get((job.job_id, batch))
            if fp is not None: result['v511_fingerprint'] = deepcopy(fp)
        return result

    def stop(self, timeout=3.):
        self._cancel.set()
        if self._worker is not None: self._worker.join(timeout)
        return self._cleanup_complete and (self._worker is None or not self._worker.is_alive())

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def _capture(self, job, batch):
        request = released_source_request(job, batch)
        metadata = prediction_metadata(job)
        request.update(raw_steps=steps_for_job(job, batch), batch_size=batch, metadata=metadata,
            hardware={'name': self.released.hardware.gpu_name,
                      'memory_bytes': int((self.released.hardware.total_vram_mb or 0) * 1024 * 1024) or None,
                      'compute_capability': self.released.hardware.compute_capability})
        source = fingerprint_source(request['source_path'], metadata)
        # Dynamic dependency analysis may abstain. The enclosing frozen harness
        # pins the complete scheduler/engine closure before and after execution;
        # an incomplete dependency receipt forbids a cache hit, not a real capture.
        key = digest({'request': request, 'source': source, 'runtime': self.runtime_identity,
                      'artifact': self.released.artifact_hash, 'normalization': self.released.normalization_hash})
        with self._lock:
            if source['complete'] and key in self._cache:
                return self._cache[key]
            context = multiprocessing.get_context('spawn')
            receiver, sender = context.Pipe(duplex=False)
            child = context.Process(target=capture_fingerprint_worker, args=(sender,
                str(self.released.runtime_src), str(self.released.normalization_path), [request],
                self.released.settings.cpu_threads), daemon=True)
            start = time.monotonic()
            child.start(); sender.close()
            try:
                deadline = start + self.released.settings.conversion_timeout_seconds
                while not receiver.poll(.05):
                    if self._cancel.is_set(): raise ValueError('optional capture cancelled at episode close')
                    rss = worker_resident_mib(child.pid)
                    if rss is not None and rss > self.released.settings.conversion_max_rss_mb:
                        raise ValueError('fingerprint capture CPU memory bound')
                    if time.monotonic() >= deadline or not child.is_alive():
                        raise ValueError('fingerprint capture deadline or child exit')
                response = receiver.recv()
                if not response.get('ok') or response.get('cuda_initialized') is not False:
                    raise ValueError('fingerprint CPU capture failed: ' + str(response.get('error')))
                audit = response['audits'][0]
                if not audit.get('numeric_parity'):
                    raise ValueError('fingerprint numeric replay failed')
                summary = audit['v511_graph']
            finally:
                receiver.close(); child.join(timeout=.2)
                if child.is_alive(): child.terminate(); child.join(timeout=2)
                if child.is_alive(): child.kill(); child.join(timeout=2)
                if child.is_alive():
                    self._cleanup_complete = False
                    raise RuntimeError('fingerprint worker cleanup incomplete')
            contract = metadata['workload_contract']
            training, adapter = contract['training'], contract.get('adapter', {}).get('context', {})
            recipe = {'batch': batch, 'accumulation': training.get('gradient_accumulation_steps', 1),
                'precision': training['precision'], 'optimizer': training['optimizer'],
                'data_path': {'backend': training['backend'], 'profile_source': self.profile_source,
                              'model_family': adapter.get('model_family'),
                              'workload_version': adapter.get('workload_version')},
                'backend': self.backend, 'ballast_mib': adapter.get('ballast_bytes', 0) / (1024 * 1024)}
            fp = make_fingerprint(summary, recipe, hardware=self.gpu_uuid,
                runtime=self.runtime_identity, artifact=self.released.artifact_hash, source=source)
            # Cache all explicit request fields including seeds; no unproved
            # source-equivalence substitution. Cache hits are provenance-checked.
            if source['complete']:
                self._cache[key] = fp
            self.capture_events.append({'job_id': job.job_id, 'batch': batch,
                'started_monotonic': start, 'completed_monotonic': time.monotonic(),
                'cache_key': key, 'fingerprint_key': fp['exact'], 'cuda_initialized': False})
            return fp

    def predict_profile_options(self, job, batch_sizes):
        outputs = deepcopy(self.predictor.predict_profile_options(job, batch_sizes))
        for batch, profile in outputs.items():
            if self.optional_async:
                key = (job.job_id, batch)
                with self._result_lock:
                    if key not in self._requested and not self._cancel.is_set():
                        try:
                            self._queue.put_nowait((deepcopy(job), batch))
                            self._requested.add(key)
                        except queue.Full:
                            self.failures.append({'job_id': job.job_id, 'batch': batch, 'reason': 'optional queue full'})
                profile.update(self.augment(job, batch, profile))
            else:
                profile['v511_fingerprint'] = deepcopy(self._capture(job, batch))
            # Conservative fallback is explicit, not a claim about reserved VRAM
            # accuracy. Fresh device checks still gate every dispatch.
            peak = profile['peak_vram_mib']
            measured = (job.metadata.get('v5_startup_envelope') or {}).get('peak_mib', 0.)
            profile['v511_reservation_mib'] = max(peak + max(256., .1 * peak), measured)
            profile['v511_reservation_basis'] = 'predicted_allocated_plus_max_256MiB_10pct_or_measured_startup'
        return outputs
