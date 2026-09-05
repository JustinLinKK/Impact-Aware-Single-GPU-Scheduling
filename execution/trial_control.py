"""Bounded, cooperative step measurements without worker restarts."""

from __future__ import annotations

import time

import torch

from ..domain import SafePointType, utc_now


def configure_trial_memory(context):
    """Limit the PyTorch allocator before a cooperative runner builds its model."""
    allowance = context.job.metadata.get("trial_memory_limit_mb")
    if allowance is None or not context.job.resource_requirements.requires_gpu:
        return
    free, total = torch.cuda.mem_get_info()
    limit = min(float(allowance) * 1024**2, max(0, free - 256 * 1024**2))
    if limit <= 0:
        raise torch.OutOfMemoryError("No live memory allowance for trial allocation")
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total))
    context.store.update_job(context.job.job_id, metadata_updates={
        "trial_allocator_limit_bytes": int(limit), "trial_allocator_guard": "pytorch_caching_allocator",
    })


class StepTrialControlMixin:
    def _trial_sync(self):
        if self.job.resource_requirements.requires_gpu and torch.cuda.is_initialized():
            started = time.perf_counter()
            torch.cuda.synchronize()
            self._step_trial_state["sync_seconds"] = self._step_trial_state.get("sync_seconds", 0.0) + time.perf_counter() - started

    def _cooperative_step(self, safe_point_type, *, epoch, global_step, steps_per_epoch, metrics):
        if not self.job.metadata.get("cooperative_trial"):
            return False
        now = time.monotonic()
        control_started = time.perf_counter()
        if not hasattr(self, "_step_trial_state"):
            self._step_trial_state = {"poll_at": 0.0, "publish_at": 0.0, "command": {}, "token": None}
        state = self._step_trial_state
        sync_before = state.get("sync_seconds", 0.0)
        yielded_seconds = 0.0
        if now >= state["poll_at"] or safe_point_type != SafePointType.STEP:
            job = self.store.get_job(self.job.job_id)
            state["command"] = dict((job.metadata if job else {}).get("trial_step_command") or {})
            state["trial"] = dict((job.metadata if job else {}).get("colocation_trial") or {})
            state["poll_at"] = now + 0.05
        if now >= state["publish_at"] or safe_point_type != SafePointType.STEP or epoch != state.get("published_epoch"):
            updates = {
                "trial_runner_ready": True, "runtime_global_step": global_step,
                "runtime_steps_per_epoch": steps_per_epoch,
                "last_completed_epoch": epoch,
                **({"trial_last_metrics": metrics} if metrics else {}),
                "trial_control_seconds": state.get("control_seconds", 0.0),
                "trial_sync_seconds": state.get("sync_seconds", 0.0),
                "trial_yield_seconds": state.get("yield_seconds", 0.0),
            }
            if not state.get("ready"):
                self.event_logger.emit("trial_runner_ready", job_id=self.job.job_id, payload={"global_step": global_step})
                state["ready"] = True
            if self.job.resource_requirements.requires_gpu and torch.cuda.is_initialized():
                updates["trial_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                updates["trial_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
            self.store.update_job(self.job.job_id, last_heartbeat_at=utc_now(), metadata_updates=updates)
            state["publish_at"] = now + 0.1
            state["published_epoch"] = epoch
        while True:
            command = state["command"]
            trial = state.get("trial") or {}
            preparing = bool(trial.get("cold_start") and trial.get("phase") == "prepare" and trial.get("decision") == "pending")
            if command and float(command.get("expires_at", 0)) <= time.time():
                self.store.update_job(self.job.job_id, metadata_updates={"trial_lease_expired": command.get("owner")})
                state["command"] = {}
                break
            at_boundary = (
                command.get("candidate") and command.get("action") == "measure"
                and epoch >= int(command.get("target_epoch", epoch + 1))
            )
            reference_done = command.get("kind") == "reference" and state.get("complete") and state.get("token") == command.get("token")
            if command.get("action") != "yield" and not preparing and not at_boundary and not reference_done:
                break
            self._trial_sync()
            token = command.get("token") or trial.get("trial_id")
            if state.get("yield_token") != token:
                if state.get("yield_token") is not None:
                    duration = time.monotonic() - state.pop("yield_started")
                    yielded_seconds += duration
                    self.event_logger.emit("trial_worker_released", job_id=self.job.job_id, payload={"token": state.pop("yield_token"), "yield_seconds": duration, "global_step": global_step})
                state["yield_token"] = token
                state["yield_started"] = time.monotonic()
                self.store.update_job(self.job.job_id, metadata_updates={"trial_yield_ack": token})
                self.event_logger.emit("trial_worker_yielded", job_id=self.job.job_id, payload={"token": token, "global_step": global_step})
            external = self.control_plane.read_command(self.job.job_id)
            if external.action in {"pause", "cancel"}:
                break
            if preparing and time.monotonic() - state["yield_started"] > 30:
                self.store.update_job(self.job.job_id, metadata_updates={"trial_lease_expired": trial.get("trial_id")})
                break
            time.sleep(0.01)
            current = self.store.get_job(self.job.job_id)
            state["command"] = dict((current.metadata if current else {}).get("trial_step_command") or {})
            state["trial"] = dict((current.metadata if current else {}).get("colocation_trial") or {})
        if state.get("yield_token") is not None:
            duration = time.monotonic() - state.pop("yield_started")
            yielded_seconds += duration
            self.event_logger.emit("trial_worker_released", job_id=self.job.job_id, payload={"token": state.pop("yield_token"), "yield_seconds": duration, "global_step": global_step})
        command = state["command"]
        if state["token"] != command.get("token"):
            state.update({"token": command.get("token"), "start_step": global_step, "samples": [], "complete": False, "block_step": global_step + 2, "block_started": None})
        if command.get("action") == "measure" and not state["complete"] and safe_point_type == SafePointType.STEP:
            if global_step >= state["start_step"] + 2 and state["block_started"] is None:
                self._trial_sync()
                state["block_started"] = state["measurement_started"] = time.perf_counter()
                state["block_step"] = global_step
                state["measurement_step"] = global_step
            elif state["block_started"] is not None and (
                global_step - state["block_step"] >= 4
                or time.perf_counter() - state["measurement_started"] >= 1.0
            ):
                self._trial_sync()
                finished = time.perf_counter()
                count = global_step - state["block_step"]
                state["samples"].append((finished - state["block_started"]) / count)
                state["block_started"], state["block_step"] = finished, global_step
                steps = global_step - state["measurement_step"]
                elapsed = finished - state["measurement_started"]
                sufficient = steps >= 8 and len(state["samples"]) >= 2
                if (sufficient and (elapsed >= 0.1 or steps >= 64)) or elapsed >= 1.0:
                    samples = state["samples"]
                    half = len(samples) // 2
                    window = {
                        "token": command["token"], "owner": command["owner"], "kind": command["kind"],
                        "complete": sufficient, "halves": [sum(samples[:half]) / half, sum(samples[half:]) / (len(samples) - half)] if sufficient else [],
                        "samples": samples, "steps": steps, "start_step": state["measurement_step"],
                        "end_step": global_step, "elapsed_seconds": elapsed, "finished_at": utc_now(),
                    }
                    self.store.update_job(self.job.job_id, metadata_updates={"trial_step_window": window})
                    self.event_logger.emit("trial_step_window", job_id=self.job.job_id, payload=window)
                    state["complete"] = True
        state["yield_seconds"] = state.get("yield_seconds", 0.0) + yielded_seconds
        state["control_seconds"] = state.get("control_seconds", 0.0) + max(0.0, time.perf_counter() - control_started - yielded_seconds - (state.get("sync_seconds", 0.0) - sync_before))
        return True
