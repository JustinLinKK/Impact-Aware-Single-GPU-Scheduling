"""Graph-oriented query and recommendation helpers."""

from __future__ import annotations

from statistics import median
from typing import Any

from .domain import RunProfile

_JOB_DESIGN_CANDIDATE_KEYS = {
    "stage",
    "task_type",
    "model_key",
    "model_family",
    "packing_family",
    "proposed_batch_size",
    "proposed_epochs",
    "requires_gpu",
    "script_signature",
    "backend_preference",
    "uses_amp",
    "notes",
}


class SchedulerKnowledgeBase:
    """Read graph-style knowledge from the scheduler state store.

    The MCP server exposes these helpers directly, so this class acts as the
    compatibility layer between:
    1. concrete storage backends such as SQLite-only or Neo4j-backed state, and
    2. stable query payloads that future agents can consume without knowing the
       storage layout.
    """

    def __init__(self, store: Any):
        self.store = store

    def _current_hardware_profile(self) -> Any | None:
        try:
            return self.store.hardware_profile()
        except Exception:
            return None

    def _current_hardware_name(self) -> str:
        profile = self._current_hardware_profile()
        return str(profile.gpu_name) if profile is not None else ""

    def _current_hardware_key(self) -> str:
        profile = self._current_hardware_profile()
        return str(profile.hardware_key) if profile is not None else ""

    def _current_toolkit(self) -> tuple[str, str]:
        profile = self._current_hardware_profile()
        if profile is not None and getattr(profile, "cuda_runtime", None):
            return "cuda", str(profile.cuda_runtime)
        return "unknown", "unknown"

    def _settings(self) -> Any | None:
        return getattr(self.store, "settings", None)

    def _safe_vram_budget_mb(self) -> int | None:
        settings = self._settings()
        gpu_scheduler = getattr(settings, "gpu_scheduler", None)
        memory = getattr(gpu_scheduler, "memory", None)
        budget_gib = getattr(memory, "safe_vram_budget_gib", None)
        if budget_gib is None:
            return None
        try:
            return int(float(budget_gib) * 1024.0)
        except (TypeError, ValueError):
            return None

    def _model_key_for_signature(self, signature: str) -> str | None:
        resolver = getattr(self.store, "_model_key_for_signature", None)
        if callable(resolver):
            try:
                return resolver(signature)
            except Exception:
                return None
        for job in self.store.list_jobs():
            if job.packing.signature == signature:
                return str(job.batch_probe.model_key or job.baseline_model_id)
        return None

    def _format_hardware_summary(self, record: dict[str, Any]) -> str:
        headline = record.get("gpu_name") or record.get("hardware_key") or "unknown hardware"
        details: list[str] = []
        if record.get("total_vram_mb") is not None:
            details.append(f"{record['total_vram_mb']} MiB VRAM")
        if record.get("compute_capability"):
            details.append(f"compute capability {record['compute_capability']}")
        if record.get("toolkit_name") and record.get("toolkit_version"):
            details.append(f"{record['toolkit_name']} {record['toolkit_version']}")
        text = headline
        if details:
            text += " with " + ", ".join(details)
        stats = record.get("_stats") or {}
        evidence: list[str] = []
        for key, label in (
            ("runtime_profiles", "runtime profile"),
            ("solo_profiles", "solo profile"),
            ("pair_profiles", "pair profile"),
            ("batch_probe_profiles", "batch probe profile"),
            ("batch_size_observations", "batch-size observation"),
        ):
            count = int(stats.get(key, 0) or 0)
            if count:
                suffix = "" if count == 1 else "s"
                evidence.append(f"{count} {label}{suffix}")
        if evidence:
            text += f"; observed in {', '.join(evidence)}"
        return text

    def _hardware_record_from_current(self, profile: Any) -> dict[str, Any]:
        # The current runtime hardware is the only source that is guaranteed to
        # exist even when the graph backend is disabled or sparsely populated.
        toolkit_name = "cuda" if getattr(profile, "cuda_runtime", None) else "unknown"
        toolkit_version = str(profile.cuda_runtime) if getattr(profile, "cuda_runtime", None) else "unknown"
        record = {
            "hardware_key": str(profile.hardware_key),
            "gpu_name": str(profile.gpu_name),
            "total_vram_mb": profile.total_vram_mb,
            "compute_capability": profile.compute_capability,
            "toolkit_name": toolkit_name,
            "toolkit_version": toolkit_version,
            "torch_version": profile.torch_version,
            "summary_text": "",
            "is_current": True,
            "source": "current_runtime",
            "accelerator": {
                "accelerator_key": str(profile.hardware_key),
                "accelerator_name": str(profile.gpu_name),
                "compute_capability": profile.compute_capability,
                "total_vram_mb": profile.total_vram_mb,
            },
            "toolkit": {
                "toolkit_key": f"{toolkit_name}:{toolkit_version}",
                "toolkit_name": toolkit_name,
                "toolkit_version": toolkit_version,
                "torch_version": profile.torch_version,
            },
            "_stats": {},
        }
        record["summary_text"] = self._format_hardware_summary(record)
        return record

    def _normalize_hardware_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Storage backends can return either flattened hardware rows or richer
        # nested records. Normalize them once so the MCP layer can stay backend
        # agnostic.
        hardware = dict(payload.get("hardware") or {})
        accelerator = dict(payload.get("accelerator") or {})
        toolkit = dict(payload.get("toolkit") or {})
        record = {
            "hardware_key": payload.get("hardware_key") or hardware.get("hardware_key"),
            "gpu_name": payload.get("gpu_name") or hardware.get("gpu_name") or accelerator.get("accelerator_name"),
            "total_vram_mb": payload.get("total_vram_mb") if payload.get("total_vram_mb") is not None else hardware.get("total_vram_mb"),
            "compute_capability": payload.get("compute_capability") or hardware.get("compute_capability") or accelerator.get("compute_capability"),
            "toolkit_name": payload.get("toolkit_name") or hardware.get("toolkit_name") or toolkit.get("toolkit_name"),
            "toolkit_version": payload.get("toolkit_version") or hardware.get("toolkit_version") or toolkit.get("toolkit_version"),
            "torch_version": payload.get("torch_version") or hardware.get("torch_version") or toolkit.get("torch_version"),
            "summary_text": payload.get("summary_text") or hardware.get("summary_text") or "",
            "is_current": bool(payload.get("is_current")),
            "source": payload.get("source") or "graph_hardware_node",
            "accelerator": accelerator or None,
            "toolkit": toolkit or None,
            "_stats": dict(payload.get("_stats") or {}),
        }
        if not record["summary_text"]:
            record["summary_text"] = self._format_hardware_summary(record)
        return record

    def _merge_hardware_record(self, inventory: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
        # Hardware can be discovered from multiple sources: current runtime,
        # persisted Hardware nodes, and profile-only evidence. Merge into one
        # record keyed by hardware_key and keep the richest known fields.
        record = self._normalize_hardware_record(payload)
        hardware_key = str(record.get("hardware_key") or "")
        if not hardware_key:
            return
        current = inventory.get(hardware_key)
        if current is None:
            inventory[hardware_key] = record
            return
        for key in (
            "gpu_name",
            "total_vram_mb",
            "compute_capability",
            "toolkit_name",
            "toolkit_version",
            "torch_version",
            "summary_text",
            "source",
        ):
            if current.get(key) in {None, "", "unknown"} and record.get(key) not in {None, "", "unknown"}:
                current[key] = record[key]
        current["is_current"] = bool(current.get("is_current") or record.get("is_current"))
        if not current.get("accelerator") and record.get("accelerator"):
            current["accelerator"] = record["accelerator"]
        if not current.get("toolkit") and record.get("toolkit"):
            current["toolkit"] = record["toolkit"]
        stats = current.setdefault("_stats", {})
        for key, value in (record.get("_stats") or {}).items():
            stats[key] = int(stats.get(key, 0) or 0) + int(value or 0)
        current["summary_text"] = self._format_hardware_summary(current)

    def _touch_hardware_stat(self, inventory: dict[str, dict[str, Any]], hardware_key: str, stat_key: str) -> None:
        # Some backends do not materialize explicit Hardware nodes for every
        # observed key. This fallback keeps hardware discoverable from profile
        # evidence alone.
        if not hardware_key:
            return
        record = inventory.setdefault(
            hardware_key,
            {
                "hardware_key": hardware_key,
                "gpu_name": None,
                "total_vram_mb": None,
                "compute_capability": None,
                "toolkit_name": None,
                "toolkit_version": None,
                "torch_version": None,
                "summary_text": "",
                "is_current": False,
                "source": "derived_profile_inventory",
                "accelerator": None,
                "toolkit": None,
                "_stats": {},
            },
        )
        stats = record.setdefault("_stats", {})
        stats[stat_key] = int(stats.get(stat_key, 0) or 0) + 1
        record["summary_text"] = self._format_hardware_summary(record)

    def _hardware_inventory(self) -> dict[str, dict[str, Any]]:
        # Build a best-effort inventory by starting with the current runtime
        # hardware, then layering persisted graph records, then finally any
        # profile-derived hardware keys that appear in observations.
        inventory: dict[str, dict[str, Any]] = {}
        profile = self._current_hardware_profile()
        if profile is not None:
            current_record = self._hardware_record_from_current(profile)
            inventory[str(profile.hardware_key)] = current_record
        list_hardware_records = getattr(self.store, "list_hardware_records", None)
        if callable(list_hardware_records):
            try:
                for raw_record in list_hardware_records():
                    self._merge_hardware_record(inventory, raw_record)
            except Exception:
                pass
        for runtime_profile in self.store.list_runtime_profiles():
            self._touch_hardware_stat(inventory, str(runtime_profile.hardware_key), "runtime_profiles")
        for solo_profile in self.store.list_solo_profiles():
            self._touch_hardware_stat(inventory, str(solo_profile.hardware_key), "solo_profiles")
        if hasattr(self.store, "list_pair_profiles"):
            for pair_profile in self.store.list_pair_profiles():
                self._touch_hardware_stat(inventory, str(pair_profile.hardware_key), "pair_profiles")
        for observation in self.store.list_batch_size_observations():
            self._touch_hardware_stat(inventory, str(observation.hardware_key), "batch_size_observations")
            record = inventory.get(str(observation.hardware_key))
            if record is not None and observation.memory_total_mb is not None:
                current_total = record.get("total_vram_mb")
                if current_total is None or int(observation.memory_total_mb) > int(current_total):
                    record["total_vram_mb"] = int(observation.memory_total_mb)
                    record["summary_text"] = self._format_hardware_summary(record)
        current_name = self._current_hardware_name()
        current_key = self._current_hardware_key()
        if current_name and current_key:
            for batch_probe_profile in self.store.list_batch_probe_profiles():
                if batch_probe_profile.device_type == current_name:
                    self._touch_hardware_stat(inventory, current_key, "batch_probe_profiles")
        return inventory

    def _lookup_hardware_record(self, hardware_key: str) -> dict[str, Any] | None:
        # "current" is resolved first through the live runtime, then we fall
        # back to persisted hardware records or synthesized inventory entries.
        if hardware_key == "current":
            current_key = self._current_hardware_key()
            if current_key:
                hardware_key = current_key
        get_hardware_record = getattr(self.store, "get_hardware_record", None)
        if callable(get_hardware_record):
            try:
                raw_record = get_hardware_record(hardware_key)
            except Exception:
                raw_record = None
            if raw_record:
                record = self._normalize_hardware_record(raw_record)
                if hardware_key == self._current_hardware_key():
                    record["is_current"] = True
                    if record.get("source") == "graph_hardware_node":
                        record["source"] = "graph_hardware_node"
                return record
        return self._hardware_inventory().get(hardware_key)

    def _hardware_query_text(self, record: dict[str, Any]) -> str:
        parts = [
            str(record.get("hardware_key") or ""),
            str(record.get("gpu_name") or ""),
            str(record.get("total_vram_mb") or ""),
            str(record.get("compute_capability") or ""),
            str(record.get("toolkit_name") or ""),
            str(record.get("toolkit_version") or ""),
            str(record.get("torch_version") or ""),
            str(record.get("summary_text") or ""),
        ]
        return " ".join(parts).lower()

    def _record_matches_hardware_filter(
        self,
        hardware: str | None,
        *,
        hardware_key: str | None = None,
        device_type: str | None = None,
    ) -> bool:
        if not hardware:
            return True
        normalized = hardware.strip().lower()
        if not normalized:
            return True
        if normalized == "current":
            current_key = self._current_hardware_key()
            if hardware_key and hardware_key == current_key:
                return True
            current_name = self._current_hardware_name().lower()
            return bool(device_type and current_name and current_name in str(device_type).lower())
        if device_type and normalized in str(device_type).lower():
            return True
        if hardware_key:
            record = self._lookup_hardware_record(hardware_key)
            if record is not None and normalized in self._hardware_query_text(record):
                return True
            if normalized in str(hardware_key).lower():
                return True
        return False

    def _matches_toolkit(self, toolkit: str | None) -> bool:
        if not toolkit:
            return True
        current_name, current_version = self._current_toolkit()
        normalized = toolkit.lower()
        return normalized in {current_name.lower(), f"{current_name.lower()}:{current_version.lower()}"}

    def _profile_summary(self, kind: str, payload: dict[str, Any]) -> str:
        if kind == "runtime_profile":
            signature = payload.get("signature") or "unknown-signature"
            backend = payload.get("backend_name") or "unknown-backend"
            batch_size = payload.get("resolved_batch_size")
            runtime = payload.get("estimated_total_runtime_seconds")
            return (
                f"Runtime profile for {signature} on backend {backend}"
                f" at batch size {batch_size}: estimated total runtime {runtime} seconds."
            )
        if kind == "batch_probe_profile":
            return (
                f"Batch probe for model {payload.get('model_key')} on {payload.get('device_type')}"
                f" resolved batch size {payload.get('resolved_batch_size')}."
            )
        if kind == "solo_profile":
            return (
                f"Solo profile for {payload.get('signature')} on hardware {payload.get('hardware_key')}"
                f" observed peak VRAM {payload.get('peak_vram_mb')} MiB."
            )
        if kind == "run_profile":
            model_key = payload.get("model_key") or payload.get("signature") or "unknown-model"
            return (
                f"Run profile {payload.get('run_profile_id')} for {model_key}"
                f" ended with status {payload.get('status')}."
            )
        return f"{kind} evidence"

    def _match_model_key(self, candidate_model_key: str | None, value: str | None) -> str | None:
        if not candidate_model_key or not value:
            return None
        left = str(candidate_model_key).strip().lower()
        right = str(value).strip().lower()
        if not left or not right:
            return None
        if left == right:
            return "model_key_exact"
        if left in right or right in left:
            return "model_key_contains"
        return None

    def _runtime_match_reason(self, signature: str | None, candidate: dict[str, Any]) -> str | None:
        # Signature matches are stronger than model-key fallbacks because they
        # preserve the scheduler's exact workload identity.
        script_signature = candidate.get("script_signature")
        if script_signature and signature == script_signature:
            return "signature_exact"
        model_key = self._model_key_for_signature(str(signature or ""))
        reason = self._match_model_key(candidate.get("model_key"), model_key)
        if reason:
            return reason
        return self._match_model_key(candidate.get("model_key"), signature)

    def _solo_match_reason(self, signature: str | None, family: str | None, candidate: dict[str, Any]) -> str | None:
        reason = self._runtime_match_reason(signature, candidate)
        if reason:
            return reason
        if candidate.get("packing_family") and family and candidate["packing_family"] == family:
            return "packing_family_exact"
        if candidate.get("model_family") and family and candidate["model_family"] == family:
            return "model_family_exact"
        return None

    def _batch_probe_match_reason(self, model_key: str | None, candidate: dict[str, Any]) -> str | None:
        reason = self._match_model_key(candidate.get("model_key"), model_key)
        if reason:
            return reason
        if candidate.get("packing_family"):
            return self._match_model_key(candidate["packing_family"], model_key)
        return None

    def _run_match_reason(self, profile: RunProfile, candidate: dict[str, Any]) -> str | None:
        if candidate.get("script_signature") and profile.signature == candidate["script_signature"]:
            return "signature_exact"
        reason = self._match_model_key(candidate.get("model_key"), profile.model_key)
        if reason:
            return reason
        return self._match_model_key(candidate.get("model_key"), profile.signature)

    def _evidence_score(self, kind: str, match_reason: str, backend_name: str | None, candidate: dict[str, Any]) -> int:
        # Ranking is intentionally heuristic: runtime profiles are the strongest
        # scheduling evidence, followed by batch probes, then broader solo/run
        # history. This keeps the aggregate MCP response deterministic without
        # forcing callers to understand every profile type.
        kind_weight = {
            "runtime_profile": 40,
            "batch_probe_profile": 30,
            "solo_profile": 20,
            "run_profile": 10,
        }.get(kind, 0)
        match_weight = {
            "signature_exact": 30,
            "model_key_exact": 22,
            "model_key_contains": 12,
            "packing_family_exact": 10,
            "model_family_exact": 8,
        }.get(match_reason, 0)
        backend_bonus = 0
        if candidate.get("backend_preference") and backend_name == candidate["backend_preference"]:
            backend_bonus = 5
        return kind_weight + match_weight + backend_bonus

    def _evidence_ref(self, kind: str, payload: dict[str, Any]) -> str:
        if kind == "runtime_profile":
            return f"runtime_profile:{payload.get('profile_key')}"
        if kind == "batch_probe_profile":
            return f"batch_probe_profile:{payload.get('probe_key')}"
        if kind == "solo_profile":
            return f"solo_profile:{payload.get('hardware_key')}:{payload.get('signature')}"
        if kind == "run_profile":
            return f"run_profile:{payload.get('run_profile_id')}"
        return f"{kind}:unknown"

    def _append_evidence(
        self,
        rows: list[tuple[int, dict[str, Any]]],
        *,
        kind: str,
        payload: dict[str, Any],
        match_reason: str,
        candidate: dict[str, Any],
        backend_name: str | None = None,
    ) -> None:
        summary_text = str(payload.get("summary_text") or self._profile_summary(kind, payload))
        entry = {
            "kind": kind,
            "match_reason": match_reason,
            "summary_text": summary_text,
            "ref": self._evidence_ref(kind, payload),
            "data": payload,
        }
        rows.append((self._evidence_score(kind, match_reason, backend_name, candidate), entry))

    def _normalized_job_design_candidate(self, candidate: dict[str, Any] | None) -> dict[str, Any]:
        # Only accept the fixed public contract here so future callers get
        # predictable behavior even if they over-send fields.
        normalized: dict[str, Any] = {}
        for key in _JOB_DESIGN_CANDIDATE_KEYS:
            if candidate and key in candidate:
                normalized[key] = candidate[key]
        for key in ("proposed_batch_size", "proposed_epochs"):
            if normalized.get(key) is None:
                continue
            try:
                normalized[key] = int(normalized[key])
            except (TypeError, ValueError):
                normalized[key] = None
        return normalized

    def _current_runtime_profiles_for_candidate(self, candidate: dict[str, Any]) -> list[tuple[str, Any]]:
        current_key = self._current_hardware_key()
        profiles = self.store.list_runtime_profiles(hardware_key=current_key or None)
        rows: list[tuple[str, Any]] = []
        for profile in profiles:
            match_reason = self._runtime_match_reason(profile.signature, candidate)
            if match_reason:
                rows.append((match_reason, profile))
        return rows

    def _runtime_estimate_for_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        runtime_matches = self._current_runtime_profiles_for_candidate(candidate)
        if not runtime_matches:
            return {"found": False, "reason": "insufficient evidence"}
        proposed_batch_size = candidate.get("proposed_batch_size")
        exact: list[tuple[str, Any]] = []
        if proposed_batch_size is not None:
            # Prefer exact batch-size matches when the candidate already has a
            # proposed value; otherwise return the best matching runtime profile.
            exact = [
                (match_reason, profile)
                for match_reason, profile in runtime_matches
                if int(profile.resolved_batch_size or 0) == int(proposed_batch_size)
            ]
        chosen_match_reason, chosen_profile = (exact[0] if exact else runtime_matches[0])
        payload = chosen_profile.to_dict()
        return {
            "found": True,
            "profile": payload,
            "matched_exact_batch_size": bool(exact),
            "match_reason": chosen_match_reason,
            "summary_text": self._profile_summary("runtime_profile", payload),
        }

    def _other_hardware_profile_match(self, candidate: dict[str, Any]) -> bool:
        current_key = self._current_hardware_key()
        if not current_key:
            return False
        for profile in self.store.list_runtime_profiles():
            if profile.hardware_key == current_key:
                continue
            if self._runtime_match_reason(profile.signature, candidate):
                return True
        for solo_profile in self.store.list_solo_profiles():
            if solo_profile.hardware_key == current_key:
                continue
            if self._solo_match_reason(solo_profile.signature, solo_profile.family, candidate):
                return True
        return False

    def _matched_profiles_for_candidate(self, candidate: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        current_key = self._current_hardware_key()
        current_name = self._current_hardware_name()
        rows: list[tuple[int, dict[str, Any]]] = []
        for profile in self.store.list_runtime_profiles(hardware_key=current_key or None):
            match_reason = self._runtime_match_reason(profile.signature, candidate)
            if not match_reason:
                continue
            self._append_evidence(
                rows,
                kind="runtime_profile",
                payload=profile.to_dict(),
                match_reason=match_reason,
                candidate=candidate,
                backend_name=profile.backend_name,
            )
        for profile in self.store.list_batch_probe_profiles():
            if current_name and profile.device_type != current_name:
                continue
            match_reason = self._batch_probe_match_reason(profile.model_key, candidate)
            if not match_reason:
                continue
            self._append_evidence(
                rows,
                kind="batch_probe_profile",
                payload=profile.to_dict(),
                match_reason=match_reason,
                candidate=candidate,
            )
        for profile in self.store.list_solo_profiles(hardware_key=current_key or None):
            match_reason = self._solo_match_reason(profile.signature, profile.family, candidate)
            if not match_reason:
                continue
            self._append_evidence(
                rows,
                kind="solo_profile",
                payload=profile.to_dict(),
                match_reason=match_reason,
                candidate=candidate,
            )
        for profile in self.list_run_profiles(
            model_key=candidate.get("model_key"),
            signature=candidate.get("script_signature"),
        ):
            match_reason = self._run_match_reason(profile, candidate)
            if not match_reason:
                continue
            self._append_evidence(
                rows,
                kind="run_profile",
                payload=profile.to_dict(),
                match_reason=match_reason,
                candidate=candidate,
                backend_name=profile.backend_name,
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows[: max(1, int(limit))]]

    def _scheduler_limits(self) -> dict[str, Any]:
        settings = self._settings()
        gpu_scheduler = getattr(settings, "gpu_scheduler", None)
        if gpu_scheduler is None:
            return {}
        memory = getattr(gpu_scheduler, "memory", None)
        thresholds = getattr(gpu_scheduler, "thresholds", None)
        submission_defaults = getattr(gpu_scheduler, "submission_defaults", None)
        return {
            "safe_vram_budget_mb": self._safe_vram_budget_mb(),
            "memory": memory.to_dict() if hasattr(memory, "to_dict") else {},
            "thresholds": thresholds.to_dict() if hasattr(thresholds, "to_dict") else {},
            "backend_priority": list(getattr(gpu_scheduler, "backend_priority", []) or []),
            "submission_defaults": submission_defaults.to_dict() if hasattr(submission_defaults, "to_dict") else {},
        }

    def _backend_capabilities(self) -> dict[str, Any]:
        settings = self._settings()
        gpu_scheduler = getattr(settings, "gpu_scheduler", None)
        if gpu_scheduler is None:
            return {}
        enabled_backends = ["exclusive"]
        if getattr(getattr(gpu_scheduler, "mps", None), "enabled", False):
            enabled_backends.append("mps")
        if getattr(getattr(gpu_scheduler, "cuda_process", None), "enabled", False):
            enabled_backends.append("cuda_process")
        if getattr(getattr(gpu_scheduler, "stream", None), "enabled", False):
            enabled_backends.append("stream")
        return {
            "enabled_backends": enabled_backends,
            "priority": list(getattr(gpu_scheduler, "backend_priority", []) or []),
            "concurrent_groups_enabled": bool(getattr(gpu_scheduler, "concurrent_groups_enabled", False)),
            "concurrent_backend_allowlist": list(getattr(gpu_scheduler, "concurrent_backend_allowlist", []) or []),
            "max_packed_jobs_per_gpu": getattr(gpu_scheduler, "max_packed_jobs_per_gpu", None),
            "allow_three_way_packing": bool(getattr(gpu_scheduler, "allow_three_way_packing", False)),
        }

    def _scheduler_guidance(self) -> dict[str, Any]:
        settings = self._settings()
        gpu_scheduler = getattr(settings, "gpu_scheduler", None)
        submission_defaults = getattr(gpu_scheduler, "submission_defaults", None)
        recommended_allowlist = []
        if submission_defaults is not None:
            recommended_allowlist = list(getattr(submission_defaults, "backend_allowlist", []) or [])
        if not recommended_allowlist and gpu_scheduler is not None:
            recommended_allowlist = list(getattr(gpu_scheduler, "backend_priority", []) or [])
        return {
            "recommended_backend_allowlist": recommended_allowlist,
            "batch_probe_enabled": bool(getattr(submission_defaults, "batch_probe_enabled", False)),
            "runtime_probe_enabled": bool(getattr(submission_defaults, "runtime_probe_enabled", False)),
            "safe_vram_budget_mb": self._safe_vram_budget_mb(),
            "packing_eligible_default": bool(getattr(submission_defaults, "packing_eligible", False)),
            "packing_family_default": getattr(submission_defaults, "packing_family", None),
        }

    def _job_design_risk_flags(
        self,
        candidate: dict[str, Any],
        *,
        matched_profiles: list[dict[str, Any]],
        runtime_estimate: dict[str, Any],
        batch_size_recommendation: dict[str, Any],
        epoch_recommendation: dict[str, Any],
        scheduler_guidance: dict[str, Any],
    ) -> list[str]:
        # Keep risk flags coarse and explainable so downstream callers can make
        # simple policy decisions without reverse-engineering internal scoring.
        flags: list[str] = []
        if not matched_profiles:
            flags.append("no_matching_profiles_for_current_hardware")
        if self._other_hardware_profile_match(candidate) and not matched_profiles:
            flags.append("matching_profiles_exist_for_other_hardware_only")
        proposed_batch_size = candidate.get("proposed_batch_size")
        recommended_batch_size = batch_size_recommendation.get("recommended_batch_size")
        if proposed_batch_size is not None and recommended_batch_size is not None:
            if int(proposed_batch_size) > int(recommended_batch_size):
                flags.append("proposed_batch_size_above_recommended_batch_size")
        proposed_epochs = candidate.get("proposed_epochs")
        recommended_epochs = epoch_recommendation.get("recommended_epochs")
        if proposed_epochs is not None and recommended_epochs is not None:
            if int(proposed_epochs) > int(recommended_epochs):
                flags.append("proposed_epochs_above_historical_median")
        if runtime_estimate.get("found") and not runtime_estimate.get("matched_exact_batch_size", False):
            flags.append("runtime_estimate_reused_nearest_batch_size")
        backend_preference = candidate.get("backend_preference")
        recommended_allowlist = scheduler_guidance.get("recommended_backend_allowlist") or []
        if backend_preference and recommended_allowlist and backend_preference not in recommended_allowlist:
            flags.append("backend_preference_not_in_recommended_allowlist")
        return flags

    def _job_design_confidence(
        self,
        *,
        matched_profiles: list[dict[str, Any]],
        runtime_estimate: dict[str, Any],
        batch_size_recommendation: dict[str, Any],
        epoch_recommendation: dict[str, Any],
        risk_flags: list[str],
    ) -> float:
        confidence = min(0.40, 0.10 * len(matched_profiles))
        if runtime_estimate.get("found"):
            confidence += 0.25
        if batch_size_recommendation.get("found"):
            confidence += 0.20
        if epoch_recommendation.get("found"):
            confidence += 0.15
        if "matching_profiles_exist_for_other_hardware_only" in risk_flags:
            confidence -= 0.20
        if "no_matching_profiles_for_current_hardware" in risk_flags:
            confidence -= 0.10
        return max(0.0, min(1.0, round(confidence, 3)))

    def list_run_profiles(self, *, job_id: str | None = None, model_key: str | None = None, signature: str | None = None) -> list[RunProfile]:
        if hasattr(self.store, "list_run_profiles"):
            return self.store.list_run_profiles(job_id=job_id, model_key=model_key, signature=signature)
        profiles: list[RunProfile] = []
        jobs = [self.store.get_job(job_id)] if job_id else self.store.list_jobs()
        toolkit_name, toolkit_version = self._current_toolkit()
        gpu_name = self._current_hardware_name()
        for job in [job for job in jobs if job is not None]:
            model = str(job.batch_probe.model_key or job.baseline_model_id)
            if model_key is not None and model != model_key:
                continue
            if signature is not None and job.packing.signature != signature:
                continue
            profiles.append(
                RunProfile(
                    run_profile_id=f"run::job::{job.job_id}",
                    run_kind="training_resume" if (job.latest_checkpoint_path or job.resume_from_checkpoint) else "training",
                    status=job.status.value,
                    job_id=job.job_id,
                    backend_name=str(job.metadata.get("placement_backend") or "exclusive"),
                    toolkit_name=toolkit_name,
                    toolkit_version=toolkit_version,
                    model_key=model,
                    signature=job.packing.signature,
                    resolved_batch_size=job.metadata.get("resolved_batch_size"),
                    epochs=job.max_epochs,
                    total_steps=job.max_steps,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                    source="synthetic",
                    metadata={"gpu_name": gpu_name},
                )
            )
        return profiles

    def search_hardware(self, *, query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        normalized = str(query or "").strip().lower()
        rows = list(self._hardware_inventory().values())
        if normalized:
            rows = [row for row in rows if normalized in self._hardware_query_text(row)]
        rows.sort(key=lambda row: (not bool(row.get("is_current")), str(row.get("gpu_name") or ""), str(row.get("hardware_key") or "")))
        trimmed = rows[: max(1, int(limit))]
        return [
            {
                "hardware_key": row.get("hardware_key"),
                "gpu_name": row.get("gpu_name"),
                "total_vram_mb": row.get("total_vram_mb"),
                "compute_capability": row.get("compute_capability"),
                "toolkit_name": row.get("toolkit_name"),
                "toolkit_version": row.get("toolkit_version"),
                "torch_version": row.get("torch_version"),
                "summary_text": row.get("summary_text"),
                "is_current": bool(row.get("is_current")),
            }
            for row in trimmed
        ]

    def get_hardware_context(self, hardware_key: str = "current", include_scheduler_limits: bool = True) -> dict[str, Any]:
        # This tool is intentionally read-only and returns a self-contained
        # payload so callers do not need follow-up graph queries for basic
        # hardware, toolkit, and scheduler-limit inspection.
        record = self._lookup_hardware_record(hardware_key)
        if record is None:
            return {
                "found": False,
                "hardware": None,
                "accelerator": None,
                "toolkit": None,
                "backend_capabilities": self._backend_capabilities(),
                "scheduler_limits": self._scheduler_limits() if include_scheduler_limits else {},
                "source": "not_found",
            }
        hardware = {
            "hardware_key": record.get("hardware_key"),
            "gpu_name": record.get("gpu_name"),
            "total_vram_mb": record.get("total_vram_mb"),
            "compute_capability": record.get("compute_capability"),
            "toolkit_name": record.get("toolkit_name"),
            "toolkit_version": record.get("toolkit_version"),
            "torch_version": record.get("torch_version"),
            "summary_text": record.get("summary_text"),
            "is_current": bool(record.get("is_current")),
        }
        accelerator = record.get("accelerator") or (
            {
                "accelerator_key": record.get("hardware_key"),
                "accelerator_name": record.get("gpu_name"),
                "compute_capability": record.get("compute_capability"),
                "total_vram_mb": record.get("total_vram_mb"),
            }
            if record.get("gpu_name") or record.get("total_vram_mb") is not None
            else None
        )
        toolkit = record.get("toolkit") or (
            {
                "toolkit_key": f"{record.get('toolkit_name')}:{record.get('toolkit_version')}",
                "toolkit_name": record.get("toolkit_name"),
                "toolkit_version": record.get("toolkit_version"),
                "torch_version": record.get("torch_version"),
            }
            if record.get("toolkit_name") or record.get("toolkit_version")
            else None
        )
        return {
            "found": True,
            "hardware": hardware,
            "accelerator": accelerator,
            "toolkit": toolkit,
            "backend_capabilities": self._backend_capabilities(),
            "scheduler_limits": self._scheduler_limits() if include_scheduler_limits else {},
            "source": record.get("source") or ("current_runtime" if hardware_key == "current" else "derived_profile_inventory"),
        }

    def get_job_graph_context(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        signature = job.packing.signature
        model_key = str(job.batch_probe.model_key or job.baseline_model_id)
        return {
            "job": job.to_dict(),
            "events": self.store.list_events(job_id=job_id),
            "run_profiles": [profile.to_dict() for profile in self.list_run_profiles(job_id=job_id)],
            "runtime_profiles": [profile.to_dict() for profile in self.store.list_runtime_profiles(signature=signature)] if signature else [],
            "solo_profile": self.store.get_solo_profile(signature).to_dict() if signature and self.store.get_solo_profile(signature) else None,
            "batch_probe_profiles": [
                profile.to_dict()
                for profile in self.store.list_batch_probe_profiles()
                if profile.model_key == model_key
            ],
        }

    def search_profiles(
        self,
        *,
        model_name: str | None = None,
        hardware: str | None = None,
        backend: str | None = None,
        toolkit: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        model_name = str(model_name or "")
        results: list[dict[str, Any]] = []
        for profile in self.store.list_runtime_profiles():
            if model_name and model_name not in str(self._model_key_for_signature(profile.signature) or profile.signature):
                continue
            if backend and backend != profile.backend_name:
                continue
            if not self._matches_toolkit(toolkit):
                continue
            if not self._record_matches_hardware_filter(hardware, hardware_key=profile.hardware_key):
                continue
            results.append({"kind": "runtime_profile", "data": profile.to_dict(), "summary_text": getattr(profile, "summary_text", None) or self._profile_summary("runtime_profile", profile.to_dict())})
        for profile in self.store.list_batch_probe_profiles():
            if model_name and model_name not in profile.model_key:
                continue
            if not self._record_matches_hardware_filter(hardware, device_type=profile.device_type):
                continue
            if not self._matches_toolkit(toolkit):
                continue
            results.append({"kind": "batch_probe_profile", "data": profile.to_dict(), "summary_text": profile.metadata.get("summary_text") or self._profile_summary("batch_probe_profile", profile.to_dict())})
        for profile in self.store.list_solo_profiles():
            if model_name and model_name not in str(self._model_key_for_signature(profile.signature) or profile.signature):
                continue
            if not self._record_matches_hardware_filter(hardware, hardware_key=profile.hardware_key):
                continue
            results.append({"kind": "solo_profile", "data": profile.to_dict(), "summary_text": profile.metadata.get("summary_text") or self._profile_summary("solo_profile", profile.to_dict())})
        return results[: max(1, int(limit))]

    def get_runtime_estimate(
        self,
        *,
        model_or_signature: str,
        batch_size: int,
        hardware: str | None = None,
        backend: str = "exclusive",
    ) -> dict[str, Any]:
        signature = model_or_signature
        profiles = self.store.list_runtime_profiles(signature=signature, backend_name=backend)
        if not profiles:
            profiles = [profile for profile in self.store.list_runtime_profiles(backend_name=backend) if model_or_signature in profile.signature]
        if hardware:
            profiles = [profile for profile in profiles if self._record_matches_hardware_filter(hardware, hardware_key=profile.hardware_key)]
        if not profiles:
            return {"found": False, "reason": "insufficient evidence"}
        exact = [profile for profile in profiles if int(profile.resolved_batch_size) == int(batch_size)]
        chosen = exact[0] if exact else profiles[0]
        return {"found": True, "profile": chosen.to_dict(), "matched_exact_batch_size": bool(exact)}

    def recommend_batch_size(
        self,
        *,
        model_or_signature: str,
        hardware: str | None = None,
        toolkit: str | None = None,
        shape_signature: str | None = None,
        current_batch_size: int | None = None,
    ) -> dict[str, Any]:
        candidates = []
        for profile in self.store.list_batch_probe_profiles():
            if model_or_signature not in {profile.model_key, shape_signature and profile.shape_signature or ""} and model_or_signature not in profile.model_key:
                continue
            if shape_signature and profile.shape_signature != shape_signature:
                continue
            if not self._record_matches_hardware_filter(hardware, device_type=profile.device_type):
                continue
            if not self._matches_toolkit(toolkit):
                continue
            candidates.append(profile)
        if not candidates:
            observations = [
                obs
                for obs in self.store.list_batch_size_observations()
                if model_or_signature in obs.model_key and self._record_matches_hardware_filter(hardware, hardware_key=obs.hardware_key)
            ]
            if not observations:
                return {"found": False, "reason": "insufficient evidence"}
            observations.sort(key=lambda item: (item.batch_size, item.observations), reverse=True)
            best = observations[0]
            return {
                "found": True,
                "recommended_batch_size": best.batch_size,
                "source": "batch_size_observation",
                "evidence": best.to_dict(),
                "current_batch_size": current_batch_size,
            }
        candidates.sort(key=lambda item: (item.resolved_batch_size, item.observations), reverse=True)
        best = candidates[0]
        return {
            "found": True,
            "recommended_batch_size": best.resolved_batch_size,
            "source": "batch_probe_profile",
            "evidence": best.to_dict(),
            "current_batch_size": current_batch_size,
        }

    def recommend_epochs(
        self,
        *,
        model_or_signature: str,
        hardware: str | None = None,
        toolkit: str | None = None,
        current_epochs: int | None = None,
    ) -> dict[str, Any]:
        if not self._matches_toolkit(toolkit):
            return {"found": False, "reason": "insufficient evidence"}
        jobs = []
        for job in self.store.list_jobs():
            if model_or_signature not in {job.baseline_model_id, job.packing.signature} and model_or_signature not in str(job.batch_probe.model_key or ""):
                continue
            if not self._record_matches_hardware_filter(hardware, hardware_key=self._current_hardware_key()):
                continue
            epochs = job.max_epochs or job.config.max_epochs
            if epochs:
                jobs.append(int(epochs))
        if not jobs:
            return {"found": False, "reason": "insufficient evidence", "current_epochs": current_epochs}
        recommendation = int(median(jobs))
        return {
            "found": True,
            "recommended_epochs": recommendation,
            "source": "historical_jobs",
            "evidence_count": len(jobs),
            "current_epochs": current_epochs,
        }

    def get_packet_compatibility(
        self,
        *,
        model_a: str,
        model_b: str,
        hardware: str | None = None,
        backend: str = "exclusive",
    ) -> dict[str, Any]:
        profiles = self.store.list_pair_profiles(backend_name=backend) if hasattr(self.store, "list_pair_profiles") else []
        for profile in profiles:
            if not self._record_matches_hardware_filter(hardware, hardware_key=profile.hardware_key):
                continue
            model_left = self._model_key_for_signature(profile.left_signature)
            model_right = self._model_key_for_signature(profile.right_signature)
            values = {str(model_left or profile.left_signature), str(model_right or profile.right_signature)}
            if {model_a, model_b} == values:
                return {"found": True, "compatible": profile.compatible, "profile": profile.to_dict()}
        return {"found": False, "reason": "insufficient evidence"}

    def search_profile_summaries(self, *, query: str, limit: int = 20) -> list[dict[str, Any]]:
        normalized = query.strip().lower()
        rows: list[dict[str, Any]] = []
        for entry in self.search_profiles(limit=max(20, limit * 2)):
            summary_text = str(entry.get("summary_text") or entry["data"].get("summary_text") or "")
            if normalized and normalized not in summary_text.lower():
                continue
            rows.append(entry)
        for profile in self.list_run_profiles():
            summary_text = profile.summary_text or self._profile_summary("run_profile", profile.to_dict())
            if normalized and normalized not in summary_text.lower():
                continue
            rows.append({"kind": "run_profile", "data": profile.to_dict(), "summary_text": summary_text})
        return rows[: max(1, int(limit))]

    def get_job_design_context(self, *, candidate: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        # Aggregate all scheduling-relevant evidence into one response instead
        # of making future agent integrations orchestrate several low-level MCP
        # calls and duplicate ranking logic on their side.
        normalized_candidate = self._normalized_job_design_candidate(candidate)
        matched_profiles = self._matched_profiles_for_candidate(normalized_candidate, limit=max(1, int(limit)))
        runtime_estimate = self._runtime_estimate_for_candidate(normalized_candidate)
        recommendation_key = str(
            normalized_candidate.get("model_key")
            or normalized_candidate.get("script_signature")
            or normalized_candidate.get("packing_family")
            or ""
        )
        if recommendation_key:
            batch_size_recommendation = self.recommend_batch_size(
                model_or_signature=recommendation_key,
                hardware="current",
                current_batch_size=normalized_candidate.get("proposed_batch_size"),
            )
            epoch_recommendation = self.recommend_epochs(
                model_or_signature=recommendation_key,
                hardware="current",
                current_epochs=normalized_candidate.get("proposed_epochs"),
            )
        else:
            batch_size_recommendation = {"found": False, "reason": "insufficient evidence"}
            epoch_recommendation = {"found": False, "reason": "insufficient evidence"}
        scheduler_guidance = self._scheduler_guidance()
        risk_flags = self._job_design_risk_flags(
            normalized_candidate,
            matched_profiles=matched_profiles,
            runtime_estimate=runtime_estimate,
            batch_size_recommendation=batch_size_recommendation,
            epoch_recommendation=epoch_recommendation,
            scheduler_guidance=scheduler_guidance,
        )
        confidence = self._job_design_confidence(
            matched_profiles=matched_profiles,
            runtime_estimate=runtime_estimate,
            batch_size_recommendation=batch_size_recommendation,
            epoch_recommendation=epoch_recommendation,
            risk_flags=risk_flags,
        )
        return {
            "hardware_context": self.get_hardware_context("current", include_scheduler_limits=True),
            "matched_profiles": matched_profiles,
            "runtime_estimate": runtime_estimate,
            "batch_size_recommendation": batch_size_recommendation,
            "epoch_recommendation": epoch_recommendation,
            "scheduler_guidance": scheduler_guidance,
            "risk_flags": risk_flags,
            "confidence": confidence,
            "evidence_refs": [entry["ref"] for entry in matched_profiles],
        }

    def record_tuning_outcome(
        self,
        *,
        job_id: str,
        chosen_batch_size: int | None,
        chosen_epochs: int | None,
        recommendation_source: str,
        outcome_metrics: dict[str, Any],
        notes: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "chosen_batch_size": chosen_batch_size,
            "chosen_epochs": chosen_epochs,
            "recommendation_source": recommendation_source,
            "outcome_metrics": outcome_metrics,
            "notes": notes,
        }
        self.store.log_event("tuning_outcome_recorded", job_id=job_id, payload=payload)
        return {"ok": True, "job_id": job_id, "payload": payload}
