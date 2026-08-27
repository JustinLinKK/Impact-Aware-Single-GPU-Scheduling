"""Neo4j-backed persistent state store."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any
import json
import os
import sqlite3

from ..config import SchedulerSettings
from ..domain import (
    BatchProbeProfile,
    BatchSizeObservation,
    CombinationProfile,
    JobStatus,
    PairProfile,
    RuntimeProfile,
    SoloProfile,
    TrainingJob,
    utc_now,
)
from ..hardware import HardwareProfile, detect_hardware_profile

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


def _json_dumps(payload: dict[str, Any] | list[Any] | None) -> str:
    return json.dumps(payload or {}, sort_keys=True)


def _toolkit_identity_from_hardware(profile: HardwareProfile) -> tuple[str, str]:
    if profile.cuda_runtime:
        return "cuda", str(profile.cuda_runtime)
    return "unknown", "unknown"


class Neo4jStateStore:
    """Mirror empirical scheduler evidence into the canonical graph schema.

    SQLite is the live scheduler state store. This class exposes graph evidence
    dimensions and write methods only; it is not a scheduler control plane.
    """

    def __init__(self, settings: SchedulerSettings):
        if (
            GraphDatabase is None
        ):  # pragma: no cover - exercised only when dependency missing
            raise RuntimeError("neo4j python driver is not installed")
        self.settings = settings
        self._hardware_profile: HardwareProfile | None = None
        self.settings.ensure_runtime_layout()
        password = os.getenv(self.settings.graph_db.password_env, "")
        auth = (
            (self.settings.graph_db.username, password)
            if self.settings.graph_db.username
            else None
        )
        self._driver = GraphDatabase.driver(self.settings.graph_db.uri, auth=auth)
        self.initialize()

    def _session(self):
        return self._driver.session(database=self.settings.graph_db.database)

    def _run(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        with self._session() as session:
            result = session.run(cypher, params or {})
            rows = [record.data() for record in result]
            result.consume()
            return rows

    def _run_write(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._run(cypher, params)

    def initialize(self) -> None:
        self._driver.verify_connectivity()
        if self.settings.graph_db.bootstrap_constraints:
            self._apply_constraints()
        self._ensure_hardware_dimensions(self.hardware_profile())
        if self.settings.graph_db.import_sqlite_evidence:
            self._import_sqlite_evidence_if_needed()

    def _apply_constraints(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "neo4j_constraints.cypher"
        )
        if not schema_path.exists():
            return
        statements = [
            statement.strip()
            for statement in schema_path.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        for statement in statements:
            self._run_write(statement)

    def _graph_has_jobs(self) -> bool:
        rows = self._run("MATCH (j:Job) RETURN count(j) AS value")
        return bool(rows and int(rows[0]["value"]) > 0)

    def _import_sqlite_evidence_if_needed(self) -> None:
        sqlite_path = Path(
            self.settings.graph_db.sqlite_evidence_path or self.settings.db_path
        )
        if self._graph_has_jobs() or not sqlite_path.exists():
            return
        with sqlite3.connect(sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            for row in connection.execute(
                "SELECT payload_json FROM jobs ORDER BY queue_sequence ASC"
            ).fetchall():
                self.record_scheduler_job_evidence(
                    TrainingJob.from_dict(
                        json.loads(row["payload_json"]), historical_read=True
                    )
                )
            for row in connection.execute("SELECT * FROM solo_profiles").fetchall():
                self.record_solo_profile_evidence(SoloProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM pair_profiles").fetchall():
                self.record_pair_profile_evidence(PairProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM runtime_profiles").fetchall():
                self.record_runtime_profile_evidence(RuntimeProfile.from_row(dict(row)))
            for row in connection.execute(
                "SELECT * FROM batch_probe_profiles"
            ).fetchall():
                self.record_batch_probe_evidence(BatchProbeProfile.from_row(dict(row)))
            for row in connection.execute(
                "SELECT * FROM batch_size_observations"
            ).fetchall():
                self.record_batch_size_observation_evidence(
                    BatchSizeObservation.from_row(dict(row))
                )
            for row in connection.execute(
                "SELECT * FROM combination_profiles"
            ).fetchall():
                self.record_combination_profile_evidence(
                    CombinationProfile.from_row(dict(row))
                )

    def hardware_profile(self) -> HardwareProfile:
        if self._hardware_profile is None:
            self._hardware_profile = detect_hardware_profile(
                device_index=self.settings.gpu_scheduler.device_index
            )
        return self._hardware_profile

    def hardware_key(self) -> str:
        return self.hardware_profile().hardware_key

    def _ensure_hardware_dimensions(self, profile: HardwareProfile) -> None:
        toolkit_name, toolkit_version = _toolkit_identity_from_hardware(profile)
        self._run_write(
            """
            MERGE (h:Hardware {hardware_key: $hardware_key})
            SET h.hardware_key = $hardware_key,
                h.hardware_kind = 'gpu',
                h.vendor = $vendor,
                h.product_name = $gpu_name,
                h.total_vram_mb = $total_vram_mb,
                h.compute_capability = $compute_capability,
                h.toolkit_name = $toolkit_name,
                h.toolkit_version = $toolkit_version,
                h.torch_version = $torch_version,
                h.device_index = $device_index,
                h.updated_at = $updated_at
            """,
            {
                "toolkit_name": toolkit_name,
                "toolkit_version": toolkit_version,
                "torch_version": profile.torch_version,
                "gpu_name": profile.gpu_name,
                "compute_capability": profile.compute_capability,
                "total_vram_mb": profile.total_vram_mb,
                "hardware_key": profile.hardware_key,
                "vendor": "nvidia" if profile.compute_capability else "unknown",
                "device_index": self.settings.gpu_scheduler.device_index,
                "updated_at": utc_now(),
            },
        )

    def _build_hardware_record(self, row: dict[str, Any]) -> dict[str, Any]:
        accelerator = None
        if row.get("accelerator_key") or row.get("accelerator_name"):
            accelerator = {
                "accelerator_key": row.get("accelerator_key"),
                "accelerator_name": row.get("accelerator_name"),
                "compute_capability": row.get("accelerator_compute_capability"),
                "total_vram_mb": row.get("accelerator_total_vram_mb"),
            }
        toolkit = None
        if row.get("toolkit_key") or row.get("toolkit_name_node"):
            toolkit = {
                "toolkit_key": row.get("toolkit_key"),
                "toolkit_name": row.get("toolkit_name_node"),
                "toolkit_version": row.get("toolkit_version_node"),
                "torch_version": row.get("toolkit_torch_version"),
            }
        return {
            "hardware_key": row.get("hardware_key"),
            "gpu_name": row.get("gpu_name"),
            "total_vram_mb": row.get("total_vram_mb"),
            "compute_capability": row.get("compute_capability"),
            "toolkit_name": row.get("toolkit_name"),
            "toolkit_version": row.get("toolkit_version"),
            "torch_version": row.get("torch_version"),
            "summary_text": row.get("summary_text"),
            "hardware": {
                "hardware_key": row.get("hardware_key"),
                "os_name": row.get("os_name"),
                "host_name": row.get("host_name"),
                "gpu_name": row.get("gpu_name"),
                "cpu_name": row.get("cpu_name"),
                "total_ram_mb": row.get("total_ram_mb"),
                "total_vram_mb": row.get("total_vram_mb"),
                "compute_capability": row.get("compute_capability"),
                "toolkit_name": row.get("toolkit_name"),
                "toolkit_version": row.get("toolkit_version"),
                "torch_version": row.get("torch_version"),
                "device_index": row.get("device_index"),
                "summary_text": row.get("summary_text"),
            },
            "accelerator": accelerator,
            "toolkit": toolkit,
            "source": "graph_hardware_node",
        }

    def get_hardware_record(self, hardware_key: str) -> dict[str, Any] | None:
        rows = self._run(
            """
            MATCH (h:Hardware {hardware_key: $hardware_key})
            OPTIONAL MATCH (h)-[:HAS_ACCELERATOR]->(a:Accelerator)
            OPTIONAL MATCH (h)-[:RUNS_TOOLKIT]->(t:Toolkit)
            RETURN
                h.hardware_key AS hardware_key,
                h.os_name AS os_name,
                h.host_name AS host_name,
                coalesce(h.gpu_name, h.product_name) AS gpu_name,
                h.cpu_name AS cpu_name,
                h.total_ram_mb AS total_ram_mb,
                h.total_vram_mb AS total_vram_mb,
                h.compute_capability AS compute_capability,
                h.toolkit_name AS toolkit_name,
                h.toolkit_version AS toolkit_version,
                h.torch_version AS torch_version,
                h.device_index AS device_index,
                h.summary_text AS summary_text,
                a.accelerator_key AS accelerator_key,
                a.accelerator_name AS accelerator_name,
                a.compute_capability AS accelerator_compute_capability,
                a.total_vram_mb AS accelerator_total_vram_mb,
                t.toolkit_key AS toolkit_key,
                t.toolkit_name AS toolkit_name_node,
                t.toolkit_version AS toolkit_version_node,
                t.torch_version AS toolkit_torch_version
            LIMIT 1
            """,
            {"hardware_key": hardware_key},
        )
        if not rows:
            return None
        return self._build_hardware_record(rows[0])

    def list_hardware_records(self) -> list[dict[str, Any]]:
        rows = self._run("""
            MATCH (h:Hardware)
            OPTIONAL MATCH (h)-[:HAS_ACCELERATOR]->(a:Accelerator)
            OPTIONAL MATCH (h)-[:RUNS_TOOLKIT]->(t:Toolkit)
            RETURN
                h.hardware_key AS hardware_key,
                h.os_name AS os_name,
                h.host_name AS host_name,
                coalesce(h.gpu_name, h.product_name) AS gpu_name,
                h.cpu_name AS cpu_name,
                h.total_ram_mb AS total_ram_mb,
                h.total_vram_mb AS total_vram_mb,
                h.compute_capability AS compute_capability,
                h.toolkit_name AS toolkit_name,
                h.toolkit_version AS toolkit_version,
                h.torch_version AS torch_version,
                h.device_index AS device_index,
                h.summary_text AS summary_text,
                a.accelerator_key AS accelerator_key,
                a.accelerator_name AS accelerator_name,
                a.compute_capability AS accelerator_compute_capability,
                a.total_vram_mb AS accelerator_total_vram_mb,
                t.toolkit_key AS toolkit_key,
                t.toolkit_name AS toolkit_name_node,
                t.toolkit_version AS toolkit_version_node,
                t.torch_version AS toolkit_torch_version
            ORDER BY h.updated_at DESC, h.hardware_key ASC
            """)
        return [self._build_hardware_record(row) for row in rows]

    def _status_to_evidence_status(self, status: str | None) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"completed", "succeeded", "success", "profiled"}:
            return "succeeded"
        if normalized in {"cancelled", "canceled", "killed"}:
            return "killed"
        if "oom" in normalized:
            return "oom"
        if "timeout" in normalized:
            return "timeout"
        if normalized in {"failed", "error"}:
            return "failed"
        return "partial"

    def _canonical_json_key(self, prefix: str, payload: dict[str, Any]) -> str:
        digest = sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        return f"{prefix}:{digest}"

    def _hardware_key_for_device_type(self, device_type: str | None = None) -> str:
        profile = self.hardware_profile()
        if not device_type or str(device_type) == profile.gpu_name:
            return profile.hardware_key
        return self._canonical_json_key("hardware", {"device_type": device_type})

    def _merge_evidence_dimensions(
        self,
        *,
        model_key: str | None,
        model_name: str | None = None,
        config: dict[str, Any],
        hardware_key: str | None,
        technology_keys: list[str] | None = None,
    ) -> str:
        config_key = self._canonical_json_key("config", config)
        profile = self.hardware_profile()
        technology_keys = [str(key) for key in (technology_keys or []) if str(key)]
        self._run_write(
            """
            MERGE (c:TrainingConfig {config_key: $config_key})
            SET c.config_key = $config_key,
                c.input_signature = $input_signature,
                c.batch_size = $batch_size,
                c.effective_batch_size = $effective_batch_size,
                c.gradient_accumulation_steps = $gradient_accumulation_steps,
                c.epochs = $epochs,
                c.max_steps = $max_steps,
                c.steps_per_epoch = $steps_per_epoch,
                c.precision = $precision,
                c.optimizer = $optimizer,
                c.learning_rate = $learning_rate,
                c.hyperparams_json = $hyperparams_json,
                c.created_at = coalesce(c.created_at, $updated_at)
            MERGE (h:Hardware {hardware_key: $hardware_key})
            SET h.hardware_key = $hardware_key,
                h.hardware_kind = 'gpu',
                h.vendor = coalesce(h.vendor, $vendor),
                h.product_name = coalesce(h.product_name, $product_name),
                h.total_vram_mb = coalesce(h.total_vram_mb, $total_vram_mb),
                h.compute_capability = coalesce(h.compute_capability, $compute_capability),
                h.toolkit_name = coalesce(h.toolkit_name, $toolkit_name),
                h.toolkit_version = coalesce(h.toolkit_version, $toolkit_version),
                h.technology_keys = coalesce(h.technology_keys, $hardware_technology_keys),
                h.updated_at = $updated_at
            """,
            {
                "config_key": config_key,
                "input_signature": config.get("input_signature"),
                "batch_size": config.get("batch_size"),
                "effective_batch_size": config.get("effective_batch_size"),
                "gradient_accumulation_steps": config.get(
                    "gradient_accumulation_steps"
                ),
                "epochs": config.get("epochs"),
                "max_steps": config.get("max_steps"),
                "steps_per_epoch": config.get("steps_per_epoch"),
                "precision": config.get("precision"),
                "optimizer": config.get("optimizer"),
                "learning_rate": config.get("learning_rate"),
                "hyperparams_json": _json_dumps(config.get("hyperparams") or {}),
                "hardware_key": hardware_key or profile.hardware_key,
                "vendor": "nvidia" if profile.compute_capability else "unknown",
                "product_name": profile.gpu_name,
                "total_vram_mb": profile.total_vram_mb,
                "compute_capability": profile.compute_capability,
                "toolkit_name": "cuda" if profile.cuda_runtime else "unknown",
                "toolkit_version": (
                    str(profile.cuda_runtime) if profile.cuda_runtime else "unknown"
                ),
                "hardware_technology_keys": ["cuda"] if profile.cuda_runtime else [],
                "updated_at": utc_now(),
            },
        )
        if model_key:
            self._run_write(
                """
                MERGE (m:Model {model_key: $model_key})
                SET m.model_key = $model_key,
                    m.model_name = coalesce(m.model_name, $model_name),
                    m.model_family = coalesce(m.model_family, $model_family),
                    m.task_type = coalesce(m.task_type, $task_type),
                    m.updated_at = $updated_at
                """,
                {
                    "model_key": model_key,
                    "model_name": model_name or model_key,
                    "model_family": config.get("model_family"),
                    "task_type": config.get("task_type"),
                    "updated_at": utc_now(),
                },
            )
        for technology_key in technology_keys:
            self._run_write(
                """
                MERGE (t:Technology {technology_key: $technology_key})
                SET t.technology_key = $technology_key,
                    t.name = coalesce(t.name, $technology_key),
                    t.updated_at = $updated_at
                """,
                {"technology_key": technology_key, "updated_at": utc_now()},
            )
        return config_key

    def _upsert_single_job_evidence(
        self,
        *,
        job_id: str,
        purpose: str,
        status: str,
        model_key: str | None,
        model_name: str | None = None,
        hardware_key: str | None = None,
        technology_keys: list[str] | None = None,
        config: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}
        technology_keys = [str(key) for key in (technology_keys or []) if str(key)]
        hardware_key = hardware_key or self.hardware_key()
        config_key = self._merge_evidence_dimensions(
            model_key=model_key,
            model_name=model_name,
            config=config,
            hardware_key=hardware_key,
            technology_keys=technology_keys,
        )
        base_props = {
            "job_id": job_id,
            "profile_key": self._canonical_json_key(
                "profile",
                {
                    "purpose": purpose,
                    "model_key": model_key,
                    "hardware_key": hardware_key,
                    "technology_keys": technology_keys,
                    "config_key": config_key,
                },
            ),
            "purpose": purpose,
            "status": status,
            "hardware_set_key": hardware_key,
            "technology_keys": technology_keys,
            "technology_set_key": (
                self._canonical_json_key("tech", {"technology_keys": technology_keys})
                if technology_keys
                else None
            ),
            "run_scope": config.get("run_scope") or "fixed_steps",
            "confidence": props.get("confidence") if props else None,
            "model_key": model_key,
            "config_key": config_key,
            "created_at": props.get("created_at") if props else None,
            "started_at": props.get("started_at") if props else None,
            "finished_at": props.get("finished_at") if props else None,
        }
        if props:
            base_props.update(
                {key: value for key, value in props.items() if value is not None}
            )
        self._run_write(
            "MERGE (j:Job:SingleJob {job_id: $job_id}) SET j += $props",
            {"job_id": job_id, "props": base_props},
        )
        self._run_write(
            """
            MATCH (j:Job:SingleJob {job_id: $job_id})
            MATCH (h:Hardware {hardware_key: $hardware_key})
            MATCH (c:TrainingConfig {config_key: $config_key})
            MERGE (j)-[:JOB_USED_HARDWARE]->(h)
            MERGE (j)-[:SINGLE_USES_CONFIG]->(c)
            """,
            {"job_id": job_id, "hardware_key": hardware_key, "config_key": config_key},
        )
        if model_key:
            self._run_write(
                """
                MATCH (j:Job:SingleJob {job_id: $job_id})
                MATCH (m:Model {model_key: $model_key})
                MERGE (j)-[:SINGLE_TRAINS_MODEL]->(m)
                """,
                {"job_id": job_id, "model_key": model_key},
            )
        for technology_key in technology_keys:
            self._run_write(
                """
                MATCH (j:Job:SingleJob {job_id: $job_id})
                MATCH (t:Technology {technology_key: $technology_key})
                MERGE (j)-[:JOB_USES_TECHNOLOGY]->(t)
                """,
                {"job_id": job_id, "technology_key": technology_key},
            )

    def record_scheduler_job_evidence(self, job: TrainingJob) -> None:
        if not job.status.is_terminal:
            return
        model_key = str(job.batch_probe.model_key or job.baseline_model_id)
        resolved_batch_size = job.metadata.get("resolved_batch_size")
        technology_keys = list(job.metadata.get("technology_keys") or [])
        if job.metadata.get("uses_amp") or job.metadata.get("amp_enabled"):
            technology_keys.append("pytorch_amp")
        config = {
            "input_signature": job.packing.signature,
            "batch_size": resolved_batch_size
            or job.config.runner_kwargs.get(job.batch_probe.batch_param_name),
            "epochs": job.max_epochs or job.config.max_epochs,
            "max_steps": job.max_steps or job.config.max_steps,
            "run_scope": (
                "full_training" if job.status == JobStatus.COMPLETED else "fixed_steps"
            ),
            "model_family": job.packing.family,
            "task_type": job.task_type,
            "hyperparams": job.config.runner_kwargs,
        }
        metrics = job.metadata.get("outcome_metrics") or {}
        self._upsert_single_job_evidence(
            job_id=f"scheduler_job::{job.job_id}",
            purpose="real_training",
            status=self._status_to_evidence_status(job.status.value),
            model_key=model_key,
            model_name=job.baseline_model_id,
            technology_keys=technology_keys,
            config=config,
            props={
                "resolved_batch_size": resolved_batch_size,
                "completed_full_training": job.status == JobStatus.COMPLETED,
                "observed_steps": job.max_steps,
                "observed_epochs": job.max_epochs,
                "primary_metric_name": metrics.get("primary_metric_name"),
                "primary_metric_value": metrics.get("primary_metric_value"),
                "metrics_json": _json_dumps(metrics),
                "error_message": (
                    job.status_reason if job.status != JobStatus.COMPLETED else None
                ),
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            },
        )

    def record_batch_probe_evidence(self, profile: BatchProbeProfile) -> None:
        self._upsert_single_job_evidence(
            job_id=f"batch_probe::{profile.probe_key}",
            purpose="batch_size_probe",
            status="succeeded",
            model_key=profile.model_key,
            hardware_key=self._hardware_key_for_device_type(profile.device_type),
            technology_keys=(
                ["power_of_two_batch_optimizer"]
                if profile.metadata.get("search_mode") == "power_of_two"
                else []
            ),
            config={
                "input_signature": profile.shape_signature,
                "batch_size": profile.resolved_batch_size,
                "run_scope": "fixed_steps",
                "hyperparams": {"batch_param_name": profile.batch_param_name},
            },
            props={
                "resolved_batch_size": profile.resolved_batch_size,
                "max_safe_batch_size": profile.resolved_batch_size,
                "peak_vram_mb": profile.peak_vram_mb,
                "avg_vram_mb": profile.avg_vram_mb,
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.observations or 0)),
                "measurement_window_steps": profile.metadata.get(
                    "measurement_window_steps"
                ),
                "finished_at": profile.updated_at,
                "metrics_json": _json_dumps(profile.metadata),
            },
        )

    def record_batch_size_observation_evidence(
        self, observation: BatchSizeObservation
    ) -> None:
        self._upsert_single_job_evidence(
            job_id=f"batch_observation::{observation.observation_key}",
            purpose="batch_size_probe",
            status="succeeded",
            model_key=observation.model_key,
            hardware_key=observation.hardware_key,
            config={
                "input_signature": observation.shape_signature,
                "batch_size": observation.batch_size,
                "run_scope": "fixed_steps",
                "hyperparams": {"batch_param_name": observation.batch_param_name},
            },
            props={
                "resolved_batch_size": observation.batch_size,
                "effective_batch_size": observation.effective_batch_size,
                "max_safe_batch_size": observation.batch_size,
                "peak_vram_mb": observation.peak_vram_mb,
                "avg_vram_mb": observation.avg_vram_mb,
                "observed_avg_step_time_ms": observation.avg_step_time_ms,
                "avg_gpu_utilization_pct": observation.avg_gpu_utilization,
                "avg_vram_utilization_pct": observation.avg_memory_utilization,
                "best_metric": observation.best_metric,
                "best_epoch": observation.best_epoch,
                "planned_epochs": observation.planned_epochs,
                "completed_epochs": observation.completed_epochs,
                "seed_variance": observation.seed_variance,
                "confidence": min(
                    1.0, 0.5 + 0.1 * float(observation.observations or 0)
                ),
                "finished_at": observation.updated_at,
                "metrics_json": _json_dumps(observation.metadata),
            },
        )

    def record_runtime_profile_evidence(self, profile: RuntimeProfile) -> None:
        model_key = self._model_key_for_signature(profile.signature)
        self._upsert_single_job_evidence(
            job_id=f"runtime_profile::{profile.profile_key}",
            purpose="runtime_probe",
            status="succeeded",
            model_key=model_key or profile.signature,
            hardware_key=profile.hardware_key,
            config={
                "input_signature": profile.signature,
                "batch_size": profile.resolved_batch_size,
                "steps_per_epoch": profile.steps_per_epoch,
                "run_scope": "full_epoch" if profile.epoch_1_seconds else "fixed_steps",
                "hyperparams": {
                    "strategy": profile.strategy,
                    "backend_name": profile.backend_name,
                },
            },
            props={
                "resolved_batch_size": profile.resolved_batch_size,
                "startup_seconds": profile.startup_seconds,
                "observed_avg_step_time_ms": profile.avg_step_time_ms,
                "estimated_epoch_seconds": profile.epoch_1_seconds,
                "estimated_total_training_seconds": profile.estimated_total_runtime_seconds,
                "estimation_method": (
                    "step_time_extrapolation"
                    if profile.avg_step_time_ms
                    else "partial_epoch_extrapolation"
                ),
                "estimate_confidence": profile.confidence,
                "confidence": profile.confidence,
                "finished_at": profile.updated_at,
                "metrics_json": _json_dumps(profile.metadata),
            },
        )

    def record_solo_profile_evidence(self, profile: SoloProfile) -> None:
        model_key = self._model_key_for_signature(profile.signature)
        self._upsert_single_job_evidence(
            job_id=f"solo_profile::{profile.hardware_key or self.hardware_key()}::{profile.signature}",
            purpose="baseline_benchmark",
            status="succeeded",
            model_key=model_key or profile.signature,
            hardware_key=profile.hardware_key or self.hardware_key(),
            config={
                "input_signature": profile.signature,
                "run_scope": "fixed_steps",
                "model_family": profile.family,
            },
            props={
                "peak_vram_mb": profile.peak_vram_mb,
                "avg_vram_mb": profile.avg_vram_mb,
                "avg_gpu_utilization_pct": profile.avg_gpu_utilization,
                "avg_vram_utilization_pct": profile.avg_memory_utilization,
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.sample_count or 0)),
                "finished_at": profile.updated_at,
                "metrics_json": _json_dumps(profile.metadata),
            },
        )

    def _upsert_packed_job_evidence(
        self,
        *,
        job_id: str,
        packing_group_key: str,
        hardware_key: str,
        backend_name: str,
        compatible: bool,
        member_signatures: list[str],
        props: dict[str, Any],
    ) -> None:
        technology_keys = [backend_name] if backend_name else []
        config_key = self._merge_evidence_dimensions(
            model_key=None,
            config={
                "input_signature": packing_group_key,
                "run_scope": "fixed_steps",
                "hyperparams": {"backend_name": backend_name},
            },
            hardware_key=hardware_key,
            technology_keys=technology_keys,
        )
        packed_props = {
            "job_id": job_id,
            "profile_key": self._canonical_json_key(
                "packed_profile",
                {
                    "packing_group_key": packing_group_key,
                    "hardware_key": hardware_key,
                    "backend_name": backend_name,
                },
            ),
            "purpose": "packed_benchmark",
            "status": "succeeded" if compatible else "failed",
            "hardware_set_key": hardware_key,
            "technology_keys": technology_keys,
            "technology_set_key": (
                self._canonical_json_key("tech", {"technology_keys": technology_keys})
                if technology_keys
                else None
            ),
            "run_scope": "fixed_steps",
            "packing_group_key": packing_group_key,
            "packing_strategy": backend_name or "scheduler_packing",
            "compatible": bool(compatible),
            "config_key": config_key,
        }
        packed_props.update(
            {key: value for key, value in props.items() if value is not None}
        )
        self._run_write(
            "MERGE (p:Job:PackedJob {job_id: $job_id}) SET p += $props",
            {"job_id": job_id, "props": packed_props},
        )
        self._run_write(
            """
            MATCH (p:Job:PackedJob {job_id: $job_id})
            MATCH (h:Hardware {hardware_key: $hardware_key})
            MERGE (p)-[:JOB_USED_HARDWARE]->(h)
            """,
            {"job_id": job_id, "hardware_key": hardware_key},
        )
        for index, signature in enumerate(member_signatures):
            model_key = self._model_key_for_signature(signature) or signature
            member_config_key = self._merge_evidence_dimensions(
                model_key=model_key,
                config={"input_signature": signature, "run_scope": "fixed_steps"},
                hardware_key=hardware_key,
                technology_keys=technology_keys,
            )
            member_id = f"{job_id}::member::{index}"
            self._run_write(
                """
                MERGE (member:PackedJobMember {member_id: $member_id})
                SET member.member_id = $member_id,
                    member.model_key = $model_key,
                    member.config_key = $config_key,
                    member.status = $status,
                    member.metrics_json = $metrics_json
                WITH member
                MATCH (p:Job:PackedJob {job_id: $job_id})
                MATCH (m:Model {model_key: $model_key})
                MATCH (c:TrainingConfig {config_key: $config_key})
                MERGE (p)-[:HAS_PACKED_MEMBER {position: $position}]->(member)
                MERGE (member)-[:MEMBER_TRAINS_MODEL]->(m)
                MERGE (member)-[:MEMBER_USES_CONFIG]->(c)
                """,
                {
                    "member_id": member_id,
                    "model_key": model_key,
                    "config_key": member_config_key,
                    "status": "succeeded" if compatible else "failed",
                    "metrics_json": _json_dumps({"signature": signature}),
                    "job_id": job_id,
                    "position": index,
                },
            )
            for technology_key in technology_keys:
                self._run_write(
                    """
                    MATCH (member:PackedJobMember {member_id: $member_id})
                    MATCH (t:Technology {technology_key: $technology_key})
                    MERGE (member)-[:MEMBER_USES_TECHNOLOGY]->(t)
                    """,
                    {"member_id": member_id, "technology_key": technology_key},
                )

    def record_pair_profile_evidence(self, profile: PairProfile) -> None:
        hardware_key = profile.hardware_key or self.hardware_key()
        self._upsert_packed_job_evidence(
            job_id=f"pair_profile::{profile.pair_key}::{hardware_key}",
            packing_group_key=profile.pair_key,
            hardware_key=hardware_key,
            backend_name=profile.backend_name,
            compatible=profile.compatible,
            member_signatures=[profile.left_signature, profile.right_signature],
            props={
                "peak_vram_mb": profile.peak_vram_mb,
                "avg_vram_mb": profile.avg_vram_mb,
                "avg_gpu_utilization_pct": profile.avg_gpu_utilization,
                "avg_vram_utilization_pct": profile.avg_memory_utilization,
                "slowdown_ratio": profile.slowdown_ratio,
                "error_message": profile.last_failure_reason,
                "finished_at": profile.updated_at,
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.observations or 0)),
                "metrics_json": _json_dumps(profile.metadata),
            },
        )

    def record_combination_profile_evidence(self, profile: CombinationProfile) -> None:
        member_signatures = list(profile.batch_vector.keys()) or [
            profile.group_signature
        ]
        self._upsert_packed_job_evidence(
            job_id=f"combination_profile::{profile.combination_key}",
            packing_group_key=profile.group_signature,
            hardware_key=profile.hardware_key,
            backend_name=profile.backend_name,
            compatible=profile.compatible,
            member_signatures=member_signatures,
            props={
                "peak_vram_mb": profile.peak_vram_mb,
                "avg_vram_mb": profile.avg_vram_mb,
                "avg_gpu_utilization_pct": profile.avg_gpu_utilization,
                "avg_vram_utilization_pct": profile.avg_memory_utilization,
                "observed_avg_step_time_ms": profile.avg_step_time_ms,
                "throughput_efficiency": profile.objective_score,
                "error_message": profile.last_failure_reason,
                "finished_at": profile.updated_at,
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.observations or 0)),
                "metrics_json": _json_dumps(
                    {"batch_vector": profile.batch_vector, **profile.metadata}
                ),
            },
        )

    def _model_key_for_signature(self, signature: str) -> str | None:
        rows = self._run(
            """
            MATCH (s:WorkloadSignature {signature: $signature})
            RETURN s.model_key AS model_key
            LIMIT 1
            """,
            {"signature": signature},
        )
        return rows[0]["model_key"] if rows and rows[0].get("model_key") else None
    def delete_job(self, job_id: str) -> None:
        """Delete terminal scheduler evidence mirrored for one job."""
        self._run_write(
            "MATCH (j:Job) WHERE j.job_id IN [$job_id, $evidence_job_id] DETACH DELETE j",
            {"job_id": job_id, "evidence_job_id": f"scheduler_job::{job_id}"},
        )
