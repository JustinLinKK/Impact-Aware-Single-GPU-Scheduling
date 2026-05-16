"""Neo4j-backed persistent state store."""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable
import json
import os
import sqlite3

from ..config import SchedulerSettings
from ..domain import (
    BatchProbeProfile,
    BatchSizeObservation,
    CombinationProfile,
    CommandType,
    JobCommand,
    JobStatus,
    PairProfile,
    RunProfile,
    RuntimeProfile,
    SchedulerReport,
    SoloProfile,
    TrainingJob,
    build_backend_scoped_pair_key,
    build_group_signature,
    parse_timestamp,
    utc_now,
)
from ..hardware import HardwareProfile, detect_hardware_profile

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


def _json_dumps(payload: dict[str, Any] | list[Any] | None) -> str:
    return json.dumps(payload or {}, sort_keys=True)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def _toolkit_identity_from_hardware(profile: HardwareProfile) -> tuple[str, str]:
    if profile.cuda_runtime:
        return "cuda", str(profile.cuda_runtime)
    return "unknown", "unknown"


def _accelerator_key(device_name: str) -> str:
    normalized = str(device_name or "unknown").strip().lower()
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _coerce_run_summary(profile: RunProfile) -> str:
    model_name = profile.model_key or "unknown-model"
    hardware = profile.metadata.get("gpu_name") or profile.metadata.get("device_type") or "unknown-hardware"
    batch = profile.resolved_batch_size if profile.resolved_batch_size is not None else "?"
    return (
        f"Model {model_name} ran on {hardware} with batch size {batch}, "
        f"avg SM {profile.avg_sm_utilization or 0.0:.1%}, "
        f"avg memory {profile.avg_memory_utilization or 0.0:.1%}, "
        f"epoch time {profile.epoch_time_seconds or 0.0}s."
    )


class Neo4jStateStore:
    """Persist empirical scheduler evidence as a property graph in Neo4j.

    SQLite remains the live scheduler state store. The older control-plane
    methods in this class are retained only for compatibility with historical
    tests and migrations; StateStore now calls the record_*_evidence methods
    for graph writes.
    """

    def __init__(self, settings: SchedulerSettings):
        if GraphDatabase is None:  # pragma: no cover - exercised only when dependency missing
            raise RuntimeError("neo4j python driver is not installed")
        self.settings = settings
        self._hardware_profile: HardwareProfile | None = None
        self.settings.ensure_runtime_layout()
        password = os.getenv(self.settings.graph_db.password_env, "")
        auth = (self.settings.graph_db.username, password) if self.settings.graph_db.username else None
        self._driver = GraphDatabase.driver(self.settings.graph_db.uri, auth=auth)
        self.initialize()

    def _session(self):
        return self._driver.session(database=self.settings.graph_db.database)

    def _run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._session() as session:
            result = session.run(cypher, params or {})
            rows = [record.data() for record in result]
            result.consume()
            return rows

    def _run_write(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._run(cypher, params)

    def initialize(self) -> None:
        self._driver.verify_connectivity()
        if self.settings.graph_db.bootstrap_constraints:
            self._apply_constraints()
        self._ensure_hardware_dimensions(self.hardware_profile())
        if self.settings.graph_db.auto_import_legacy_sqlite:
            self._import_legacy_sqlite_if_needed()

    def _apply_constraints(self) -> None:
        schema_path = Path(__file__).resolve().parents[2] / "schema" / "neo4j_constraints.cypher"
        if not schema_path.exists():
            return
        statements = [statement.strip() for statement in schema_path.read_text(encoding="utf-8").split(";") if statement.strip()]
        for statement in statements:
            self._run_write(statement)

    def _graph_has_jobs(self) -> bool:
        rows = self._run("MATCH (j:Job) RETURN count(j) AS value")
        return bool(rows and int(rows[0]["value"]) > 0)

    def _import_legacy_sqlite_if_needed(self) -> None:
        legacy_path = Path(self.settings.graph_db.legacy_sqlite_path or self.settings.db_path)
        if self._graph_has_jobs() or not legacy_path.exists():
            return
        with sqlite3.connect(legacy_path) as connection:
            connection.row_factory = sqlite3.Row
            for row in connection.execute("SELECT payload_json FROM jobs ORDER BY queue_sequence ASC").fetchall():
                self.record_scheduler_job_evidence(TrainingJob.from_dict(json.loads(row["payload_json"])))
            for row in connection.execute("SELECT * FROM solo_profiles").fetchall():
                self.record_solo_profile_evidence(SoloProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM pair_profiles").fetchall():
                self.record_pair_profile_evidence(PairProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM runtime_profiles").fetchall():
                self.record_runtime_profile_evidence(RuntimeProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM batch_probe_profiles").fetchall():
                self.record_batch_probe_evidence(BatchProbeProfile.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM batch_size_observations").fetchall():
                self.record_batch_size_observation_evidence(BatchSizeObservation.from_row(dict(row)))
            for row in connection.execute("SELECT * FROM combination_profiles").fetchall():
                self.record_combination_profile_evidence(CombinationProfile.from_row(dict(row)))

    def hardware_profile(self) -> HardwareProfile:
        if self._hardware_profile is None:
            self._hardware_profile = detect_hardware_profile(device_index=self.settings.gpu_scheduler.device_index)
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
        rows = self._run(
            """
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
            """
        )
        return [self._build_hardware_record(row) for row in rows]

    def next_queue_sequence(self) -> int:
        rows = self._run("MATCH (j:Job) RETURN coalesce(max(j.queue_sequence), 0) + 1 AS value")
        return int(rows[0]["value"]) if rows else 1

    def _ensure_model_and_signature(self, job: TrainingJob) -> None:
        model_key = str(job.batch_probe.model_key or job.baseline_model_id)
        params = {
            "model_key": model_key,
            "model_uid": f"model::{model_key}",
            "model_name": job.baseline_model_id or model_key,
            "task_type": job.task_type,
            "runner_target": job.config.runner_target,
            "signature": job.packing.signature,
            "signature_uid": f"signature::{job.packing.signature}" if job.packing.signature else None,
            "family": job.packing.family,
            "default_batch_size": int(job.config.runner_kwargs.get(job.batch_probe.batch_param_name, 0) or 0) or None,
            "default_epochs": job.max_epochs or job.config.max_epochs,
            "default_steps": job.max_steps or job.config.max_steps,
            "updated_at": utc_now(),
        }
        self._run_write(
            """
            MERGE (m:Model {model_key: $model_key})
            SET m.uid = $model_uid,
                m.model_key = $model_key,
                m.model_name = $model_name,
                m.baseline_model_id = $model_name,
                m.task_type = $task_type,
                m.updated_at = $updated_at,
                m.summary_text = 'Model ' + $model_name
            """,
            params,
        )
        if job.packing.signature:
            self._run_write(
                """
                MATCH (m:Model {model_key: $model_key})
                MERGE (s:WorkloadSignature {signature: $signature})
                SET s.uid = $signature_uid,
                    s.signature = $signature,
                    s.family = $family,
                    s.runner_target = $runner_target,
                    s.task_type = $task_type,
                    s.default_batch_size = $default_batch_size,
                    s.default_epochs = $default_epochs,
                    s.default_steps = $default_steps,
                    s.model_key = $model_key,
                    s.updated_at = $updated_at,
                    s.summary_text = 'Signature ' + $signature
                MERGE (s)-[:DESCRIBES_MODEL]->(m)
                """,
                params,
            )

    def save_job(self, job: TrainingJob) -> None:
        now = utc_now()
        self._ensure_model_and_signature(job)
        payload_json = job.to_json()
        props = {
            "uid": f"job::{job.job_id}",
            "job_id": job.job_id,
            "workflow_id": job.workflow_id,
            "agent_id": job.agent_id,
            "baseline_model_id": job.baseline_model_id,
            "baseline_model_path": job.baseline_model_path,
            "task_type": job.task_type,
            "runner_target": job.config.runner_target,
            "loader_target": job.config.loader_target,
            "priority": job.priority,
            "status": job.status.value,
            "status_reason": job.status_reason,
            "queue_sequence": job.queue_sequence,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "latest_checkpoint_path": job.latest_checkpoint_path,
            "hold": job.hold,
            "requires_gpu": job.resource_requirements.requires_gpu,
            "estimated_vram_mb": job.resource_requirements.estimated_vram_mb,
            "estimated_ram_mb": job.resource_requirements.estimated_ram_mb,
            "gpu_slots": job.resource_requirements.gpu_slots,
            "packing_eligible": job.packing.eligible,
            "packing_family": job.packing.family,
            "packing_signature": job.packing.signature,
            "packing_max_slowdown_ratio": job.packing.max_slowdown_ratio,
            "batch_probe_enabled": job.batch_probe.enabled,
            "batch_probe_model_key": job.batch_probe.model_key,
            "batch_probe_search_mode": job.batch_probe.search_mode,
            "runtime_probe_enabled": job.runtime_probe.enabled,
            "runtime_probe_strategy": job.runtime_probe.strategy,
            "max_steps": job.max_steps,
            "max_epochs": job.max_epochs,
            "payload_json": payload_json,
            "metadata_json": _json_dumps(job.metadata),
            "updated_at": now,
            "summary_text": f"Job {job.job_id} for model {job.baseline_model_id} is {job.status.value}.",
        }
        self._run_write("MERGE (j:Job {job_id: $job_id}) SET j += $props", {"job_id": job.job_id, "props": props})
        if job.workflow_id:
            self._run_write(
                """
                MERGE (w:Workflow {workflow_id: $workflow_id})
                SET w.uid = $workflow_uid,
                    w.workflow_id = $workflow_id,
                    w.agent_id = $agent_id,
                    w.last_updated_at = $updated_at
                WITH w
                MATCH (j:Job {job_id: $job_id})
                MERGE (w)-[:HAS_JOB]->(j)
                """,
                {
                    "workflow_id": job.workflow_id,
                    "workflow_uid": f"workflow::{job.workflow_id}",
                    "agent_id": job.agent_id,
                    "updated_at": now,
                    "job_id": job.job_id,
                },
            )
        model_key = str(job.batch_probe.model_key or job.baseline_model_id)
        self._run_write(
            """
            MATCH (j:Job {job_id: $job_id})
            MATCH (m:Model {model_key: $model_key})
            MERGE (j)-[:USES_MODEL]->(m)
            """,
            {"job_id": job.job_id, "model_key": model_key},
        )
        if job.packing.signature:
            self._run_write(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (s:WorkloadSignature {signature: $signature})
                MERGE (j)-[:USES_SIGNATURE]->(s)
                """,
                {"job_id": job.job_id, "signature": job.packing.signature},
            )

    def submit_job(self, job: TrainingJob) -> TrainingJob:
        if not job.queue_sequence:
            job.queue_sequence = self.next_queue_sequence()
        job.mark_status(JobStatus.PENDING, reason=job.status_reason)
        self.save_job(job)
        self.enqueue_command(CommandType.SUBMIT, job_id=job.job_id)
        self.log_event("job_submitted", job_id=job.job_id, payload={"priority": job.priority})
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        rows = self._run("MATCH (j:Job {job_id: $job_id}) RETURN j.payload_json AS payload_json", {"job_id": job_id})
        if not rows:
            return None
        return TrainingJob.from_dict(json.loads(rows[0]["payload_json"]))

    def list_jobs(self, statuses: Iterable[JobStatus | str] | None = None) -> list[TrainingJob]:
        params: dict[str, Any] = {}
        if statuses:
            normalized = [status.value if isinstance(status, JobStatus) else status for status in statuses]
            rows = self._run(
                """
                MATCH (j:Job)
                WHERE j.status IN $statuses
                RETURN j.payload_json AS payload_json
                ORDER BY j.priority DESC, j.queue_sequence ASC
                """,
                {"statuses": normalized},
            )
        else:
            rows = self._run(
                """
                MATCH (j:Job)
                RETURN j.payload_json AS payload_json
                ORDER BY j.priority DESC, j.queue_sequence ASC
                """,
                params,
            )
        return [TrainingJob.from_dict(json.loads(row["payload_json"])) for row in rows]

    def runnable_jobs(self) -> list[TrainingJob]:
        jobs = self.list_jobs(statuses=[JobStatus.PENDING, JobStatus.READY, JobStatus.PAUSED, JobStatus.RECOVERABLE])
        return [job for job in jobs if job.is_runnable()]

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        reason: str | None = None,
        hold: bool | None = None,
        latest_checkpoint_path: str | None = None,
        last_heartbeat_at: str | None = None,
        last_dispatched_at: str | None = None,
        status_timestamps: dict[str, str] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> TrainingJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        if status is not None:
            job.mark_status(status, reason=reason)
        elif reason is not None:
            job.status_reason = reason
        if hold is not None:
            job.hold = hold
        if latest_checkpoint_path is not None:
            job.latest_checkpoint_path = latest_checkpoint_path
        if last_heartbeat_at is not None:
            job.last_heartbeat_at = last_heartbeat_at
        if last_dispatched_at is not None:
            job.last_dispatched_at = last_dispatched_at
        if status_timestamps:
            job.status_timestamps.update(status_timestamps)
        if metadata_updates:
            job.metadata.update(metadata_updates)
        self.save_job(job)
        return job

    def set_job_status(self, job_id: str, status: JobStatus, *, reason: str | None = None, hold: bool | None = None) -> TrainingJob:
        return self.update_job(job_id, status=status, reason=reason, hold=hold)

    def delete_job(self, job_id: str) -> None:
        self._run_write("MATCH (j:Job {job_id: $job_id}) DETACH DELETE j", {"job_id": job_id})

    def _next_sequence(self, name: str) -> int:
        rows = self._run_write(
            """
            MERGE (s:Sequence {name: $name})
            ON CREATE SET s.value = 0
            SET s.value = coalesce(s.value, 0) + 1
            RETURN s.value AS value
            """,
            {"name": name},
        )
        return int(rows[0]["value"])

    def _upsert_command(
        self,
        *,
        command_id: int,
        job_id: str | None,
        command_type: str,
        payload: dict[str, Any],
        created_at: str,
        processed_at: str | None,
    ) -> None:
        props = {
            "uid": f"command::{command_id}",
            "command_id": command_id,
            "job_id": job_id,
            "command_type": command_type,
            "payload_json": _json_dumps(payload),
            "created_at": created_at,
            "processed_at": processed_at,
            "updated_at": utc_now(),
        }
        self._run_write("MERGE (c:Command {command_id: $command_id}) SET c += $props", {"command_id": command_id, "props": props})
        if job_id:
            self._run_write(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (c:Command {command_id: $command_id})
                MERGE (j)-[:EXECUTED_COMMAND]->(c)
                """,
                {"job_id": job_id, "command_id": command_id},
            )

    def enqueue_command(self, command_type: CommandType, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> int:
        command_id = self._next_sequence("command_id")
        self._upsert_command(
            command_id=command_id,
            job_id=job_id,
            command_type=command_type.value,
            payload=payload or {},
            created_at=utc_now(),
            processed_at=None,
        )
        return command_id

    def fetch_pending_commands(self, limit: int = 100) -> list[JobCommand]:
        rows = self._run(
            """
            MATCH (c:Command)
            WHERE c.processed_at IS NULL
            RETURN c.command_id AS command_id, c.job_id AS job_id, c.command_type AS command_type,
                   c.payload_json AS payload_json, c.created_at AS created_at, c.processed_at AS processed_at
            ORDER BY c.command_id ASC
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
        return [JobCommand.from_row(row) for row in rows]

    def mark_command_processed(self, command_id: int) -> None:
        self._run_write(
            "MATCH (c:Command {command_id: $command_id}) SET c.processed_at = $processed_at, c.updated_at = $processed_at",
            {"command_id": int(command_id), "processed_at": utc_now()},
        )

    def _upsert_event(
        self,
        *,
        event_id: int,
        job_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        props = {
            "uid": f"event::{event_id}",
            "event_id": event_id,
            "job_id": job_id,
            "event_type": event_type,
            "payload_json": _json_dumps(payload),
            "created_at": created_at,
            "updated_at": created_at,
            "summary_text": f"Event {event_type} for job {job_id or 'n/a'}",
        }
        self._run_write("MERGE (e:Event {event_id: $event_id}) SET e += $props", {"event_id": event_id, "props": props})
        if job_id:
            self._run_write(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (e:Event {event_id: $event_id})
                MERGE (j)-[:EMITTED_EVENT]->(e)
                """,
                {"job_id": job_id, "event_id": event_id},
            )
        self._record_run_profile_from_event(event_id=event_id, event_type=event_type, job_id=job_id, payload=payload, created_at=created_at)

    def log_event(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        event_id = self._next_sequence("event_id")
        self._upsert_event(
            event_id=event_id,
            job_id=job_id,
            event_type=event_type,
            payload=payload or {},
            created_at=utc_now(),
        )

    def list_events(self, *, job_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        clauses = ["MATCH (e:Event) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if job_id is not None:
            clauses.append("AND e.job_id = $job_id")
            params["job_id"] = job_id
        if event_type is not None:
            clauses.append("AND e.event_type = $event_type")
            params["event_type"] = event_type
        clauses.append(
            "RETURN e.event_id AS event_id, e.job_id AS job_id, e.event_type AS event_type, "
            "e.payload_json AS payload_json, e.created_at AS created_at ORDER BY e.event_id ASC"
        )
        rows = self._run("\n".join(clauses), params)
        return [
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "payload": _json_loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _upsert_checkpoint(
        self,
        *,
        checkpoint_id: int,
        job_id: str,
        checkpoint_path: str,
        created_at: str,
        metadata: dict[str, Any],
        is_latest: bool,
    ) -> None:
        if is_latest:
            self._run_write("MATCH (c:Checkpoint {job_id: $job_id}) SET c.is_latest = false", {"job_id": job_id})
        props = {
            "uid": f"checkpoint::{checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "job_id": job_id,
            "checkpoint_path": checkpoint_path,
            "created_at": created_at,
            "metadata_json": _json_dumps(metadata),
            "is_latest": bool(is_latest),
            "updated_at": created_at,
        }
        self._run_write("MERGE (c:Checkpoint {checkpoint_id: $checkpoint_id}) SET c += $props", {"checkpoint_id": checkpoint_id, "props": props})
        self._run_write(
            """
            MATCH (j:Job {job_id: $job_id})
            MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
            MERGE (j)-[:PRODUCED_CHECKPOINT]->(c)
            """,
            {"job_id": job_id, "checkpoint_id": checkpoint_id},
        )

    def record_checkpoint(self, job_id: str, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        checkpoint_id = self._next_sequence("checkpoint_id")
        self._upsert_checkpoint(
            checkpoint_id=checkpoint_id,
            job_id=job_id,
            checkpoint_path=checkpoint_path,
            created_at=utc_now(),
            metadata=metadata or {},
            is_latest=True,
        )
        self.update_job(job_id, latest_checkpoint_path=checkpoint_path)

    def latest_checkpoint(self, job_id: str) -> str | None:
        rows = self._run(
            """
            MATCH (c:Checkpoint {job_id: $job_id})
            WHERE c.is_latest = true
            RETURN c.checkpoint_path AS checkpoint_path
            ORDER BY c.checkpoint_id DESC
            LIMIT 1
            """,
            {"job_id": job_id},
        )
        if rows:
            return str(rows[0]["checkpoint_path"])
        job = self.get_job(job_id)
        return job.latest_checkpoint_path if job else None

    def update_cache_metadata(
        self,
        model_id: str,
        baseline_model_path: str,
        *,
        size_bytes: int,
        pinned: bool,
        hits: int,
        misses: int,
        last_loaded_at: str | None,
        last_accessed_at: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        props = {
            "uid": f"cache::{model_id}",
            "cache_key": f"cache::{model_id}",
            "model_id": model_id,
            "baseline_model_path": baseline_model_path,
            "size_bytes": int(size_bytes),
            "pinned": bool(pinned),
            "hits": int(hits),
            "misses": int(misses),
            "last_loaded_at": last_loaded_at,
            "last_accessed_at": last_accessed_at,
            "metadata_json": _json_dumps(metadata),
            "updated_at": utc_now(),
        }
        self._run_write("MERGE (c:CacheEntry {cache_key: $cache_key}) SET c += $props", {"cache_key": props["cache_key"], "props": props})
        self._run_write(
            """
            MERGE (m:Model {model_key: $model_id})
            SET m.uid = coalesce(m.uid, $model_uid), m.model_name = coalesce(m.model_name, $model_id)
            WITH m
            MATCH (c:CacheEntry {cache_key: $cache_key})
            MERGE (c)-[:CACHES_MODEL]->(m)
            """,
            {"model_id": model_id, "model_uid": f"model::{model_id}", "cache_key": props["cache_key"]},
        )

    def cache_metadata_summary(self) -> dict[str, Any]:
        rows = self._run(
            """
            MATCH (c:CacheEntry)
            RETURN count(c) AS entries,
                   coalesce(sum(c.size_bytes), 0) AS used_bytes,
                   coalesce(sum(CASE WHEN c.pinned THEN 1 ELSE 0 END), 0) AS pinned_entries,
                   coalesce(sum(c.hits), 0) AS hits,
                   coalesce(sum(c.misses), 0) AS misses
            """
        )
        return dict(rows[0]) if rows else {"entries": 0, "used_bytes": 0, "pinned_entries": 0, "hits": 0, "misses": 0}

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
        digest = sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
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
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
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
                "toolkit_version": str(profile.cuda_runtime) if profile.cuda_runtime else "unknown",
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
            "technology_set_key": self._canonical_json_key("tech", {"technology_keys": technology_keys}) if technology_keys else None,
            "run_scope": config.get("run_scope") or "fixed_steps",
            "confidence": props.get("confidence") if props else None,
            "model_key": model_key,
            "config_key": config_key,
            "created_at": props.get("created_at") if props else None,
            "started_at": props.get("started_at") if props else None,
            "finished_at": props.get("finished_at") if props else None,
        }
        if props:
            base_props.update({key: value for key, value in props.items() if value is not None})
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
            "batch_size": resolved_batch_size or job.config.runner_kwargs.get(job.batch_probe.batch_param_name),
            "epochs": job.max_epochs or job.config.max_epochs,
            "max_steps": job.max_steps or job.config.max_steps,
            "run_scope": "full_training" if job.status == JobStatus.COMPLETED else "fixed_steps",
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
                "error_message": job.status_reason if job.status != JobStatus.COMPLETED else None,
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
            technology_keys=["power_of_two_batch_optimizer"] if profile.metadata.get("search_mode") == "power_of_two" else [],
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
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.observations or 0)),
                "measurement_window_steps": profile.metadata.get("measurement_window_steps"),
                "finished_at": profile.updated_at,
                "metrics_json": _json_dumps(profile.metadata),
            },
        )

    def record_batch_size_observation_evidence(self, observation: BatchSizeObservation) -> None:
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
                "max_safe_batch_size": observation.batch_size,
                "peak_vram_mb": observation.peak_vram_mb,
                "observed_avg_step_time_ms": observation.avg_step_time_ms,
                "avg_gpu_utilization_pct": observation.avg_gpu_utilization,
                "avg_vram_utilization_pct": observation.avg_memory_utilization,
                "confidence": min(1.0, 0.5 + 0.1 * float(observation.observations or 0)),
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
                "hyperparams": {"strategy": profile.strategy, "backend_name": profile.backend_name},
            },
            props={
                "resolved_batch_size": profile.resolved_batch_size,
                "startup_seconds": profile.startup_seconds,
                "observed_avg_step_time_ms": profile.avg_step_time_ms,
                "estimated_epoch_seconds": profile.epoch_1_seconds,
                "estimated_total_training_seconds": profile.estimated_total_runtime_seconds,
                "estimation_method": "step_time_extrapolation" if profile.avg_step_time_ms else "partial_epoch_extrapolation",
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
            config={"input_signature": packing_group_key, "run_scope": "fixed_steps", "hyperparams": {"backend_name": backend_name}},
            hardware_key=hardware_key,
            technology_keys=technology_keys,
        )
        packed_props = {
            "job_id": job_id,
            "profile_key": self._canonical_json_key("packed_profile", {"packing_group_key": packing_group_key, "hardware_key": hardware_key, "backend_name": backend_name}),
            "purpose": "packed_benchmark",
            "status": "succeeded" if compatible else "failed",
            "hardware_set_key": hardware_key,
            "technology_keys": technology_keys,
            "technology_set_key": self._canonical_json_key("tech", {"technology_keys": technology_keys}) if technology_keys else None,
            "run_scope": "fixed_steps",
            "packing_group_key": packing_group_key,
            "packing_strategy": backend_name or "scheduler_packing",
            "compatible": bool(compatible),
            "config_key": config_key,
        }
        packed_props.update({key: value for key, value in props.items() if value is not None})
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
        member_signatures = list(profile.batch_vector.keys()) or [profile.group_signature]
        self._upsert_packed_job_evidence(
            job_id=f"combination_profile::{profile.combination_key}",
            packing_group_key=profile.group_signature,
            hardware_key=profile.hardware_key,
            backend_name=profile.backend_name,
            compatible=profile.compatible,
            member_signatures=member_signatures,
            props={
                "peak_vram_mb": profile.peak_vram_mb,
                "avg_gpu_utilization_pct": profile.avg_gpu_utilization,
                "avg_vram_utilization_pct": profile.avg_memory_utilization,
                "observed_avg_step_time_ms": profile.avg_step_time_ms,
                "throughput_efficiency": profile.objective_score,
                "error_message": profile.last_failure_reason,
                "finished_at": profile.updated_at,
                "confidence": min(1.0, 0.5 + 0.1 * float(profile.observations or 0)),
                "metrics_json": _json_dumps({"batch_vector": profile.batch_vector, **profile.metadata}),
            },
        )

    def _upsert_run_profile(self, profile: RunProfile) -> RunProfile:
        profile.updated_at = utc_now()
        if not profile.summary_text:
            profile.summary_text = _coerce_run_summary(profile)
        props = profile.to_dict()
        props["metadata_json"] = _json_dumps(profile.metadata)
        props["result_payload_json"] = _json_dumps(profile.result_payload)
        props.pop("metadata", None)
        props.pop("result_payload", None)
        self._run_write(
            "MERGE (r:RunProfile {run_profile_id: $run_profile_id}) SET r += $props",
            {"run_profile_id": profile.run_profile_id, "props": props},
        )
        if profile.job_id:
            self._run_write(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (r:RunProfile {run_profile_id: $run_profile_id})
                MERGE (j)-[:HAS_RUN_PROFILE]->(r)
                """,
                {"job_id": profile.job_id, "run_profile_id": profile.run_profile_id},
            )
        if profile.model_key:
            self._run_write(
                """
                MERGE (m:Model {model_key: $model_key})
                SET m.uid = coalesce(m.uid, $model_uid), m.model_name = coalesce(m.model_name, $model_key)
                WITH m
                MATCH (r:RunProfile {run_profile_id: $run_profile_id})
                MERGE (r)-[:FOR_MODEL]->(m)
                """,
                {"model_key": profile.model_key, "model_uid": f"model::{profile.model_key}", "run_profile_id": profile.run_profile_id},
            )
        if profile.signature:
            self._run_write(
                """
                MERGE (s:WorkloadSignature {signature: $signature})
                SET s.uid = coalesce(s.uid, $signature_uid), s.signature = $signature
                WITH s
                MATCH (r:RunProfile {run_profile_id: $run_profile_id})
                MERGE (r)-[:FOR_SIGNATURE]->(s)
                """,
                {"signature": profile.signature, "signature_uid": f"signature::{profile.signature}", "run_profile_id": profile.run_profile_id},
            )
        return profile

    def _record_run_profile_from_event(
        self,
        *,
        event_id: int,
        event_type: str,
        job_id: str | None,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        if event_type not in {
            "job_completed",
            "job_failed",
            "job_paused",
            "job_cancelled",
            "batch_probe_selected",
            "batch_probe_failed",
            "packed_pair_dispatched",
            "packed_group_dispatched",
            "packed_group_fallback",
        }:
            return
        job = self.get_job(job_id) if job_id else None
        if event_type.startswith("batch_probe"):
            run_kind = "batch_probe"
            probe_kind = "batch"
        elif event_type.startswith("packed_"):
            run_kind = "packed_training"
            probe_kind = None
        else:
            run_kind = "training_resume" if job and (job.latest_checkpoint_path or job.resume_from_checkpoint) else "training"
            probe_kind = None
        model_key = None
        signature = None
        if job is not None:
            model_key = str(job.batch_probe.model_key or job.baseline_model_id)
            signature = job.packing.signature
        run_profile = RunProfile(
            run_profile_id=f"run::{job_id or 'global'}::{event_type}::{event_id}",
            run_kind=run_kind,
            probe_kind=probe_kind,
            status=(job.status.value if job else event_type.upper()),
            job_id=job_id,
            backend_name=str((job.metadata.get("placement_backend") if job else None) or payload.get("backend_name") or "exclusive"),
            model_key=model_key,
            signature=signature,
            resolved_batch_size=(job.metadata.get("resolved_batch_size") if job else None) or payload.get("resolved_batch_size"),
            epochs=job.max_epochs if job else None,
            total_steps=job.max_steps if job else None,
            avg_sm_utilization=payload.get("avg_gpu_utilization"),
            avg_gpu_utilization=payload.get("avg_gpu_utilization"),
            avg_memory_utilization=payload.get("avg_memory_utilization"),
            peak_vram_mb=payload.get("peak_vram_mb"),
            target_budget_mb=payload.get("target_budget_mb"),
            started_at=job.started_at if job else None,
            finished_at=job.finished_at if job else created_at,
            updated_at=created_at,
            source="event",
            result_payload=payload,
            metadata={
                "event_type": event_type,
                "gpu_name": self.hardware_profile().gpu_name,
            },
        )
        self._upsert_run_profile(run_profile)

    def list_run_profiles(self, *, job_id: str | None = None, model_key: str | None = None, signature: str | None = None) -> list[RunProfile]:
        clauses = ["MATCH (r:RunProfile) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if job_id is not None:
            clauses.append("AND r.job_id = $job_id")
            params["job_id"] = job_id
        if model_key is not None:
            clauses.append("AND r.model_key = $model_key")
            params["model_key"] = model_key
        if signature is not None:
            clauses.append("AND r.signature = $signature")
            params["signature"] = signature
        clauses.append("RETURN properties(r) AS row ORDER BY r.updated_at DESC")
        rows = self._run("\n".join(clauses), params)
        return [RunProfile.from_row(row["row"]) for row in rows]

    def upsert_solo_profile(self, profile: SoloProfile) -> SoloProfile:
        profile.updated_at = utc_now()
        if not profile.hardware_key:
            profile.hardware_key = self.hardware_key()
        solo_profile_id = f"solo::{profile.hardware_key}::{profile.signature}"
        props = {
            "uid": solo_profile_id,
            "solo_profile_id": solo_profile_id,
            "signature": profile.signature,
            "hardware_key": profile.hardware_key,
            "family": profile.family,
            "peak_vram_mb": profile.peak_vram_mb,
            "avg_sm_utilization": profile.avg_gpu_utilization,
            "avg_gpu_utilization": profile.avg_gpu_utilization,
            "avg_ram_utilization": None,
            "avg_memory_utilization": profile.avg_memory_utilization,
            "sample_count": profile.sample_count,
            "last_job_id": profile.last_job_id,
            "updated_at": profile.updated_at,
            "metadata_json": _json_dumps(profile.metadata),
            "summary_text": (
                f"Solo profile for {profile.signature} on {self.hardware_profile().gpu_name}: "
                f"peak VRAM {profile.peak_vram_mb or 0} MB, avg SM {profile.avg_gpu_utilization or 0.0:.1%}."
            ),
        }
        self._run_write("MERGE (s:SoloProfile {solo_profile_id: $solo_profile_id}) SET s += $props", {"solo_profile_id": solo_profile_id, "props": props})
        self._run_write(
            """
            MERGE (w:WorkloadSignature {signature: $signature})
            SET w.uid = coalesce(w.uid, $signature_uid), w.signature = $signature
            MERGE (h:Hardware {hardware_key: $hardware_key})
            SET h.uid = coalesce(h.uid, $hardware_uid), h.hardware_key = $hardware_key
            WITH w, h
            MATCH (s:SoloProfile {solo_profile_id: $solo_profile_id})
            MERGE (s)-[:SOLO_FOR_SIGNATURE]->(w)
            MERGE (s)-[:SOLO_ON_HARDWARE]->(h)
            """,
            {
                "signature": profile.signature,
                "signature_uid": f"signature::{profile.signature}",
                "hardware_key": profile.hardware_key,
                "hardware_uid": f"hardware::{profile.hardware_key}",
                "solo_profile_id": solo_profile_id,
            },
        )
        if profile.last_job_id:
            self._upsert_run_profile(
                RunProfile(
                    run_profile_id=f"run::solo::{solo_profile_id}",
                    run_kind="training",
                    status="PROFILED",
                    job_id=profile.last_job_id,
                    backend_name=profile.metadata.get("backend_name"),
                    model_key=self._model_key_for_signature(profile.signature),
                    signature=profile.signature,
                    avg_sm_utilization=profile.avg_gpu_utilization,
                    avg_gpu_utilization=profile.avg_gpu_utilization,
                    avg_memory_utilization=profile.avg_memory_utilization,
                    peak_vram_mb=profile.peak_vram_mb,
                    observation_count=profile.sample_count,
                    source="solo_profile",
                    metadata={"gpu_name": self.hardware_profile().gpu_name},
                )
            )
        return profile

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

    def get_solo_profile(self, signature: str, *, hardware_key: str | None = None) -> SoloProfile | None:
        hardware_key = hardware_key or self.hardware_key()
        rows = self._run(
            """
            MATCH (s:SoloProfile {signature: $signature, hardware_key: $hardware_key})
            RETURN {
                signature: s.signature,
                hardware_key: s.hardware_key,
                family: s.family,
                peak_vram_mb: s.peak_vram_mb,
                avg_gpu_utilization: s.avg_gpu_utilization,
                avg_memory_utilization: s.avg_memory_utilization,
                sample_count: s.sample_count,
                last_job_id: s.last_job_id,
                updated_at: s.updated_at,
                metadata_json: s.metadata_json
            } AS row
            LIMIT 1
            """,
            {"signature": signature, "hardware_key": hardware_key},
        )
        return SoloProfile.from_row(rows[0]["row"]) if rows else None

    def list_solo_profiles(self, *, hardware_key: str | None = None) -> list[SoloProfile]:
        if hardware_key is None:
            rows = self._run(
                """
                MATCH (s:SoloProfile)
                RETURN {
                    signature: s.signature,
                    hardware_key: s.hardware_key,
                    family: s.family,
                    peak_vram_mb: s.peak_vram_mb,
                    avg_gpu_utilization: s.avg_gpu_utilization,
                    avg_memory_utilization: s.avg_memory_utilization,
                    sample_count: s.sample_count,
                    last_job_id: s.last_job_id,
                    updated_at: s.updated_at,
                    metadata_json: s.metadata_json
                } AS row
                ORDER BY s.updated_at DESC
                """
            )
        else:
            rows = self._run(
                """
                MATCH (s:SoloProfile {hardware_key: $hardware_key})
                RETURN {
                    signature: s.signature,
                    hardware_key: s.hardware_key,
                    family: s.family,
                    peak_vram_mb: s.peak_vram_mb,
                    avg_gpu_utilization: s.avg_gpu_utilization,
                    avg_memory_utilization: s.avg_memory_utilization,
                    sample_count: s.sample_count,
                    last_job_id: s.last_job_id,
                    updated_at: s.updated_at,
                    metadata_json: s.metadata_json
                } AS row
                ORDER BY s.updated_at DESC
                """,
                {"hardware_key": hardware_key},
            )
        return [SoloProfile.from_row(row["row"]) for row in rows]

    def upsert_pair_profile(self, profile: PairProfile) -> PairProfile:
        profile.updated_at = utc_now()
        if not profile.hardware_key:
            profile.hardware_key = self.hardware_key()
        packet_profile_id = f"pair::{profile.hardware_key}::{profile.pair_key}"
        props = {
            "uid": packet_profile_id,
            "packet_profile_id": packet_profile_id,
            "profile_scope": "pair",
            "pair_key": profile.pair_key,
            "hardware_key": profile.hardware_key,
            "backend_name": profile.backend_name,
            "compatible": profile.compatible,
            "observations": profile.observations,
            "peak_vram_mb": profile.peak_vram_mb,
            "avg_sm_utilization": profile.avg_gpu_utilization,
            "avg_gpu_utilization": profile.avg_gpu_utilization,
            "avg_memory_utilization": profile.avg_memory_utilization,
            "slowdown_ratio": profile.slowdown_ratio,
            "cooldown_until": profile.cooldown_until,
            "last_failure_reason": profile.last_failure_reason,
            "updated_at": profile.updated_at,
            "metadata_json": _json_dumps(profile.metadata),
            "left_signature": profile.left_signature,
            "right_signature": profile.right_signature,
            "summary_text": (
                f"Packed profile for {profile.left_signature} and {profile.right_signature} on {self.hardware_profile().gpu_name} "
                f"using {profile.backend_name}: compatible={profile.compatible}."
            ),
        }
        self._run_write("MERGE (p:PacketProfile {packet_profile_id: $packet_profile_id}) SET p += $props", {"packet_profile_id": packet_profile_id, "props": props})
        self._link_packet_profile(packet_profile_id, [profile.left_signature, profile.right_signature], profile.hardware_key, profile.backend_name)
        self._upsert_run_profile(
            RunProfile(
                run_profile_id=f"run::packet::{packet_profile_id}",
                run_kind="packed_training",
                status="COMPATIBLE" if profile.compatible else "INCOMPATIBLE",
                backend_name=profile.backend_name,
                signature=build_group_signature([profile.left_signature, profile.right_signature]),
                avg_sm_utilization=profile.avg_gpu_utilization,
                avg_gpu_utilization=profile.avg_gpu_utilization,
                avg_memory_utilization=profile.avg_memory_utilization,
                peak_vram_mb=profile.peak_vram_mb,
                slowdown_ratio=profile.slowdown_ratio,
                observation_count=profile.observations,
                source="pair_profile",
                metadata={"gpu_name": self.hardware_profile().gpu_name},
            )
        )
        return profile

    def _link_packet_profile(self, packet_profile_id: str, signatures: list[str], hardware_key: str, backend_name: str) -> None:
        for index, signature in enumerate(signatures):
            self._run_write(
                """
                MERGE (s:WorkloadSignature {signature: $signature})
                SET s.uid = coalesce(s.uid, $signature_uid), s.signature = $signature
                MERGE (h:Hardware {hardware_key: $hardware_key})
                SET h.uid = coalesce(h.uid, $hardware_uid), h.hardware_key = $hardware_key
                MERGE (b:Backend {backend_name: $backend_name})
                SET b.uid = coalesce(b.uid, $backend_uid), b.backend_name = $backend_name
                WITH s, h, b
                MATCH (p:PacketProfile {packet_profile_id: $packet_profile_id})
                MERGE (p)-[r:INVOLVES_SIGNATURE]->(s)
                SET r.position = $position
                MERGE (p)-[:PACKET_ON_HARDWARE]->(h)
                MERGE (p)-[:PACKET_WITH_BACKEND]->(b)
                """,
                {
                    "signature": signature,
                    "signature_uid": f"signature::{signature}",
                    "hardware_key": hardware_key,
                    "hardware_uid": f"hardware::{hardware_key}",
                    "backend_name": backend_name,
                    "backend_uid": f"backend::{backend_name}",
                    "packet_profile_id": packet_profile_id,
                    "position": index,
                },
            )
            model_key = self._model_key_for_signature(signature)
            if model_key:
                self._run_write(
                    """
                    MERGE (m:Model {model_key: $model_key})
                    SET m.uid = coalesce(m.uid, $model_uid), m.model_name = coalesce(m.model_name, $model_key)
                    WITH m
                    MATCH (p:PacketProfile {packet_profile_id: $packet_profile_id})
                    MERGE (p)-[:INVOLVES_MODEL]->(m)
                    """,
                    {"model_key": model_key, "model_uid": f"model::{model_key}", "packet_profile_id": packet_profile_id},
                )

    def get_pair_profile(
        self,
        left_signature: str,
        right_signature: str,
        *,
        hardware_key: str | None = None,
        backend_name: str | None = None,
    ) -> PairProfile | None:
        hardware_key = hardware_key or self.hardware_key()
        if backend_name is None:
            rows = self._run(
                """
                MATCH (p:PacketProfile {profile_scope: 'pair', hardware_key: $hardware_key})
                WHERE (p.left_signature = $left_signature AND p.right_signature = $right_signature)
                   OR (p.left_signature = $right_signature AND p.right_signature = $left_signature)
                RETURN {
                    pair_key: p.pair_key,
                    left_signature: p.left_signature,
                    right_signature: p.right_signature,
                    hardware_key: p.hardware_key,
                    backend_name: p.backend_name,
                    compatible: p.compatible,
                    observations: p.observations,
                    peak_vram_mb: p.peak_vram_mb,
                    avg_gpu_utilization: p.avg_gpu_utilization,
                    avg_memory_utilization: p.avg_memory_utilization,
                    slowdown_ratio: p.slowdown_ratio,
                    cooldown_until: p.cooldown_until,
                    last_failure_reason: p.last_failure_reason,
                    updated_at: p.updated_at,
                    metadata_json: p.metadata_json
                } AS row
                ORDER BY p.updated_at DESC
                LIMIT 1
                """,
                {"hardware_key": hardware_key, "left_signature": left_signature, "right_signature": right_signature},
            )
        else:
            pair_key = build_backend_scoped_pair_key(left_signature, right_signature, backend_name=backend_name)
            rows = self._run(
                """
                MATCH (p:PacketProfile {profile_scope: 'pair', hardware_key: $hardware_key, pair_key: $pair_key})
                RETURN {
                    pair_key: p.pair_key,
                    left_signature: p.left_signature,
                    right_signature: p.right_signature,
                    hardware_key: p.hardware_key,
                    backend_name: p.backend_name,
                    compatible: p.compatible,
                    observations: p.observations,
                    peak_vram_mb: p.peak_vram_mb,
                    avg_gpu_utilization: p.avg_gpu_utilization,
                    avg_memory_utilization: p.avg_memory_utilization,
                    slowdown_ratio: p.slowdown_ratio,
                    cooldown_until: p.cooldown_until,
                    last_failure_reason: p.last_failure_reason,
                    updated_at: p.updated_at,
                    metadata_json: p.metadata_json
                } AS row
                LIMIT 1
                """,
                {"hardware_key": hardware_key, "pair_key": pair_key},
            )
        return PairProfile.from_row(rows[0]["row"]) if rows else None

    def list_pair_profiles(self, *, hardware_key: str | None = None, backend_name: str | None = None) -> list[PairProfile]:
        clauses = ["MATCH (p:PacketProfile {profile_scope: 'pair'}) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if hardware_key is not None:
            clauses.append("AND p.hardware_key = $hardware_key")
            params["hardware_key"] = hardware_key
        if backend_name is not None:
            clauses.append("AND p.backend_name = $backend_name")
            params["backend_name"] = backend_name
        clauses.append(
            """
            RETURN {
                pair_key: p.pair_key,
                left_signature: p.left_signature,
                right_signature: p.right_signature,
                hardware_key: p.hardware_key,
                backend_name: p.backend_name,
                compatible: p.compatible,
                observations: p.observations,
                peak_vram_mb: p.peak_vram_mb,
                avg_gpu_utilization: p.avg_gpu_utilization,
                avg_memory_utilization: p.avg_memory_utilization,
                slowdown_ratio: p.slowdown_ratio,
                cooldown_until: p.cooldown_until,
                last_failure_reason: p.last_failure_reason,
                updated_at: p.updated_at,
                metadata_json: p.metadata_json
            } AS row
            ORDER BY p.updated_at DESC
            """
        )
        rows = self._run("\n".join(clauses), params)
        return [PairProfile.from_row(row["row"]) for row in rows]

    def upsert_runtime_profile(self, profile: RuntimeProfile) -> RuntimeProfile:
        profile.updated_at = utc_now()
        if not profile.hardware_key:
            profile.hardware_key = self.hardware_key()
        props = {
            "uid": f"runtime::{profile.profile_key}",
            "profile_key": profile.profile_key,
            "signature": profile.signature,
            "hardware_key": profile.hardware_key,
            "backend_name": profile.backend_name,
            "resolved_batch_size": profile.resolved_batch_size,
            "strategy": profile.strategy,
            "startup_seconds": profile.startup_seconds,
            "epoch_1_seconds": profile.epoch_1_seconds,
            "steps_per_epoch": profile.steps_per_epoch,
            "avg_step_time_ms": profile.avg_step_time_ms,
            "estimated_total_runtime_seconds": profile.estimated_total_runtime_seconds,
            "confidence": profile.confidence,
            "observations": profile.observations,
            "last_job_id": profile.last_job_id,
            "updated_at": profile.updated_at,
            "source": profile.source,
            "metadata_json": _json_dumps(profile.metadata),
            "summary_text": (
                f"Runtime profile for signature {profile.signature} on {self.hardware_profile().gpu_name} "
                f"with backend {profile.backend_name}: startup {profile.startup_seconds or 0.0}s, "
                f"epoch 1 {profile.epoch_1_seconds or 0.0}s, estimated total {profile.estimated_total_runtime_seconds or 0.0}s."
            ),
        }
        self._run_write("MERGE (r:RuntimeProfile {profile_key: $profile_key}) SET r += $props", {"profile_key": profile.profile_key, "props": props})
        self._run_write(
            """
            MERGE (s:WorkloadSignature {signature: $signature})
            SET s.uid = coalesce(s.uid, $signature_uid), s.signature = $signature
            MERGE (h:Hardware {hardware_key: $hardware_key})
            SET h.uid = coalesce(h.uid, $hardware_uid), h.hardware_key = $hardware_key
            MERGE (b:Backend {backend_name: $backend_name})
            SET b.uid = coalesce(b.uid, $backend_uid), b.backend_name = $backend_name
            WITH s, h, b
            MATCH (r:RuntimeProfile {profile_key: $profile_key})
            MERGE (r)-[:RUNTIME_FOR_SIGNATURE]->(s)
            MERGE (r)-[:RUNTIME_ON_HARDWARE]->(h)
            MERGE (r)-[:RUNTIME_WITH_BACKEND]->(b)
            """,
            {
                "signature": profile.signature,
                "signature_uid": f"signature::{profile.signature}",
                "hardware_key": profile.hardware_key,
                "hardware_uid": f"hardware::{profile.hardware_key}",
                "backend_name": profile.backend_name,
                "backend_uid": f"backend::{profile.backend_name}",
                "profile_key": profile.profile_key,
            },
        )
        self._upsert_run_profile(
            RunProfile(
                run_profile_id=f"run::runtime::{profile.profile_key}",
                run_kind="runtime_probe",
                probe_kind="runtime",
                status="PROFILED",
                job_id=profile.last_job_id,
                backend_name=profile.backend_name,
                model_key=self._model_key_for_signature(profile.signature),
                signature=profile.signature,
                resolved_batch_size=profile.resolved_batch_size,
                startup_seconds=profile.startup_seconds,
                epoch_time_seconds=profile.epoch_1_seconds,
                avg_step_time_ms=profile.avg_step_time_ms,
                estimated_total_runtime_seconds=profile.estimated_total_runtime_seconds,
                confidence=profile.confidence,
                observation_count=profile.observations,
                source="runtime_profile",
                metadata={"gpu_name": self.hardware_profile().gpu_name},
            )
        )
        return profile

    def get_runtime_profile(
        self,
        signature: str,
        *,
        resolved_batch_size: int,
        backend_name: str,
        hardware_key: str | None = None,
    ) -> RuntimeProfile | None:
        hardware_key = hardware_key or self.hardware_key()
        rows = self._run(
            """
            MATCH (r:RuntimeProfile {signature: $signature, hardware_key: $hardware_key, backend_name: $backend_name, resolved_batch_size: $resolved_batch_size})
            RETURN properties(r) AS row
            ORDER BY CASE r.source WHEN 'probe' THEN 0 ELSE 1 END, r.confidence DESC, r.updated_at DESC
            LIMIT 1
            """,
            {
                "signature": signature,
                "hardware_key": hardware_key,
                "backend_name": backend_name,
                "resolved_batch_size": int(resolved_batch_size),
            },
        )
        return RuntimeProfile.from_row(rows[0]["row"]) if rows else None

    def list_runtime_profiles(self, *, signature: str | None = None, hardware_key: str | None = None, backend_name: str | None = None) -> list[RuntimeProfile]:
        clauses = ["MATCH (r:RuntimeProfile) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if signature is not None:
            clauses.append("AND r.signature = $signature")
            params["signature"] = signature
        if hardware_key is not None:
            clauses.append("AND r.hardware_key = $hardware_key")
            params["hardware_key"] = hardware_key
        if backend_name is not None:
            clauses.append("AND r.backend_name = $backend_name")
            params["backend_name"] = backend_name
        clauses.append("RETURN properties(r) AS row ORDER BY CASE r.source WHEN 'probe' THEN 0 ELSE 1 END, r.confidence DESC, r.updated_at DESC")
        rows = self._run("\n".join(clauses), params)
        return [RuntimeProfile.from_row(row["row"]) for row in rows]

    def upsert_batch_probe_profile(self, profile: BatchProbeProfile) -> BatchProbeProfile:
        profile.updated_at = utc_now()
        props = {
            "uid": f"batch_probe::{profile.probe_key}",
            "probe_key": profile.probe_key,
            "model_key": profile.model_key,
            "device_type": profile.device_type,
            "shape_signature": profile.shape_signature,
            "batch_param_name": profile.batch_param_name,
            "resolved_batch_size": profile.resolved_batch_size,
            "peak_vram_mb": profile.peak_vram_mb,
            "memory_total_mb": profile.memory_total_mb,
            "target_budget_mb": profile.target_budget_mb,
            "observations": profile.observations,
            "last_job_id": profile.last_job_id,
            "updated_at": profile.updated_at,
            "metadata_json": _json_dumps(profile.metadata),
            "summary_text": (
                f"Batch probe for {profile.model_key} on {profile.device_type} resolved batch size "
                f"{profile.resolved_batch_size} under target budget {profile.target_budget_mb or 0} MB."
            ),
        }
        self._run_write("MERGE (p:BatchProbeProfile {probe_key: $probe_key}) SET p += $props", {"probe_key": profile.probe_key, "props": props})
        self._run_write(
            """
            MERGE (m:Model {model_key: $model_key})
            SET m.uid = coalesce(m.uid, $model_uid), m.model_name = coalesce(m.model_name, $model_key)
            MERGE (a:Accelerator {accelerator_key: $accelerator_key})
            SET a.uid = coalesce(a.uid, $accelerator_uid), a.accelerator_name = $device_type
            MERGE (s:BatchShape {shape_signature: $shape_signature})
            SET s.uid = coalesce(s.uid, $shape_uid), s.shape_signature = $shape_signature, s.batch_param_name = $batch_param_name
            WITH m, a, s
            MATCH (p:BatchProbeProfile {probe_key: $probe_key})
            MERGE (p)-[:OBSERVES_MODEL]->(m)
            MERGE (p)-[:FOR_ACCELERATOR]->(a)
            MERGE (p)-[:OBSERVES_SHAPE]->(s)
            """,
            {
                "model_key": profile.model_key,
                "model_uid": f"model::{profile.model_key}",
                "accelerator_key": _accelerator_key(profile.device_type),
                "accelerator_uid": f"accelerator::{_accelerator_key(profile.device_type)}",
                "device_type": profile.device_type,
                "shape_signature": profile.shape_signature,
                "shape_uid": f"shape::{profile.shape_signature}",
                "batch_param_name": profile.batch_param_name,
                "probe_key": profile.probe_key,
            },
        )
        self._upsert_run_profile(
            RunProfile(
                run_profile_id=f"run::batch_probe::{profile.probe_key}",
                run_kind="batch_probe",
                probe_kind="batch",
                status="PROFILED",
                job_id=profile.last_job_id,
                model_key=profile.model_key,
                shape_signature=profile.shape_signature,
                resolved_batch_size=profile.resolved_batch_size,
                peak_vram_mb=profile.peak_vram_mb,
                memory_total_mb=profile.memory_total_mb,
                target_budget_mb=profile.target_budget_mb,
                observation_count=profile.observations,
                source="batch_probe_profile",
                metadata={"device_type": profile.device_type},
            )
        )
        return profile

    def get_batch_probe_profile(self, probe_key: str) -> BatchProbeProfile | None:
        rows = self._run("MATCH (p:BatchProbeProfile {probe_key: $probe_key}) RETURN properties(p) AS row LIMIT 1", {"probe_key": probe_key})
        return BatchProbeProfile.from_row(rows[0]["row"]) if rows else None

    def list_batch_probe_profiles(self) -> list[BatchProbeProfile]:
        rows = self._run("MATCH (p:BatchProbeProfile) RETURN properties(p) AS row ORDER BY p.updated_at DESC")
        return [BatchProbeProfile.from_row(row["row"]) for row in rows]

    def upsert_batch_size_observation(self, observation: BatchSizeObservation) -> BatchSizeObservation:
        observation.updated_at = utc_now()
        props = {
            "uid": f"batch_obs::{observation.observation_key}",
            "observation_key": observation.observation_key,
            "model_key": observation.model_key,
            "shape_signature": observation.shape_signature,
            "hardware_key": observation.hardware_key,
            "backend_name": observation.backend_name,
            "batch_param_name": observation.batch_param_name,
            "batch_size": observation.batch_size,
            "peak_vram_mb": observation.peak_vram_mb,
            "memory_total_mb": observation.memory_total_mb,
            "avg_step_time_ms": observation.avg_step_time_ms,
            "avg_sm_utilization": observation.avg_gpu_utilization,
            "avg_gpu_utilization": observation.avg_gpu_utilization,
            "avg_memory_utilization": observation.avg_memory_utilization,
            "observations": observation.observations,
            "last_job_id": observation.last_job_id,
            "updated_at": observation.updated_at,
            "metadata_json": _json_dumps(observation.metadata),
            "summary_text": (
                f"Batch observation for {observation.model_key} at batch size {observation.batch_size} on "
                f"{observation.hardware_key}: avg step {observation.avg_step_time_ms or 0.0} ms."
            ),
        }
        self._run_write("MERGE (o:BatchSizeObservation {observation_key: $observation_key}) SET o += $props", {"observation_key": observation.observation_key, "props": props})
        self._run_write(
            """
            MERGE (m:Model {model_key: $model_key})
            SET m.uid = coalesce(m.uid, $model_uid), m.model_name = coalesce(m.model_name, $model_key)
            MERGE (s:BatchShape {shape_signature: $shape_signature})
            SET s.uid = coalesce(s.uid, $shape_uid), s.shape_signature = $shape_signature
            MERGE (h:Hardware {hardware_key: $hardware_key})
            SET h.uid = coalesce(h.uid, $hardware_uid), h.hardware_key = $hardware_key
            MERGE (b:Backend {backend_name: $backend_name})
            SET b.uid = coalesce(b.uid, $backend_uid), b.backend_name = $backend_name
            WITH m, s, h, b
            MATCH (o:BatchSizeObservation {observation_key: $observation_key})
            MERGE (o)-[:OBSERVES_MODEL]->(m)
            MERGE (o)-[:OBSERVES_SHAPE]->(s)
            MERGE (o)-[:OBSERVED_ON_HARDWARE]->(h)
            MERGE (o)-[:OBSERVED_WITH_BACKEND]->(b)
            """,
            {
                "model_key": observation.model_key,
                "model_uid": f"model::{observation.model_key}",
                "shape_signature": observation.shape_signature,
                "shape_uid": f"shape::{observation.shape_signature}",
                "hardware_key": observation.hardware_key,
                "hardware_uid": f"hardware::{observation.hardware_key}",
                "backend_name": observation.backend_name,
                "backend_uid": f"backend::{observation.backend_name}",
                "observation_key": observation.observation_key,
            },
        )
        return observation

    def get_batch_size_observation(
        self,
        *,
        model_key: str,
        shape_signature: str,
        hardware_key: str,
        backend_name: str,
        batch_size: int,
    ) -> BatchSizeObservation | None:
        rows = self._run(
            """
            MATCH (o:BatchSizeObservation)
            WHERE o.model_key = $model_key AND o.shape_signature = $shape_signature
              AND o.hardware_key = $hardware_key AND o.backend_name = $backend_name
              AND o.batch_size = $batch_size
            RETURN properties(o) AS row
            ORDER BY o.updated_at DESC
            LIMIT 1
            """,
            {
                "model_key": model_key,
                "shape_signature": shape_signature,
                "hardware_key": hardware_key,
                "backend_name": backend_name,
                "batch_size": int(batch_size),
            },
        )
        return BatchSizeObservation.from_row(rows[0]["row"]) if rows else None

    def list_batch_size_observations(
        self,
        *,
        model_key: str | None = None,
        shape_signature: str | None = None,
        hardware_key: str | None = None,
        backend_name: str | None = None,
    ) -> list[BatchSizeObservation]:
        clauses = ["MATCH (o:BatchSizeObservation) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if model_key is not None:
            clauses.append("AND o.model_key = $model_key")
            params["model_key"] = model_key
        if shape_signature is not None:
            clauses.append("AND o.shape_signature = $shape_signature")
            params["shape_signature"] = shape_signature
        if hardware_key is not None:
            clauses.append("AND o.hardware_key = $hardware_key")
            params["hardware_key"] = hardware_key
        if backend_name is not None:
            clauses.append("AND o.backend_name = $backend_name")
            params["backend_name"] = backend_name
        clauses.append("RETURN properties(o) AS row ORDER BY o.updated_at DESC")
        rows = self._run("\n".join(clauses), params)
        return [BatchSizeObservation.from_row(row["row"]) for row in rows]

    def upsert_combination_profile(self, profile: CombinationProfile) -> CombinationProfile:
        profile.updated_at = utc_now()
        packet_profile_id = f"group::{profile.combination_key}"
        props = {
            "uid": packet_profile_id,
            "packet_profile_id": packet_profile_id,
            "profile_scope": "group",
            "combination_key": profile.combination_key,
            "group_signature": profile.group_signature,
            "hardware_key": profile.hardware_key,
            "backend_name": profile.backend_name,
            "scheduler_mode": profile.scheduler_mode,
            "batch_vector_json": _json_dumps(profile.batch_vector),
            "compatible": profile.compatible,
            "observations": profile.observations,
            "peak_vram_mb": profile.peak_vram_mb,
            "memory_total_mb": profile.memory_total_mb,
            "avg_sm_utilization": profile.avg_gpu_utilization,
            "avg_gpu_utilization": profile.avg_gpu_utilization,
            "avg_memory_utilization": profile.avg_memory_utilization,
            "avg_step_time_ms": profile.avg_step_time_ms,
            "objective_score": profile.objective_score,
            "resolved_optimal": profile.resolved_optimal,
            "last_failure_reason": profile.last_failure_reason,
            "fallback_order_json": _json_dumps(profile.fallback_order),
            "updated_at": profile.updated_at,
            "metadata_json": _json_dumps(profile.metadata),
            "summary_text": (
                f"Packed group profile for {profile.group_signature} on {self.hardware_profile().gpu_name} "
                f"using {profile.backend_name}: compatible={profile.compatible}."
            ),
        }
        self._run_write("MERGE (p:PacketProfile {packet_profile_id: $packet_profile_id}) SET p += $props", {"packet_profile_id": packet_profile_id, "props": props})
        signatures = [sig for sig in str(profile.group_signature).split("::") if sig]
        self._link_packet_profile(packet_profile_id, signatures, profile.hardware_key, profile.backend_name)
        return profile

    def best_combination_profile(
        self,
        *,
        group_signature: str,
        hardware_key: str,
        backend_name: str,
        scheduler_mode: str,
    ) -> CombinationProfile | None:
        rows = self._run(
            """
            MATCH (p:PacketProfile {profile_scope: 'group', group_signature: $group_signature, hardware_key: $hardware_key, backend_name: $backend_name, scheduler_mode: $scheduler_mode})
            WHERE p.compatible = true
            RETURN {
                combination_key: p.combination_key,
                group_signature: p.group_signature,
                hardware_key: p.hardware_key,
                backend_name: p.backend_name,
                scheduler_mode: p.scheduler_mode,
                batch_vector_json: p.batch_vector_json,
                compatible: p.compatible,
                observations: p.observations,
                peak_vram_mb: p.peak_vram_mb,
                memory_total_mb: p.memory_total_mb,
                avg_gpu_utilization: p.avg_gpu_utilization,
                avg_memory_utilization: p.avg_memory_utilization,
                avg_step_time_ms: p.avg_step_time_ms,
                objective_score: p.objective_score,
                resolved_optimal: p.resolved_optimal,
                last_failure_reason: p.last_failure_reason,
                fallback_order_json: p.fallback_order_json,
                updated_at: p.updated_at,
                metadata_json: p.metadata_json
            } AS row
            ORDER BY p.resolved_optimal DESC, p.objective_score DESC, p.updated_at DESC
            LIMIT 1
            """,
            {
                "group_signature": group_signature,
                "hardware_key": hardware_key,
                "backend_name": backend_name,
                "scheduler_mode": scheduler_mode,
            },
        )
        return CombinationProfile.from_row(rows[0]["row"]) if rows else None

    def list_combination_profiles(
        self,
        *,
        group_signature: str | None = None,
        hardware_key: str | None = None,
        backend_name: str | None = None,
        scheduler_mode: str | None = None,
    ) -> list[CombinationProfile]:
        clauses = ["MATCH (p:PacketProfile {profile_scope: 'group'}) WHERE 1 = 1"]
        params: dict[str, Any] = {}
        if group_signature is not None:
            clauses.append("AND p.group_signature = $group_signature")
            params["group_signature"] = group_signature
        if hardware_key is not None:
            clauses.append("AND p.hardware_key = $hardware_key")
            params["hardware_key"] = hardware_key
        if backend_name is not None:
            clauses.append("AND p.backend_name = $backend_name")
            params["backend_name"] = backend_name
        if scheduler_mode is not None:
            clauses.append("AND p.scheduler_mode = $scheduler_mode")
            params["scheduler_mode"] = scheduler_mode
        clauses.append(
            """
            RETURN {
                combination_key: p.combination_key,
                group_signature: p.group_signature,
                hardware_key: p.hardware_key,
                backend_name: p.backend_name,
                scheduler_mode: p.scheduler_mode,
                batch_vector_json: p.batch_vector_json,
                compatible: p.compatible,
                observations: p.observations,
                peak_vram_mb: p.peak_vram_mb,
                memory_total_mb: p.memory_total_mb,
                avg_gpu_utilization: p.avg_gpu_utilization,
                avg_memory_utilization: p.avg_memory_utilization,
                avg_step_time_ms: p.avg_step_time_ms,
                objective_score: p.objective_score,
                resolved_optimal: p.resolved_optimal,
                last_failure_reason: p.last_failure_reason,
                fallback_order_json: p.fallback_order_json,
                updated_at: p.updated_at,
                metadata_json: p.metadata_json
            } AS row
            ORDER BY p.updated_at DESC
            """
        )
        rows = self._run("\n".join(clauses), params)
        return [CombinationProfile.from_row(row["row"]) for row in rows]

    def mark_pair_incompatible(
        self,
        left_signature: str,
        right_signature: str,
        *,
        backend_name: str,
        reason: str,
        cooldown_seconds: int,
        peak_vram_mb: int | None = None,
        avg_gpu_utilization: float | None = None,
        avg_memory_utilization: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PairProfile:
        hardware_key = self.hardware_key()
        existing = self.get_pair_profile(left_signature, right_signature, hardware_key=hardware_key, backend_name=backend_name)
        cooldown_until = None
        if cooldown_seconds > 0:
            cooldown_until = (parse_timestamp(utc_now()) + timedelta(seconds=cooldown_seconds)).isoformat()
        profile = PairProfile.create(
            left_signature,
            right_signature,
            backend_name=backend_name,
            hardware_key=hardware_key,
            compatible=False,
            observations=(existing.observations + 1) if existing else 1,
            peak_vram_mb=peak_vram_mb if peak_vram_mb is not None else (existing.peak_vram_mb if existing else None),
            avg_gpu_utilization=avg_gpu_utilization if avg_gpu_utilization is not None else (existing.avg_gpu_utilization if existing else None),
            avg_memory_utilization=avg_memory_utilization if avg_memory_utilization is not None else (existing.avg_memory_utilization if existing else None),
            slowdown_ratio=existing.slowdown_ratio if existing else None,
            cooldown_until=cooldown_until,
            last_failure_reason=reason,
            metadata={**(existing.metadata if existing else {}), **(metadata or {})},
        )
        return self.upsert_pair_profile(profile)

    def reconcile_incomplete_jobs(self) -> list[TrainingJob]:
        stale_jobs = self.list_jobs(statuses=[JobStatus.RUNNING, JobStatus.PAUSING])
        reconciled: list[TrainingJob] = []
        for job in stale_jobs:
            checkpoint_path = self.latest_checkpoint(job.job_id)
            if checkpoint_path and Path(checkpoint_path).exists():
                updated = self.update_job(
                    job.job_id,
                    status=JobStatus.RECOVERABLE,
                    reason="scheduler restarted while job was active; checkpoint available",
                    latest_checkpoint_path=checkpoint_path,
                )
            else:
                updated = self.update_job(
                    job.job_id,
                    status=JobStatus.FAILED,
                    reason="scheduler restarted while job was active; no checkpoint found",
                )
            reconciled.append(updated)
        return reconciled

    def report(self) -> SchedulerReport:
        jobs = self.list_jobs()
        events = self.list_events(event_type=None)
        wait_times: list[float] = []
        runtimes: list[float] = []
        for job in jobs:
            submitted = parse_timestamp(job.submitted_at)
            started = parse_timestamp(job.started_at)
            finished = parse_timestamp(job.finished_at)
            if submitted and started:
                wait_times.append((started - submitted).total_seconds())
            if started and finished:
                runtimes.append((finished - started).total_seconds())
        cache_summary = self.cache_metadata_summary()
        total_cache = int(cache_summary["hits"]) + int(cache_summary["misses"])
        return SchedulerReport(
            total_jobs=len(jobs),
            completed_jobs=sum(job.status == JobStatus.COMPLETED for job in jobs),
            failed_jobs=sum(job.status == JobStatus.FAILED for job in jobs),
            cancelled_jobs=sum(job.status == JobStatus.CANCELLED for job in jobs),
            average_queue_wait_seconds=sum(wait_times) / len(wait_times) if wait_times else 0.0,
            average_runtime_seconds=sum(runtimes) / len(runtimes) if runtimes else 0.0,
            cache_hit_rate=(int(cache_summary["hits"]) / total_cache) if total_cache else 0.0,
            cache_hits=int(cache_summary["hits"]),
            cache_misses=int(cache_summary["misses"]),
            cache_evictions=sum(event["event_type"] == "cache_evicted" for event in events),
            packed_dispatches=sum(event["event_type"] in {"packed_pair_dispatched", "packed_group_dispatched"} for event in events),
            packed_fallbacks=sum(event["event_type"] in {"packed_pair_fallback", "packed_group_fallback"} for event in events),
        )
