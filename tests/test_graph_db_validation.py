from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.domain import BatchProbeProfile, BatchProbeSpec, PackingSpec, RuntimeProfile, SoloProfile, TrainingJob
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.mcp_server import build_mcp_server
from localml_scheduler.storage import LegacySQLiteStateStore
from localml_scheduler.storage.models import SCHEMA_STATEMENTS
from localml_scheduler.storage.neo4j_store import Neo4jStateStore
from localml_scheduler.storage.state_store import StateStore


def _test_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        hardware_key="hw-test-001",
        os_name="linux",
        gpu_name="Test GPU",
        total_vram_mb=24576,
        compute_capability="9.0",
        cuda_runtime="12.4",
        torch_version="2.5.1",
    )


def _sqlite_client_with_test_hardware(tmpdir: str) -> SchedulerClient:
    settings = SchedulerConfig(runtime_root=tmpdir)
    settings.graph_db.enabled = False
    settings.graph_db.mode = "off"
    client = SchedulerClient(settings)
    client.store.backend._hardware_profile = _test_hardware_profile()
    return client


class _FakeNeo4jRecord:
    def __init__(self, row: dict[str, object]):
        self._row = dict(row)

    def data(self) -> dict[str, object]:
        return dict(self._row)


class _FakeNeo4jResult:
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __iter__(self):
        for row in self._rows:
            yield _FakeNeo4jRecord(row)

    def consume(self) -> None:
        return None


class _FakeNeo4jSession:
    def __init__(self, driver: "_FakeNeo4jDriver"):
        self.driver = driver

    def __enter__(self) -> "_FakeNeo4jSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def run(self, cypher: str, params: dict[str, object] | None = None) -> _FakeNeo4jResult:
        return _FakeNeo4jResult(self.driver.handle(cypher, params or {}))


class _FakeNeo4jDriver:
    def __init__(self):
        self.jobs: dict[str, dict[str, object]] = {}
        self.commands: dict[int, dict[str, object]] = {}
        self.events: dict[int, dict[str, object]] = {}
        self.batch_probe_profiles: dict[str, dict[str, object]] = {}
        self.runtime_profiles: dict[str, dict[str, object]] = {}
        self.run_profiles: dict[str, dict[str, object]] = {}
        self.signatures: dict[str, str] = {}
        self.sequences: dict[str, int] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def verify_connectivity(self) -> None:
        return None

    def session(self, database: str | None = None) -> _FakeNeo4jSession:
        return _FakeNeo4jSession(self)

    def handle(self, cypher: str, params: dict[str, object]) -> list[dict[str, object]]:
        normalized = " ".join(cypher.split())
        self.calls.append((normalized, dict(params)))

        if normalized == "MATCH (j:Job) RETURN count(j) AS value":
            return [{"value": len(self.jobs)}]
        if normalized == "MATCH (j:Job) RETURN coalesce(max(j.queue_sequence), 0) + 1 AS value":
            next_value = max((int(job.get("queue_sequence") or 0) for job in self.jobs.values()), default=0) + 1
            return [{"value": next_value}]
        if "MERGE (s:Sequence {name: $name})" in normalized and "RETURN s.value AS value" in normalized:
            name = str(params["name"])
            next_value = self.sequences.get(name, 0) + 1
            self.sequences[name] = next_value
            return [{"value": next_value}]
        if normalized == "MERGE (j:Job {job_id: $job_id}) SET j += $props":
            self.jobs[str(params["job_id"])] = dict(params["props"])
            return []
        if normalized == "MATCH (j:Job {job_id: $job_id}) RETURN j.payload_json AS payload_json":
            job = self.jobs.get(str(params["job_id"]))
            return [{"payload_json": job["payload_json"]}] if job else []
        if normalized.startswith("MATCH (j:Job) WHERE j.status IN $statuses RETURN j.payload_json AS payload_json"):
            statuses = {str(item) for item in params["statuses"]}
            rows = [
                {"payload_json": job["payload_json"]}
                for job in self.jobs.values()
                if str(job.get("status")) in statuses
            ]
            return rows
        if normalized.startswith("MATCH (j:Job) RETURN j.payload_json AS payload_json"):
            return [{"payload_json": job["payload_json"]} for job in self.jobs.values()]
        if normalized == "MERGE (c:Command {command_id: $command_id}) SET c += $props":
            self.commands[int(params["command_id"])] = dict(params["props"])
            return []
        if normalized.startswith("MATCH (c:Command) WHERE c.processed_at IS NULL RETURN c.command_id AS command_id"):
            rows = [
                {
                    "command_id": command["command_id"],
                    "job_id": command["job_id"],
                    "command_type": command["command_type"],
                    "payload_json": command["payload_json"],
                    "created_at": command["created_at"],
                    "processed_at": command["processed_at"],
                }
                for command in sorted(self.commands.values(), key=lambda item: int(item["command_id"]))
                if not command.get("processed_at")
            ]
            return rows[: int(params["limit"])]
        if normalized.startswith("MATCH (c:Command {command_id: $command_id}) SET c.processed_at = $processed_at"):
            command = self.commands.get(int(params["command_id"]))
            if command is not None:
                command["processed_at"] = params["processed_at"]
                command["updated_at"] = params["processed_at"]
            return []
        if normalized == "MERGE (e:Event {event_id: $event_id}) SET e += $props":
            self.events[int(params["event_id"])] = dict(params["props"])
            return []
        if normalized.startswith("MATCH (e:Event) WHERE 1 = 1"):
            rows = []
            for event in sorted(self.events.values(), key=lambda item: int(item["event_id"])):
                if params.get("job_id") is not None and event.get("job_id") != params["job_id"]:
                    continue
                if params.get("event_type") is not None and event.get("event_type") != params["event_type"]:
                    continue
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "job_id": event["job_id"],
                        "event_type": event["event_type"],
                        "payload_json": event["payload_json"],
                        "created_at": event["created_at"],
                    }
                )
            return rows
        if normalized == "MERGE (p:BatchProbeProfile {probe_key: $probe_key}) SET p += $props":
            self.batch_probe_profiles[str(params["probe_key"])] = dict(params["props"])
            return []
        if normalized == "MATCH (p:BatchProbeProfile {probe_key: $probe_key}) RETURN properties(p) AS row LIMIT 1":
            profile = self.batch_probe_profiles.get(str(params["probe_key"]))
            return [{"row": dict(profile)}] if profile else []
        if normalized.startswith("MATCH (p:BatchProbeProfile) RETURN properties(p) AS row"):
            return [{"row": dict(profile)} for profile in self.batch_probe_profiles.values()]
        if normalized == "MERGE (r:RuntimeProfile {profile_key: $profile_key}) SET r += $props":
            self.runtime_profiles[str(params["profile_key"])] = dict(params["props"])
            return []
        if normalized.startswith("MATCH (r:RuntimeProfile {signature: $signature, hardware_key: $hardware_key, backend_name: $backend_name, resolved_batch_size: $resolved_batch_size})"):
            rows = []
            for profile in self.runtime_profiles.values():
                if profile.get("signature") != params["signature"]:
                    continue
                if profile.get("hardware_key") != params["hardware_key"]:
                    continue
                if profile.get("backend_name") != params["backend_name"]:
                    continue
                if int(profile.get("resolved_batch_size") or 0) != int(params["resolved_batch_size"]):
                    continue
                rows.append({"row": dict(profile)})
            return rows[:1]
        if normalized.startswith("MATCH (r:RuntimeProfile) WHERE 1 = 1") and "RETURN properties(r) AS row" in normalized:
            rows = []
            for profile in self.runtime_profiles.values():
                if params.get("signature") is not None and profile.get("signature") != params["signature"]:
                    continue
                if params.get("hardware_key") is not None and profile.get("hardware_key") != params["hardware_key"]:
                    continue
                if params.get("backend_name") is not None and profile.get("backend_name") != params["backend_name"]:
                    continue
                rows.append({"row": dict(profile)})
            return rows
        if normalized == "MERGE (r:RunProfile {run_profile_id: $run_profile_id}) SET r += $props":
            self.run_profiles[str(params["run_profile_id"])] = dict(params["props"])
            return []
        if normalized.startswith("MATCH (r:RunProfile) WHERE 1 = 1") and "RETURN properties(r) AS row" in normalized:
            rows = []
            for profile in self.run_profiles.values():
                if params.get("job_id") is not None and profile.get("job_id") != params["job_id"]:
                    continue
                if params.get("model_key") is not None and profile.get("model_key") != params["model_key"]:
                    continue
                if params.get("signature") is not None and profile.get("signature") != params["signature"]:
                    continue
                rows.append({"row": dict(profile)})
            return rows
        if normalized == "MATCH (s:WorkloadSignature {signature: $signature}) RETURN s.model_key AS model_key LIMIT 1":
            model_key = self.signatures.get(str(params["signature"]))
            return [{"model_key": model_key}] if model_key else []
        if "MERGE (s:WorkloadSignature {signature: $signature})" in normalized and "s.model_key = $model_key" in normalized:
            signature = params.get("signature")
            model_key = params.get("model_key")
            if signature and model_key:
                self.signatures[str(signature)] = str(model_key)
            return []
        return []


class _FakeCursor:
    def __init__(self, statements: list[tuple[str, object]]):
        self.statements = statements

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params: object | None = None) -> None:
        self.statements.append((" ".join(sql.split()), params))


class _FakeConnection:
    def __init__(self, statements: list[tuple[str, object]]):
        self.statements = statements

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.statements)


class _FakePsycopg:
    def __init__(self, statements: list[tuple[str, object]]):
        self.statements = statements

    def connect(self, dsn: str, autocommit: bool = True) -> _FakeConnection:
        self.statements.append((f"CONNECT {dsn}", autocommit))
        return _FakeConnection(self.statements)


class _FailingPsycopg:
    def connect(self, dsn: str, autocommit: bool = True):
        del dsn, autocommit
        raise RuntimeError("unreachable")


class GraphDatabaseValidationTest(unittest.TestCase):
    def test_neo4j_store_round_trips_jobs_profiles_and_run_profiles(self) -> None:
        driver = _FakeNeo4jDriver()
        graph_database = SimpleNamespace(driver=lambda uri, auth=None: driver)
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.graph_db.bootstrap_constraints = False
            settings.graph_db.auto_import_legacy_sqlite = False
            settings.graph_db.allow_legacy_fallback = False
            job = TrainingJob.create(
                "module:runner",
                "baseline-a",
                "/tmp/a.pt",
                packing=PackingSpec(eligible=True, signature="sig-1", family="family-a"),
                batch_probe=BatchProbeSpec(enabled=True, model_key="baseline-a"),
                max_epochs=4,
                runner_kwargs={"batch_size": 2},
            )
            with patch("localml_scheduler.storage.neo4j_store.GraphDatabase", graph_database), patch(
                "localml_scheduler.storage.neo4j_store.detect_hardware_profile",
                return_value=_test_hardware_profile(),
            ):
                store = Neo4jStateStore(settings)
                submitted = store.submit_job(job)
                batch_probe = BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="baseline-a",
                    device_type="Test GPU",
                    shape_signature="shape-1",
                    batch_param_name="batch_size",
                    resolved_batch_size=8,
                    observations=2,
                    last_job_id=submitted.job_id,
                )
                runtime_profile = RuntimeProfile.create(
                    signature="sig-1",
                    hardware_key=_test_hardware_profile().hardware_key,
                    backend_name="exclusive",
                    resolved_batch_size=8,
                    strategy="epoch_1",
                    epoch_1_seconds=12.0,
                    estimated_total_runtime_seconds=48.0,
                    confidence=0.9,
                    observations=1,
                    last_job_id=submitted.job_id,
                )

                store.upsert_batch_probe_profile(batch_probe)
                store.upsert_runtime_profile(runtime_profile)

                self.assertEqual(store.get_job(submitted.job_id).job_id, submitted.job_id)
                self.assertEqual(len(store.list_jobs()), 1)
                self.assertEqual(len(store.fetch_pending_commands()), 1)
                self.assertEqual(len(store.list_events(job_id=submitted.job_id, event_type="job_submitted")), 1)
                self.assertEqual(store.get_batch_probe_profile("probe-1").resolved_batch_size, 8)
                self.assertEqual(
                    store.get_runtime_profile(
                        "sig-1",
                        resolved_batch_size=8,
                        backend_name="exclusive",
                    ).estimated_total_runtime_seconds,
                    48.0,
                )
                run_profiles = store.list_run_profiles(job_id=submitted.job_id)
                self.assertEqual(len(run_profiles), 2)
                self.assertEqual({profile.run_kind for profile in run_profiles}, {"batch_probe", "runtime_probe"})

                pending = store.fetch_pending_commands()
                store.mark_command_processed(pending[0].command_id)
                self.assertEqual(store.fetch_pending_commands(), [])

    def test_state_store_falls_back_to_legacy_sqlite_when_neo4j_boot_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.graph_db.allow_legacy_fallback = True
            with patch("localml_scheduler.storage.state_store.Neo4jStateStore", side_effect=RuntimeError("boom")):
                store = StateStore(settings)
            self.assertIsInstance(store.backend, LegacySQLiteStateStore)

    def test_state_store_uses_sqlite_primary_with_best_effort_graph_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.graph_db.mode = "mirror"
            settings.graph_db.enabled = True
            mirror = MagicMock()
            with patch("localml_scheduler.storage.state_store.Neo4jStateStore", side_effect=lambda current_settings: mirror):
                store = StateStore(settings)

            job = TrainingJob.create(
                "module:runner",
                "baseline-a",
                "/tmp/a.pt",
                runner_kwargs={"batch_size": 2},
            )
            submitted = store.submit_job(job)
            profile = BatchProbeProfile(
                probe_key="probe-1",
                model_key="baseline-a",
                device_type="Test GPU",
                shape_signature="shape-1",
                batch_param_name="batch_size",
                resolved_batch_size=8,
                observations=1,
                last_job_id=submitted.job_id,
            )

            store.upsert_batch_probe_profile(profile)

            self.assertIsInstance(store.backend, LegacySQLiteStateStore)
            self.assertEqual(store.get_job(submitted.job_id).job_id, submitted.job_id)
            mirror.record_scheduler_job_evidence.assert_called()
            mirror.log_event.assert_not_called()
            mirror.record_batch_probe_evidence.assert_called_once()

    def test_state_store_mirror_mode_degrades_to_sqlite_when_graph_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.graph_db.mode = "mirror"
            settings.graph_db.enabled = True
            with patch("localml_scheduler.storage.state_store.Neo4jStateStore", side_effect=RuntimeError("boom")):
                store = StateStore(settings)
            self.assertIsInstance(store.backend, LegacySQLiteStateStore)

    def test_legacy_sqlite_import_bootstraps_evidence_records_only(self) -> None:
        driver = _FakeNeo4jDriver()
        graph_database = SimpleNamespace(driver=lambda uri, auth=None: driver)
        imported: dict[str, list[object]] = {
            "jobs": [],
            "batch_probe": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = os.path.join(tmpdir, "runtime")
            legacy_path = os.path.join(tmpdir, "legacy.sqlite3")
            settings = SchedulerConfig(runtime_root=runtime_root)
            settings.graph_db.bootstrap_constraints = False
            settings.graph_db.auto_import_legacy_sqlite = True
            settings.graph_db.allow_legacy_fallback = False
            settings.graph_db.legacy_sqlite_path = legacy_path

            with sqlite3.connect(legacy_path) as connection:
                for statement in SCHEMA_STATEMENTS:
                    connection.execute(statement)
                job = TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt", runner_kwargs={"batch_size": 2})
                connection.execute(
                    """
                    INSERT INTO jobs(job_id, status, priority, baseline_model_id, submitted_at, queue_sequence, payload_json, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.status.value,
                        job.priority,
                        job.baseline_model_id,
                        job.submitted_at,
                        1,
                        job.to_json(),
                        job.submitted_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO commands(command_id, job_id, command_type, payload_json, created_at, processed_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (1, job.job_id, "submit", json.dumps({"priority": job.priority}, sort_keys=True), job.submitted_at, None),
                )
                connection.execute(
                    """
                    INSERT INTO events(event_id, job_id, event_type, payload_json, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (1, job.job_id, "job_submitted", json.dumps({"priority": job.priority}, sort_keys=True), job.submitted_at),
                )
                connection.execute(
                    """
                    INSERT INTO checkpoints(checkpoint_id, job_id, checkpoint_path, created_at, metadata_json, is_latest)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (1, job.job_id, "/tmp/checkpoint.pt", job.submitted_at, json.dumps({"epoch": 1}, sort_keys=True), 1),
                )
                connection.execute(
                    """
                    INSERT INTO cache_entries(model_id, baseline_model_path, size_bytes, pinned, hits, misses, last_loaded_at, last_accessed_at, metadata_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("baseline-a", "/tmp/a.pt", 1024, 1, 3, 1, job.submitted_at, job.submitted_at, json.dumps({"source": "test"}, sort_keys=True)),
                )
                connection.commit()

            def _record_job(self, job: TrainingJob) -> None:
                imported["jobs"].append(job)

            def _record_batch_probe(self, profile: BatchProbeProfile) -> None:
                imported["batch_probe"].append(profile)

            with patch("localml_scheduler.storage.neo4j_store.GraphDatabase", graph_database), patch(
                "localml_scheduler.storage.neo4j_store.detect_hardware_profile",
                return_value=_test_hardware_profile(),
            ), patch.object(Neo4jStateStore, "record_scheduler_job_evidence", _record_job), patch.object(
                Neo4jStateStore, "record_batch_probe_evidence", _record_batch_probe
            ):
                Neo4jStateStore(settings)

        self.assertEqual(len(imported["jobs"]), 1)
        self.assertEqual(imported["jobs"][0].baseline_model_id, "baseline-a")
        self.assertEqual(imported["batch_probe"], [])

    def test_log_store_routes_events_and_metrics_into_expected_tables(self) -> None:
        statements: list[tuple[str, object]] = []
        fake_psycopg = _FakePsycopg(statements)
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.log_db.enabled = True
            with patch.dict(os.environ, {settings.log_db.dsn_env: "postgresql://scheduler:test@localhost/scheduler"}, clear=False), patch(
                "localml_scheduler.storage.log_store.psycopg",
                fake_psycopg,
            ):
                from localml_scheduler.storage.log_store import SchedulerLogStore

                store = SchedulerLogStore(settings)
                session_id = store.start_session(
                    status="running",
                    pid=1234,
                    runtime_root=tmpdir,
                    host_identity={"host": "test-host"},
                    config_json={"mode": "test"},
                    started_at="2026-05-11T00:00:00+00:00",
                )
                store.record_event(
                    job_id="job-1",
                    event_type="job_started",
                    created_at="2026-05-11T00:00:01+00:00",
                    payload={"status": "started"},
                )
                store.record_event(
                    job_id="job-1",
                    event_type="batch_probe_selected",
                    created_at="2026-05-11T00:00:02+00:00",
                    payload={"resolved_batch_size": 8},
                )
                store.record_event(
                    job_id=None,
                    event_type="cache_warm",
                    created_at="2026-05-11T00:00:03+00:00",
                    payload={"model_id": "baseline-a"},
                )
                store.record_event(
                    job_id=None,
                    event_type="planner_decision_trace",
                    created_at="2026-05-11T00:00:03.100000+00:00",
                    payload={
                        "scheduler_mode": "parallel_auto_pack",
                        "safe_vram_budget_mb": 28672.0,
                        "active_gpu_occupancy": {"vram_mb": 1024.0, "sm_utilization": 0.1},
                        "selected_plan": {
                            "mode": "packed_pair",
                            "backend_name": "stream",
                            "job_ids": ["job-1", "job-2"],
                            "reason": "auto-pack group selected",
                            "expected_runtime_seconds": 12.0,
                        },
                        "candidates": [],
                    },
                )
                store.record_event(
                    job_id="job-1",
                    event_type="runtime_probe_profiled",
                    created_at="2026-05-11T00:00:03.200000+00:00",
                    payload={
                        "profile_key": "runtime-profile-1",
                        "signature": "sig-1",
                        "hardware_key": "hw-test-001",
                        "backend_name": "stream",
                        "strategy": "epoch_1",
                        "resolved_batch_size": 8,
                        "confidence": 0.8,
                        "estimated_total_runtime_seconds": 48.0,
                        "avg_step_time_ms": 12.5,
                        "source": "probe",
                    },
                )
                store.record_event(
                    job_id="job-1",
                    event_type="worker_launched",
                    created_at="2026-05-11T00:00:03.300000+00:00",
                    payload={
                        "group_id": "group-1",
                        "backend_name": "stream",
                        "placement_mode": "packed_pair",
                        "pid": 456,
                        "stdout_path": "/tmp/stdout.log",
                        "stderr_path": "/tmp/stderr.log",
                    },
                )
                store.record_job_metric_sample(
                    job_id="job-1",
                    created_at="2026-05-11T00:00:04+00:00",
                    epoch=1,
                    global_step=10,
                    avg_step_time_ms=12.5,
                    estimated_total_runtime_seconds=48.0,
                    remaining_runtime_seconds=35.5,
                    metrics={"loss": 0.1},
                )
                store.open_run_group(
                    group_id="group-1",
                    mode="solo",
                    backend_name="exclusive",
                    hardware_key="hw-test-001",
                    group_signature="sig-1",
                    opened_at="2026-05-11T00:00:05+00:00",
                    overlapped=False,
                    metadata={"probe_task": False},
                )
                store.upsert_run_group_member(
                    group_id="group-1",
                    job_id="job-1",
                    role="primary",
                    batch_size=8,
                    joined_at="2026-05-11T00:00:05+00:00",
                    metadata={"order": 0},
                )
                store.record_gpu_metric_sample(
                    group_id="group-1",
                    created_at="2026-05-11T00:00:06+00:00",
                    backend_name="exclusive",
                    hardware_key="hw-test-001",
                    memory_used_mb=1024,
                    memory_total_mb=24576,
                    gpu_utilization=0.8,
                    memory_utilization=0.4,
                    job_ids=["job-1"],
                )
                store.close_run_group(
                    group_id="group-1",
                    closed_at="2026-05-11T00:00:07+00:00",
                    overlapped=False,
                    fallback_triggered=False,
                    fallback_reason=None,
                    exit_reason="completed",
                )

        rendered = "\n".join(sql for sql, _ in statements if isinstance(sql, str))
        self.assertIsNotNone(session_id)
        self.assertIn("scheduler_sessions", rendered)
        self.assertIn("job_activity_log", rendered)
        self.assertIn("probe_activity_log", rendered)
        self.assertIn("cache_activity_log", rendered)
        self.assertIn("job_metric_samples", rendered)
        self.assertIn("run_groups", rendered)
        self.assertIn("run_group_members", rendered)
        self.assertIn("gpu_metric_samples", rendered)
        self.assertIn("planner_decision_log", rendered)
        self.assertIn("runtime_probe_summaries", rendered)
        self.assertIn("worker_execution_log", rendered)

    def test_log_store_warns_but_does_not_fail_when_dsn_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.log_db.enabled = True
            with patch.dict(os.environ, {}, clear=True):
                from localml_scheduler.storage.log_store import SchedulerLogStore

                store = SchedulerLogStore(settings)
                session_id = store.start_session(
                    status="running",
                    pid=1234,
                    runtime_root=tmpdir,
                    host_identity={"host": "test-host"},
                    config_json={"mode": "test"},
                    started_at="2026-05-11T00:00:00+00:00",
                )
                store.record_event(
                    job_id="job-1",
                    event_type="job_started",
                    created_at="2026-05-11T00:00:01+00:00",
                    payload={"status": "started"},
                )

        self.assertIsNotNone(session_id)
        self.assertTrue(store._warned)

    def test_log_store_warns_but_does_not_fail_when_postgres_is_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            settings.log_db.enabled = True
            with patch.dict(os.environ, {settings.log_db.dsn_env: "postgresql://scheduler:test@localhost/scheduler"}, clear=False), patch(
                "localml_scheduler.storage.log_store.psycopg",
                _FailingPsycopg(),
            ):
                from localml_scheduler.storage.log_store import SchedulerLogStore

                store = SchedulerLogStore(settings)
                store.record_event(
                    job_id="job-1",
                    event_type="job_started",
                    created_at="2026-05-11T00:00:01+00:00",
                    payload={"status": "started"},
                )

        self.assertTrue(store._warned)

    def _seed_job_design_evidence(self, api: SchedulerClient) -> None:
        submitted = api.submit(
            TrainingJob.create(
                "module:runner",
                "resnet50-baseline",
                "/tmp/resnet50.pt",
                packing=PackingSpec(eligible=True, signature="sig-resnet50", family="cnn"),
                batch_probe=BatchProbeSpec(enabled=True, model_key="resnet50"),
                max_epochs=4,
                runner_kwargs={"batch_size": 32},
            )
        )
        api.upsert_runtime_profile(
            RuntimeProfile.create(
                signature="sig-resnet50",
                hardware_key=_test_hardware_profile().hardware_key,
                backend_name="exclusive",
                resolved_batch_size=32,
                strategy="epoch_1",
                epoch_1_seconds=12.0,
                estimated_total_runtime_seconds=48.0,
                confidence=0.9,
                observations=2,
                last_job_id=submitted.job_id,
            )
        )
        api.upsert_batch_probe_profile(
            BatchProbeProfile(
                probe_key="probe-resnet50",
                model_key="resnet50",
                device_type=_test_hardware_profile().gpu_name,
                shape_signature="shape-resnet50",
                batch_param_name="batch_size",
                resolved_batch_size=32,
                peak_vram_mb=8192,
                memory_total_mb=24576,
                target_budget_mb=23855,
                observations=2,
                last_job_id=submitted.job_id,
            )
        )
        api.upsert_solo_profile(
            SoloProfile(
                signature="sig-resnet50",
                hardware_key=_test_hardware_profile().hardware_key,
                family="cnn",
                peak_vram_mb=8192,
                avg_gpu_utilization=0.72,
                avg_memory_utilization=0.41,
                sample_count=2,
                last_job_id=submitted.job_id,
            )
        )

    def test_search_hardware_returns_current_and_derived_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            api.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature="sig-other",
                    hardware_key="hw-other-002",
                    backend_name="exclusive",
                    resolved_batch_size=16,
                    strategy="epoch_1",
                    epoch_1_seconds=18.0,
                    estimated_total_runtime_seconds=72.0,
                    confidence=0.8,
                    observations=1,
                )
            )

            results = api.search_hardware(limit=10)
            rows = {row["hardware_key"]: row for row in results}
            self.assertIn(_test_hardware_profile().hardware_key, rows)
            self.assertIn("hw-other-002", rows)
            self.assertTrue(rows[_test_hardware_profile().hardware_key]["is_current"])
            self.assertFalse(rows["hw-other-002"]["is_current"])

            filtered = api.search_hardware(query="hw-other-002", limit=10)
            self.assertEqual([row["hardware_key"] for row in filtered], ["hw-other-002"])
            self.assertEqual(api.search_hardware(query="does-not-exist", limit=10), [])

    def test_get_hardware_context_supports_current_explicit_and_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            api.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature="sig-other",
                    hardware_key="hw-other-002",
                    backend_name="exclusive",
                    resolved_batch_size=16,
                    strategy="epoch_1",
                    epoch_1_seconds=18.0,
                    estimated_total_runtime_seconds=72.0,
                    confidence=0.8,
                    observations=1,
                )
            )

            current = api.get_hardware_context()
            self.assertTrue(current["found"])
            self.assertEqual(current["hardware"]["hardware_key"], _test_hardware_profile().hardware_key)
            self.assertIn("safe_vram_budget_mb", current["scheduler_limits"])
            self.assertIn("enabled_backends", current["backend_capabilities"])

            explicit = api.get_hardware_context("hw-other-002")
            self.assertTrue(explicit["found"])
            self.assertEqual(explicit["hardware"]["hardware_key"], "hw-other-002")

            missing = api.get_hardware_context("missing-hw")
            self.assertFalse(missing["found"])
            self.assertIsNone(missing["hardware"])

    def test_get_job_design_context_returns_rich_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            self._seed_job_design_evidence(api)

            context = api.get_job_design_context(
                candidate={
                    "stage": "draft",
                    "task_type": "classification",
                    "model_key": "resnet50",
                    "model_family": "cnn",
                    "packing_family": "cnn",
                    "proposed_batch_size": 32,
                    "proposed_epochs": 4,
                    "requires_gpu": True,
                    "script_signature": "sig-resnet50",
                    "backend_preference": "exclusive",
                    "uses_amp": True,
                    "notes": "baseline resnet50 training job",
                },
                limit=5,
            )

            self.assertTrue(context["hardware_context"]["found"])
            self.assertGreater(len(context["matched_profiles"]), 0)
            self.assertTrue(context["runtime_estimate"]["found"])
            self.assertTrue(context["batch_size_recommendation"]["found"])
            self.assertTrue(context["epoch_recommendation"]["found"])
            self.assertGreater(context["confidence"], 0.0)
            self.assertGreater(len(context["evidence_refs"]), 0)

    def test_get_job_design_context_handles_sparse_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)

            context = api.get_job_design_context(
                candidate={
                    "model_key": "missing-model",
                    "proposed_batch_size": 8,
                    "proposed_epochs": 2,
                    "requires_gpu": True,
                },
                limit=5,
            )

            self.assertEqual(context["matched_profiles"], [])
            self.assertFalse(context["runtime_estimate"]["found"])
            self.assertFalse(context["batch_size_recommendation"]["found"])
            self.assertIn("no_matching_profiles_for_current_hardware", context["risk_flags"])

    def test_get_job_design_context_supports_model_key_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            self._seed_job_design_evidence(api)

            context = api.get_job_design_context(
                candidate={
                    "model_key": "resnet50",
                    "proposed_batch_size": 32,
                    "proposed_epochs": 4,
                    "requires_gpu": True,
                },
                limit=5,
            )

            self.assertTrue(context["batch_size_recommendation"]["found"])
            self.assertTrue(any(entry["kind"] == "batch_probe_profile" for entry in context["matched_profiles"]))

    def test_get_job_design_context_supports_script_signature_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            self._seed_job_design_evidence(api)

            context = api.get_job_design_context(
                candidate={
                    "script_signature": "sig-resnet50",
                    "proposed_batch_size": 32,
                    "requires_gpu": True,
                },
                limit=5,
            )

            self.assertTrue(context["runtime_estimate"]["found"])
            self.assertTrue(any(entry["match_reason"] == "signature_exact" for entry in context["matched_profiles"]))

    def test_get_job_design_context_flags_other_hardware_only_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            api.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature="sig-foreign",
                    hardware_key="hw-other-002",
                    backend_name="exclusive",
                    resolved_batch_size=16,
                    strategy="epoch_1",
                    epoch_1_seconds=18.0,
                    estimated_total_runtime_seconds=72.0,
                    confidence=0.8,
                    observations=1,
                )
            )

            context = api.get_job_design_context(
                candidate={
                    "script_signature": "sig-foreign",
                    "proposed_batch_size": 16,
                    "requires_gpu": True,
                },
                limit=5,
            )

            self.assertEqual(context["matched_profiles"], [])
            self.assertIn("matching_profiles_exist_for_other_hardware_only", context["risk_flags"])

    def test_get_optimization_context_returns_new_agent_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            self._seed_job_design_evidence(api)

            context = api.get_optimization_context(
                candidate={
                    "stage": "improve",
                    "task_type": "vision_training",
                    "model_key": "resnet50",
                    "model_family": "cnn",
                    "script_signature": "sig-resnet50",
                    "proposed_batch_size": 64,
                    "proposed_epochs": 4,
                    "requires_gpu": True,
                    "uses_amp": False,
                    "framework": "pytorch",
                },
                limit=5,
            )

            self.assertTrue(context["hardware_context"]["found"])
            self.assertIn("exact_profiles", context["graph_evidence"])
            self.assertIn("similar_profiles", context["graph_evidence"])
            self.assertIn("packed_profiles", context["graph_evidence"])
            self.assertIn("precision_not_optimized", context["derived_diagnosis"]["profile_symptoms"])
            self.assertIn("enable_tensor_core", context["derived_diagnosis"]["optimization_targets"])
            self.assertIn("recipes", context["vector_evidence"])
            self.assertIn("recommendations", context)
            self.assertIn("risk_flags", context)
            self.assertGreaterEqual(context["confidence"], 0.0)

    def test_new_graph_queries_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api = _sqlite_client_with_test_hardware(tmpdir)
            self._seed_job_design_evidence(api)
            before_events = list(api.list_events())

            api.search_hardware(limit=10)
            api.get_hardware_context()
            api.get_job_design_context(
                candidate={"model_key": "resnet50", "script_signature": "sig-resnet50", "proposed_batch_size": 32},
                limit=5,
            )
            api.get_profile_evidence(
                candidate={"model_key": "resnet50", "script_signature": "sig-resnet50", "proposed_batch_size": 32},
                limit=5,
            )
            api.search_code_knowledge(query="pytorch amp", filters={"framework": "pytorch"}, limit=3)
            api.get_code_optimization_context(
                candidate={"model_key": "resnet50", "script_signature": "sig-resnet50", "proposed_batch_size": 32},
                limit=3,
            )
            api.get_optimization_context(
                candidate={"model_key": "resnet50", "script_signature": "sig-resnet50", "proposed_batch_size": 32},
                limit=3,
            )
            api.search_hardware_features(query="cuda training", limit=3)
            api.get_hardware_feature_context(workload_type="vision_training", limit=3)
            api.get_hardware_optimization_context(
                candidate={"model_key": "resnet50", "script_signature": "sig-resnet50", "proposed_batch_size": 32},
                limit=3,
            )

            after_events = list(api.list_events())
            self.assertEqual(before_events, after_events)

    def test_mcp_server_registers_expected_graph_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            server = build_mcp_server(SchedulerConfig(runtime_root=tmpdir))
        tool_names = set(server._tool_manager._tools.keys())
        self.assertEqual(
            tool_names,
            {
                "get_job_graph_context",
                "search_hardware",
                "get_hardware_context",
                "get_job_design_context",
                "search_profiles",
                "get_profile_evidence",
                "get_runtime_estimate",
                "recommend_batch_size",
                "recommend_epochs",
                "get_packet_compatibility",
                "search_profile_summaries",
                "search_code_knowledge",
                "get_code_optimization_context",
                "get_optimization_context",
                "plan_job_packet",
                "optimize_job_packet",
                "get_model_design_hardware_context",
                "search_hardware_features",
                "get_hardware_feature_context",
                "get_hardware_optimization_context",
                "record_tuning_outcome",
            },
        )


if __name__ == "__main__":
    unittest.main()
