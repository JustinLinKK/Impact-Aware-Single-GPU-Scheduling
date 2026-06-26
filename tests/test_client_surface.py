from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.domain import BatchProbeProfile, BatchResolution, JobRun, RuntimeProfile, TrainingJob
from localml_scheduler.dto import SubmitJobRequest
from localml_scheduler.graph_knowledge import SchedulerKnowledgeBase


class _MemoryCache:
    def __init__(self):
        self.values = {}

    def _key(self, namespace, payload):
        return (namespace, repr(sorted(payload.items())))

    def get(self, namespace, payload):
        return self.values.get(self._key(namespace, payload))

    def set(self, namespace, payload, value):
        self.values[self._key(namespace, payload)] = value

    def invalidate_namespace(self, namespace):
        for key in [key for key in self.values if key[0] == namespace]:
            self.values.pop(key, None)


class _FakeGraphStore:
    def __init__(self):
        self.settings = SchedulerConfig(runtime_root=tempfile.mkdtemp())
        self.hardware_calls = 0
        self.events = []

    def hardware_profile(self):
        self.hardware_calls += 1
        return SimpleNamespace(
            hardware_key="test-hw",
            gpu_name="Test GPU",
            total_vram_mb=24576,
            compute_capability="9.0",
            cuda_runtime="12.4",
            torch_version="2.5.0",
        )

    def list_runtime_profiles(self, **kwargs):
        del kwargs
        return []

    def list_solo_profiles(self, **kwargs):
        del kwargs
        return []

    def list_pair_profiles(self, **kwargs):
        del kwargs
        return []

    def list_batch_size_observations(self, **kwargs):
        del kwargs
        return []

    def list_batch_probe_profiles(self):
        return []

    def list_events(self, *, job_id=None, event_type=None):
        del job_id
        if event_type is None:
            return list(self.events)
        return [event for event in self.events if event.get("event_type") == event_type]


class _FakeHardwareKnowledgeStore:
    def __init__(self):
        self.neighborhood_calls = 0
        self.detail_calls = []

    def get_feature_neighborhood(self, *, hardware_terms, limit):
        self.neighborhood_calls += 1
        return {
            "found": True,
            "hardware": {"hardware_id": "nvidia.blackwell.geforce_rtx_5090.spec", "name": "GeForce RTX 5090"},
            "features": [
                {
                    "feature_id": "bf16",
                    "feature_name": "BF16",
                    "category": "precision_optimization",
                    "support_level": "native",
                    "recommended": True,
                    "performance_impact": "high",
                    "frameworks": ["PyTorch"],
                    "detail_text": "Use bf16 autocast.",
                    "recommended_patterns": ["Use torch.autocast with bf16."],
                },
                {
                    "feature_id": "fp8",
                    "feature_name": "FP8",
                    "category": "precision_optimization",
                    "support_level": "supported",
                    "recommended": False,
                    "performance_impact": "medium",
                    "frameworks": ["PyTorch"],
                    "detail_text": "Use only when transformer engine is available.",
                },
            ],
        }

    def get_feature_details(self, *, hardware_terms, feature_ids, limit):
        self.detail_calls.append((list(hardware_terms), list(feature_ids), limit))
        return {
            "found": True,
            "hardware": {"hardware_id": "nvidia.blackwell.geforce_rtx_5090.spec", "name": "GeForce RTX 5090"},
            "features": [
                {
                    "feature_id": feature_id,
                    "feature_name": feature_id.upper(),
                    "detail_text": f"detail for {feature_id}",
                }
                for feature_id in feature_ids
            ],
            "missing_feature_ids": [],
        }


class SchedulerClientSurfaceTest(unittest.TestCase):
    def test_submit_job_request_accepts_split_spec_and_run_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            seed_job = TrainingJob.create(
                "module:runner",
                "baseline-a",
                "/tmp/a.pt",
                priority=4,
                runner_kwargs={"batch_size": 2},
            )
            request = SubmitJobRequest(
                spec=seed_job.to_job_spec(),
                run=JobRun(metadata={"source": "client-test"}),
            )

            stored = client.submit(request)

            self.assertEqual(stored.job_id, seed_job.job_id)
            self.assertEqual(stored.priority, 4)
            self.assertEqual(stored.metadata["source"], "client-test")
            self.assertIsNotNone(client.inspect(stored.job_id))

    def test_submit_many_submits_full_round_before_polling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            jobs = [
                TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt"),
                TrainingJob.create("module:runner", "baseline-b", "/tmp/b.pt"),
            ]

            submitted = client.submit_many(jobs)

            self.assertEqual([job.job_id for job in submitted], [job.job_id for job in jobs])
            self.assertEqual(len(client.list_jobs()), 2)

    def test_plan_job_packet_returns_per_candidate_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            calls = []

            def fake_context(*, candidate, limit):
                calls.append((candidate, limit))
                return {
                    "hardware_context": {"found": True},
                    "graph_evidence": {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
                    "recommendations": ["Use batch size 8."],
                    "risk_flags": [],
                    "evidence_refs": [f"graph:{candidate['node_id']}"],
                    "confidence": 0.6,
                }

            client.get_optimization_context = fake_context  # type: ignore[method-assign]
            client.get_packet_compatibility = lambda **_: {"found": False, "reason": "unit-test"}  # type: ignore[method-assign]

            packet = client.plan_job_packet(
                candidates=[
                    {"node_id": "n1", "model_key": "m1"},
                    {"node_id": "n2", "model_key": "m2"},
                ],
                limit=3,
            )

            self.assertTrue(packet["found"])
            self.assertEqual(len(packet["jobs"]), 2)
            self.assertEqual(len(calls), 2)
            self.assertIn("graph:n1", packet["evidence_refs"])
            self.assertEqual(len(packet["packet_compatibility"]), 1)

    def test_model_design_hardware_context_ranks_candidate_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            client.get_hardware_context = lambda *_, **__: {  # type: ignore[method-assign]
                "found": True,
                "hardware": {"gpu_name": "RTX 5090", "summary_text": "RTX 5090"},
                "scheduler_limits": {"safe_vram_budget_mb": 24000},
            }
            client.get_hardware_feature_index = lambda *_, **__: {"found": False, "features": []}  # type: ignore[method-assign]
            client.search_code_knowledge = lambda **_: [  # type: ignore[method-assign]
                {
                    "record_id": "bf16-doc",
                    "record_type": "code_doc_chunks",
                    "title": "BF16 tensor core training",
                    "summary_text": "Use bf16 tensor core paths for transformer throughput.",
                    "recommended_patterns": ["Use bf16 autocast."],
                    "confidence": 0.8,
                }
            ]

            context = client.get_model_design_hardware_context(
                workload_type="vision_training",
                candidate_families=["vision_transformer", "convnet"],
                limit=2,
            )

            self.assertTrue(context["found"])
            self.assertEqual(context["model_options"][0]["model_family"], "vision_transformer")
            self.assertIn("code_knowledge:code_doc_chunks:bf16-doc", context["evidence_refs"])
            self.assertGreaterEqual(context["confidence"], 0.8)

    def test_hardware_feature_index_prewarm_and_details_use_neighborhood_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(
                runtime_root=tmpdir,
                graph_db={"enabled": False, "mode": "off"},
                redis_cache={"enabled": True},
            )
            client = SchedulerClient(settings)
            fake_store = _FakeHardwareKnowledgeStore()
            client._hardware_knowledge_store = fake_store
            client._hardware_neighborhood_cache = _MemoryCache()
            client.get_hardware_context = lambda *_, **__: {  # type: ignore[method-assign]
                "found": True,
                "hardware": {
                    "hardware_key": "hw-current",
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "compute_capability": "12.0",
                },
            }

            prewarm = client.prewarm_current_hardware_neighborhood()
            index = client.get_hardware_feature_index()
            details = client.get_hardware_feature_details(feature_ids=["bf16"])

            self.assertTrue(prewarm["ok"])
            self.assertEqual(prewarm["feature_count"], 2)
            self.assertEqual(index["source"], "redis")
            self.assertEqual([item["feature_id"] for item in index["features"]], ["bf16", "fp8"])
            self.assertEqual(details["source"], "redis")
            self.assertEqual([item["feature_id"] for item in details["features"]], ["bf16"])
            self.assertEqual(fake_store.neighborhood_calls, 1)
            self.assertEqual(fake_store.detail_calls, [])

    def test_stage_hardware_features_use_static_pipeline_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SchedulerClient(SchedulerConfig(runtime_root=tmpdir))
            client.get_hardware_context = lambda *_, **__: {  # type: ignore[method-assign]
                "found": True,
                "hardware": {
                    "hardware_key": "current",
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "compute_capability": "12.0",
                },
            }

            context = client.get_stage_hardware_features(pipeline_stage="optimizer", limit=12)

            self.assertTrue(context["found"])
            self.assertEqual(context["stage_filter"], "optimizer")
            feature_ids = [item["feature_id"] for item in context["features"]]
            self.assertIn("soap_optimizer", feature_ids)
            self.assertIn("ademamix_optimizer", feature_ids)
            self.assertNotIn("bf16", feature_ids)
            node = context["stages"][0]["node"]
            stage_keys = {item[0] for item in node["stage_feature_keys"]}
            not_recommended_keys = {item[0] for item in node["not_recommended_feature_keys"]}
            recommended_keys = {item[0] for item in node.get("recommended_feature_keys", [])}
            self.assertIn("soap_optimizer", stage_keys)
            self.assertIn("soap_optimizer", not_recommended_keys)
            self.assertNotIn("soap_optimizer", recommended_keys)
            for row in node["stage_feature_keys"]:
                self.assertEqual(len(row), 2)
                self.assertTrue(row[1])
            self.assertNotIn("node_id", node)
            self.assertIn("recommended_patterns", node)
            self.assertIn("avoid_patterns", node)
            self.assertNotIn("https://", str(node))
            self.assertNotIn("http://", str(node))

    def test_stage_hardware_features_map_composite_workflow_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SchedulerClient(SchedulerConfig(runtime_root=tmpdir))
            client.get_hardware_context = lambda *_, **__: {  # type: ignore[method-assign]
                "found": True,
                "hardware": {
                    "hardware_key": "current",
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "compute_capability": "12.0",
                },
            }

            stage1 = client.get_stage_hardware_features(
                pipeline_stage="stage1_candidate_construction",
                limit=32,
            )
            stage1_ids = {item["feature_id"] for item in stage1["features"]}

            self.assertEqual(stage1["stage_filter"], ["datatype", "model"])
            self.assertIn("dataset_decomposition", stage1_ids)
            self.assertIn("tensor_cores", stage1_ids)
            self.assertIn("sm_120", stage1_ids)
            self.assertNotIn("bf16", stage1_ids)
            self.assertNotIn("muon_optimizer", stage1_ids)

            datatype_precision = client.get_stage_hardware_features(
                pipeline_stage="datatype_precision",
                limit=32,
            )
            datatype_precision_ids = {
                item["feature_id"] for item in datatype_precision["features"]
            }

            self.assertEqual(datatype_precision["stage_filter"], ["datatype", "tuning"])
            self.assertIn("dataset_decomposition", datatype_precision_ids)
            self.assertIn("nvimagecodec_gpu_decode", datatype_precision_ids)
            self.assertIn("bf16", datatype_precision_ids)
            self.assertIn("fp8_rowwise_scaling", datatype_precision_ids)
            self.assertNotIn("tensor_cores", datatype_precision_ids)
            self.assertNotIn("muon_optimizer", datatype_precision_ids)
            self.assertNotIn("gram_newton_schulz_symmetric_gemm", datatype_precision_ids)
            self.assertNotIn("async_tensor_parallel", datatype_precision_ids)

            training = client.get_stage_hardware_features(
                pipeline_stage="training_evaluation",
                limit=32,
            )
            training_ids = {item["feature_id"] for item in training["features"]}

            self.assertEqual(training["stage_filter"], ["optimizer", "tuning"])
            self.assertIn("muon_optimizer", training_ids)
            self.assertIn("gram_newton_schulz_symmetric_gemm", training_ids)
            self.assertIn("bf16", training_ids)
            self.assertIn("fp8_rowwise_scaling", training_ids)
            self.assertIn("async_tensor_parallel", training_ids)

    def test_optimization_context_attaches_candidate_stage_hardware_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SchedulerClient(SchedulerConfig(runtime_root=tmpdir))
            stage_calls = []
            code_context_inputs = []

            client.get_profile_evidence = lambda **_: {  # type: ignore[method-assign]
                "hardware_context": {"found": True},
                "graph_evidence": {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
                "derived_diagnosis": {"profile_symptoms": [], "optimization_targets": []},
                "risk_flags": [],
                "evidence_refs": [],
                "confidence": 0.0,
            }

            def fake_stage_features(hardware_id, *, pipeline_stage, limit):
                stage_calls.append((hardware_id, list(pipeline_stage), limit))
                return {
                    "found": True,
                    "stage_filter": list(pipeline_stage),
                    "stages": [
                        {
                            "stage": "tuning",
                            "features": [{"feature_id": "bf16", "name": "BF16", "category": "precision"}],
                            "feature_count": 1,
                        }
                    ],
                    "features": [{"feature_id": "bf16", "name": "BF16", "category": "precision"}],
                    "feature_count": 1,
                    "source": "unit-test",
                }

            def fake_code_context(*, candidate, graph_context, limit):
                code_context_inputs.append((candidate, graph_context, limit))
                return {"vector_evidence": {}, "evidence_refs": []}

            client.get_stage_hardware_features = fake_stage_features  # type: ignore[method-assign]
            client.get_code_optimization_context = fake_code_context  # type: ignore[method-assign]

            context = client.get_optimization_context(
                candidate={"stage": "training_evaluation", "framework": "pytorch"},
                limit=3,
            )

            self.assertEqual(stage_calls, [("current", ["optimizer", "tuning"], 3)])
            self.assertEqual(code_context_inputs[0][1]["stage_hardware_features"]["stage_filter"], ["optimizer", "tuning"])
            self.assertEqual(context["stage_hardware_features"]["features"][0]["feature_id"], "bf16")

    def test_batch_resolution_apply_updates_runner_kwargs_and_metadata(self) -> None:
        job = TrainingJob.create(
            "module:runner",
            "baseline-a",
            "/tmp/a.pt",
            runner_kwargs={"batch_size": 2},
        )

        updated = BatchResolution.apply(job, 8)

        self.assertEqual(updated.config.runner_kwargs["batch_size"], 8)
        self.assertEqual(updated.metadata["resolved_batch_size"], 8)
        self.assertEqual(updated.metadata["placement_batch_param_name"], "batch_size")
        self.assertEqual(BatchResolution.resolved_batch_size(updated), 8)

    def test_graph_recommendation_surface_uses_persisted_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            job = client.submit(
                TrainingJob.create(
                    "module:runner",
                    "baseline-a",
                    "/tmp/a.pt",
                    max_epochs=4,
                    runner_kwargs={"batch_size": 2},
                )
            )
            client.upsert_batch_probe_profile(
                BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="baseline-a",
                    device_type="test-gpu",
                    shape_signature="shape-1",
                    batch_param_name="batch_size",
                    resolved_batch_size=8,
                    observations=2,
                    last_job_id=job.job_id,
                )
            )
            client.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature or job.job_id,
                    hardware_key=client.store.hardware_key(),
                    backend_name="exclusive",
                    resolved_batch_size=8,
                    strategy="epoch_1",
                    epoch_1_seconds=12.0,
                    estimated_total_runtime_seconds=48.0,
                    confidence=0.9,
                    observations=1,
                    last_job_id=job.job_id,
                )
            )

            recommendation = client.recommend_batch_size(
                model_or_signature="baseline-a",
                hardware="test-gpu",
                shape_signature="shape-1",
                current_batch_size=2,
            )
            runtime_estimate = client.get_runtime_estimate(
                model_or_signature=job.packing.signature or job.job_id,
                batch_size=8,
                backend="exclusive",
            )
            tuning_outcome = client.record_tuning_outcome(
                job_id=job.job_id,
                chosen_batch_size=8,
                chosen_epochs=4,
                recommendation_source="unit-test",
                outcome_metrics={"loss": 0.1},
            )

            self.assertTrue(recommendation["found"])
            self.assertEqual(recommendation["recommended_batch_size"], 8)
            self.assertTrue(runtime_estimate["found"])
            self.assertTrue(tuning_outcome["ok"])
            self.assertEqual(len(client.list_events(job_id=job.job_id, event_type="tuning_outcome_recorded")), 1)

    def test_graph_knowledge_context_uses_redis_cache_when_available(self) -> None:
        store = _FakeGraphStore()
        knowledge = SchedulerKnowledgeBase(store, redis_cache=_MemoryCache())

        first = knowledge.get_hardware_context("current")
        calls_after_first_read = store.hardware_calls
        second = knowledge.get_hardware_context("current")

        self.assertEqual(first, second)
        self.assertGreater(calls_after_first_read, 0)
        self.assertEqual(store.hardware_calls, calls_after_first_read)

    def test_graph_knowledge_preserves_empty_auto_probe_backend_lists(self) -> None:
        store = _FakeGraphStore()
        store.events = [
            {
                "event_type": "scheduler_auto_backend_probe",
                "payload": {
                    "configured_mode": "auto",
                    "effective_scheduler_mode": "parallel_auto_pack",
                    "backend_priority": [],
                    "concurrent_backend_allowlist": [],
                },
            }
        ]
        knowledge = SchedulerKnowledgeBase(store)

        limits = knowledge._scheduler_limits()
        capabilities = knowledge._backend_capabilities()

        self.assertEqual(limits["backend_priority"], [])
        self.assertEqual(limits["concurrent_backend_allowlist"], [])
        self.assertEqual(capabilities["backend_priority"], [])
        self.assertEqual(capabilities["concurrent_backend_allowlist"], [])


if __name__ == "__main__":
    unittest.main()
