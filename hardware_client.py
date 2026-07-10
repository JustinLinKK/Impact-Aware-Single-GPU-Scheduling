"""Independent hardware-knowledge facade for MLEvolve agents."""

from __future__ import annotations

from typing import Any
import json
import os
from pathlib import Path
import subprocess
import sys

from .client import SchedulerClient
from .config import SchedulerConfig
from .domain import JobSpec, TrainingJob
from .dto import SubmitJobRequest
from .graph_knowledge import SchedulerKnowledgeBase
from .hardware import HardwareProfile


_PROFILE_LIST_METHODS = {
    "list_jobs",
    "list_runtime_profiles",
    "list_solo_profiles",
    "list_pair_profiles",
    "list_batch_probe_profiles",
    "list_batch_size_observations",
    "list_combination_profiles",
}


class _HardwareKnowledgeStateStore:
    """Read-only-ish store proxy that supplies current hardware from a probe."""

    def __init__(self, backend: Any, client: "HardwareKnowledgeClient") -> None:
        self._backend = backend
        self._client = client
        self.settings = getattr(backend, "settings", None)

    def hardware_profile(self) -> HardwareProfile:
        return self._client.hardware_profile()

    def get_job(self, *args: Any, **kwargs: Any) -> Any:
        if not self._client.include_profile_evidence:
            return None
        return self._backend.get_job(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in _PROFILE_LIST_METHODS and not self._client.include_profile_evidence:
            return lambda *args, **kwargs: []
        return getattr(self._backend, name)


class HardwareKnowledgeClient(SchedulerClient):
    """Prompt/evidence client that does not start scheduler services or jobs."""

    def __init__(
        self,
        settings: SchedulerConfig | None = None,
        *,
        include_profile_evidence: bool = True,
        probe_timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__(settings)
        self.include_profile_evidence = bool(include_profile_evidence)
        self.probe_timeout_seconds = max(0.1, float(probe_timeout_seconds or 10.0))
        self._probe_status: dict[str, Any] | None = None
        self._scheduler_client: Any | None = None
        self.store = _HardwareKnowledgeStateStore(self.store, self)
        self.knowledge = SchedulerKnowledgeBase(self.store)
        self.hardware_knowledge_client = None
        self.profile_evidence_used = False

    def attach_scheduler_client(self, scheduler_client: Any | None) -> None:
        self._scheduler_client = scheduler_client

    @property
    def probe_status(self) -> dict[str, Any]:
        return dict(self.probe_current_hardware())

    @property
    def scheduler_context_attached(self) -> bool:
        return self._scheduler_client is not None

    def create_service(self, **kwargs: Any) -> Any:  # pragma: no cover - guardrail
        raise RuntimeError("HardwareKnowledgeClient does not start scheduler services")

    def submit(self, request: SubmitJobRequest | JobSpec | TrainingJob) -> TrainingJob:  # pragma: no cover - guardrail
        raise RuntimeError("HardwareKnowledgeClient cannot submit scheduler jobs")

    def submit_many(self, requests: list[SubmitJobRequest | JobSpec | TrainingJob]) -> list[TrainingJob]:  # pragma: no cover - guardrail
        raise RuntimeError("HardwareKnowledgeClient cannot submit scheduler jobs")

    def probe_current_hardware(self) -> dict[str, Any]:
        if self._probe_status is not None:
            return self._probe_status

        device_index = int(getattr(getattr(self.settings, "gpu_scheduler", None), "device_index", 0) or 0)
        script = (
            "import json, sys\n"
            "from localml_scheduler.hardware import detect_hardware_profile\n"
            "profile = detect_hardware_profile(device_index=int(sys.argv[1]))\n"
            "print(json.dumps(profile.to_dict(), sort_keys=True))\n"
        )
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, str(device_index)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.probe_timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe timed out after {self.probe_timeout_seconds:g}s",
                "stderr": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
            }
            return self._probe_status
        except Exception as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": str(exc),
            }
            return self._probe_status

        if completed.returncode != 0:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe exited with code {completed.returncode}",
                "stderr": (completed.stderr or "")[-1000:],
            }
            return self._probe_status
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe produced invalid JSON: {exc}",
                "stdout": (completed.stdout or "")[-1000:],
                "stderr": (completed.stderr or "")[-1000:],
            }
            return self._probe_status

        self._probe_status = {
            "ok": True,
            "source": "hardware_probe_subprocess",
            "device_index": device_index,
            "hardware_profile": payload,
        }
        return self._probe_status

    def hardware_profile(self) -> HardwareProfile:
        status = self.probe_current_hardware()
        if not status.get("ok"):
            raise RuntimeError(str(status.get("reason") or "hardware probe failed"))
        payload = dict(status.get("hardware_profile") or {})
        return HardwareProfile(
            hardware_key=str(payload.get("hardware_key") or ""),
            os_name=str(payload.get("os_name") or ""),
            gpu_name=str(payload.get("gpu_name") or ""),
            total_vram_mb=payload.get("total_vram_mb"),
            compute_capability=payload.get("compute_capability"),
            cuda_runtime=payload.get("cuda_runtime"),
            torch_version=str(payload.get("torch_version") or ""),
        )

    def get_hardware_context(self, hardware_key: str = "current", include_scheduler_limits: bool = True) -> dict[str, Any]:
        if str(hardware_key or "current") == "current":
            probe = self.probe_current_hardware()
            if not probe.get("ok"):
                return {
                    "found": False,
                    "hardware": None,
                    "accelerator": None,
                    "toolkit": None,
                    "backend_capabilities": {},
                    "scheduler_limits": {},
                    "source": "hardware_probe_subprocess",
                    "hardware_probe_source": "hardware_probe_subprocess",
                    "hardware_probe_success": False,
                    "reason": probe.get("reason"),
                }
        expose_scheduler = bool(include_scheduler_limits and self._scheduler_client is not None)
        result = super().get_hardware_context(hardware_key=hardware_key, include_scheduler_limits=expose_scheduler)
        if not expose_scheduler:
            result["backend_capabilities"] = {}
            result["scheduler_limits"] = {}
        status = self.probe_current_hardware() if str(hardware_key or "current") == "current" else {}
        result["hardware_probe_source"] = status.get("source")
        result["hardware_probe_success"] = status.get("ok")
        result["profile_evidence_enabled"] = self.include_profile_evidence
        return result

    def get_profile_evidence(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        result = super().get_profile_evidence(candidate=candidate, limit=limit)
        graph = result.get("graph_evidence") or {}
        self.profile_evidence_used = any(
            graph.get(key)
            for key in ("exact_profiles", "similar_profiles", "packed_profiles")
        )
        return result
