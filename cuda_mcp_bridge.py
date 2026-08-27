"""Merge NVIDIA's CUDA MCP documentation into the hardware knowledge database.

The two knowledge sources are complementary and neither is sufficient alone:

  HWKD  knows what happened on *this* machine -- measured peak VRAM per script
        signature, the scheduler's memory budget, the card's compute
        capability, the installed CUDA and torch versions. It has no idea how
        to fix anything.

  CUDA MCP knows what NVIDIA says -- current, first-party CUDA documentation
        and code samples, served from
        https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs. It has no idea
        which GPU this is, how much VRAM is left, or what this model already
        measured.

Composing them matters because the failures worth fixing are memory failures.
Across the cassava traces in this repo, 15.2% of buggy nodes were CUDA memory
errors (11.6% torch OutOfMemoryError, 3.6% "CUDA error: out of memory"), while
the leaf traces had none -- leaf peaks at 388 MiB and never pressures the card.
A generic documentation answer cannot size a batch for a 31 GB budget, and a
measured VRAM number cannot tell the agent to enable AMP. Together they can.

The second reason to merge rather than co-register: HWKD already carries a
curated knowledge store (hardware_features/seed_records.yaml) that is written by
hand and stamped with a retrieved_or_verified_date. That is precisely the
content CUDA MCP can keep current automatically, with source_refs pointing at
real NVIDIA pages instead of a human's recollection.

Hardware gating is what stops this from being a plain documentation dump.
Records are tagged with the compute capabilities they apply to, so advice that
does not fit the installed card is filtered out before the agent ever sees it.
On the V100 in this project that means TF32 (needs SM 8.0) and FP8 (needs
SM 8.9) guidance is excluded, while AMP fp16, gradient checkpointing, and
channels_last remain. Note that torch reports bf16 as supported on SM 7.0 even
though there is no native tensor-core path, so bf16 is deliberately not treated
as an exclusion here.

This bridge remains transport-independent. The role-gated client under
``localml_scheduler.cuda_docs`` supplies normalized content and exact source
references; this module turns it into source-preserving raw documentation
chunks. Recipe synthesis is intentionally outside the agent critical path.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable

from .backend_mode import (
    BACKEND_NEUTRAL,
    RUNNER_CONTRACT_SUBPROCESS_V1,
    normalize_packing_backend,
)
from .code_knowledge.records import (
    CODE_DOC_SCHEMA_VERSION,
    is_recognized_nvidia_source_url,
)
from .cuda_docs.models import (
    CUDA_DOCS_QUERY_TEMPLATE_VERSION,
    CapabilitySupport,
)

CUDA_MCP_ENDPOINT = "https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs"

# Functional support and native acceleration are distinct. Values are
# (minimum functional capability, minimum native-acceleration capability).
TECHNIQUE_CAPABILITY: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "bf16": ((7, 0), (8, 0)),
    "tf32": ((8, 0), (8, 0)),
    "fp8": ((8, 9), (8, 9)),
    "float8": ((8, 9), (8, 9)),
    "transformer engine": ((8, 9), (8, 9)),
    "flash attention 3": ((9, 0), (9, 0)),
    "thread block cluster": ((9, 0), (9, 0)),
    "tma": ((9, 0), (9, 0)),
}
TECHNIQUE_MIN_CAPABILITY = {
    key: value[0] for key, value in TECHNIQUE_CAPABILITY.items()
}

# Failure signatures seen in this repo's traces, mapped to the documentation
# topic worth asking CUDA MCP about. Derived from the measured taxonomy rather
# than guessed: cassava n=112 gave 11.6% OutOfMemoryError and 3.6% CUDA API
# out-of-memory, and those are the two entries that matter most.
ERROR_TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"CUDA out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY", "reduce peak GPU memory during training"),
    (r"CUDA error: out of memory", "reduce peak GPU memory during training"),
    (
        r"CUBLAS_STATUS_(?:ALLOC_FAILED|EXECUTION_FAILED)",
        "resolve cuBLAS allocation or execution failures",
    ),
    (r"device-side assert", "diagnose device-side assertion in CUDA kernels"),
    (r"no kernel image is available", "match compiled architectures to the installed GPU"),
    (r"cuDNN error|CUDNN_STATUS", "resolve cuDNN failures in convolution workloads"),
    (r"Expected all tensors to be on the same device", "keep tensors on one CUDA device"),
]


@dataclass
class HardwareFacts:
    """What HWKD knows about the machine, used to constrain documentation.

    Attributes:
        gpu_name: e.g. "Tesla V100-SXM2-32GB".
        compute_capability: (major, minor), e.g. (7, 0).
        total_vram_mb: Physical card memory.
        budget_vram_mb: What the scheduler will actually commit, which is
            lower than physical and is the number a batch size must respect.
        cuda_version / torch_version: Toolkit context for the query.
        measured_peak_vram_mb: Highest peak this signature or family has
            actually reached here, from solo_profiles. None when unmeasured.
        measured_samples: How many observations back that peak.
    """

    gpu_name: str = ""
    compute_capability: tuple[int, int] | None = None
    total_vram_mb: float | None = None
    budget_vram_mb: float | None = None
    cuda_version: str = ""
    driver_version: str = ""
    torch_version: str = ""
    gpu_architecture: str = ""
    backend_config_hash: str = ""
    measured_peak_vram_mb: float | None = None
    measured_samples: int = 0
    residual_group_budget_mb: float | None = None
    active_group_usage_mb: float | None = None
    safety_reserve_mb: float | None = None
    backend_overhead_mb: float | None = None
    active_backend_allocation: str = ""

    @property
    def capability_str(self) -> str:
        if not self.compute_capability:
            return ""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    def supports(self, technique: str) -> bool:
        """Whether the installed card meets a technique's minimum capability."""
        need = TECHNIQUE_MIN_CAPABILITY.get(technique.lower())
        if need is None:
            return True
        if self.compute_capability is None:
            return False
        return self.compute_capability >= need

    def support_status(self, technique: str) -> str:
        requirements = TECHNIQUE_CAPABILITY.get(technique.lower())
        if requirements is None or self.compute_capability is None:
            return CapabilitySupport.UNKNOWN.value
        functional, native = requirements
        if self.compute_capability < functional:
            return CapabilitySupport.UNSUPPORTED.value
        if self.compute_capability >= native:
            return CapabilitySupport.NATIVELY_ACCELERATED.value
        return CapabilitySupport.FUNCTIONALLY_SUPPORTED.value

    def excluded_techniques(self) -> list[str]:
        return sorted(
            technique
            for technique in TECHNIQUE_CAPABILITY
            if self.support_status(technique) == CapabilitySupport.UNSUPPORTED.value
        )


def facts_from_knowledge_base(client: Any, *, signature: str | None = None) -> HardwareFacts:
    """Read hardware facts out of a live SchedulerClient / knowledge base.

    Args:
        client: Anything exposing get_hardware_context() and a .store with
            list_solo_profiles(), which SchedulerClient satisfies.
        signature: Packing signature to pull a measured peak for. When absent
            or unmatched, the highest peak across all profiles is used, since
            an over-estimate is the safe direction for a memory budget.
    """
    facts = HardwareFacts()
    try:
        ctx = client.get_hardware_context("current", include_scheduler_limits=True)
    except Exception:
        ctx = {}
    hardware = (ctx.get("hardware") or {}) if isinstance(ctx, dict) else {}
    facts.gpu_name = str(hardware.get("gpu_name") or "")
    facts.total_vram_mb = hardware.get("total_vram_mb")
    facts.cuda_version = str(hardware.get("toolkit_version") or "")
    facts.driver_version = str(hardware.get("driver_version") or "")
    facts.torch_version = str(hardware.get("torch_version") or "")
    facts.gpu_architecture = str(
        hardware.get("architecture") or hardware.get("gpu_architecture") or ""
    )
    cap = hardware.get("compute_capability")
    if isinstance(cap, str) and "." in cap:
        major, _, minor = cap.partition(".")
        if major.strip().isdigit() and minor.strip().isdigit():
            facts.compute_capability = (int(major), int(minor))
    elif isinstance(cap, (list, tuple)) and len(cap) == 2:
        facts.compute_capability = (int(cap[0]), int(cap[1]))
    if not facts.gpu_architecture:
        facts.gpu_architecture = _architecture_from_capability(
            facts.compute_capability
        )
    # Fill gaps from local probes only. These values never originate in the
    # hosted response and are safe stable applicability facts.
    try:
        import torch

        facts.cuda_version = facts.cuda_version or str(torch.version.cuda or "")
        facts.torch_version = facts.torch_version or str(torch.__version__ or "")
        if facts.compute_capability is None and torch.cuda.is_available():
            facts.compute_capability = tuple(
                int(value) for value in torch.cuda.get_device_capability()
            )
            facts.gpu_architecture = facts.gpu_architecture or _architecture_from_capability(
                facts.compute_capability
            )
    except Exception:
        pass
    if not facts.driver_version:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=1.0,
            )
            facts.driver_version = (probe.stdout.splitlines() or [""])[0].strip()
        except Exception:
            pass

    # Mirror ResourceEstimator.safe_budget_mb (resource_estimator.py:43-50):
    # in parallel_time_aware mode the budget is gpu_vram_gib scaled by
    # predicted_budget_fraction, and safe_vram_budget_gib is only the fallback.
    # Quoting the fallback would understate the ceiling by ~2 GB and push the
    # agent to shrink batches further than the scheduler requires.
    limits = (ctx.get("scheduler_limits") or {}) if isinstance(ctx, dict) else {}
    memory = limits.get("memory") or {}
    gpu_vram_gib = memory.get("gpu_vram_gib")
    fraction = memory.get("predicted_budget_fraction")
    if gpu_vram_gib and fraction:
        facts.budget_vram_mb = float(gpu_vram_gib) * 1024.0 * float(fraction)
    elif limits.get("safe_vram_budget_mb"):
        facts.budget_vram_mb = float(limits["safe_vram_budget_mb"])
    elif memory.get("safe_vram_budget_gib"):
        facts.budget_vram_mb = float(memory["safe_vram_budget_gib"]) * 1024.0
    facts.active_group_usage_mb = _optional_number(
        limits.get("active_group_usage_mb")
    )
    facts.safety_reserve_mb = _optional_number(
        limits.get("safety_reserve_mb") or memory.get("safety_reserve_mb")
    )
    facts.backend_overhead_mb = _optional_number(
        limits.get("backend_overhead_mb")
    )
    facts.active_backend_allocation = str(
        limits.get("active_backend_allocation")
        or limits.get("packing_backend")
        or ""
    )
    if facts.budget_vram_mb is not None:
        deductions = sum(
            value or 0.0
            for value in (
                facts.active_group_usage_mb,
                facts.safety_reserve_mb,
                facts.backend_overhead_mb,
            )
        )
        facts.residual_group_budget_mb = max(0.0, facts.budget_vram_mb - deductions)
    backend_config = {
        "packing_backend": limits.get("packing_backend"),
        "runner_contract": limits.get("runner_contract"),
        "mps": limits.get("mps"),
        "cuda_process": limits.get("cuda_process"),
    }
    facts.backend_config_hash = hashlib.sha256(
        json.dumps(
            backend_config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    try:
        profiles = list(client.store.list_solo_profiles())
    except Exception:
        profiles = []
    matched = [p for p in profiles if signature and getattr(p, "signature", None) == signature]
    pool = matched or profiles
    peaks = [float(getattr(p, "peak_vram_mb", 0) or 0) for p in pool]
    peaks = [p for p in peaks if p > 0]
    if peaks:
        facts.measured_peak_vram_mb = max(peaks)
        facts.measured_samples = sum(int(getattr(p, "sample_count", 0) or 0) for p in pool)
    return facts


def _optional_number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _architecture_from_capability(
    capability: tuple[int, int] | None,
) -> str:
    return {
        (7, 0): "volta",
        (7, 2): "volta",
        (7, 5): "turing",
        (8, 0): "ampere",
        (8, 6): "ampere",
        (8, 7): "ampere",
        (8, 9): "ada",
        (9, 0): "hopper",
        (10, 0): "blackwell",
        (10, 1): "blackwell",
        (12, 0): "blackwell",
    }.get(capability, "")


def topic_for_error(error_text: str) -> str | None:
    """Documentation topic implied by a failure, or None if not CUDA-related.

    Returning None matters as much as returning a topic: most failures in this
    repo's traces are syntax errors and missing packages, which CUDA
    documentation cannot help with, and querying for them would only add noise.
    """
    for pattern, topic in ERROR_TOPIC_PATTERNS:
        if re.search(pattern, error_text, re.I):
            return topic
    return None


def build_query(
    topic: str,
    facts: HardwareFacts,
    *,
    effective_backend: str | None = None,
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
) -> str:
    """Compose a CUDA MCP query that carries this machine's constraints.

    Only stable applicability is sent remotely. Measured utilization, paths,
    source code, and exact memory usage are composed locally after retrieval.
    """
    if runner_contract != RUNNER_CONTRACT_SUBPROCESS_V1:
        raise ValueError(
            f"Unsupported runner contract {runner_contract!r}; expected {RUNNER_CONTRACT_SUBPROCESS_V1}"
        )
    parts = [topic + "."]
    if effective_backend:
        parts.append(
            "Execution contract: backend {}, runner {}; return only job-code guidance and leave deployment controls to the scheduler.".format(
                normalize_packing_backend(effective_backend), runner_contract
            )
        )
    if facts.gpu_name:
        cap = f" (compute capability {facts.capability_str})" if facts.capability_str else ""
        parts.append(f"Target GPU is {facts.gpu_name}{cap}.")
    if facts.cuda_version or facts.torch_version:
        parts.append(
            "Toolkit: CUDA {}, PyTorch {}.".format(facts.cuda_version or "unknown", facts.torch_version or "unknown")
        )
    if facts.driver_version:
        parts.append(f"Installed NVIDIA driver is {facts.driver_version}.")
    if facts.gpu_architecture:
        parts.append(f"GPU architecture is {facts.gpu_architecture}.")
    parts.append(
        "Return current first-party NVIDIA documentation context, exact source URLs, and relevant code examples."
    )
    return " ".join(parts)


def compose_local_runtime_facts(facts: HardwareFacts) -> str:
    """Format volatile measured facts locally; this text is never sent remotely."""

    parts: list[str] = []
    if facts.residual_group_budget_mb is not None:
        parts.append(
            f"Residual scheduler group budget: {facts.residual_group_budget_mb:.0f} MB"
        )
    if facts.active_group_usage_mb is not None:
        parts.append(f"active group usage: {facts.active_group_usage_mb:.0f} MB")
    if facts.safety_reserve_mb is not None:
        parts.append(f"safety reserve: {facts.safety_reserve_mb:.0f} MB")
    if facts.backend_overhead_mb is not None:
        parts.append(f"backend overhead: {facts.backend_overhead_mb:.0f} MB")
    if facts.active_backend_allocation:
        parts.append(f"active backend allocation: {facts.active_backend_allocation}")
    if facts.measured_peak_vram_mb is not None:
        parts.append(
            "measured workload peak: {:.0f} MB across {} observation(s)".format(
                facts.measured_peak_vram_mb, facts.measured_samples
            )
        )
    return "; ".join(parts)


def _record_id(
    topic: str,
    facts: HardwareFacts,
    *,
    effective_backend: str | None,
    runner_contract: str,
    source_version: str = "",
    source_identity: str = "",
    content_hash: str = "",
    query_template_version: str = CUDA_DOCS_QUERY_TEMPLATE_VERSION,
    remote_tool_schema_hash: str = "unknown",
) -> str:
    backend = (
        normalize_packing_backend(effective_backend)
        if effective_backend
        else BACKEND_NEUTRAL
    )
    identity = "|".join(
        [
            topic,
            facts.gpu_name,
            facts.gpu_architecture,
            facts.capability_str,
            _major_minor(facts.driver_version),
            _major_minor(facts.cuda_version),
            _major_minor(facts.torch_version),
            backend,
            facts.backend_config_hash,
            runner_contract,
            query_template_version,
            remote_tool_schema_hash,
            source_version,
            source_identity,
            content_hash,
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return f"nvidia.cuda_mcp.{digest}"


def _major_minor(value: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", str(value or ""))
    return f"{match.group(1)}.{match.group(2)}" if match else "unknown"


def to_records(
    *,
    topic: str,
    answer: str,
    facts: HardwareFacts,
    source_refs: Iterable[dict[str, Any]] | None = None,
    verified_date: str,
    confidence: float = 0.7,
    effective_backend: str | None = None,
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
    cache_key: str = "",
    query_template_version: str = CUDA_DOCS_QUERY_TEMPLATE_VERSION,
    remote_tool_schema_hash: str = "unknown",
    allow_unverified: bool = False,
) -> list[dict[str, Any]]:
    """Shape normalized CUDA MCP content into source-preserving doc chunks.

    The record is tagged with this card's compute capability so HWKD's existing
    filters will not serve it to different hardware later, and every record
    keeps its NVIDIA source_refs so a reader can check the claim.

    Raw context is preserved regardless of Markdown structure. Recipe creation
    is an asynchronous curator responsibility and is never inferred here.
    """
    if runner_contract != RUNNER_CONTRACT_SUBPROCESS_V1:
        raise ValueError(
            f"Unsupported runner contract {runner_contract!r}; expected {RUNNER_CONTRACT_SUBPROCESS_V1}"
        )
    unsafe_job_control = re.search(
        r"(?:start|stop|configure|set).{0,80}(?:MPS daemon|active.thread percentage|client priority|compute mode)|"
        r"(?:CUDA_MPS_[A-Z_]+|nvidia-cuda-mps-control)|"
        r"(?:cross.job|between jobs|multiple jobs).{0,80}(?:torch\.cuda\.Stream|shared CUDA context)|"
        r"(?:torch\.cuda\.Stream|shared CUDA context).{0,80}(?:cross.job|between jobs|multiple jobs)",
        answer,
        re.I | re.S,
    )
    if unsafe_job_control:
        return []
    text = str(answer or "").strip()
    if not text:
        return []
    refs = [dict(item) for item in (source_refs or []) if isinstance(item, dict)]
    verified_refs = [
        item
        for item in refs
        if is_recognized_nvidia_source_url(str(item.get("url") or ""))
    ]
    if not verified_refs and not allow_unverified:
        return []
    if not refs:
        return []
    for ref in refs:
        ref.setdefault("retrieved_or_verified_date", verified_date)
        ref.setdefault("source_type", "vendor_documentation")
    primary = verified_refs[0] if verified_refs else refs[0]
    source_version = str(
        primary.get("source_version") or primary.get("version") or ""
    )
    backend = (
        normalize_packing_backend(effective_backend)
        if effective_backend
        else BACKEND_NEUTRAL
    )
    applicability_support = _classify_applicability(text, facts)
    statuses = set(applicability_support.values())
    if CapabilitySupport.UNSUPPORTED.value in statuses:
        support_status = CapabilitySupport.UNSUPPORTED.value
    elif CapabilitySupport.FUNCTIONALLY_SUPPORTED.value in statuses:
        support_status = CapabilitySupport.FUNCTIONALLY_SUPPORTED.value
    elif statuses and statuses == {CapabilitySupport.NATIVELY_ACCELERATED.value}:
        support_status = CapabilitySupport.NATIVELY_ACCELERATED.value
    else:
        support_status = CapabilitySupport.UNKNOWN.value
    record_id = _record_id(
        topic,
        facts,
        effective_backend=effective_backend,
        runner_contract=runner_contract,
        source_version=source_version,
        source_identity=str(primary.get("url") or ""),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        query_template_version=query_template_version,
        remote_tool_schema_hash=remote_tool_schema_hash,
    )
    record = {
        "schema_version": CODE_DOC_SCHEMA_VERSION,
        "chunk_id": record_id,
        "title": f"{topic} on {facts.gpu_name or 'CUDA GPU'}",
        "text": text[:32000],
        "source_id": record_id,
        "source_type": "nvidia_cuda_docs" if verified_refs else "unverified_cuda_docs",
        "source_title": str(primary.get("title") or "NVIDIA CUDA documentation"),
        "source_url": str(primary.get("url") or ""),
        "source_version": source_version,
        "source_refs": refs,
        "retrieved_or_verified_date": verified_date,
        "vendor": "nvidia",
        "frameworks": ["pytorch"],
        "framework": "pytorch",
        "framework_version": facts.torch_version,
        "framework_versions": [facts.torch_version] if facts.torch_version else [],
        "toolkits": ["cuda"],
        "toolkit_versions": [facts.cuda_version] if facts.cuda_version else [],
        "driver_versions": [facts.driver_version] if facts.driver_version else [],
        "backend_modes": [backend],
        "backend_keys": [backend],
        "runner_contracts": [runner_contract],
        "pipeline_stages": [
            "model_design",
            "datatype_precision",
            "training_evaluation",
        ],
        "rule_type": "recommendation",
        "owner": "job_code",
        "strength": "preferred",
        "transferability": (
            "backend_neutral" if backend == BACKEND_NEUTRAL else "exact_backend"
        ),
        "confidence": confidence,
        "gpu_architectures": [facts.gpu_architecture] if facts.gpu_architecture else [],
        "backend_config_hash": facts.backend_config_hash or "unknown",
        "query_template_version": query_template_version,
        "remote_tool_schema_hash": remote_tool_schema_hash,
        "cuda_docs_cache_key": cache_key,
        "applicability": {
            "gpu_architecture": facts.gpu_architecture,
            "compute_capability": facts.capability_str or "unknown",
            "driver_major_minor": _major_minor(facts.driver_version),
            "cuda_major_minor": _major_minor(facts.cuda_version),
            "framework": "pytorch",
            "framework_major_minor": _major_minor(facts.torch_version),
            "backend_mode": backend,
            "backend_config_hash": facts.backend_config_hash or "unknown",
            "runner_contract": runner_contract,
            "remote_tool_schema_hash": remote_tool_schema_hash,
        },
        "support_status": support_status,
        "applicability_support": applicability_support,
        "verified_source": bool(verified_refs),
    }
    if facts.capability_str:
        record["compute_capabilities"] = [facts.capability_str]
    if facts.gpu_name:
        record["accelerator_names"] = [facts.gpu_name.lower().replace(" ", "_")]
    return [record]


def _classify_applicability(text: str, facts: HardwareFacts) -> dict[str, str]:
    lowered = text.lower()
    return {
        technique: facts.support_status(technique)
        for technique in TECHNIQUE_CAPABILITY
        if technique in lowered
    }


def ingest(store: Any, records: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
    """Write records into HWKD's CodeKnowledgeStore.

    Once written, HWKD's existing MCP tools (search_code_knowledge,
    get_code_optimization_context and get_optimization_context serve this
    content joined with measured profiles, so the agent reaches NVIDIA guidance
    and this machine's numbers through one query.
    """
    if not records:
        return {"ingested": 0, "skipped": "no records"}
    return store.ingest_records(records, dry_run=dry_run)
