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

This module does not call the MCP server itself; the caller supplies the
answer text, because the CUDA MCP server requires an interactive NVIDIA
Developer login that a headless run cannot perform. Everything downstream of
that -- query construction, hardware gating, record shaping, ingestion -- is
handled here and is testable without network access.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

CUDA_MCP_ENDPOINT = "https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs"

OPTIMIZATION_RECIPE_SCHEMA_VERSION = "optimization_recipe_chunk_v1"

# Minimum compute capability each technique needs. A record mentioning one of
# these is dropped when the installed card cannot run it, which is the whole
# point of gating documentation on measured hardware.
TECHNIQUE_MIN_CAPABILITY: dict[str, tuple[int, int]] = {
    "tf32": (8, 0),
    "fp8": (8, 9),
    "float8": (8, 9),
    "transformer engine": (8, 9),
    "flash attention 3": (9, 0),
    "thread block cluster": (9, 0),
    "tma": (9, 0),
}

# Failure signatures seen in this repo's traces, mapped to the documentation
# topic worth asking CUDA MCP about. Derived from the measured taxonomy rather
# than guessed: cassava n=112 gave 11.6% OutOfMemoryError and 3.6% CUDA API
# out-of-memory, and those are the two entries that matter most.
ERROR_TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"CUDA out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY", "reduce peak GPU memory during training"),
    (r"CUDA error: out of memory", "reduce peak GPU memory during training"),
    (r"CUBLAS_STATUS_ALLOC_FAILED", "reduce workspace memory for cuBLAS matmul"),
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
    torch_version: str = ""
    measured_peak_vram_mb: float | None = None
    measured_samples: int = 0

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

    def excluded_techniques(self) -> list[str]:
        return sorted(t for t in TECHNIQUE_MIN_CAPABILITY if not self.supports(t))


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
    facts.torch_version = str(hardware.get("torch_version") or "")
    cap = hardware.get("compute_capability")
    if isinstance(cap, str) and "." in cap:
        major, _, minor = cap.partition(".")
        if major.strip().isdigit() and minor.strip().isdigit():
            facts.compute_capability = (int(major), int(minor))
    elif isinstance(cap, (list, tuple)) and len(cap) == 2:
        facts.compute_capability = (int(cap[0]), int(cap[1]))

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


def build_query(topic: str, facts: HardwareFacts) -> str:
    """Compose a CUDA MCP query that carries this machine's constraints.

    The measured numbers are what a bare documentation search cannot supply,
    so they go in the query rather than being applied to the answer afterwards.
    """
    parts = [topic + "."]
    if facts.gpu_name:
        cap = f" (compute capability {facts.capability_str})" if facts.capability_str else ""
        parts.append(f"Target GPU is {facts.gpu_name}{cap}.")
    if facts.cuda_version or facts.torch_version:
        parts.append(
            "Toolkit: CUDA {}, PyTorch {}.".format(facts.cuda_version or "unknown", facts.torch_version or "unknown")
        )
    if facts.budget_vram_mb:
        parts.append(f"The scheduler commits at most {facts.budget_vram_mb:.0f} MB of VRAM per job.")
    if facts.measured_peak_vram_mb:
        parts.append(
            "This workload has measured a peak of {:.0f} MB across {} observations on this hardware.".format(
                facts.measured_peak_vram_mb, facts.measured_samples
            )
        )
    excluded = facts.excluded_techniques()
    if excluded:
        parts.append("Exclude techniques unavailable on this card: " + ", ".join(excluded) + ".")
    parts.append("Answer with concrete PyTorch changes and cite the CUDA documentation section.")
    return " ".join(parts)


def _record_id(topic: str, facts: HardwareFacts) -> str:
    digest = hashlib.sha1(f"{topic}|{facts.gpu_name}|{facts.capability_str}".encode()).hexdigest()[:12]
    return f"nvidia.cuda_mcp.{digest}"


def _split_patterns(answer: str) -> tuple[list[str], list[str]]:
    """Pull recommended and avoid bullets out of a documentation answer.

    Bullets that read as prohibitions become avoid_patterns; the rest become
    recommended_patterns. Anything unbulleted stays in the summary only.
    """
    recommended: list[str] = []
    avoid: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] not in "-*•":
            continue
        text = stripped.lstrip("-*• ").strip()
        if not text:
            continue
        if re.match(r"(do not|don't|avoid|never|not supported|unsupported)\b", text, re.I):
            avoid.append(text)
        else:
            recommended.append(text)
    return recommended, avoid


def to_records(
    *,
    topic: str,
    answer: str,
    facts: HardwareFacts,
    source_refs: Iterable[dict[str, Any]] | None = None,
    verified_date: str,
    confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """Shape a CUDA MCP answer into HWKD optimization-recipe records.

    The record is tagged with this card's compute capability so HWKD's existing
    filters will not serve it to different hardware later, and every record
    keeps its NVIDIA source_refs so a reader can check the claim.

    Returns an empty list when the answer carries no actionable bullets, since
    an empty recipe would dilute search results without adding knowledge.
    """
    recommended, avoid = _split_patterns(answer)
    if not recommended and not avoid:
        return []

    # Drop guidance the installed card cannot execute. Doing this at ingestion
    # keeps the store honest for every later query, rather than re-filtering
    # at each read.
    def _keep(line: str) -> bool:
        lowered = line.lower()
        return all(facts.supports(t) for t in TECHNIQUE_MIN_CAPABILITY if t in lowered)

    recommended = [line for line in recommended if _keep(line)]

    summary = answer.strip().split("\n\n")[0][:600]
    # The validator keys each schema by its own id field; optimization recipes
    # use recipe_id, not a generic record_id (code_knowledge/records.py:18-22).
    record = {
        "schema_version": OPTIMIZATION_RECIPE_SCHEMA_VERSION,
        "recipe_id": _record_id(topic, facts),
        "title": f"{topic} on {facts.gpu_name or 'CUDA GPU'}",
        "problem_statement": topic,
        "solution_summary": summary,
        "text": summary,
        "optimization_targets": ["gpu_memory"] if "memory" in topic.lower() else ["correctness"],
        "recommended_patterns": recommended,
        "avoid_patterns": avoid,
        "vendor": "nvidia",
        "frameworks": ["pytorch"],
        "toolkits": ["cuda"],
        "confidence": confidence,
    }
    if facts.capability_str:
        record["compute_capabilities"] = [facts.capability_str]
    if facts.gpu_name:
        record["accelerator_names"] = [facts.gpu_name.lower().replace(" ", "_")]
    refs = list(source_refs or [])
    if not refs:
        refs = [
            {
                "title": "NVIDIA CUDA MCP Server",
                "url": CUDA_MCP_ENDPOINT,
                "source_type": "vendor_mcp_server",
                "retrieved_or_verified_date": verified_date,
            }
        ]
    else:
        for ref in refs:
            ref.setdefault("retrieved_or_verified_date", verified_date)
            ref.setdefault("source_type", "vendor_documentation")
    record["source_refs"] = refs
    return [record]


def ingest(store: Any, records: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
    """Write records into HWKD's CodeKnowledgeStore.

    Once written, HWKD's existing MCP tools (search_code_knowledge,
    get_code_optimization_context, get_hardware_optimization_context) serve this
    content joined with measured profiles, so the agent reaches NVIDIA guidance
    and this machine's numbers through one query.
    """
    if not records:
        return {"ingested": 0, "skipped": "no records"}
    return store.ingest_records(records, dry_run=dry_run)
