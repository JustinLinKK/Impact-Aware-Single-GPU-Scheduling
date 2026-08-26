"""Role-gated local-first NVIDIA CUDA documentation integration."""

from .client import CudaDocsMCPClient, RemoteCallResult
from .models import (
    CapabilitySupport,
    CudaDocsApplicability,
    CudaDocsContext,
    CudaDocsRequest,
    CudaDocsSettings,
    DocChunk,
    RouteOutcome,
    SourceRef,
)
from .curator import (
    STRUCTURED_RECIPE_JSON_SCHEMA,
    synthesize_structured_recipe_records,
)

__all__ = [
    "CapabilitySupport",
    "CudaDocsApplicability",
    "CudaDocsContext",
    "CudaDocsMCPClient",
    "CudaDocsRequest",
    "CudaDocsService",
    "CudaDocsSettings",
    "DocChunk",
    "RemoteCallResult",
    "RouteOutcome",
    "SourceRef",
    "STRUCTURED_RECIPE_JSON_SCHEMA",
    "synthesize_structured_recipe_records",
]


def __getattr__(name: str):
    """Load the orchestrator lazily to keep bridge/model imports acyclic."""

    if name == "CudaDocsService":
        from .service import CudaDocsService

        return CudaDocsService
    raise AttributeError(name)
