"""Normalize MCP content blocks without assuming Markdown response shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import re

from ..code_knowledge.records import (
    canonicalize_nvidia_source_url,
    is_recognized_nvidia_source_url,
)
from .models import CapabilitySupport, DocChunk, SourceRef

_URL = re.compile(r"https://[^\s<>\"')\]]+", re.I)
_TEXT_KEYS = ("text", "content", "snippet", "body", "summary", "markdown")
_URL_KEYS = ("url", "source_url", "document_url", "uri", "href")
_TITLE_KEYS = ("title", "source_title", "document_title", "name")


@dataclass(frozen=True, slots=True)
class NormalizedCudaDocsResult:
    chunks: tuple[DocChunk, ...]
    source_refs: tuple[SourceRef, ...]
    raw_chars: int
    rejected_reason: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.chunks and self.source_refs and not self.rejected_reason)


def normalize_mcp_result(
    result: Any,
    *,
    retrieved_date: str,
    max_raw_chars: int = 32000,
    max_chunk_chars: int = 4000,
    max_chunks: int = 8,
) -> NormalizedCudaDocsResult:
    """Extract bounded source-labelled chunks from SDK or mapping results."""

    structured = _value(result, "structuredContent", "structured_content")
    content = _value(result, "content")
    candidates: list[tuple[str, str, str, str]] = []
    if structured is not None:
        candidates.extend(_walk_candidates(structured))
    if content is not None:
        for block in content if isinstance(content, (list, tuple)) else [content]:
            block_type = str(_value(block, "type") or "")
            if block_type and block_type not in {"text", "resource", "resource_link"}:
                continue
            text = str(_value(block, "text") or "")
            resource = _value(block, "resource")
            if resource is not None:
                text = text or str(_value(resource, "text") or "")
            url = str(
                _value(block, "uri", "url")
                or (_value(resource, "uri", "url") if resource is not None else "")
                or ""
            )
            title = str(_value(block, "title", "name") or "")
            if text:
                candidates.extend(_text_candidates(text, url=url, title=title))
    if not candidates and isinstance(result, str):
        candidates.extend(_text_candidates(result))
    if not candidates and isinstance(result, dict):
        candidates.extend(_walk_candidates(result))

    deduped: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    raw_chars = 0
    for text, url, title, version in candidates:
        text = _bounded_text(text, max_chunk_chars)
        raw_chars += len(text)
        if raw_chars > max_raw_chars:
            remaining = max(0, max_raw_chars - (raw_chars - len(text)))
            text = text[:remaining]
        url = canonicalize_nvidia_source_url(url.strip().rstrip(".,;:"))
        if not text or not is_recognized_nvidia_source_url(url):
            continue
        key = (hashlib.sha256(text.encode()).hexdigest(), url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, url, title.strip(), version.strip()))
        if len(deduped) >= max(1, int(max_chunks)) or raw_chars >= max_raw_chars:
            break

    if not deduped:
        return NormalizedCudaDocsResult(
            chunks=(),
            source_refs=(),
            raw_chars=min(raw_chars, max_raw_chars),
            rejected_reason="missing_valid_nvidia_source_url_or_text",
        )

    chunks: list[DocChunk] = []
    refs: list[SourceRef] = []
    seen_refs: set[str] = set()
    for text, url, title, version in deduped:
        display_title = title or _title_from_url(url)
        chunk_id = (
            "cuda-doc:"
            + hashlib.sha256(f"{url}|{version}|{text}".encode()).hexdigest()[:24]
        )
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                text=text,
                title=display_title,
                source_url=url,
                source_version=version,
                retrieved_or_verified_date=retrieved_date,
                support_status=CapabilitySupport.UNKNOWN.value,
            )
        )
        if url not in seen_refs:
            refs.append(
                SourceRef(
                    title=display_title,
                    url=url,
                    source_version=version,
                    retrieved_or_verified_date=retrieved_date,
                )
            )
            seen_refs.add(url)
    return NormalizedCudaDocsResult(
        chunks=tuple(chunks),
        source_refs=tuple(refs),
        raw_chars=min(raw_chars, max_raw_chars),
    )


def _walk_candidates(value: Any) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    if isinstance(value, dict):
        text = next(
            (str(value[key]) for key in _TEXT_KEYS if isinstance(value.get(key), str)),
            "",
        )
        url = next(
            (str(value[key]) for key in _URL_KEYS if isinstance(value.get(key), str)),
            "",
        )
        title = next(
            (str(value[key]) for key in _TITLE_KEYS if isinstance(value.get(key), str)),
            "",
        )
        version = str(value.get("source_version") or value.get("version") or "")
        if text:
            rows.extend(_text_candidates(text, url=url, title=title, version=version))
        for child in value.values():
            if isinstance(child, (dict, list, tuple)):
                rows.extend(_walk_candidates(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_walk_candidates(child))
    return rows


def _text_candidates(
    text: str,
    *,
    url: str = "",
    title: str = "",
    version: str = "",
) -> list[tuple[str, str, str, str]]:
    urls = [url] if url else _URL.findall(text)
    return [(text, item, title, version) for item in dict.fromkeys(urls) if item]


def _bounded_text(text: str, limit: int) -> str:
    value = str(text or "").replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[: max(1, int(limit))]


def _value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _title_from_url(url: str) -> str:
    path = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return (
        path.replace("-", " ").replace("_", " ").strip() or "NVIDIA CUDA documentation"
    )[:160]
