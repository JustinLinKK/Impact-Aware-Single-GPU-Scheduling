"""Neo4j-backed hardware capability knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

from .records import load_hardware_knowledge_from_schema

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


_CONSTRAINTS = [
    "CREATE CONSTRAINT hardware_spec_id_unique IF NOT EXISTS FOR (n:HardwareSpec) REQUIRE n.hardware_id IS UNIQUE",
    "CREATE CONSTRAINT hardware_spec_name_key_unique IF NOT EXISTS FOR (n:HardwareSpec) REQUIRE n.name_key IS UNIQUE",
    "CREATE CONSTRAINT hardware_id_unique IF NOT EXISTS FOR (n:Hardware) REQUIRE n.hardware_id IS UNIQUE",
    "CREATE CONSTRAINT hardware_name_key_unique IF NOT EXISTS FOR (n:Hardware) REQUIRE n.name_key IS UNIQUE",
    "CREATE CONSTRAINT feature_id_unique IF NOT EXISTS FOR (n:Feature) REQUIRE n.feature_id IS UNIQUE",
    "CREATE INDEX hardware_spec_lookup IF NOT EXISTS FOR (n:HardwareSpec) ON (n.vendor, n.hardware_type, n.architecture, n.name_key)",
    "CREATE INDEX hardware_spec_name_lookup IF NOT EXISTS FOR (n:HardwareSpec) ON (n.name)",
    "CREATE INDEX hardware_lookup IF NOT EXISTS FOR (n:Hardware) ON (n.vendor, n.hardware_type, n.architecture, n.name_key)",
    "CREATE INDEX hardware_name_lookup IF NOT EXISTS FOR (n:Hardware) ON (n.name)",
    "CREATE INDEX feature_lookup IF NOT EXISTS FOR (n:Feature) ON (n.category, n.maturity, n.name)",
]


class HardwareKnowledgeGraphStore:
    """Store and query hardware capability facts in an isolated graph submodel."""

    def __init__(self, settings: Any, *, driver: Any | None = None):
        self.settings = settings
        self.config = getattr(settings, "hardware_knowledge_graph", None) or getattr(settings, "graph_db", None)
        self._driver = driver

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False)) and str(getattr(self.config, "provider", "neo4j")).lower() == "neo4j"

    def _connect(self) -> Any:
        if self._driver is None:
            if GraphDatabase is None:  # pragma: no cover - exercised only without dependency
                raise RuntimeError("neo4j python driver is not installed")
            password = os.getenv(getattr(self.config, "password_env", "LOCALML_SCHEDULER_NEO4J_PASSWORD"), "")
            username = getattr(self.config, "username", "neo4j")
            auth = (username, password) if username else None
            self._driver = GraphDatabase.driver(getattr(self.config, "uri", "bolt://127.0.0.1:7687"), auth=auth)
        return self._driver

    def _session(self) -> Any:
        database = getattr(self.config, "database", "neo4j") or None
        return self._connect().session(database=database)

    def _run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._session() as session:
            result = session.run(cypher, params or {})
            rows = [record.data() for record in result]
            result.consume()
            return rows

    def _run_write(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._run(cypher, params)

    def ensure_schema(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "hardware knowledge graph disabled"}
        for statement in _CONSTRAINTS:
            self._run_write(statement)
        return {"ok": True, "constraints": len(_CONSTRAINTS)}

    def _wipe(self) -> None:
        self._run_write("MATCH (h) WHERE h:HardwareSpec OR h:Hardware DETACH DELETE h")
        self._run_write("MATCH (f:Feature) WHERE NOT ()-[:HAS_FEATURE]->(f) DELETE f")

    def ingest_bundle(
        self,
        bundle: dict[str, list[dict[str, Any]]],
        *,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        hardware = list(bundle.get("hardware") or [])
        features = list(bundle.get("features") or [])
        relationships = list(bundle.get("relationships") or [])
        counts = {
            "hardware": len(hardware),
            "features": len(features),
            "relationships": len(relationships),
        }
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "would_write": counts,
                "hardware_ids": [item["hardware_id"] for item in hardware],
                "feature_ids": [item["feature_id"] for item in features],
            }
        if not self.enabled:
            return {"ok": False, "reason": "hardware knowledge graph disabled", "would_write": counts}
        schema = self.ensure_schema()
        if recreate:
            self._wipe()
        for item in hardware:
            self._run_write(
                "MERGE (h:HardwareSpec {hardware_id: $hardware_id}) SET h += $props",
                {"hardware_id": item["hardware_id"], "props": item},
            )
        for item in features:
            self._run_write(
                "MERGE (f:Feature {feature_id: $feature_id}) SET f += $props",
                {"feature_id": item["feature_id"], "props": item},
            )
        for item in relationships:
            props = {key: value for key, value in item.items() if key not in {"hardware_id", "feature_id"}}
            self._run_write(
                """
                MATCH (h:HardwareSpec {hardware_id: $hardware_id})
                MATCH (f:Feature {feature_id: $feature_id})
                MERGE (h)-[r:HAS_FEATURE]->(f)
                SET r += $props
                """,
                {"hardware_id": item["hardware_id"], "feature_id": item["feature_id"], "props": props},
            )
        return {
            "ok": True,
            "dry_run": False,
            "written": counts,
            "schema": schema,
            "recreate": bool(recreate),
        }

    def ingest_schema_root(
        self,
        schema_root: str | Path = "schema",
        *,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        bundle = load_hardware_knowledge_from_schema(schema_root)
        result = self.ingest_bundle(bundle, recreate=recreate, dry_run=dry_run)
        result["schema_root"] = str(schema_root)
        return result

    def _query_rows(
        self,
        *,
        hardware: str | None,
        feature: str | None,
        architecture: str | None,
        vendor: str | None,
        framework: str | None,
        row_limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["MATCH (h)-[r:HAS_FEATURE]->(f:Feature)", "WHERE (h:HardwareSpec OR h:Hardware)"]
        params: dict[str, Any] = {}
        if hardware:
            clauses.append(
                """
                AND (
                    toLower(h.hardware_id) CONTAINS $hardware
                    OR toLower(h.name) CONTAINS $hardware
                    OR h.name_key CONTAINS $hardware
                    OR any(alias IN coalesce(h.aliases, []) WHERE toLower(alias) CONTAINS $hardware)
                )
                """
            )
            params["hardware"] = hardware.lower()
        if feature:
            clauses.append(
                """
                AND (
                    toLower(f.feature_id) CONTAINS $feature
                    OR toLower(f.name) CONTAINS $feature
                    OR toLower(f.category) CONTAINS $feature
                )
                """
            )
            params["feature"] = feature.lower()
        if architecture:
            clauses.append("AND toLower(h.architecture) CONTAINS $architecture")
            params["architecture"] = architecture.lower()
        if vendor:
            clauses.append("AND toLower(h.vendor) CONTAINS $vendor")
            params["vendor"] = vendor.lower()
        if framework:
            clauses.append(
                """
                AND (
                    any(item IN coalesce(f.frameworks, []) WHERE toLower(item) = $framework)
                    OR any(item IN coalesce(r.software_requirements, []) WHERE toLower(item) = $framework)
                )
                """
            )
            params["framework"] = framework.lower()
        clauses.append(
            """
            RETURN properties(h) AS hardware, properties(f) AS feature, properties(r) AS relationship
            LIMIT $row_limit
            """
        )
        params["row_limit"] = max(1, int(row_limit))
        return self._run("\n".join(clauses), params)

    def _query_neighborhood_rows(
        self,
        *,
        hardware_terms: list[str],
        feature_ids: list[str] | None = None,
        row_limit: int = 256,
    ) -> list[dict[str, Any]]:
        terms = [str(term or "").strip().lower() for term in hardware_terms if str(term or "").strip()]
        params: dict[str, Any] = {
            "hardware_terms": terms,
            "feature_ids": [str(item) for item in feature_ids or [] if str(item).strip()],
            "row_limit": max(1, int(row_limit)),
        }
        feature_filter = "AND f.feature_id IN $feature_ids" if feature_ids else ""
        if terms:
            hardware_filter = """
                AND any(term IN $hardware_terms WHERE
                    toLower(coalesce(h.hardware_id, '')) = term
                    OR toLower(coalesce(h.hardware_id, '')) CONTAINS term
                    OR toLower(coalesce(h.name, '')) = term
                    OR toLower(coalesce(h.name, '')) CONTAINS term
                    OR toLower(coalesce(h.name_key, '')) = term
                    OR toLower(coalesce(h.name_key, '')) CONTAINS term
                    OR any(alias IN coalesce(h.aliases, []) WHERE toLower(alias) = term OR toLower(alias) CONTAINS term)
                )
            """
        else:
            hardware_filter = "AND false"
        return self._run(
            f"""
            MATCH (h)-[r:HAS_FEATURE]->(f:Feature)
            WHERE (h:HardwareSpec OR h:Hardware)
            {hardware_filter}
            {feature_filter}
            RETURN properties(h) AS hardware, properties(f) AS feature, properties(r) AS relationship
            LIMIT $row_limit
            """,
            params,
        )

    @staticmethod
    def _terms(query: str | None) -> list[str]:
        return [term for term in re.split(r"[^a-z0-9]+", str(query or "").lower()) if len(term) > 1]

    @staticmethod
    def _row_text(row: dict[str, Any]) -> str:
        hardware = row.get("hardware") or {}
        feature = row.get("feature") or {}
        relationship = row.get("relationship") or {}
        values: list[str] = []
        for payload in (hardware, feature, relationship):
            for value in payload.values():
                if isinstance(value, list):
                    values.extend(str(item) for item in value)
                else:
                    values.append(str(value))
        return " ".join(values).lower()

    @staticmethod
    def _impact_score(value: str | None) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)

    def _rank_rows(self, rows: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
        terms = self._terms(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            relationship = row.get("relationship") or {}
            text = self._row_text(row)
            score = float(sum(1 for term in terms if term in text))
            if relationship.get("recommended"):
                score += 1.5
            score += self._impact_score(relationship.get("performance_impact")) * 0.25
            ranked.append((score, row))
        ranked.sort(
            key=lambda item: (
                item[0],
                bool((item[1].get("relationship") or {}).get("recommended")),
                self._impact_score((item[1].get("relationship") or {}).get("performance_impact")),
                str((item[1].get("feature") or {}).get("feature_id") or ""),
            ),
            reverse=True,
        )
        return [row for _, row in ranked]

    def search(
        self,
        *,
        query: str | None = None,
        hardware: str | None = None,
        feature: str | None = None,
        architecture: str | None = None,
        vendor: str | None = None,
        framework: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            rows = self._query_rows(
                hardware=hardware,
                feature=feature,
                architecture=architecture,
                vendor=vendor,
                framework=framework,
                row_limit=max(50, int(limit) * 10),
            )
        except Exception:
            return []
        ranked = self._rank_rows(rows, query)
        return [self._public_result(row) for row in ranked[: max(1, int(limit))]]

    def get_feature_neighborhood(
        self,
        *,
        hardware_terms: list[str],
        limit: int = 256,
    ) -> dict[str, Any]:
        """Return one hardware node and all directly linked feature records."""
        if not self.enabled:
            return {"found": False, "hardware": None, "features": [], "reason": "hardware knowledge graph disabled"}
        try:
            rows = self._query_neighborhood_rows(hardware_terms=hardware_terms, row_limit=limit)
        except Exception as exc:
            return {"found": False, "hardware": None, "features": [], "reason": str(exc)}
        if not rows:
            return {"found": False, "hardware": None, "features": [], "reason": "hardware not found"}
        ranked = self._rank_rows(rows, None)
        hardware = dict((ranked[0].get("hardware") or {}))
        features = [self._public_result(row) for row in ranked]
        return {
            "found": True,
            "hardware": self._public_hardware(hardware),
            "features": features[: max(1, int(limit))],
        }

    def get_feature_index(
        self,
        *,
        hardware_terms: list[str],
        limit: int = 256,
    ) -> dict[str, Any]:
        neighborhood = self.get_feature_neighborhood(hardware_terms=hardware_terms, limit=limit)
        return {
            "found": bool(neighborhood.get("found")),
            "hardware": neighborhood.get("hardware"),
            "features": [self._index_result(item) for item in neighborhood.get("features") or []],
            "reason": neighborhood.get("reason"),
        }

    def get_feature_details(
        self,
        *,
        hardware_terms: list[str],
        feature_ids: list[str],
        limit: int = 64,
    ) -> dict[str, Any]:
        requested = [str(item).strip() for item in feature_ids if str(item).strip()]
        if not requested:
            return {"found": False, "hardware": None, "features": [], "missing_feature_ids": []}
        if not self.enabled:
            return {
                "found": False,
                "hardware": None,
                "features": [],
                "missing_feature_ids": requested,
                "reason": "hardware knowledge graph disabled",
            }
        try:
            rows = self._query_neighborhood_rows(
                hardware_terms=hardware_terms,
                feature_ids=requested,
                row_limit=max(int(limit), len(requested)),
            )
        except Exception as exc:
            return {"found": False, "hardware": None, "features": [], "missing_feature_ids": requested, "reason": str(exc)}
        if not rows:
            return {"found": False, "hardware": None, "features": [], "missing_feature_ids": requested, "reason": "features not found"}
        by_id = {str((row.get("feature") or {}).get("feature_id")): self._public_result(row) for row in rows}
        ordered = [by_id[feature_id] for feature_id in requested if feature_id in by_id]
        hardware = dict((rows[0].get("hardware") or {}))
        return {
            "found": bool(ordered),
            "hardware": self._public_hardware(hardware),
            "features": ordered[: max(1, int(limit))],
            "missing_feature_ids": [feature_id for feature_id in requested if feature_id not in by_id],
        }

    @staticmethod
    def _public_hardware(hardware: dict[str, Any]) -> dict[str, Any]:
        return {
            "hardware_id": hardware.get("hardware_id"),
            "name": hardware.get("name"),
            "name_key": hardware.get("name_key"),
            "aliases": list(hardware.get("aliases") or []),
            "vendor": hardware.get("vendor"),
            "hardware_type": hardware.get("hardware_type"),
            "architecture": hardware.get("architecture"),
            "description": hardware.get("description"),
            "memory_gb": hardware.get("memory_gb"),
            "memory_type": hardware.get("memory_type"),
            "supported_precisions": list(hardware.get("supported_precisions") or []),
            "software_stack": list(hardware.get("software_stack") or []),
        }

    @staticmethod
    def _index_result(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "feature_id": item.get("feature_id"),
            "feature_name": item.get("feature_name") or item.get("title"),
            "category": item.get("category"),
            "support_level": item.get("support_level"),
            "recommended": bool(item.get("recommended")),
            "performance_impact": item.get("performance_impact"),
            "frameworks": list(item.get("frameworks") or []),
            "tags": list(item.get("tags") or []),
            "confidence": item.get("confidence"),
        }

    def _public_result(self, row: dict[str, Any]) -> dict[str, Any]:
        hardware = dict(row.get("hardware") or {})
        feature = dict(row.get("feature") or {})
        relationship = dict(row.get("relationship") or {})
        recommended = relationship.get("hardware_specific_how_to_use") or feature.get("how_to_use")
        limitation = relationship.get("limitations") or feature.get("when_not_to_use")
        hardware_id = hardware.get("hardware_id")
        feature_id = feature.get("feature_id")
        return {
            "record_id": f"{hardware_id}:{feature_id}",
            "record_type": "hardware_knowledge_graph",
            "title": feature.get("name"),
            "summary_text": feature.get("description"),
            "detail_text": feature.get("how_to_use"),
            "hardware_id": hardware_id,
            "hardware_name": hardware.get("name"),
            "feature_id": feature_id,
            "feature_name": feature.get("name"),
            "category": feature.get("category"),
            "frameworks": list(feature.get("frameworks") or []),
            "support_level": relationship.get("support_level"),
            "recommended": bool(relationship.get("recommended")),
            "performance_impact": relationship.get("performance_impact"),
            "software_requirements": list(relationship.get("software_requirements") or []),
            "recommended_patterns": [recommended] if recommended else [],
            "avoid_patterns": [limitation] if limitation else [],
            "sample_code": relationship.get("hardware_specific_sample_code") or feature.get("sample_code"),
            "hardware_match": {
                "hardware_id": hardware_id,
                "name": hardware.get("name"),
                "name_key": hardware.get("name_key"),
                "architecture": hardware.get("architecture"),
                "matched": True,
            },
            "tags": [value for value in (feature_id, feature.get("category"), hardware.get("architecture")) if value],
            "last_verified": relationship.get("last_verified_at"),
            "confidence": 1.0 if relationship.get("verified") else 0.6,
            "evidence_ref": f"hardware_knowledge:HAS_FEATURE:{hardware_id}:{feature_id}",
        }
