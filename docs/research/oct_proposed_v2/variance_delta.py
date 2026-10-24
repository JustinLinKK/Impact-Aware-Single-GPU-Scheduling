"""CPU-only research reference: descriptor drift, NOT an admission decision.

Inputs are already-extracted, fixed-schema features. No model parsing, execution,
identity verification, learned calibration, GPU access or scheduler integration.
Missing observations keep the full distance unknown; partial distances remain
visible. A zero distance never certifies profile reuse or equivalent performance.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

SCHEMA_VERSION = 1
# None means a categorical field. Numeric values are nonnegative, with unit floors.
SCHEMA = {
    "scale": {"parameters": 1, "activation_bytes": 1, "optimizer_bytes": 1,
              "flops_per_update": 1, "input_elements": 1, "launches_per_update": 1},
    "recipe": {"optimizer": None, "precision": None, "compile_mode": None,
               "trainable_mask": None, "checkpointing": None,
               "physical_batch": 1, "accumulation": 1},
    "pipeline": {"data_revision": None, "preprocess_revision": None,
                 "execution_plan": None, "workers": 1},
    "observed": {"median_update_s": 1e-6, "p95_update_s": 1e-6,
                 "mad_update_s": 1e-6, "allocated_peak_bytes": 1,
                 "reserved_peak_bytes": 1, "input_wait_s": 1e-6},
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a finite nonnegative number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("Number exceeds supported range") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative number")
    return value


def scalar_delta(current, reference, floor):
    """Symmetric [0,1] distance, plus asymmetric reference-relative change."""
    current, reference, floor = map(_number, (current, reference, floor))
    if floor == 0:
        raise ValueError("Unit floor must be positive")
    relative = (current-reference) / max(reference, floor)
    if not math.isfinite(relative):
        raise ValueError("Relative change exceeds supported range")
    return {"distance": abs(current-reference) / max(current, reference, floor),
            "signed_relative": relative}


def histogram_delta(current, reference):
    """Raw count distance; caller must certify complete extraction on both sides."""
    for histogram in (current, reference):
        if not isinstance(histogram, dict):
            raise ValueError("Histogram must be a mapping")
        for key, value in histogram.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Histogram keys must be nonempty labels")
            _number(value)
    keys = current.keys() | reference.keys()
    scale = max([*current.values(), *reference.values()], default=0)
    if scale == 0:
        return 0.0
    # Normalize before summing to avoid overflow from individually finite counts.
    total = sum(current.get(k, 0)/scale + reference.get(k, 0)/scale for k in keys)
    return sum(abs(current.get(k, 0)/scale-reference.get(k, 0)/scale) for k in keys)/total


def neighborhood_histogram(nodes, edges, rounds=2):
    """Small directed/edge-typed WL-like histogram, lossy and ID invariant.

    Node labels must encode operator semantics; edges include ports/types. This
    function does not discover shapes, gradients or hidden control flow. Limits
    bound this reference's work; exceeding them means extraction is unavailable.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 3:
        raise ValueError("Use 0 to 3 rounds")
    if not isinstance(nodes, dict) or len(nodes) > 10000:
        raise ValueError("Expected at most 10000 labeled nodes")
    for key, label in nodes.items():
        if not isinstance(key, str) or not isinstance(label, str) or not label:
            raise ValueError("Node IDs and nonempty labels must be strings")
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for index, (source, target, edge_type) in enumerate(edges):
        if index >= 50000:
            raise ValueError("At most 50000 edges")
        if source not in nodes or target not in nodes or not isinstance(edge_type, str):
            raise ValueError("Invalid edge")
        incoming[target].append((edge_type, source))
        outgoing[source].append((edge_type, target))
    def digest(value):
        return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
    labels = {key: digest(label) for key, label in nodes.items()}
    counts = Counter()
    for iteration in range(rounds+1):
        counts.update(f"{iteration}:{label}" for label in labels.values())
        if iteration < rounds:
            labels = {key: digest([labels[key],
                      sorted((kind, labels[other]) for kind, other in incoming[key]),
                      sorted((kind, labels[other]) for kind, other in outgoing[key])])
                      for key in nodes}
    return dict(sorted(counts.items()))


def compare_descriptors(current, reference):
    """Return drift and missing-data diagnostics, with no reuse/admission flag.

    Each component must set complete=True. Structure also supplies histogram;
    others supply values using SCHEMA. None/missing values remain unknown even
    when complete=True. Unknown extra fields require a schema upgrade; they are
    rejected rather than silently ignored. Completeness is a caller assertion,
    not verified by this demonstration.
    """
    allowed = {"schema_version", "structure", *SCHEMA}
    for descriptor in (current, reference):
        if descriptor.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible descriptor schema")
        if set(descriptor) - allowed:
            raise ValueError("Unexpected descriptor fields; version the schema")
    components, unknown = {}, []
    for name in ("structure", *SCHEMA):
        pair = [d.get(name, {}) for d in (current, reference)]
        if any(not isinstance(x, dict) for x in pair):
            raise ValueError("Components must be mappings")
        permitted = {"complete", "histogram" if name == "structure" else "values"}
        if any(set(x)-permitted for x in pair):
            raise ValueError("Unexpected component fields")
        missing, distances, signed = [], {}, {}
        if any(x.get("complete") is not True for x in pair):
            missing.append(f"{name}.coverage")
        if name == "structure":
            if missing or any(x.get("histogram") is None for x in pair):
                missing.append("structure.histogram")
            else:
                distances["histogram"] = histogram_delta(*(x["histogram"] for x in pair))
        else:
            values = [x.get("values", {}) for x in pair]
            if any(not isinstance(x, dict) or set(x)-SCHEMA[name].keys() for x in values):
                raise ValueError("Invalid or unexpected feature fields")
            for field, floor in SCHEMA[name].items():
                a, b = (x.get(field) for x in values)
                if a is None or b is None or a == "" or b == "":
                    missing.append(f"{name}.{field}")
                    continue
                if floor is None:
                    if not isinstance(a, str) or not isinstance(b, str):
                        raise ValueError("Categories must be nonempty strings")
                    distances[field] = float(a != b)
                else:
                    value = scalar_delta(a, b, floor)
                    distances[field] = value["distance"]
                    signed[field] = value["signed_relative"]
        known_max = max(distances.values(), default=None)
        components[name] = {"delta": None if missing else known_max,
                            "known_max": known_max, "fields": distances,
                            "signed_relative": signed, "unknown": missing}
        unknown.extend(missing)
    known = [x["known_max"] for x in components.values() if x["known_max"] is not None]
    return {"schema_version": SCHEMA_VERSION, "components": components,
            "delta": None if unknown else max(known),
            "known_max": max(known, default=None), "unknown": unknown}
