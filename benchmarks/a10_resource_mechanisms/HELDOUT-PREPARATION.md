# Third-iteration controller preparation

This branch prepares the held-out controller while the first recovery experiment
is running. No revised policy has been selected, no third-iteration executable
plan has been frozen, and no A10 work has been dispatched from this branch.

`iteration-3-blueprint.json` preserves the exact predeclared bytes (SHA256
`a95a277a9c5969c9ec698d7bd85c67322f87818f523b4928abf3bee8e17962ce`).
`heldout.candidate_plan` requires an explicit supported revised arm, a complete
64-cohort/384-receipt iteration-two review identity, a source commit and a written
rationale. It always produces candidate status. A syntactically valid review
hash is only an identity: the later deployment handoff must verify the actual
review, source and raw evidence. There is no automatic policy selection.

The controller supports the declared three-arm, four-condition comparison:
48 cohorts and 288 jobs, both feedback modes, two reversed repeats. Every failed
or unattempted cell remains in coverage. Coverage checks against the full sealed
order; a 48-row subset of the 64-row second iteration cannot be relabeled as a
complete third iteration. Correct request denominators remain 12 individual
jobs or four all-required brackets per condition/arm across the two repeats.
The existing four-arm second-iteration path remains covered by regression tests.

Held-out configurations, mixtures, duration targets and seeds must match the
blueprint. An explicit configuration order preserves calibration seed mapping
when a plan is serialized with sorted JSON keys. Source/runtime specifications
and budgets are generated on the actual host from fresh dedicated calibration;
no synthetic unit-test rate or fabricated runtime signature is a deployment
input. The immutable blueprint is checked when sealing and between episodes.

Third-iteration calibration manifests now carry the explicit
`resource_solo_runtime_v1` identity marker. Before GPU calibration starts, every
specification and the manifest must match the controller's actual workload
metadata, including Torch version and source hashes. The table identity adds
that runtime metadata and a direct `torch_version` field to the raw snapshot;
the original before/after source snapshots retain their existing representation.
The runtime is rechecked before fitting the table, and held-out budget generation
rejects tables from another runtime or tables missing this identity. Legacy
calibration manifests retain their original representation.

Offline audit replay must explicitly supply the retained, SHA-verified remote
metadata to `calibration_source_identity`; substituting the analyst machine's
Torch build would be incorrect. This helper validates identity consistency, not
the provenance of the supplied bytes. The operational collector and reviewer
must still verify their file hashes, source/runtime proof and raw measurements.

Both controller sizes retain serial bounded execution, owned-process cleanup,
exclusive output creation, source checks, per-condition comparison statistics
and Gantt-over-update/VRAM contact sheets. The contact-sheet height follows the
actual number of planned cohorts rather than leaving rows for nonexistent jobs.

Still required after the complete second-iteration analysis: select and freeze
the revised policy (or explicitly implement/review a new candidate), bind the
actual source/runtime and calibration identity, adapt the separate operational
package/collector/exporter to the 48-cell protocol, reconcile finite lease and
budget headroom, run actual-host preflight/calibration, and dispatch once. The
local candidate-declaration helper's explicit Torch/source binding is supported
by the source preparation above; the operational report must preserve and verify
it. No actual-host calibration evidence is claimed here.

Local validation: 69 CPU tests passed across the held-out controller, existing
study planner and resource campaign. This includes the actual six-worker CPU
receipt-auditor integration; accelerator measurements are synthetic fixtures and
CUDA is hidden. Run from the task workspace with explicit package/root import
paths so the scheduler's engine.py cannot shadow the parent engine package.
The verification had zero failures, errors or skips. Actual A10 preflight
remains required.

The subsequent runtime-identity change passed 62 focused CPU checks across the
held-out controller, calibration evidence and study planner. These include
runtime/source mismatch rejection, remote-metadata replay, legacy representation
preservation and held-out budget rejection. This run does not repeat the earlier
six-worker integration or claim actual A10 calibration.
