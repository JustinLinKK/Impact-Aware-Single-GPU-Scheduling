# October 24 scheduler research proposal v2

1. Read the implemented busy-anchor policy and the separate oct_proposed completion-time proposal.
2. Research primary scheduler papers for single-GPU transferable mechanisms, distinguishing published behavior from our adaptation.
3. Design a distinct policy using measured normalized progress, persistent service deficit, exact-group evidence, and bounded exploration; keep CPU TorchScript predictions as hints and profile/live-memory admission authoritative.
4. Specify mathematical units, failure and recovery cases, integration points, worked examples, and a fair evaluation plan without disturbing active training or deploying a scheduler.
5. Verify source links, analytical examples, document consistency, branch isolation, and absence of runtime edits. Commit the reviewable proposal on local oct_proposed_v2 and provide a short readable document.

Scope: research/design only, based on paper commit 607d5e2 in a separate worktree. The uncommitted busy-anchor/student implementation remains in the paper checkout and is a future integration dependency; it is not silently included here. Existing oct_proposed is preserved. No GPU experiments or production changes are part of this task.

Completed: six primary sources reviewed (Gandiva, Tiresias, Themis, Gavel, Salus and Shockwave). Added full proposal, short brief, source/applicability ledger and an 11-check arithmetic verifier. Independent read-only review identified and resolved the least-attained-service degeneracy for never-started jobs, durable service accounting, trial-versus-fairness preemption distinction, planning-horizon versus polling distinction, and lack of a guaranteed safe-pause deadline. Proposed empirical envelopes are not claimed as statistical confidence guarantees. Final review found no remaining material contradiction after the two last clarifications.

Verification: arithmetic examples pass; primary source URLs and research/adaptation scope are recorded; only new documentation, source notes and verification files are staged. Original paper checkout stays on paper with its existing dirty work, and oct_proposed remains at 05b3e86. A one-page PDF and editable Word brief were exported to the user's outputs folder; PDF page count, text and rendered layout were checked. No scheduler or GPU experiment was run, so no measured performance improvement is claimed.

## Agent-generated workload revision

User correction: MLE Agent produces evolving candidate programs, so an exact historical sharing group may rarely recur. Review actual MLEvolve lineage/mutation/submission paths, research unknown-workload scheduling and graph/performance shift, and revise v2 in place. Define an auditable variance delta that separates architecture, workload, runtime and measurement drift; unknown fields must not imply similarity. Add a bounded CPU-only reference implementation and tests for the drift contract if useful, clearly separate from live admission or calibrated performance claims. Preserve active training and existing scheduler runtime. Export an updated short explanation and commit this revision on oct_proposed_v2.


Revision completed: audited local MLEvolve mutation/lineage/submission and scheduler
profile paths; reviewed seven additional papers plus official PyTorch allocator
scope (13 papers, 2 runtime documentation sources overall). Revised the main policy
around novel-candidate measurements rather than likely exact-group reuse. Added
an agent-workload evidence contract and a bounded, CPU-only drift reference with
19 passing tests. No extractor, calibrated reuse threshold, group predictor or
live scheduling policy is implemented. The original 11 arithmetic checks still
pass. Independent review resolved cold-start eligibility/containment, unpriced
progress credit, composite duration, exact lesson-bucket semantics, empty
histograms and intrinsic I/O accounting. It found no other material blockers.

Updated the one-page PDF and Word brief (402 extracted PDF words) in outputs;
page count, text and rendered layout were checked. Runtime/GPU jobs and both
other scheduler checkouts remain outside this revision. The branch contains a
reviewable research contract, not measured evidence of scheduler improvement.
