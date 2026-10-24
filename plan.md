# October 24 scheduler research proposal v2

1. Read the implemented busy-anchor policy and the separate oct_proposed completion-time proposal.
2. Research primary scheduler papers for single-GPU transferable mechanisms, distinguishing published behavior from our adaptation.
3. Design a distinct policy using measured normalized progress, persistent service deficit, exact-group evidence, and bounded exploration; keep CPU TorchScript predictions as hints and profile/live-memory admission authoritative.
4. Specify mathematical units, failure and recovery cases, integration points, worked examples, and a fair evaluation plan without disturbing active training or deploying a scheduler.
5. Verify source links, analytical examples, document consistency, branch isolation, and absence of runtime edits. Commit the reviewable proposal on local oct_proposed_v2 and provide a short readable document.

Scope: research/design only, based on paper commit 607d5e2 in a separate worktree. The uncommitted busy-anchor/student implementation remains in the paper checkout and is a future integration dependency; it is not silently included here. Existing oct_proposed is preserved. No GPU experiments or production changes are part of this task.

Completed: six primary sources reviewed (Gandiva, Tiresias, Themis, Gavel, Salus and Shockwave). Added full proposal, short brief, source/applicability ledger and an 11-check arithmetic verifier. Independent read-only review identified and resolved the least-attained-service degeneracy for never-started jobs, durable service accounting, trial-versus-fairness preemption distinction, planning-horizon versus polling distinction, and lack of a guaranteed safe-pause deadline. Proposed empirical envelopes are not claimed as statistical confidence guarantees. Final review found no remaining material contradiction after the two last clarifications.

Verification: arithmetic examples pass; primary source URLs and research/adaptation scope are recorded; only new documentation, source notes and verification files are staged. Original paper checkout stays on paper with its existing dirty work, and oct_proposed remains at 05b3e86. A one-page PDF and editable Word brief were exported to the user's outputs folder; PDF page count, text and rendered layout were checked. No scheduler or GPU experiment was run, so no measured performance improvement is claimed.
