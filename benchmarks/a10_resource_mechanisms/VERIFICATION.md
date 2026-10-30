# Candidate implementation verification

Checked locally on October 30, 2026, before any iteration-2 GPU dispatch.

- 100 focused CPU checks passed after the final reporting changes: resource
  campaign, study manifests, calibration evidence/cleanup, profile identities,
  and bounded measurement windows. The expensive CPU subprocess integration
  was excluded from this repeated run because it had already passed separately.
- The full campaign auditor passed a real six-worker CPU episode using the
  incremental policy, exact two-epoch budgets and two causal brackets. All GPU
  readings in that test were explicitly labeled fixtures, with CUDA disabled.
- A prior integration run covered all four scheduler arms with actual CPU
  subprocesses. Together with existing add-only trial and asynchronous-cache
  checks, that run passed 61 tests.
- Deterministic epoch-permutation caching preserves the original per-update
  sample hashes. Existing pause/resume training-state tests and designed
  workload tests passed in the separate 13-test runner check.
- Critical design review required stronger profile/cache identity binding,
  source/phase/process evidence, allocator peaks across launches and bounded
  cleanup. Those changes are implemented. Exact rented-host namespace mapping
  and actual CUDA measurements remain to be verified during preflight and
  dedicated calibration. Unknown GPU process ownership fails calibration.

The plan is still `candidate_protocol_pending_iteration1_review`; the campaign
controller refuses to launch that status. Its final bounds, source commit and
sealed manifests must be recorded after the iteration-1 review. This document
does not claim A10 performance, a better neural predictor, or completion of the
three-iteration experiment.

Focused command from the parent worktree:

```bash
PYTHONPATH=/home/justin/MLEvolve/worktrees/a10_causal_iteration2:/home/justin/MLEvolve \
  /home/justin/MLEvolve/.venv/bin/python -m pytest \
  --rootdir=localml_scheduler --confcutdir=localml_scheduler/tests --import-mode=importlib -q \
  localml_scheduler/tests/test_a10_resource_campaign.py \
  localml_scheduler/tests/test_a10_resource_study.py \
  localml_scheduler/tests/test_a10_resource_calibration.py \
  localml_scheduler/tests/test_a10_resource_profiles.py \
  localml_scheduler/tests/test_calibration_window_bound.py \
  -k 'not real_cpu_resource'
```
