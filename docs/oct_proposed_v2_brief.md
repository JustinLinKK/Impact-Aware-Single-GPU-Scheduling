# A second scheduler proposal

**Choose jobs by fair progress, and share the GPU only when measured gains repay the cost.**

The current scheduler starts heavy jobs and tries lighter additions. The first
proposal aims to reduce total completion delays. This second proposal, on
`oct_proposed_v2`, balances extra training progress, waiting jobs and trial overhead.

## What we can borrow

| Scheduler | Useful lesson |
|---|---|
| [Gandiva](https://www.usenix.org/system/files/osdi18-xiao.pdf) | Measure how jobs perform together; low busy alone is insufficient. |
| [Gavel](https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf) | Keep performance evidence for each job combination. |
| [Tiresias](https://www.usenix.org/system/files/nsdi19-gu.pdf) | Consider work already received and protect waiting jobs. |
| [Themis](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf) | Evaluate fairness alongside overall efficiency. |

[Salus](https://arxiv.org/abs/1902.04610) also shows that fast switching needs runtime
support. [Shockwave](https://www.usenix.org/system/files/nsdi23-zheng.pdf) motivates
refreshing measurements when training behavior changes. We borrow mechanisms;
their reported speedups are not evidence of a speedup on our GPU.

## How v2 decides

1. **Choose a main job:** within existing priority rules, prefer the job furthest
   behind its reference share of progress. If tied, use remaining time and arrival order.
2. **Choose a partner:** reuse fresh measurements for the exact group. Predictors
   help order new trials; measured progress decides whether sharing works.
3. **Pay for the trial:** skip additions whose likely overlap is too short to
   repay startup and measurement costs. Keep a budget for exploratory trial time.
4. **Protect each job:** reject a group that harms one member too much, even if
   total throughput improves. Memory and backend checks still apply.
5. **Prevent repeated waiting:** reserve a turn for an overdue job. If it cannot
   join, stop adding more work and let incumbents finish. No new fairness-driven
   preemption is assumed.

## A simple example

A runs alone at **1.0×** its solo rate. Adding low-busy B leaves A at **0.55×**
and B at **0.15×**: together they deliver **0.70×**, so reject the pair.

A busier C instead gives A **0.90×** and C **0.75×**: total **1.65×**. Over 120
seconds, a cost of 12 solo-equivalent seconds leaves **66 seconds of extra useful
work**. With only 10 seconds of overlap, the gain is **−5.5**, so skip it.
These are illustrative numbers, not benchmark results; each member must also
pass its slowdown limit.

The model stays on CPU through TorchScript. Branch profiles and live VRAM remain
authoritative, with no fixed job-count cap. This is a research design, not a
deployed scheduler. Its next test is a controlled comparison against baseline,
heavy-first and the first proposal, including trial overhead and tail waiting time.

Full specification: [oct_proposed_v2.md](oct_proposed_v2.md).
