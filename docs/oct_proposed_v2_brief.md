# Scheduling models an agent has never tried

**Revised v2: use past runs as clues, then measure the new candidate before sharing the GPU.**

MLE Agent can change the backbone, optimizer, input size, data pipeline or combine
several models. Two jobs called “ResNet” can therefore behave very differently.
Our local audit also found that “improve” may replace a pipeline, while “debug”
may retry identical code. Stage and family labels are not reliable drift measures.

## What “variance delta” means

It is a breakdown of change from a reference job, not a confidence percentage.

| Change | What we check |
|---|---|
| Structure | Operators, connections, trainable layers and optimizer work. |
| Size | Tensor shapes, activation/state size and amount of computation. |
| Recipe | Batch, precision, accumulation, optimizer and checkpointing. |
| Pipeline | Augmentation, input waits, folds and sequential/concurrent models. |
| Observed | Current step time, variability and peak memory. |

Missing information stays **unknown**. We keep predicted resource increases and
uncertainty separate. A small structural delta never proves that sharing is safe.
For example, wider layers may keep the same topology but need much more memory;
heavier augmentation can change GPU utilization without changing the model.

## How the revised scheduler would decide

1. **Identify the actual job.** Record final code/configuration, all parents,
   execution phases and what the descriptor could not capture.
2. **Use history to choose a trial.** Compare the new job with its parents and
   similar workloads. CPU TorchScript predictions help order candidates.
3. **Measure useful training.** Collect the candidate's current solo rate, then
   test a promising sharing group within the existing trial budget.
4. **Keep only worthwhile sharing.** Extra progress must repay measurement cost,
   every job must pass slowdown checks, and memory/backend gates still apply.
5. **Refresh when behavior changes.** A new phase or changed configuration retires
   old evidence. Waiting jobs retain v2's service and reservation rules.

An unchanged retry can reuse eligible fresh evidence. A new model cannot inherit
a group's performance just because its family or delta looks similar.

## Research behind the change

[Quasar](https://csl.stanford.edu/~christos/publications/2014.quasar.asplos.pdf)
already addresses unfamiliar workloads using brief measurements plus history.
[NeuSight](https://arxiv.org/abs/2407.13853) motivates operator/tensor descriptions;
[MLEvolve](https://arxiv.org/html/2606.06473v1) motivates tracking complete evolving
pipelines. These are adaptations, not evidence of a speedup on our GPU.

Saved on local branch **oct_proposed_v2**. This is a research design with a tested
CPU metric reference; it is not deployed. Validation must measure both scheduler
overhead and agent search quality over wall time, because job completion affects
which candidate the agent generates next.

Full specification: [oct_proposed_v2.md](oct_proposed_v2.md).
