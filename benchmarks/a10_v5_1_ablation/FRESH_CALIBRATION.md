# V5.1 fresh-calibration rerun

The earlier run kept one calibration long enough for its four-hour freshness limit to expire. V5.1 then correctly held its concurrency setting instead of trusting old measurements. This repair gives the campaign a calibration refresh lifecycle; it does not weaken the freshness check.

## What changes

Before each matched block of fixed-three FIFO, combined V5 and V5.1, the runner checks the oldest actual calibration observation. The same measurement generation must cover all three bounded episodes plus a five-minute margin. If it cannot, the runner first proves the GPU is idle, repeats the eight dedicated calibration jobs, converts the released S_A10 outputs on CPU, and seals a new immutable generation. Every arm in that block uses the same generation. It never refreshes partway through a block or changes timestamps on old evidence.

FIFO acceptance and first-dispatch order, adaptive-alpha policy, four-hour TTL, predictor weights, labels, seeds, workload recipes, update counts and the fixed-three comparator remain unchanged. A new generation changes measured costs, not policy thresholds. Calibration uses the existing disjoint calibration workloads, never evaluation outcomes.

## Comparison and audit

One freshly qualified A10 runs all three arms across five conditions, individual and bracket modes, and two within-host paired repeats: **60 cohorts and 432 jobs**. The reused seeds make this a corrected replication of the earlier study. It is separate from prior diagnostics and failures.

The independent final replay checks every calibration generation, source and input hashes, measured observation clocks, successful exits, receipts, optimizer updates, epochs, causal arrivals, FIFO order, and exact complete coverage. Within-block profile equality is mandatory. Results retain failed and unattempted denominators and separate measured-template controls from the published-predictor pipeline.

Report mean individual JCT and all-required bracket feedback latency, tails and makespan, paired differences, alpha versus occupancy, holds and actual activation. Report calibration refresh wall time and billing separately from request JCT; refresh time is included in campaign duration and resource spending. The new run is not an isolated causal estimate of TTL effects because it also uses a new host and newly measured costs. No production or model-promotion claim follows automatically.

## Finite resource contract

The same period-2 **$30** allowance and original cutover remain in force. All historical charges and unresolved billing liabilities remain recorded. The pre-rental reconciliation leaves about **$24.20** of conservative headroom, including the existing $1 cleanup/transfer reserve.

Use one verified A10 with 40GB storage, a finite 24-hour lease, and an hourly reservation covering the quote plus at least 10%. A $0.30/hour reserve would add $7.20 and keep the total reserved upper bound near $13.00. Fresh billing, inventory and guard verification are required before rental. The campaign bound is 19.5 hours including refreshes; job and episode limits remain 1,800 and 3,600 seconds. Full preparation, review, campaign and recovery must fit the original lease cutoff. No silent deadline extension or budget reset is permitted.

The independent budget guard checks every 20 seconds; artifact recovery runs every 60 seconds. A lightweight subagent performs one bounded hourly monitoring check. After review and SHA-verified recovery, cleanup targets only the exact owned instance and requires fresh provider proof of absence.
