---
name: profiling
description: Profile a kernel on CUDA/NVIDIA with Nsight Compute through the eval service (ncu_summary / ncu_report artifacts) to find why it is slow. Use after correctness passes and you want to know what to optimize next.
---

# Profiling (CUDA / NVIDIA, Nsight Compute)

Profiling is optional but powerful: it tells you *why* a kernel is slow instead of
guessing. Profiling only runs on a **correct** kernel, so get correctness first.

## Request a profile at submit time

```bash
bench-submit kernel.cu --artifacts ncu_summary ncu_report
```

Useful artifact kinds for CUDA:
- `ncu_summary` — a compact text digest of the Nsight Compute report (read this first).
- `ncu_report` — the full `.ncu-rep` profile.
- `cubin`, `ptx`, `resource_usage` — compiled SASS/PTX and register/shared-mem usage.

## Fetch and read the artifacts

```bash
bench-artifact <job_id> ncu_summary --out ncu_summary.txt
bench-artifact <job_id> resource_usage --out resource_usage.txt
```

## What the summary tells you (per kernel)

- **Kernel**: `GPU Time Avg/Max/Min`, `Replay Passes`.
- **Launch**: `Block Size`, `Grid Size`, `Registers / Thread`, `Shared Mem / Block`.
- **Speed Of Light**: `SM Throughput`, `Compute Memory Throughput`, `DRAM Throughput`,
  `L1/TEX Throughput`, `L2 Throughput`, `Instruction Throughput` (all % of peak).

## How to act on it

- **DRAM Throughput high, SM/Compute low** → memory-bound. Improve coalescing, use wider
  (vectorized) loads/stores, cut redundant global traffic, reuse data in shared mem/registers.
- **Compute/Instruction Throughput high, DRAM low** → compute-bound. Reduce instruction
  count, use faster math, raise ILP, avoid divergence.
- **Both low** → latency/occupancy bound. Check `Registers / Thread` and `Shared Mem / Block`
  (they cap occupancy), reduce them or retune block size; look for stalls/serialization.
- **High `Replay Passes`** → the metric set forced replays; ignore for timing, it does not
  affect the benchmark.

Re-profile after each change and compare against the leaderboard best to see what a faster
kernel does differently.
