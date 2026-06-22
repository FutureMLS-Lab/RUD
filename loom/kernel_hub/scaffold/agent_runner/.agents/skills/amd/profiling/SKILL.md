---
name: profiling
description: Profile a kernel on AMD/ROCm with rocprofv3 through the eval service (rocprof_summary / rocprof_report artifacts) to find why it is slow. Use after correctness passes and you want to know what to optimize next.
---

# Profiling (AMD / ROCm, rocprofv3)

Profiling is optional but powerful: it tells you *why* a kernel is slow instead of
guessing. Profiling only runs on a **correct** kernel, so get correctness first.
NCU is NVIDIA-only and is not available here; on AMD we use rocprofv3.

## Request a profile at submit time

```bash
bench-submit kernel.cu --artifacts rocprof_summary rocprof_report
```

Artifact kinds for AMD:
- `rocprof_summary` — a compact text digest of the PMC counters (read this first).
- `rocprof_report` — the full profile as a `.tar.gz` (per-kernel
  `counter_collection.csv`, and an ATT thread trace when enabled).

## Fetch and read the artifacts

```bash
bench-artifact <job_id> rocprof_summary --out rocprof_summary.txt
bench-artifact <job_id> rocprof_report  --out rocprof_report.tar.gz
tar -tzf rocprof_report.tar.gz          # list contents
```

## What the summary tells you (per kernel, by `kernel_index`)

Derived from hardware PMC counters:
- **Instruction mix**: `Frac_INSTS_VALU`, `Frac_INSTS_SALU`, `Frac_INSTS_SMEM`,
  `Frac_INSTS_VMEM` (fractions of total instructions), `Frac_Active_VMEM`.
- **Occupancy / utilization**: `Waves_per_WG`, `SQ_Busy_Ratio`.
- **Caches / memory**: `Icache_Hit_Rate`, `Scache_Hit_Rate`, `L2_Hit_Rate`, `L2_Miss_Rate`,
  `LDS_Conflict_Ratio`, plus raw `TCP_*` (vector L1) and `TCC_*` (L2) access counts.

## How to act on it

- **High `Frac_INSTS_VMEM` + high `L2_Miss_Rate`** → memory-bound. Use wider (`uint4`/128-bit)
  loads/stores, improve coalescing, reuse data in registers/LDS, cut redundant global traffic.
- **High `Frac_INSTS_VALU`, low VMEM** → compute/ALU-bound. Reduce instruction count, use
  packed math (e.g. `v_pk_*`), raise ILP, avoid divergence.
- **High `LDS_Conflict_Ratio`** → LDS bank conflicts; pad/restripe shared arrays.
- **Low `Waves_per_WG` / low `SQ_Busy_Ratio`** → occupancy/latency bound. Check VGPR and LDS
  usage (they cap waves), retune block size, increase in-flight work.
- **Low `L2_Hit_Rate` on reused data** → consider blocking/tiling so reuse stays resident.

The ATT trace in `rocprof_report` gives per-wave instruction timelines for the hottest
kernel — use it to find stalls and hot instructions once the PMC summary points you at the
bottleneck. Re-profile after each change and compare against the leaderboard best.
