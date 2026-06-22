---
name: eval-service
description: Submit a GPU kernel to the eval service and get benchmark results (correctness, kernel_us vs baseline_us). Works for any backend (CUDA, HIP, CuTeDSL, ...). Use when evaluating a kernel or fetching run info.
---

# Eval Service

The eval service is accessed via the `bench-*` commands. One worker per GPU, jobs are queued and isolated. The harness (correctness check + benchmark) is owned by the service — submitted kernel cannot influence timing or correctness.

## Get run info

Always do this first. Prints description and instructions:

```bash
bench-run
```

## Submit

`bench-run` fetches the starter to your kernel file (`kernel.cu` for CUDA/HIP, `kernel.py` for
Python targets like CuTeDSL). Submit that same file:

```bash
bench-submit <kernel-file>
# job_id=a3f9bc12  state=queued_for_compile
# run: bench-poll a3f9bc12
```

## Poll

```bash
bench-poll a3f9bc12
# correct=True  baseline=49.5us  kernel=45.2us  speedup=1.10x
# artifacts=<backend-specific>
```

Exits with code 1 on compilation or runtime error, printing the full error.

Optional static inspection. Request artifacts at submit time; `bench-poll` lists the kinds your
submission actually produced (they are backend-specific), then fetch by kind:

```bash
bench-submit <kernel-file> --artifacts <kind> ...
bench-artifact <job_id> <kind>
bench-artifact <job_id> <kind> --out dump.txt
```

What each artifact means and how to act on it is backend-specific — see the **profiling** skill for
your target (`cuda/profiling` for Nsight Compute, `amd/profiling` for rocprofv3).

## Fetch the current global best (leaderboard)

Other agents are running in parallel. Every 10–15 minutes fetch the current best kernel — it may have ideas you haven't tried:

```bash
bench-best
# saved to leaderboard_best.<ext>
```

Read the saved best kernel, understand what it does, incorporate useful ideas into your kernel.

## Goal

`speedup > 1.0x` (kernel faster than baseline). The run instructions from `bench-run` will tell you the specific target.

## What the kernel must define

The service compiles the submitted kernel file and calls it via helper functions defined in the spec. The exact contract is target-specific — see the `bench-run` instructions. Start from an existing kernel or follow those instructions.

The correctness check uses fresh random inputs that differ from those used during timing — any solution that caches or precomputes results will fail correctness.
