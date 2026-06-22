# Skills and profiling

You have skills available in `.agents/skills/`. Read a skill's `SKILL.md` before using it:

- **`eval-service`** — how to submit/poll/inspect kernels through the eval service (`bench-*`).
- **`profiling`** — how to profile your kernel to find *why* it is slow. This skill is already
  set up for your backend (rocprofv3 on AMD, Nsight Compute on CUDA) — you do not choose the tool.

## Profile, don't guess

Once your kernel is correct, **do not optimize by blindly sweeping block sizes / vector widths**.
Profile first, then change the thing the profile says is the bottleneck:

1. Get a correct kernel (`speedup` reported, even if < 1.0x).
2. Read `.agents/skills/profiling/SKILL.md` and profile that kernel.
3. Use the profile (occupancy, memory throughput vs peak, stalls) to decide the next change —
   e.g. memory-bound → improve coalescing / vectorization / occupancy; launch/dispatch-bound at
   small shapes → reduce overhead, fuse, tune grid.
4. Re-profile after a meaningful change to confirm the bottleneck moved.

Sweeping is a fallback once profiling no longer points at an obvious bottleneck — not the first move.
