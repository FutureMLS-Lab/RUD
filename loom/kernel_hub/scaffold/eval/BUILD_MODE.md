

## BUILD MODE — correctness first (overrides the speed goal above)

You are in **build mode**: the goal is to produce a **correct** kernel, NOT a fast one. The
"work until speedup > 1.0x / do not stop early" guidance above does **not** apply here.

- Your target is a kernel that **passes the correctness check** (`bench-poll` shows `correct=True`).
  Speed is irrelevant in this mode — a correct-but-slow kernel is a success.
- Iterate on **compilation and correctness only**: get it to compile, then make the output match the
  reference within tolerance. Use `bench-submit` + `bench-poll` and read compile/runtime errors.
- **Stop as soon as you have one correct kernel.** Do not spend further iterations optimizing latency.
- If you cannot reach correctness, keep fixing compile/numerical errors — do not switch focus to speed.
