# Task

Optimize a GPU kernel for a specific task shape. Your target backend, the exact shape, and the
contract your kernel must satisfy all come from `bench-run` — always start there.

## Workflow

1. Get your run info and starter code:
   ```bash
   bench-run
   # prints run details, instructions, scalars, dtype
   # also fetches starter code to $KERNEL_FILE (if configured)
   ```
   If starter code is provided, the output tells you what kind:
   - **exact match**: Optimized for your exact shape. Good starting point.
   - **function_only match**: Optimized for a different shape. May need tile size adjustments.
   - **generic template**: Needs significant adaptation to your shape.
   - **preset**: A specific starter kernel copied into `$KERNEL_FILE` for you. Read and
     understand it, then optimize from there.

2. Edit `$KERNEL_FILE`, then submit and poll:
   ```bash
   bench-submit $KERNEL_FILE
   bench-poll <job_id>
   ```

3. Iterate until kernel_us < baseline_us (speedup > 1.0x).

## Collaborative environment

Other agents are running in parallel on the same task (you cannot see them). This is a collaborative effort — the global leaderboard is shared. Every 10–15 minutes, fetch the current best kernel from the leaderboard — it may be better than yours and give you new ideas:

```bash
bench-best
# saves current global best to leaderboard_best.<ext>
```

Read it, understand what it does differently, and incorporate the best ideas into your own work.

You can also browse all improvements made during this run:

```bash
bench-archive
# lists all improvements with speedup and job_id

bench-kernel-source <job_id> --out winner.<ext>
# fetch the source of a specific kernel
```

## Working with preset starter kernels

When `bench-run` shows preset mode, a specific starter kernel has been copied into `$KERNEL_FILE`
(the source file is also mounted read-only at `/preset/<filename>`). This is a known-good starting
point chosen for your task — it already compiles and passes the contract.

Your goal: optimize it to beat the baseline timing while maintaining correctness.

## The kernel contract

The exact signature, entry point, or export your kernel must define is **target-specific** — the
`bench-run` instructions for your task spell it out. Follow that contract exactly; the service
compiles `$KERNEL_FILE` and drives it through helper functions baked into the spec.

## Rules

- Always start by running `bench-run` to get starter code and understand the task
- Do not stop until you have exhausted your optimization ideas
- You can rewrite the kernel from scratch if useful (e.g., you have a completely different algorithmic idea that you want to try)
- Run evaluation a few times to confirm results are stable before finishing
- Search online! This can be very useful.
- Remember that we are optimizing for a specific shape.
- Check the leaderboard every 10–15 minutes with `bench-best`
- You must work until kernel_us < baseline_us (speedup > 1.0x). Do not stop early. Do not try to monkey patch your way into a solution.
