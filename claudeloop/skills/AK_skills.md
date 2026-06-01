# K8S

在开始实现或 review 前，先把任务目标、当前 plan、成功标准整理清楚，只保留对后续执行有用的信息。默认遵守下面的工作习惯：

1. Task state and documentation
- If working inside a claudeloop task, keep `PLAN.md` updated with current useful progress, decisions, blockers, and next steps. Do not record noisy or obsolete details.
- Save useful results, logs, configs, and notes under the task directory. Keep them concise and organized.
- If the task is a code review, create a sibling `REVIEW.md` next to `PLAN.md`. Each bullet should be one concrete review comment with file/line context, the issue, and the recommended fix. Focus on correctness bugs, likely runtime errors, regressions, and important missing tests.

2. Code, worktrees, and branches
- Use the task worktree for all experiments, fixes, and PR reviews. Reuse an existing worktree if it already exists.
- Name branches, worktrees, and related review branches with the `zhongzhu/<task-name>` convention.
- Do not create PRs unless explicitly asked. Leave code changes in the local worktree so the user can review them directly in GitHub/Cursor changes.
- Do not create commits unless explicitly asked. If asked to commit, do not add Claude/Cursor/AI as author or co-author.

3. Experiments and compute
- Prefer Kubernetes for tests, evals, and training so the local machine stays responsive. Use pods/jobs/nodes as needed, and clean up old completed or failed jobs after use.
- Be careful with flaky or unhealthy cluster resources: some nodes may have NCCL, IB, or GPU issues. Check node/job status when results look suspicious.
- If a local GPU is free and safe to use, it is okay to run lightweight checks locally; otherwise prefer Kubernetes.
- Use wandb for experiment tracking; the user is already logged in.

4. Secrets and caches
- Use the existing `HF_TOKEN` environment variable only. Never write token values into files, logs, prompts, commits, or PR text.
- Use `/shared/huggingface` as the Hugging Face cache location (`HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`). Do not use `~/.cache/huggingface`.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
