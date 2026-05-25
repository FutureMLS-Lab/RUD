# K8S

请你先整理组织下我们的plan,task,success，现在写的比较混乱，保留总结重要内容。然后记得加入一下内容：

- Use the task worktree for experiments and PR review. If the worktree already exists, reuse it.
- Prefer Kubernetes for tests/evals/training so the local machine stays free.
- Use pods/jobs/nodes to scale experiments, and clean up old completed or failed jobs when done.
- Save all results, logs, configs, and notes under the task directory.
- Use wandb for run tracking; the user is already logged in.
- Use the existing `HF_TOKEN` 
- Use `/shared/huggingface` for Hugging Face cache (`HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`)! DO NOT USE ~/.cache/huggingface!
- some node NCCL/IB/GPU might be inhealthy, the cluster has these information, please be careful to check
- branch/worktree name use zhongzhu/xxxx as the name, also for PR, etc.
- don't create PR, review by your self; just leave everything in the local worktree.
- for review, list review like "- code line xxx comment review xxx" simple & clean
- don't make author / co-author as claude / cursor or ai agent.
- 如果在claudeloop文件夹中 记得更新文件夹最外面的 PLAN TASK SUCESS 三个markdown；主要是PLAN 我们会根据PLAN检查进度。
- 如果本机local 有free GPU 没有人使用的 你可以尝试使用。

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
