# Kernel Evaluator

`kernel_evaluator` is a compile-and-benchmark service for candidate GPU kernels. It exposes a small HTTP API where clients create an evaluation run, submit candidate source text, and poll for correctness and timing results. The service owns compilation, input generation, correctness checks, benchmark policy, repeat aggregation, and artifact cleanup, so submitted kernels are evaluated as black boxes.

## What The Service Guarantees

- Candidates submit source text only. The service materializes the file inside an isolated job artifact directory.
- Correctness uses fresh random inputs that are separate from timing inputs.
- Benchmarking is service-owned; candidates cannot choose timing loops or correctness data.
- Results are conservative: the final baseline is the best observed baseline repeat, and the final candidate time is the worst observed candidate repeat.
- Evaluation runs and successful kernels are persisted in Postgres.
- User API keys are rate limited and have a small in-flight submission cap.
- Terminal jobs and artifact directories expire through TTL cleanup.

## Runtime Configuration

The service fails fast if required environment variables are missing.

```bash
export DATABASE_URL=postgresql://kernel_evaluator:kernel_evaluator@localhost:5432/kernel_evaluator
export KERNEL_EVALUATOR_USER_SUBMIT_LIMIT=10
export KERNEL_EVALUATOR_USER_SUBMIT_WINDOW_S=60
export KERNEL_EVALUATOR_USER_IN_FLIGHT_SUBMIT_LIMIT=2
export KERNEL_EVALUATOR_COMPILE_TIMEOUT_S=300
export KERNEL_EVALUATOR_BENCHMARK_TIMEOUT_S=600
```

For local development in this repo, root `.env` contains the expected values plus:

```bash
export KERNEL_EVALUATOR_API=http://localhost:8000
export KERNEL_EVALUATOR_ADMIN_API_KEY=admin1234
```

`admin1234` is the bootstrap admin key seeded by migration `003_seed_admin_api_key.py`.

## Running

From the kernel hub (`loom/kernel_hub/`):

```bash
docker compose up -d --build
```

For local Python development, use the repo virtualenv from the root workspace:

```bash
/data/fede/turbo-gemm/.venv/bin/python -m uvicorn kernel_evaluator.app:app --host 0.0.0.0 --port 8000
```

Run migrations before serving against a fresh DB:

```bash
DATABASE_URL=postgresql://kernel_evaluator:kernel_evaluator@localhost:5432/kernel_evaluator \
  /data/fede/turbo-gemm/.venv/bin/alembic -c kernel_evaluator/alembic.ini upgrade head
```

## API Keys And Roles

There is no login/session system. Every protected request sends:

```bash
X-API-Key: <key>
```

Roles:

- `admin`: can create evaluation runs, create/revoke API keys, submit jobs, and read/mutate all resources.
- `user`: can submit jobs to existing runs and read only its own in-memory evaluation jobs/results.

Create a user key with the bootstrap admin key:

```bash
curl -s -X POST http://localhost:8000/api-keys \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: admin1234' \
  -d '{"role":"user"}'
```

The raw key is returned once. The database stores only the SHA-256 hash.

## Evaluation Flow

Create a run. This is admin-only:

```bash
curl -s -X POST http://localhost:8000/evaluation/runs \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: admin1234' \
  -d '{
    "plugin": "torch.linear",
    "target": "cutedsl",
    "shape": {"m": 1, "n": 2048, "k": 2048, "dtype": "bf16"}
  }'
```

Submit a candidate. This can use a user key:

```bash
curl -s -X POST http://localhost:8000/evaluation/runs/cutedsl_torch_linear_bf16_m1_n2048_k2048/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <user-key>' \
  -d "{\"source_text\": $(python -c 'import json; print(json.dumps(open(\"kernel.py\").read()))'), \"artifacts\": []}"
```

Poll the job:

```bash
curl -s -H 'X-API-Key: <user-key>' \
  http://localhost:8000/evaluation/jobs/<job_id>
```

Fetch the completed result:

```bash
curl -s -H 'X-API-Key: <user-key>' \
  http://localhost:8000/evaluation/jobs/<job_id>/result
```

## Artifacts

Submissions can request compile artifacts:

```json
{"artifacts": ["ptx", "resource_usage", "cubin"]}
```

Produced artifact names appear in the job summary:

```json
{"artifacts": ["cubin", "ptx", "resource_usage"]}
```

Fetch artifacts with the same job visibility rules as job results:

```bash
curl -s -H 'X-API-Key: <user-key>' \
  http://localhost:8000/evaluation/jobs/<job_id>/artifacts/ptx > kernel.ptx

curl -s -H 'X-API-Key: <user-key>' \
  http://localhost:8000/evaluation/jobs/<job_id>/artifacts/resource_usage

curl -s -H 'X-API-Key: <user-key>' \
  http://localhost:8000/evaluation/jobs/<job_id>/artifacts/cubin > kernel.cubin
```

The public artifact model is intentionally generic so future profiling artifacts such as `ncu_report` and `ncu_summary` can use the same request and fetch API.

## Agent CLI

Agents should use the helper installed by `agent_runner/Dockerfile`:

```bash
kernel-eval run <run_id>
kernel-eval submit <run_id> kernel.cu
kernel-eval poll <job_id>
kernel-eval best-kernel <run_id> --out leaderboard_best.cu
```

`agent_runner/run_agents.sh` creates one user API key per Codex agent using the admin key, then passes only the user key into that agent container as `KERNEL_EVALUATOR_API_KEY`.

## Plugins And Targets

Operation plugins define the math contract. Targets define how submitted code is executed.

Current operation plugins include:

- `torch.linear`
- `torch.sdpa`
- `torch.fp8_gemm`
- `cuda.int4_matmul`
- `cuda.fa3_paged_decode`

Current targets include:

- `cuda`: submitted CUDA source must export `kernel`, `globals_size`, `make_globals`, `grid_dims`, `block_dim`, and `shmem_bytes`.
- `cutedsl`: submitted Python source must export `prepare(inputs)` and return a zero-argument launch function.

The central plugin registry composes operation contracts with target contracts to produce a validated run contract.

## Benchmark Semantics

Default benchmark policy:

- timing mode: CUDA graph timing
- warmup: `10`
- iterations: `50`
- graph calls per timing event: `100`
- repeats: `3`
- sleep between repeats: `10` seconds

Each repeat records a median baseline time and a median candidate time. Final aggregation is:

```text
baseline_us = min(repeat.baseline_us)
candidate_us = max(repeat.candidate_us)
speedup = baseline_us / candidate_us
correct = all(repeat.correct)
```

This matches the leaderboard-style policy: best observed baseline, worst observed candidate.

## Persistence, Job Lifecycle, And TTL

Evaluation runs are stored in `eval_runs`. The row contains denormalized query fields plus the validated `run_contract` JSON used to recreate submission jobs. Successful correct benchmark jobs are stored in `kernel_library`.

Jobs live in memory in `queue_state.jobs` because they are runtime queue state. Each job gets an artifact directory under `/tmp/kernel_evaluator`.

Workers move jobs through:

```text
queued_compile -> compiling -> queued_benchmark -> benchmarking -> completed
```

Failures end in:

```text
compile_failed
benchmark_failed
```

Terminal jobs are evicted after `terminal_job_ttl_s` seconds, default `3600`. The cleanup worker runs every `cleanup_interval_s` seconds, default `60`, and removes both the in-memory job and its artifact directory.

## Tests

Focused non-timing tests:

```bash
/data/fede/turbo-gemm/.venv/bin/python -m pytest kernel_evaluator/tests/non_timing
```

API-level timing smoke test:

```bash
/data/fede/turbo-gemm/.venv/bin/python -m pytest \
  kernel_evaluator/tests/timing/test_api_cutedsl_chunked_gemv_bf16_m1_n2048_k2048.py
```

That test exercises the black-box API path: create run, submit source text, poll job, fetch result, and assert expected timing.
