# Engine Design

`kernel_evaluator.services.evaluation` is the benchmark-controller slice for kernel evaluation. Its job is to compare reference baselines and candidate kernels through one execution contract, even when the implementation language differs.

The current scope is deliberately narrow: prove that the controller can reproduce trusted harness numbers for known tasks, then use the same path to evaluate new candidate kernels. The service wiring can stay thin if task contracts, builders, and benchmark semantics are explicit here.

## Core Contracts

`types.py` defines the stable vocabulary:

- `ReferenceTask` describes the task baseline: deterministic input generation, reference output, tolerances, optional custom reference timing, and the expected kernel ABI.
- `CandidateSubmission` describes a candidate package: kind, ABI, source/artifacts, optional entrypoint, and metadata such as shape scalars.
- `BuiltCandidate` is the normalized output of a builder. The benchmark controller consumes this object without backend-specific branches.
- `CandidateExecutor` is the runtime interface: `prepare(inputs)`, `launch()`, and `close()`.
- `BenchmarkPolicy` and `BenchmarkResult` capture timing policy and repeat aggregation.

The important boundary is:

- `prepare(inputs)` binds concrete tensors and creates backend launch state. It is not timed.
- `launch()` performs the operation being benchmarked. It is timed.
- Output tensors are cleared after `prepare()` and before timing as an input precondition.

This lets CUDA, CuTeDSL, Triton, and Python candidates share the same controller path while still doing backend-specific setup correctly.

## Controller Flow

`BenchmarkController` in `benchmark.py` owns the main orchestration:

1. Resolve a builder from `BuilderRegistry`.
2. Build the submitted candidate into a `BuiltCandidate`.
3. Create deterministic reference and candidate inputs for each repeat.
4. Benchmark the reference baseline.
5. Prepare the candidate executor on candidate inputs.
6. Clear candidate output tensors.
7. Benchmark candidate `launch()`.
8. Run correctness on fresh inputs against the reference output.
9. Return aggregated repeat results and speedup.

Reference timing may use a custom `benchmark_reference(inputs) -> launch` callable when the baseline has nontrivial setup. Otherwise the controller times `task.reference(inputs)` directly.

## Builders

`builders.py` currently includes:

- `CudaSourceBuilder`: compiles a `.cu` source artifact into a helper `.so` plus `.cubin`, captures build logs/resource usage, and launches the cubin through the CUDA driver API.
- `ExternalPtxSoBuilder`: consumes prebuilt shared objects and optional cubins using the same helper ABI.
- `CuTeDSLBuilder`: loads a `module:symbol` entrypoint that prepares a zero-argument CuTeDSL launch callable.
- `TritonCallableBuilder`: loads Triton-backed prepared callables.
- `PythonCallableBuilder`: loads direct Python callables.

`registry.py` provides `BuilderRegistry` and `default_registry()` so the service can add or override backend support without changing the benchmark controller.

## Reference Plugins

Reference plugins define task-specific inputs and trusted outputs. Their filenames should describe the backend/operation:

- `torch_linear.py`
- `torch_sdpa.py`
- `torch_fp8_gemm.py`
- `cuda_int4_matmul.py`
- `fa3_paged_decode.py`
- `cutedsl_chunked_gemv.py`

Registry keys should mirror this naming style, for example `torch.linear`, `cuda.int4_matmul`, `fa3.paged_decode`, and `cutedsl.chunked_gemv`.

Reference plugins may keep in-process caches for expensive reference setup, but those caches are only optimizations. Candidate submissions should not rely on hidden module-global state.

## Candidate Fixtures

Known-good candidate fixtures live in `kernel_evaluator/tests/testing_kernels/`. Use:

```text
{language}_{operation}_{dtype}_{shape_or_variant}
```

Examples:

- `cuda_int4_matmul_fp16_m1_o_proj.cu`
- `cuda_fa3_paged_decode_bf16_b1_hq16_hkv4_d64_s128.cu`
- `cutedsl_chunked_gemv_bf16_m1_n2048_k2048.py`

Fixture files should look like submitted kernels. They should not import candidate code from reference plugins. Tests can import reference plugins, but candidate fixtures should carry the candidate implementation and expose the required entrypoint.

## Task Contracts

Task contracts should eventually be source-controlled, not invented by agents at submission time. The scalable shape is:

- A template defines the common operation contract: tensor names, roles, dtype rules, tolerances, timing policy, candidate kind, and instructions.
- A task instance defines dimensions, variant name, expected timings, and any task-specific entrypoint name.

This avoids repeating a full contract for every shape while keeping the source of truth reviewable. The DB should store runtime state and results; task YAML should store canonical task contracts or a version/hash of them.

## Current Validation Targets

The current tests prove these paths:

- CUDA int4 matmul candidates against the int4 reference plugin.
- CUDA FA3 paged-decode candidate against the FA3 reference plugin.
- CuTeDSL chunked-GEMV candidate against a cuBLAS/`F.linear` reference for `M=1, N=2048, K=2048, bf16`.
- Torch linear and SDPA smoke tests for Python-callable plumbing.

## Next Work

- Add a task-template parser that expands YAML into `ReferenceTask`, `CandidateSubmission` prompts, and benchmark policy.
- Decide which plugin keys need backwards-compatible aliases before service migration.
- Move eval-service task loading toward the same task contract schema.
- Add clearer failure messages for timing drift and correctness mismatches.
