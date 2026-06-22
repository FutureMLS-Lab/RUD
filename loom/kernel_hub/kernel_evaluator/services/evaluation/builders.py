import importlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_evaluator.services.evaluation.artifacts import artifact_by_kind, materialize_artifact, normalize_artifacts, write_json, write_log
from kernel_evaluator.services.evaluation.cuda_launcher import make_cubin_so_setup, make_so_setup
from kernel_evaluator.services.evaluation.executors import CallableExecutor, PreparedFunctionExecutor
from kernel_evaluator.services.evaluation.runtime import DEFAULT_RUNTIME_ENV, RuntimePolicy, validate_runtime_policy
from kernel_evaluator.services.evaluation.specs import require_cubin_abi, require_helper_abi
from kernel_evaluator.services.evaluation.types import ArtifactKind, BuildContext, BuiltCandidate, CandidateKind, CandidateSubmission


def load_entrypoint(entrypoint: str) -> Any:
    module_name, symbol_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _runtime_policy(submission: CandidateSubmission, default_env: str, default_profile: str) -> RuntimePolicy:
    env = submission.metadata["runtime_env"] if "runtime_env" in submission.metadata else default_env
    profile = submission.metadata["build_profile"] if "build_profile" in submission.metadata else default_profile
    roots = submission.metadata["import_roots"] if "import_roots" in submission.metadata else ()
    if not isinstance(env, str):
        raise ValueError("runtime_env must be a string")
    if not isinstance(profile, str):
        raise ValueError("build_profile must be a string")
    if not isinstance(roots, tuple):
        raise ValueError("import_roots must be a tuple")
    return validate_runtime_policy(RuntimePolicy(env, profile, roots))


@dataclass(frozen=True)
class PassthroughBuilder:
    kind: CandidateKind

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        del context
        factory = submission.metadata["executor_factory"]
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=normalize_artifacts(submission.artifacts),
            executor_factory=factory,
            metadata=submission.metadata,
        )


@dataclass(frozen=True)
class PythonCallableBuilder:
    kind: CandidateKind = CandidateKind.PYTHON_CALLABLE

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        del context
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "python_entrypoint")
        if submission.entrypoint is None:
            raise ValueError("python callable submissions require entrypoint")
        fn = load_entrypoint(submission.entrypoint)
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=normalize_artifacts(submission.artifacts),
            executor_factory=lambda: CallableExecutor(fn),
            metadata=submission.metadata,
        )


@dataclass(frozen=True)
class TritonCallableBuilder:
    kind: CandidateKind = CandidateKind.TRITON_CALLABLE

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        del context
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "triton_entrypoint")
        if submission.entrypoint is None:
            raise ValueError("triton callable submissions require entrypoint")
        fn = load_entrypoint(submission.entrypoint)
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=normalize_artifacts(submission.artifacts),
            executor_factory=lambda: PreparedFunctionExecutor(fn),
            metadata=submission.metadata,
        )


@dataclass(frozen=True)
class CuTeDSLBuilder:
    kind: CandidateKind = CandidateKind.CUTEDSL_AOT

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        del context
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "cutedsl_entrypoint")
        if submission.entrypoint is None:
            raise ValueError("cutedsl submissions require entrypoint")
        package_artifacts = tuple(
            artifact for artifact in submission.artifacts
            if artifact.kind in (ArtifactKind.SOURCE, ArtifactKind.MANIFEST)
        )
        if not package_artifacts:
            raise ValueError("cutedsl submissions require source or manifest artifacts")
        for artifact in package_artifacts:
            if not artifact.path.exists():
                raise ValueError(f"cutedsl artifact does not exist: {artifact.path}")
        fn = load_entrypoint(submission.entrypoint)
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=normalize_artifacts(submission.artifacts),
            executor_factory=lambda: PreparedFunctionExecutor(fn),
            metadata=submission.metadata,
        )


def _metadata_scalars(submission: CandidateSubmission) -> dict:
    if "scalars" not in submission.metadata:
        raise ValueError(f"{submission.kind} submissions require metadata['scalars']")
    scalars = submission.metadata["scalars"]
    if not isinstance(scalars, dict):
        raise ValueError("metadata['scalars'] must be a dict")
    return scalars


@dataclass(frozen=True)
class ExternalPtxSoBuilder:
    kind: CandidateKind = CandidateKind.EXTERNAL_PTX_SO

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        del context
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "prebuilt")
        scalars = _metadata_scalars(submission)
        so = artifact_by_kind(submission.artifacts, ArtifactKind.SHARED_OBJECT).path
        cubins = tuple(artifact for artifact in submission.artifacts if artifact.kind == ArtifactKind.CUBIN)
        if cubins:
            if len(cubins) != 1:
                raise ValueError("external_ptx_so submissions support exactly one cubin artifact")
            require_cubin_abi(submission.abi)
            setup_fn = make_cubin_so_setup(so, cubins[0].path, submission.abi, scalars)
        else:
            require_helper_abi(submission.abi)
            setup_fn = make_so_setup(so, submission.abi, scalars)
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=normalize_artifacts(submission.artifacts),
            executor_factory=lambda: PreparedFunctionExecutor(setup_fn),
            metadata=submission.metadata,
        )


def _nvcc_flags(context: BuildContext) -> list[str]:
    flags = [
        "nvcc",
        "-std=c++20",
        "-O3",
        "--use_fast_math",
        "-gencode",
        f"arch=compute_{context.cuda_arch},code=sm_{context.cuda_arch}",
        "-lcuda",
        "-lcudart",
    ]
    for include_dir in context.include_dirs:
        flags.extend(["-I", str(include_dir)])
    tk_root = os.environ["THUNDERKITTENS_ROOT"] if "THUNDERKITTENS_ROOT" in os.environ else ""
    if tk_root:
        flags.extend([
            "-DKITTENS_HOPPER",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-I",
            str(Path(tk_root) / "include"),
        ])
    for library_dir in context.library_dirs:
        flags.extend(["-L", str(library_dir)])
    flags.extend(context.extra_flags)
    return flags


def _run_build(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output)
    return output


def _python_extension_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or suffix == "":
        raise RuntimeError("Python extension suffix is unavailable")
    return suffix


def _pybind11_includes() -> list[str]:
    return subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).split()


def _hipcc_flags(context: BuildContext) -> list[str]:
    from torch.utils.cpp_extension import include_paths

    target_arch = context.cuda_arch if context.cuda_arch.startswith("gfx") else "gfx950"
    flags = [
        "hipcc",
        "-std=c++20",
        "-O3",
        "-shared",
        "-fPIC",
        f"--offload-arch={target_arch}",
        *_pybind11_includes(),
        "-I",
        sysconfig.get_path("include"),
    ]
    for path in include_paths():
        flags.extend(["-I", path])
    for include_dir in context.include_dirs:
        flags.extend(["-I", str(include_dir)])
    hk_root = os.environ["HIPKITTENS_ROOT"]
    flags.extend([
        "-DKITTENS_CDNA4",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-ffast-math",
        "-Wl,--allow-multiple-definition",
        "-I",
        str(Path(hk_root) / "include"),
        "-I",
        str(Path(hk_root) / "prototype"),
    ])
    for library_dir in context.library_dirs:
        flags.extend(["-L", str(library_dir)])
    flags.extend(context.extra_flags)
    return flags


def _hipcc_link_flags() -> list[str]:
    from torch.utils.cpp_extension import library_paths

    flags = []
    for path in library_paths():
        flags.extend(["-L", path])
    flags.extend(["-ltorch_python", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-lc10", "-lc10_hip"])
    return flags


def _load_pybind_function(so: Path, entrypoint: str) -> Any:
    module_name, symbol_name = entrypoint.split(":", 1)
    spec = importlib.util.spec_from_file_location(module_name, so)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load extension module: {so}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


@dataclass(frozen=True)
class HipSourceBuilder:
    kind: CandidateKind = CandidateKind.HIP_SOURCE

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "hip_pybind")
        if submission.entrypoint is None:
            raise ValueError("hip source submissions require entrypoint")
        shared_objects = tuple(artifact for artifact in submission.artifacts if artifact.kind == ArtifactKind.SHARED_OBJECT)
        if shared_objects:
            if len(shared_objects) != 1:
                raise ValueError("hip source submissions support exactly one shared object artifact")
            so = shared_objects[0].path
            artifacts = normalize_artifacts(submission.artifacts)
        else:
            if submission.source is not None:
                source = Path(submission.source)
            else:
                source = artifact_by_kind(submission.artifacts, ArtifactKind.SOURCE).path
            if not source.exists():
                raise ValueError(f"source artifact does not exist: {source}")
            build_dir = context.work_dir / "engine_build" / source.stem
            build_dir.mkdir(parents=True, exist_ok=True)
            so = build_dir / f"candidate{_python_extension_suffix()}"
            log = build_dir / "build.log"
            manifest = build_dir / "manifest.normalized.json"
            flags = _hipcc_flags(context)
            link_flags = _hipcc_link_flags()
            output = _run_build(flags + [str(source), "-o", str(so), *link_flags])
            write_log(log, output)
            manifest_artifact = write_json(
                manifest,
                {
                    "kind": submission.kind,
                    "function_name": submission.abi.function_name,
                    "source": str(source),
                    "shared_object": str(so),
                    "target_arch": context.cuda_arch,
                    "flags": flags,
                    "link_flags": link_flags,
                },
            )
            artifacts = (
                materialize_artifact(ArtifactKind.SOURCE, source),
                materialize_artifact(ArtifactKind.SHARED_OBJECT, so),
                materialize_artifact(ArtifactKind.LOG, log),
                manifest_artifact,
            )
        dispatch = _load_pybind_function(so, submission.entrypoint)

        def run(inputs):
            dispatch(
                *(inputs.tensors[arg.name] for arg in submission.abi.tensor_args),
                *(inputs.scalars[arg.name] for arg in submission.abi.scalar_args),
            )

        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=artifacts,
            executor_factory=lambda: CallableExecutor(run),
            metadata=submission.metadata,
        )


@dataclass(frozen=True)
class CudaSourceBuilder:
    kind: CandidateKind = CandidateKind.CUDA_SOURCE

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        _runtime_policy(submission, DEFAULT_RUNTIME_ENV, "cuda_sm90a_default")
        scalars = _metadata_scalars(submission)
        require_cubin_abi(submission.abi)
        if submission.source is not None:
            source = Path(submission.source)
        else:
            source = artifact_by_kind(submission.artifacts, ArtifactKind.SOURCE).path
        if not source.exists():
            raise ValueError(f"source artifact does not exist: {source}")

        build_dir = context.work_dir / "engine_build" / source.stem
        build_dir.mkdir(parents=True, exist_ok=True)
        so = build_dir / "candidate.so"
        cubin = build_dir / "candidate.cubin"
        ptx = build_dir / "candidate.ptx"
        log = build_dir / "build.log"
        resource_usage = build_dir / "resource_usage.txt"
        manifest = build_dir / "manifest.normalized.json"
        flags = _nvcc_flags(context)

        output = ""
        output += _run_build(flags + ["-Xcompiler", "-fPIC", "--shared", "-o", str(so), str(source)])
        output += _run_build(flags + ["--ptx", "-o", str(ptx), str(source)])
        output += _run_build(flags + ["--cubin", "-o", str(cubin), str(source)])
        write_log(log, output)

        resource_result = subprocess.run(["cuobjdump", "--dump-resource-usage", str(cubin)], capture_output=True, text=True)
        resource_artifacts = ()
        if resource_result.returncode == 0:
            write_log(resource_usage, resource_result.stdout + resource_result.stderr)
            resource_artifacts = (materialize_artifact(ArtifactKind.RESOURCE_USAGE, resource_usage),)
        manifest_artifact = write_json(
            manifest,
            {
                "kind": submission.kind,
                "function_name": submission.abi.function_name,
                "source": str(source),
                "shared_object": str(so),
                "cubin": str(cubin),
                "cuda_arch": context.cuda_arch,
                "flags": flags,
            },
        )

        artifacts = (
            materialize_artifact(ArtifactKind.SOURCE, source),
            materialize_artifact(ArtifactKind.SHARED_OBJECT, so),
            materialize_artifact(ArtifactKind.CUBIN, cubin),
            materialize_artifact(ArtifactKind.PTX, ptx),
            materialize_artifact(ArtifactKind.LOG, log),
            manifest_artifact,
            *resource_artifacts,
        )
        setup_fn = make_cubin_so_setup(so, cubin, submission.abi, scalars)
        return BuiltCandidate(
            kind=submission.kind,
            abi=submission.abi,
            artifacts=artifacts,
            executor_factory=lambda: PreparedFunctionExecutor(setup_fn),
            metadata={**submission.metadata, "build_dir": build_dir},
        )
