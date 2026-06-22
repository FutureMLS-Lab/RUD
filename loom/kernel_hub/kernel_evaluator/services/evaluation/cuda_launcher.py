import ctypes
from pathlib import Path

import torch

from kernel_evaluator.services.evaluation.specs import require_cubin_abi, require_helper_abi
from kernel_evaluator.services.evaluation.types import ExecutionInputs, KernelABI, ScalarType

_SCALAR_CTYPES = {
    ScalarType.INT: ctypes.c_int,
    ScalarType.LONG_LONG: ctypes.c_longlong,
    ScalarType.FLOAT: ctypes.c_float,
    ScalarType.DOUBLE: ctypes.c_double,
    ScalarType.BOOL: ctypes.c_bool,
}


class CUlaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p),
        ("numAttrs", ctypes.c_uint),
    ]


def configure_helper_library(lib, abi: KernelABI) -> None:
    require_helper_abi(abi)
    scalar_ctypes = [_SCALAR_CTYPES[arg.dtype] for arg in abi.scalar_args]
    tensor_voidp = [ctypes.c_void_p] * len(abi.tensor_args)

    getattr(lib, abi.globals_size_fn).restype = ctypes.c_int
    getattr(lib, abi.globals_size_fn).argtypes = []
    getattr(lib, abi.block_dim_fn).restype = ctypes.c_int
    getattr(lib, abi.block_dim_fn).argtypes = []
    getattr(lib, abi.shmem_bytes_fn).restype = ctypes.c_int
    getattr(lib, abi.shmem_bytes_fn).argtypes = []
    getattr(lib, abi.make_globals_fn).restype = None
    getattr(lib, abi.make_globals_fn).argtypes = [ctypes.c_void_p] + tensor_voidp + scalar_ctypes
    getattr(lib, abi.grid_dims_fn).restype = None
    getattr(lib, abi.grid_dims_fn).argtypes = scalar_ctypes + [ctypes.POINTER(ctypes.c_int)] * 3

    if abi.run_fn is not None:
        getattr(lib, abi.run_fn).restype = None
        getattr(lib, abi.run_fn).argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]


def _scalar_cargs(abi: KernelABI, scalars: dict) -> list:
    return [_SCALAR_CTYPES[arg.dtype](scalars[arg.name]) for arg in abi.scalar_args]


def _prepare_globals(lib, abi: KernelABI, scalars: dict, inputs: ExecutionInputs):
    scalar_cargs = _scalar_cargs(abi, scalars)
    globals_size = getattr(lib, abi.globals_size_fn)()
    gbuf = (ctypes.c_uint8 * globals_size)()
    tensor_ptrs = [ctypes.c_void_p(inputs.tensors[tensor.name].data_ptr()) for tensor in abi.tensor_args]
    getattr(lib, abi.make_globals_fn)(ctypes.addressof(gbuf), *tensor_ptrs, *scalar_cargs)
    gx, gy, gz = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    getattr(lib, abi.grid_dims_fn)(*scalar_cargs, ctypes.byref(gx), ctypes.byref(gy), ctypes.byref(gz))
    block = getattr(lib, abi.block_dim_fn)()
    shmem = getattr(lib, abi.shmem_bytes_fn)()
    return gbuf, gx, gy, gz, block, shmem


def make_so_setup(so_path: Path, abi: KernelABI, scalars: dict):
    require_helper_abi(abi)

    def setup(inputs: ExecutionInputs):
        lib = ctypes.CDLL(str(so_path))
        configure_helper_library(lib, abi)
        gbuf, gx, gy, gz, block, shmem = _prepare_globals(lib, abi, scalars, inputs)

        if abi.run_fn is not None:
            def call():
                getattr(lib, abi.run_fn)(
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                    gx.value,
                    gy.value,
                    gz.value,
                    block,
                    shmem,
                    ctypes.addressof(gbuf),
                )

            return call

        cuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
        cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cuda.cuModuleLoad.restype = ctypes.c_int
        cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cuda.cuModuleGetFunction.restype = ctypes.c_int
        cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cuda.cuFuncSetAttribute.restype = ctypes.c_int
        cuda.cuLaunchKernelEx.argtypes = [ctypes.POINTER(CUlaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cuda.cuLaunchKernelEx.restype = ctypes.c_int

        if abi.kernel_symbol is None:
            raise ValueError("SO driver launch requires kernel_symbol")
        module = ctypes.c_void_p()
        rc = cuda.cuModuleLoad(ctypes.byref(module), str(so_path).encode())
        if rc != 0:
            raise RuntimeError(f"cuModuleLoad failed: {rc}")
        func = ctypes.c_void_p()
        rc = cuda.cuModuleGetFunction(ctypes.byref(func), module, abi.kernel_symbol.encode())
        if rc != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {rc}")
        rc = cuda.cuFuncSetAttribute(func, 8, ctypes.c_int(shmem))
        if rc != 0:
            raise RuntimeError(f"cuFuncSetAttribute failed: {rc}")
        kernel_args = (ctypes.c_void_p * 1)(ctypes.cast(gbuf, ctypes.c_void_p))

        def call():
            cfg = CUlaunchConfig(
                gridDimX=gx.value,
                gridDimY=gy.value,
                gridDimZ=gz.value,
                blockDimX=block,
                blockDimY=1,
                blockDimZ=1,
                sharedMemBytes=shmem,
                hStream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                attrs=None,
                numAttrs=0,
            )
            rc = cuda.cuLaunchKernelEx(ctypes.byref(cfg), func, kernel_args, None)
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernelEx failed: {rc}")

        return call

    return setup


def make_cubin_so_setup(so_path: Path, cubin_path: Path, abi: KernelABI, scalars: dict):
    require_cubin_abi(abi)

    def setup(inputs: ExecutionInputs):
        lib = ctypes.CDLL(str(so_path))
        configure_helper_library(lib, abi)
        gbuf, gx, gy, gz, block, shmem = _prepare_globals(lib, abi, scalars, inputs)

        cuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
        cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cuda.cuModuleLoad.restype = ctypes.c_int
        cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cuda.cuModuleGetFunction.restype = ctypes.c_int
        cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cuda.cuFuncSetAttribute.restype = ctypes.c_int
        cuda.cuLaunchKernelEx.argtypes = [ctypes.POINTER(CUlaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cuda.cuLaunchKernelEx.restype = ctypes.c_int

        module = ctypes.c_void_p()
        rc = cuda.cuModuleLoad(ctypes.byref(module), str(cubin_path).encode())
        if rc != 0:
            raise RuntimeError(f"cuModuleLoad failed: {rc}")
        func = ctypes.c_void_p()
        rc = cuda.cuModuleGetFunction(ctypes.byref(func), module, abi.kernel_symbol.encode())
        if rc != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {rc}")
        rc = cuda.cuFuncSetAttribute(func, 8, ctypes.c_int(shmem))
        if rc != 0:
            raise RuntimeError(f"cuFuncSetAttribute failed: {rc}")
        kernel_args = (ctypes.c_void_p * 1)(ctypes.cast(gbuf, ctypes.c_void_p))

        def call():
            cfg = CUlaunchConfig(
                gridDimX=gx.value,
                gridDimY=gy.value,
                gridDimZ=gz.value,
                blockDimX=block,
                blockDimY=1,
                blockDimZ=1,
                sharedMemBytes=shmem,
                hStream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                attrs=None,
                numAttrs=0,
            )
            rc = cuda.cuLaunchKernelEx(ctypes.byref(cfg), func, kernel_args, None)
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernelEx failed: {rc}")

        return call

    return setup
