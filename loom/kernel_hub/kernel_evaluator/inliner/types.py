import ctypes
from dataclasses import dataclass, replace
from enum import Enum


class CScalarType(Enum):
    INT       = ("int",       ctypes.c_int)
    LONG_LONG = ("long long", ctypes.c_longlong)
    FLOAT     = ("float",     ctypes.c_float)
    DOUBLE    = ("double",    ctypes.c_double)
    BOOL      = ("bool",      ctypes.c_bool)

    def __new__(cls, c_str: str, ctype: type):
        obj = object.__new__(cls)
        obj._value_ = c_str
        obj.ctype = ctype
        return obj

    @property
    def ctype_str(self) -> str:
        return f"ctypes.{self.ctype.__name__}"

    def parse_value(self, raw) -> int | float | bool:
        return self.ctype(raw).value


@dataclass
class ExternalReplacementKernelSpec:
    ptx_path: str
    library_path: str
    function_name: str
    tensor_args: list[str]
    scalar_args: dict[str, CScalarType]
    kernel_symbol: str
    globals_size_fn: str
    make_globals_fn: str
    grid_dims_fn: str
    block_dim_fn: str
    shmem_bytes_fn: str
    pointer_arg_roles: dict[str, str] | None = None
    profiling_physical_stride_scalars: dict[str, list[str]] | None = None
    codegen_stride_override_scalars: dict[str, list[str]] | None = None
    max_barrier_slots: int = -1
    num_tma_descriptors_fn: str | None = None
    describe_tma_descriptors_fn: str | None = None
    cluster_shape: tuple[int, int, int] = (1, 1, 1)
    run_fn: str | None = None

    def _load_lib(self) -> ctypes.CDLL:
        return ctypes.CDLL(self.library_path)

    def get_shmem_bytes(self) -> int:
        lib = self._load_lib()
        return getattr(lib, self.shmem_bytes_fn)()

    def get_block_dim(self) -> int:
        lib = self._load_lib()
        return getattr(lib, self.block_dim_fn)()

    def get_grid_size(self, kwargs: dict) -> int:
        lib = self._load_lib()
        scalar_args = [ctype.ctype(kwargs.get(s, 0)) for s, ctype in self.scalar_args.items()]
        gx, gy, gz = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        getattr(lib, self.grid_dims_fn)(*scalar_args, ctypes.byref(gx), ctypes.byref(gy), ctypes.byref(gz))
        return gx.value * gy.value * gz.value

    def resolve_grid(self, lp):
        gx, gy, gz = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        lib = self._load_lib()
        scalar_args = [ctype.ctype(lp.kwargs.get(s, 0)) for s, ctype in self.scalar_args.items()]
        getattr(lib, self.grid_dims_fn)(*scalar_args, ctypes.byref(gx), ctypes.byref(gy), ctypes.byref(gz))
        return replace(lp, grid=(gx.value, gy.value, gz.value))
