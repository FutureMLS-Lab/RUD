// CUTLASS SM90 BF16 ABt GEMM — StreamK scheduler: C(M,N) = A(M,K) @ B(N,K)^T
// TileShape <128,256,64>, ClusterShape <1,1,1>, KernelTmaWarpSpecializedCooperative,
// StreamKScheduler with DecompositionMode::Heuristic.
//
// The StreamK scheduler dynamically distributes K-slices across all SMs when the
// number of output tiles is too small to fill the GPU (e.g. M=512 → only 4 M-tiles).
// The heuristic mode auto-selects between data-parallel and StreamK decomposition
// based on occupancy analysis, so it is safe to use for all shapes.
//
// Compiled to cubin and launched via CUDA driver API (run_fn path).

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_runtime.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Kernel type definition
// ---------------------------------------------------------------------------

using ElementA           = cutlass::bfloat16_t;
using ElementB           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;

// A(M,K) row-major, B(N,K) stored row-major → ABt pattern.
// LayoutB must be ColumnMajor so CUTLASS 3 places the static stride-1 on
// the K dimension and the dynamic stride (=K) on the N dimension, matching
// the physical B[n,k] = B_ptr[n*K + k] layout.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
static constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// StreamK dynamically splits K across SMs; heuristic mode auto-selects
// data-parallel vs. StreamK based on occupancy.
using TileScheduler    = cutlass::gemm::StreamKScheduler;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementD, float,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementD, LayoutD, AlignmentD,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler
>;

using Gemm   = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Params = typename GemmKernel::Params;

// ---------------------------------------------------------------------------
// Hardware info helper — queries current device SM count
// ---------------------------------------------------------------------------

static cutlass::KernelHardwareInfo get_hw_info() {
    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    return hw_info;
}

// ---------------------------------------------------------------------------
// Host-side helpers for TKCC ExternalReplacementKernelSpec
// ---------------------------------------------------------------------------

extern "C" int cutlass_gemm_abt_streamk_globals_size() {
    return (int)sizeof(Params);
}

// Persistent device-memory workspace for StreamK reduction.
static void*  s_streamk_workspace      = nullptr;
static size_t s_streamk_workspace_size = 0;

extern "C" void cutlass_gemm_abt_streamk_make_globals(
    void* out_buf,
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K
) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA*>(d_A), stride_A,
            static_cast<ElementB*>(d_Bt), stride_B,
        },
        {
            {1.0f, 0.0f},
            static_cast<ElementD*>(d_C), stride_D,
            static_cast<ElementD*>(d_C), stride_D,
        },
        get_hw_info()
    };

    size_t ws_size = Gemm::get_workspace_size(arguments);
    if (ws_size > 0) {
        if (s_streamk_workspace_size < ws_size) {
            if (s_streamk_workspace) cudaFree(s_streamk_workspace);
            cudaMalloc(&s_streamk_workspace, ws_size);
            s_streamk_workspace_size = ws_size;
        }
        cudaMemset(s_streamk_workspace, 0, ws_size);
    }

    Params params = GemmKernel::to_underlying_arguments(arguments, s_streamk_workspace);
    memcpy(out_buf, &params, sizeof(Params));
}

extern "C" void cutlass_gemm_abt_streamk_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B},
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D},
        get_hw_info()
    };

    Params params = GemmKernel::to_underlying_arguments(arguments, nullptr);
    dim3 grid = GemmKernel::get_grid_shape(params);
    *x = grid.x; *y = grid.y; *z = grid.z;
}

extern "C" int cutlass_gemm_abt_streamk_block_dim() {
    dim3 block = GemmKernel::get_block_shape();
    return block.x * block.y * block.z;
}

extern "C" int cutlass_gemm_abt_streamk_shmem_bytes() {
    return GemmKernel::SharedStorageSize;
}

// Force instantiation so the kernel symbol appears in the cubin
template __global__ void cutlass::device_kernel<GemmKernel>(CUTLASS_GRID_CONSTANT Params const);

// ---------------------------------------------------------------------------
// Self-contained run function — uses CUTLASS's GemmUniversalAdapter::run()
// which handles workspace allocation and correct launch configuration.
// Signature: run(A, Bt, C, M, N, K, stream) → void
// ---------------------------------------------------------------------------

extern "C" void cutlass_gemm_abt_streamk_run(
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K,
    void* stream
) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA*>(d_A), stride_A,
            static_cast<ElementB*>(d_Bt), stride_B,
        },
        {
            {1.0f, 0.0f},
            static_cast<ElementD*>(d_C), stride_D,
            static_cast<ElementD*>(d_C), stride_D,
        },
        get_hw_info()
    };

    size_t ws_size = Gemm::get_workspace_size(arguments);
    if (ws_size > 0) {
        if (s_streamk_workspace_size < ws_size) {
            if (s_streamk_workspace) cudaFree(s_streamk_workspace);
            cudaMalloc(&s_streamk_workspace, ws_size);
            s_streamk_workspace_size = ws_size;
        }
        cudaMemset(s_streamk_workspace, 0, ws_size);
    }

    Gemm gemm;
    gemm.run(arguments, s_streamk_workspace, static_cast<cudaStream_t>(stream));
}
