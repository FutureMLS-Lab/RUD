// CUTLASS SM90 BF16 ABt GEMM — Cooperative, 2×1×1 cluster: C(M,N) = A(M,K) @ B(N,K)^T
// TileShape <128,256,64>, ClusterShape <2,1,1>, KernelTmaWarpSpecializedCooperative.
//
// 2-CTA cluster allows the TMA to broadcast the B tile across both CTAs (only 1
// B tile load for 2 output tiles), cutting B-load traffic in half and improving
// utilisation of the 2nd-gen TMA engine on H100.  Both CTAs share the same B
// slice; each handles a different M-slice.  Requires M divisible by 256 (2×128).
//
// Compiled to cubin and launched via CUDA driver API.

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

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
using ClusterShape = Shape<_2, _1, _1>;  // 2-CTA cluster along M

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using TileScheduler    = cutlass::gemm::PersistentScheduler;

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
// Host-side helpers for TKCC ExternalReplacementKernelSpec
// ---------------------------------------------------------------------------

extern "C" int cutlass_gemm_abt_coop2_globals_size() {
    return (int)sizeof(Params);
}

// Persistent device-memory workspace for the cluster-aware PersistentScheduler.
// The tile counter is a small atomic (typically 4 bytes) that must be zeroed
// before each GEMM call so the scheduler starts from tile 0.
static void*  s_coop2_workspace      = nullptr;
static size_t s_coop2_workspace_size = 0;

extern "C" void cutlass_gemm_abt_coop2_make_globals(
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
        }
    };

    // Allocate (once) and zero the scheduler workspace before each launch.
    size_t ws_size = Gemm::get_workspace_size(arguments);
    if (ws_size > 0) {
        if (s_coop2_workspace_size < ws_size) {
            if (s_coop2_workspace) cudaFree(s_coop2_workspace);
            cudaMalloc(&s_coop2_workspace, ws_size);
            s_coop2_workspace_size = ws_size;
        }
        cudaMemset(s_coop2_workspace, 0, ws_size);
    }

    Params params = GemmKernel::to_underlying_arguments(arguments, s_coop2_workspace);
    memcpy(out_buf, &params, sizeof(Params));
}

extern "C" void cutlass_gemm_abt_coop2_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
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
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D}
    };

    Params params = GemmKernel::to_underlying_arguments(arguments, nullptr);
    dim3 grid = GemmKernel::get_grid_shape(params);
    *x = grid.x; *y = grid.y; *z = grid.z;
}

extern "C" int cutlass_gemm_abt_coop2_block_dim() {
    dim3 block = GemmKernel::get_block_shape();
    return block.x * block.y * block.z;
}

extern "C" int cutlass_gemm_abt_coop2_shmem_bytes() {
    return GemmKernel::SharedStorageSize;
}

// Force instantiation so the kernel symbol appears in the cubin
template __global__ void cutlass::device_kernel<GemmKernel>(CUTLASS_GRID_CONSTANT Params const);

// ---------------------------------------------------------------------------
// Self-contained run function — uses CUTLASS's GemmUniversalAdapter::run()
// which calls cudaLaunchKernelExC internally and correctly sets the
// CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION attribute for cluster kernels.
// Signature: run(A, Bt, C, M, N, K, stream) → void
// ---------------------------------------------------------------------------

extern "C" void cutlass_gemm_abt_coop2_run(
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
        }
    };

    size_t ws_size = Gemm::get_workspace_size(arguments);
    if (ws_size > 0) {
        if (s_coop2_workspace_size < ws_size) {
            if (s_coop2_workspace) cudaFree(s_coop2_workspace);
            cudaMalloc(&s_coop2_workspace, ws_size);
            s_coop2_workspace_size = ws_size;
        }
        cudaMemset(s_coop2_workspace, 0, ws_size);
    }

    Gemm gemm;
    gemm.run(arguments, s_coop2_workspace, static_cast<cudaStream_t>(stream));
}
