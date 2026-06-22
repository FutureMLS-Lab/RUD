/*
CURRENTLY NOT WORKING!
*/

// mm_tk_abt_4096x3072x3072.cu — CUTLASS SM90 cluster GEMM for 4096×3072×3072
//
// C(M,N) = A(M,K) @ B(N,K)^T   (B row-major N×K, matching F.linear)
//
// Config: 128×256 tiles, 2×1 CTA cluster (cooperative A multicast),
// persistent scheduler, warp-specialized cooperative mainloop.
// Benchmarked ~110 us on H100 (standalone), beats F.linear (~115 us via PyTorch).

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

using ElementA           = cutlass::bfloat16_t;
using ElementB           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int Align = 16 / sizeof(ElementA);  // 8

// 128×256 tiles with 2×1 cluster — best config for 4096×3072×3072
using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;

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
    ElementD, LayoutD, Align,
    ElementD, LayoutD, Align,
    EpilogueSchedule, FusionOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, Align,
    ElementB, LayoutB, Align,
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

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Params = typename GemmKernel::Params;

// ---------------------------------------------------------------------------
// Host-side helpers for TKCC ExternalReplacementKernelSpec
// ---------------------------------------------------------------------------

extern "C" int cutlass_gemm_4096x3072x3072_globals_size() {
    return (int)sizeof(Params);
}

extern "C" void cutlass_gemm_4096x3072x3072_make_globals(
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

    Params params = GemmKernel::to_underlying_arguments(arguments, nullptr);
    memcpy(out_buf, &params, sizeof(Params));
}

extern "C" void cutlass_gemm_4096x3072x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { nullptr, stride_A, nullptr, stride_B },
        { {1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D }
    };

    Params params = GemmKernel::to_underlying_arguments(arguments, nullptr);
    dim3 grid = GemmKernel::get_grid_shape(params);
    *x = grid.x;
    *y = grid.y;
    *z = grid.z;
}

extern "C" int cutlass_gemm_4096x3072x3072_block_dim() {
    dim3 block = GemmKernel::get_block_shape();
    return block.x * block.y * block.z;
}

extern "C" int cutlass_gemm_4096x3072x3072_shmem_bytes() {
    return GemmKernel::SharedStorageSize;
}

// Force instantiation
template __global__ void cutlass::device_kernel<GemmKernel>(CUTLASS_GRID_CONSTANT Params const);
