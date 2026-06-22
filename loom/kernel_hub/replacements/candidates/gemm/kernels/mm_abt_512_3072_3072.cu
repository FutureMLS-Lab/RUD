// mm_abt_512x3072x3072.cu — TK persistent bf16 ABt GEMM for M=512, N=3072, K=3072
//
// C[m,n] = sum_k A[m,k] * B[n,k]   (B row-major N×K, matching F.linear weight)
//
// Config:
//   Tile:   128M × 96N × 64K
//   Grid:   144 blocks (persistent, M-major super-tile ordering)
//   Threads: 384 (3 WGs: 1 producer, 2 consumers)
//   Pipeline depth: auto-selected (up to 12 stages)
//   TMA loads: A[64×64] per consumer WG + B[96×64] per stage
//   K-iterations: 3072/64 = 48

#define ROW_BLOCK_VAL 128
#define COL_BLOCK_VAL 96
#define RED_BLOCK_VAL 64
#define PIPELINE_STAGES_VAL 12
#define SUPER_M_VAL 2
#define PRODUCER_REGS_VAL 56
#define CONSUMER_REGS_VAL 160
#define MMA_WAIT_GROUPS_VAL 2
#define GRID_BLOCKS_VAL 144

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"
#include "tma_desc_meta.cuh"

using namespace kittens;
using namespace kittens::prototype;

static constexpr int kRowBlock = ROW_BLOCK_VAL;
static constexpr int kColBlock = COL_BLOCK_VAL;
static constexpr int kRedBlock = RED_BLOCK_VAL;
static constexpr int kRequestedStages = PIPELINE_STAGES_VAL;
static constexpr int kSuperM = SUPER_M_VAL;
static constexpr int kProducerRegs = PRODUCER_REGS_VAL;
static constexpr int kConsumerRegs = CONSUMER_REGS_VAL;
static constexpr int kMmaWaitGroups = MMA_WAIT_GROUPS_VAL;
static constexpr int kConsumerWarpgroups = kRowBlock / 64;
static constexpr int kConsumerWarps = kConsumerWarpgroups * 4;
static constexpr int kProducerWarps = 4;
static constexpr int kNumWarps = kConsumerWarps + kProducerWarps;
static constexpr int kNumThreads = kNumWarps * kittens::WARP_THREADS;
static constexpr int kRowBlocks = 512 / kRowBlock;
static constexpr int kColBlocks = 3072 / kColBlock;
static constexpr int kNumTasks = kRowBlocks * kColBlocks;
static constexpr int kNumIters = 3072 / kRedBlock;

static_assert(kConsumerWarpgroups >= 1 && kConsumerWarpgroups <= 2);
static_assert(kRowBlocks % kSuperM == 0);

struct globals {
    using a_tile = st_bf<kRowBlock / kConsumerWarpgroups, kRedBlock>;
    using b_tile = st_bf<kColBlock, kRedBlock>;
    using c_tile = st_bf<kRowBlock / kConsumerWarpgroups, kColBlock>;
    using a_layout = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_layout = gl<bf16, 1, 1, -1, -1, b_tile>;
    using c_layout = gl<bf16, 1, 1, -1, -1, c_tile>;

    a_layout A;
    b_layout B;
    c_layout C;
};

struct pipeline_inputs {
    globals::a_tile A[kConsumerWarpgroups];
    globals::b_tile B;
};

struct pipeline_outputs {
    globals::c_tile C[kConsumerWarpgroups];
};

constexpr int align_up_const(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static constexpr bool kOverlayOutputs = sizeof(pipeline_outputs) <= sizeof(pipeline_inputs);
static constexpr int kSharedMemoryBudget = MAX_SHARED_MEMORY - 1024;
static constexpr int kInputBytes = (int)sizeof(pipeline_inputs);
static constexpr int kOutputBytes = (int)sizeof(pipeline_outputs);

constexpr int required_smem_bytes(int stages) {
    return kOverlayOutputs
        ? (kInputBytes * stages)
        : (align_up_const(kInputBytes * stages, 1024) + kOutputBytes);
}

constexpr bool stage_supported(int stages) {
    return stages >= 2 &&
           stages <= kRequestedStages &&
           (kNumIters % stages == 0) &&
           required_smem_bytes(stages) <= kSharedMemoryBudget;
}

constexpr int pick_stage_count() {
    return stage_supported(10) ? 10 :
           stage_supported(9) ? 9 :
           stage_supported(8) ? 8 :
           stage_supported(6) ? 6 :
           stage_supported(4) ? 4 :
           stage_supported(3) ? 3 :
           stage_supported(2) ? 2 :
           0;
}

static constexpr int kStages = pick_stage_count();
static_assert(kStages >= 2, "Tile shape exceeds shared memory budget");
static_assert(kNumIters % kStages == 0);
static constexpr int kShmemBytes = required_smem_bytes(kStages) + 1024;

__device__ inline void task_coords(int task_id, int &row_idx, int &col_idx) {
    constexpr int kSuperBlocks = kSuperM * kColBlocks;
    row_idx = kSuperM * (task_id / kSuperBlocks) + task_id % kSuperM;
    col_idx = (task_id % kSuperBlocks) / kSuperM;
}

extern "C" __global__ __launch_bounds__(kNumThreads, 1)
void gemm_512x3072x3072_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    pipeline_inputs (&inputs)[kStages] = allocator.allocate<pipeline_inputs, kStages>();

    pipeline_outputs *outputs_ptr;
    if constexpr (kOverlayOutputs) {
        outputs_ptr = reinterpret_cast<pipeline_outputs*>(&inputs[kStages - 1]);
    }
    else {
        outputs_ptr = &allocator.allocate<pipeline_outputs>();
    }
    pipeline_outputs &outputs = *outputs_ptr;

    __shared__ semaphore inputs_arrived[kStages];
    __shared__ semaphore inputs_finished[kStages];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;

    if (threadIdx.x == 0) {
        G.A.template prefetch_tma<globals::a_tile>();
        G.B.template prefetch_tma<globals::b_tile>();
        G.C.template prefetch_tma<globals::c_tile>();
        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, kConsumerWarps);
        }
        init_semaphore(outputs_arrived, 0, kConsumerWarpgroups);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    if (warpgroup_id == kConsumerWarpgroups) {
        warpgroup::decrease_registers<kProducerRegs>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < kNumTasks; task_id += gridDim.x) {
                int row_idx, col_idx;
                task_coords(task_id, row_idx, col_idx);
                const int global_row = row_idx * kConsumerWarpgroups;

                for (int red_idx = 0; red_idx < kNumIters; ++red_idx) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(pipeline_inputs));

                    if constexpr (kOverlayOutputs) {
                        if (red_idx == kStages - 1) {
                            wait(outputs_finished, get_phasebit<1>(phasebits, kStages));
                            update_phasebit<1>(phasebits, kStages);
                        }
                    }
                    else if (red_idx == 0) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, kStages));
                        update_phasebit<1>(phasebits, kStages);
                    }

                    #pragma unroll
                    for (int i = 0; i < kConsumerWarpgroups; ++i) {
                        tma::load_async(
                            inputs[stage].A[i],
                            G.A,
                            {global_row + i, red_idx},
                            inputs_arrived[stage]
                        );
                    }
                    tma::load_async(inputs[stage].B, G.B, {col_idx, red_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % kStages;
                }
            }
        }
        else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < kNumTasks; task_id += gridDim.x) {
                int row_idx, col_idx;
                task_coords(task_id, row_idx, col_idx);
                const int global_row = row_idx * kConsumerWarpgroups;

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                #pragma unroll
                for (int i = 0; i < kConsumerWarpgroups; ++i) {
                    tma::store_async(G.C, outputs.C[i], {global_row + i, col_idx});
                }
                tma::store_async_read_wait();
                arrive(outputs_finished);
            }
        }
    }
    else {
        warpgroup::increase_registers<kConsumerRegs>();

        for (int task_id = blockIdx.x; task_id < kNumTasks; task_id += gridDim.x) {
            rt_fl<16, kColBlock> accum;
            warp::zero(accum);

            for (int red_idx = 0; red_idx < kNumIters; ++red_idx) {
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                warpgroup::mma_ABt(accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
                warpgroup::mma_async_wait<kMmaWaitGroups>();
                warp::arrive(inputs_finished[stage]);
                stage = (stage + 1) % kStages;
            }

            warpgroup::mma_async_wait();
            group<kConsumerWarps>::sync(3);
            warpgroup::store(outputs.C[warpgroup_id], accum);
            warpgroup::sync(warpgroup_id + 1);
            warpgroup::arrive(outputs_arrived);
        }
    }
}

extern "C" int gemm_512x3072x3072_globals_size() {
    return (int)sizeof(globals);
}

extern "C" void gemm_512x3072x3072_make_globals(
    void* out_buf,
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K
) {
    globals::a_layout Ag{(bf16*)d_A,  nullptr, nullptr, (size_t)M, (size_t)K};
    globals::b_layout Bg{(bf16*)d_Bt, nullptr, nullptr, (size_t)N, (size_t)K};
    globals::c_layout Cg{(bf16*)d_C,  nullptr, nullptr, (size_t)M, (size_t)N};
    globals G{Ag, Bg, Cg};
    memcpy(out_buf, &G, sizeof(globals));
}

extern "C" void gemm_512x3072x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = kNumTasks < GRID_BLOCKS_VAL ? kNumTasks : GRID_BLOCKS_VAL;
    *y = 1;
    *z = 1;
}

extern "C" int gemm_512x3072x3072_block_dim() {
    return kNumThreads;
}

extern "C" int gemm_512x3072x3072_shmem_bytes() {
    return kShmemBytes;
}

extern "C" int gemm_512x3072x3072_num_tma_descriptors() {
    return 2;
}

extern "C" void gemm_512x3072x3072_describe_tma_descriptors(
    void *d_A, void *d_Bt, void *d_C,
    int M, int N, int K,
    void *out_meta
) {
    _fill_tma_desc_meta_2d(
        (char*)out_meta + 0 * 96,
        (uint64_t)(uintptr_t)d_A,
        2,
        (uint64_t)M, (uint64_t)K,
        64, 64
    );
    _fill_tma_desc_meta_2d(
        (char*)out_meta + 1 * 96,
        (uint64_t)(uintptr_t)d_Bt,
        2,
        (uint64_t)N, (uint64_t)K,
        96, 64
    );
}
