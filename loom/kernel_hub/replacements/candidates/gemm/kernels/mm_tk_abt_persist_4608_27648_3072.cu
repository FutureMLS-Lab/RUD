// mm_tk_abt_persist_4608_27648_3072.cu — Persistent bf16 ABt GEMM
//
// C = A(M,K) @ Bt(N,K)^T   (B stored row-major as N×K, matching F.linear)
//
// Persistent 132-block grid with dedicated producer warpgroup.
// 2 consumer WGs + 1 producer WG, 4 pipeline stages.
// SUPER_M=36 swizzle pattern for L2 locality.
//
// Tile:     128M × 256N × 64K
// Grid:     132 CTAs
// Threads:  288 (3 WGs: 2 consumers, 1 producer)
// Pipeline: 4 stages

#include "kittens.cuh"
#include "common.cuh"
#include "../include/tma_desc_meta.cuh"

using namespace kittens;

namespace {

constexpr int TASK_M = 4608;
constexpr int TASK_N = 27648;
constexpr int TASK_K = 3072;

struct config {
    static constexpr int NUM_BLOCKS = 132;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

struct globals {
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 36;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    static_assert(TASK_M % ROW_BLOCK == 0);
    static_assert(TASK_N % COL_BLOCK == 0);
    static_assert(TASK_K % RED_BLOCK == 0);

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_subtile = st_bf<RED_BLOCK, RED_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using A_gl = gl<bf16, 1, 1, TASK_M, TASK_K, A_tile>;
    using B_gl = gl<bf16, 1, 1, TASK_N, TASK_K, B_subtile>;
    using C_gl = gl<bf16, 1, 1, TASK_M, TASK_N, C_tile>;

    A_gl A;
    B_gl B;
    C_gl C;

    struct pipeline_inputs {
        A_tile A[2];
        B_subtile B[4];
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

static constexpr int TASK_ROW_BLOCKS = TASK_M / globals::ROW_BLOCK;
static constexpr int TASK_COL_BLOCKS = TASK_N / globals::COL_BLOCK;
static constexpr int TASK_SUPER_REPEAT = globals::SUPER_M * TASK_COL_BLOCKS;
static constexpr int TASK_NUM_BLOCKS = TASK_ROW_BLOCKS * TASK_COL_BLOCKS;
static constexpr int TASK_NUM_ITERS = TASK_K / globals::RED_BLOCK;
static constexpr int TASK_SHMEM_BYTES =
    int(sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) +
        sizeof(globals::pipeline_outputs));

static_assert(TASK_SHMEM_BYTES <= config::DYNAMIC_SHARED_MEMORY);

__device__ __forceinline__ void decode_task(int task_id, int &row_idx, int &col_idx) {
    row_idx = globals::SUPER_M * (task_id / TASK_SUPER_REPEAT) + (task_id % globals::SUPER_M);
    col_idx = (task_id % TASK_SUPER_REPEAT) / globals::SUPER_M;
}

extern "C" __global__ __launch_bounds__(config::NUM_THREADS, 1)
void tk_gemm_persist_4608x27648x3072_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] =
        allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs =
        *reinterpret_cast<globals::pipeline_outputs*>(&inputs[globals::PIPELINE_STAGES - 1]);

    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;

    if (threadIdx.x == 0) {
        G.A.template prefetch_tma<globals::A_tile>();
        G.B.template prefetch_tma<globals::B_subtile>();
        G.C.template prefetch_tma<globals::C_tile>();

#pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < TASK_NUM_BLOCKS; task_id += gridDim.x) {
                int row_idx, col_idx;
                decode_task(task_id, row_idx, col_idx);

                for (int red_idx = 0; red_idx < TASK_NUM_ITERS; red_idx++) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                    if (red_idx == globals::PIPELINE_STAGES - 1) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                    }
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        tma::load_async(inputs[stage].A[i], G.A, {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                    }
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        tma::load_async(inputs[stage].B[i], G.B, {col_idx * 4 + i, red_idx}, inputs_arrived[stage]);
                    }
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
        else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < TASK_NUM_BLOCKS; task_id += gridDim.x) {
                int row_idx, col_idx;
                decode_task(task_id, row_idx, col_idx);

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        G.C, outputs.C[i], {row_idx * 2 + i, col_idx}
                    );
                }
                tma::store_async_read_wait();
                arrive(outputs_finished);
            }
        }
    }
    else {
        using accum_tile = rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK>;
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < TASK_NUM_BLOCKS; task_id += gridDim.x) {
            accum_tile C_accum;
            using B_tile = st_bf<globals::COL_BLOCK, globals::RED_BLOCK>;
            warp::zero(C_accum);

            for (int red_idx = 0; red_idx < TASK_NUM_ITERS; red_idx++) {
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                warpgroup::mma_ABt(C_accum, inputs[stage].A[warpgroup_id], reinterpret_cast<B_tile&>(inputs[stage].B));
                warpgroup::mma_async_wait();
                warp::arrive(inputs_finished[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }

            group<8>::sync(3);
            warpgroup::store(outputs.C[warpgroup_id], C_accum);
            warpgroup::sync(warpgroup_id + 1);
            warpgroup::arrive(outputs_arrived);
        }
    }
}

} // namespace

extern "C" int tk_gemm_persist_4608x27648x3072_globals_size() {
    return (int)sizeof(globals);
}

extern "C" void tk_gemm_persist_4608x27648x3072_make_globals(
    void* out_buf,
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K
) {
    (void)M;
    (void)N;
    (void)K;
    globals::A_gl Ag{(bf16*)d_A, nullptr, nullptr, nullptr, nullptr};
    globals::B_gl Bg{(bf16*)d_Bt, nullptr, nullptr, nullptr, nullptr};
    globals::C_gl Cg{(bf16*)d_C, nullptr, nullptr, nullptr, nullptr};
    globals G{Ag, Bg, Cg};
    memcpy(out_buf, &G, sizeof(globals));
}

extern "C" void tk_gemm_persist_4608x27648x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    (void)M;
    (void)N;
    (void)K;
    *x = config::NUM_BLOCKS;
    *y = 1;
    *z = 1;
}

extern "C" int tk_gemm_persist_4608x27648x3072_block_dim() {
    return config::NUM_THREADS;
}

extern "C" int tk_gemm_persist_4608x27648x3072_shmem_bytes() {
    return config::DYNAMIC_SHARED_MEMORY;
}

extern "C" int tk_gemm_persist_4608x27648x3072_num_tma_descriptors() { return 3; }

extern "C" void tk_gemm_persist_4608x27648x3072_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K,
    void* out_meta
) {
    // A[M, K] bf16, tile 64×64 (A_tile = st_bf<ROW_BLOCK/2, RED_BLOCK> = st_bf<64,64>)
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)d_A, 2, 1, 1, M, K, 64, 64);
    // Bt[N, K] bf16, tile 64×64 (B_subtile = st_bf<RED_BLOCK, RED_BLOCK> = st_bf<64,64>)
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, 1, 1, N, K, 64, 64);
    // C[M, N] bf16, tile 64×256 (C_tile = st_bf<ROW_BLOCK/2, COL_BLOCK> = st_bf<64,256>)
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)d_C, 2, 1, 1, M, N, 64, 256);
}
