// mm_tk_abt_4096_3072_128.cu — TK persistent ABt GEMM for M=4096, N=3072, K=128
//
// C[m,n] = sum_k A[m,k] * B[n,k]   (B row-major N×K, matching F.linear weight)
//
// Tile:  128M × 256N × 64K (2 K_TILES), 3 warpgroups (2 consumer + 1 producer)
// Grid:  132 persistent CTAs, SUPER_M=3 swizzle
// SMEM:  3-stage pipeline

#include "kittens.cuh"
#include "common.cuh"
#include "tma_desc_meta.cuh"

using namespace kittens;

#ifndef SUPER_M_VAL
#define SUPER_M_VAL 3
#endif

#ifndef GRID_BLOCKS_VAL
#define GRID_BLOCKS_VAL 132
#endif

#ifndef MIN_BLOCKS_PER_SM_VAL
#define MIN_BLOCKS_PER_SM_VAL 1
#endif

#ifndef INPUT_STAGES_VAL
#define INPUT_STAGES_VAL 3
#endif

#ifndef PRODUCER_REGS_VAL
#define PRODUCER_REGS_VAL 32
#endif

#ifndef CONSUMER_REGS_VAL
#define CONSUMER_REGS_VAL 192
#endif

namespace {

constexpr int ROW_BLOCK = 128;
constexpr int COL_BLOCK = 256;
constexpr int RED_BLOCK = 64;
constexpr int K_TILES = 2;
constexpr int INPUT_STAGES = INPUT_STAGES_VAL;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int ROW_BLOCKS = 4096 / ROW_BLOCK;
constexpr int COL_BLOCKS = 3072 / COL_BLOCK;
constexpr int TOTAL_TASKS = ROW_BLOCKS * COL_BLOCKS;
constexpr int SUPER_M =
    SUPER_M_VAL < 1 ? 1 : (SUPER_M_VAL < ROW_BLOCKS ? SUPER_M_VAL : ROW_BLOCKS);

using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
using B_tile = st_bf<COL_BLOCK, RED_BLOCK>;
using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;

struct globals {
    A_gl A;
    B_gl B;
    C_gl C;
};

struct pipeline_input {
    A_tile A[2];
    B_tile B;
};

constexpr int SHMEM_BYTES =
    INPUT_STAGES * (int)sizeof(pipeline_input) +
    CONSUMER_WARPGROUPS * (int)sizeof(C_tile) +
    1024;

template<typename T>
__device__ inline void toggle(T& value) {
    value ^= 1;
}

__device__ inline bool task_coords(int task_id, int& row_idx, int& col_idx) {
    if (task_id >= TOTAL_TASKS)
        return false;

    constexpr int SUPER_ROWS = (ROW_BLOCKS / SUPER_M) * SUPER_M;
    constexpr int FINAL_ROWS = ROW_BLOCKS - SUPER_ROWS;
    constexpr int SUPER_REPEAT = SUPER_M * COL_BLOCKS;

    if constexpr (SUPER_ROWS > 0) {
        if (task_id < SUPER_ROWS * COL_BLOCKS) {
            const int super_group = task_id / SUPER_REPEAT;
            const int within_group = task_id - super_group * SUPER_REPEAT;
            row_idx = super_group * SUPER_M + (within_group % SUPER_M);
            col_idx = within_group / SUPER_M;
            return true;
        }
    }

    const int tail_rows = FINAL_ROWS == 0 ? ROW_BLOCKS : FINAL_ROWS;
    const int remainder = task_id - SUPER_ROWS * COL_BLOCKS;
    row_idx = SUPER_ROWS + (remainder % tail_rows);
    col_idx = remainder / tail_rows;
    return true;
}

__device__ inline int input_slot(int task_iter, int k_tile) {
    return (task_iter * K_TILES + k_tile) % INPUT_STAGES;
}

} // namespace

__global__ __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM_VAL)
void tk_gemm_4096x3072x128_kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    pipeline_input (&inputs)[INPUT_STAGES] = allocator.allocate<pipeline_input, INPUT_STAGES>();
    C_tile (&outputs)[CONSUMER_WARPGROUPS] = allocator.allocate<C_tile, CONSUMER_WARPGROUPS>();

    __shared__ semaphore inputs_arrived[INPUT_STAGES];
    __shared__ semaphore inputs_finished[INPUT_STAGES];
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<A_tile>();
        g.B.template prefetch_tma<B_tile>();
        g.C.template prefetch_tma<C_tile>();
        #pragma unroll
        for (int i = 0; i < INPUT_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
    }
    __syncthreads();

    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();

    if (warpgroup_id == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<PRODUCER_REGS_VAL>();

        if (warp_id == 0 && lane_id == 0) {
            int input_phase[INPUT_STAGES] = {};
            for (int task_iter = 0;; ++task_iter) {
                int task_id = task_iter * gridDim.x + blockIdx.x;
                int row_idx, col_idx;
                if (!task_coords(task_id, row_idx, col_idx))
                    break;

                #pragma unroll
                for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
                    const int global_input = task_iter * K_TILES + k_tile;
                    const int slot = input_slot(task_iter, k_tile);
                    if (global_input >= INPUT_STAGES) {
                        wait(inputs_finished[slot], input_phase[slot]);
                        toggle(input_phase[slot]);
                    }
                    tma::expect_bytes(inputs_arrived[slot], sizeof(pipeline_input));
                    tma::load_async(inputs[slot].A[0], g.A, {row_idx * 2 + 0, k_tile}, inputs_arrived[slot]);
                    tma::load_async(inputs[slot].A[1], g.A, {row_idx * 2 + 1, k_tile}, inputs_arrived[slot]);
                    tma::load_async(inputs[slot].B, g.B, {col_idx, k_tile}, inputs_arrived[slot]);
                }
            }
        }
    } else {
        warpgroup::increase_registers<CONSUMER_REGS_VAL>();

        int input_phase[INPUT_STAGES] = {};

        for (int task_iter = 0;; ++task_iter) {
            int task_id = task_iter * gridDim.x + blockIdx.x;
            int row_idx, col_idx;
            if (!task_coords(task_id, row_idx, col_idx))
                break;

            rt_fl<ROW_BLOCK / 8, COL_BLOCK> accum;
            const int slot0 = input_slot(task_iter, 0);
            const int slot1 = input_slot(task_iter, 1);

            wait(inputs_arrived[slot0], input_phase[slot0]);
            toggle(input_phase[slot0]);
            warpgroup::mm_ABt(accum, inputs[slot0].A[warpgroup_id], inputs[slot0].B);

            wait(inputs_arrived[slot1], input_phase[slot1]);
            toggle(input_phase[slot1]);
            warpgroup::mma_ABt(accum, inputs[slot1].A[warpgroup_id], inputs[slot1].B);
            warpgroup::mma_async_wait();
            warp::arrive(inputs_finished[slot0]);
            warp::arrive(inputs_finished[slot1]);

            warpgroup::store(outputs[warpgroup_id], accum);
            warpgroup::sync(warpgroup_id + 1);
            if (warpgroup::laneid() == 0) {
                tma::store_async(g.C, outputs[warpgroup_id], {row_idx * 2 + warpgroup_id, col_idx});
                tma::store_async_read_wait();
            }
            warpgroup::sync(warpgroup_id + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers for cubin-based driver API launch.
// ---------------------------------------------------------------------------

extern "C" int tk_gemm_4096x3072x128_globals_size() {
    return (int)sizeof(globals);
}

extern "C" void tk_gemm_4096x3072x128_make_globals(
    void* out_buf,
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K
) {
    A_gl Ag {(bf16*)d_A,  nullptr, nullptr, (size_t)M, (size_t)K};
    B_gl Btg{(bf16*)d_Bt, nullptr, nullptr, (size_t)N, (size_t)K};
    C_gl Cg {(bf16*)d_C,  nullptr, nullptr, (size_t)M, (size_t)N};
    globals G{Ag, Btg, Cg};
    memcpy(out_buf, &G, sizeof(globals));
}

extern "C" void tk_gemm_4096x3072x128_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = GRID_BLOCKS_VAL;
    *y = 1;
    *z = 1;
}

extern "C" int tk_gemm_4096x3072x128_block_dim() {
    return NUM_THREADS;
}

extern "C" int tk_gemm_4096x3072x128_shmem_bytes() {
    return SHMEM_BYTES;
}

extern "C" int tk_gemm_4096x3072x128_num_tma_descriptors() { return 3; }

extern "C" void tk_gemm_4096x3072x128_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K,
    void* out_meta
) {
    // Descriptor 0: A[M, K] bf16, tile 64×64 (A_tile = st_bf<64, 64>)
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)d_A, 2, 1, 1, M, K, 64, 64);
    // Descriptor 1: Bt[N, K] bf16, tile 256×64 (B_tile = st_bf<256, 64>)
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, 1, 1, N, K, 256, 64);
    // Descriptor 2: C[M, N] bf16, tile 64×256 (C_tile = st_bf<64, 256>)
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)d_C, 2, 1, 1, M, N, 64, 256);
}
