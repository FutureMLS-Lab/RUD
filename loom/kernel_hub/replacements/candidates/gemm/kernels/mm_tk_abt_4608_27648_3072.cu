// mm_tk_abt_4608_27648_3072.cu — Cluster-based persistent bf16 ABt GEMM
//
// C = A(M,K) @ Bt(N,K)^T   (B stored row-major as N×K, matching F.linear)
//
// Dedicated-producer architecture with 2×1 cluster (no multicast yet).
// Both CTAs load A and B independently. Cluster enables future multicast.
//
// Tile:     128M × 256N × 64K (M_BLOCK=2, N_BLOCK=4)
// Cluster:  2×1 (132 CTAs = 66 clusters)
// Grid:     dim3(2, 66, 1)
// Threads:  384 (3 WGs: 1 producer, 2 consumers)
// Pipeline: 4 stages
// Regs:     168/thread
//
// Performance: 1063 us (0.970x F.linear) on H100, correct output.
// Alignment: M%128==0, N%256==0, K%64==0

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

using namespace kittens;
using namespace kittens::prototype;

static constexpr int M_BLOCK = 2, N_BLOCK = 4;
static constexpr int PIPE = 4;
static constexpr int SUPER_M = 12;
static constexpr int CLUSTER_M = 2;
static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;  // 8
static constexpr int NUM_PRODUCER_WARPS = 4;
static constexpr int NUM_THREADS_K = (NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * 32;  // 384

using base_tile    = st_bf<64, 64>;
using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;

struct cluster_globals {
    global_layout A, B, C;
};

struct input_block {
    base_tile a[M_BLOCK], b[N_BLOCK];
};

struct finish_block {
    base_tile c[M_BLOCK][N_BLOCK];
};

__global__ __launch_bounds__(NUM_THREADS_K, 1)
__cluster_dims__(CLUSTER_M, 1, 1)
void gemm_cluster(const __grid_constant__ cluster_globals globals) {
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]);
    input_block (&input_smem)[PIPE] = alloc.allocate<input_block, PIPE>();

    constexpr int FINISH_OFFSET = (MAX_SHARED_MEMORY - 1024) - sizeof(finish_block);
    finish_block *finish_smem = reinterpret_cast<finish_block*>(
        ((reinterpret_cast<uint64_t>(&__shm[0]) + FINISH_OFFSET) / 1024) * 1024);

    constexpr int SAFE_STAGES_RAW = (FINISH_OFFSET - 1024) / (int)sizeof(input_block);
    constexpr int SAFE_STAGES = SAFE_STAGES_RAW < PIPE ? SAFE_STAGES_RAW : PIPE;

    __shared__ semaphore inputs_arrived[PIPE], inputs_finished[PIPE], finish_finished;
    uint32_t semaphore_bitfield = 0xFFFF0000;

    int cta_rank = cluster_ctarank();
    int num_clusters = gridDim.y;

    int tiles_m = globals.C.rows() / (M_BLOCK * 64);
    int tiles_n = globals.C.cols() / (N_BLOCK * 64);
    int tiles_k = globals.A.cols() / 64;
    int cluster_tiles_m = tiles_m / CLUSTER_M;
    int total_cluster_tiles = cluster_tiles_m * tiles_n;

    if (warpid() >= NUM_CONSUMER_WARPS) {
        // ====== PRODUCER WARP GROUP ======
        warpgroup::decrease_registers<40>();
        if (warpid() == NUM_CONSUMER_WARPS) {
            for (int i = 0; i < PIPE; i++) {
                init_semaphore(inputs_arrived[i], 1, 0);
                init_semaphore(inputs_finished[i], NUM_CONSUMER_WARPS, 0);
            }
            init_semaphore(finish_finished, NUM_CONSUMER_WARPS, 0);
        }
        everyone::tma::cluster::sync();

        int input_ring = 0;
        for (int task_iter = 0; true; task_iter++) {
            int ctile = (int)blockIdx.y + task_iter * num_clusters;
            if (ctile >= total_cluster_tiles) break;

            // Tile scheduling
            int super_m = SUPER_M / CLUSTER_M;
            int super_rows = (cluster_tiles_m / super_m) * super_m;
            int final_rows = cluster_tiles_m - super_rows;
            int super_repeat = super_m * tiles_n;
            int cluster_m, block_n;
            if (ctile < super_rows * tiles_n) {
                cluster_m = super_m * (ctile / super_repeat) + ctile % super_m;
                block_n = (ctile % super_repeat) / super_m;
            } else {
                int rid = ctile - super_rows * tiles_n;
                cluster_m = super_rows + (rid % final_rows);
                block_n = rid / final_rows;
            }
            int block_m = (cluster_m * CLUSTER_M + cta_rank) * M_BLOCK;
            int block_n_base = block_n * N_BLOCK;

            int load_iter = 0;
            for (; load_iter < SAFE_STAGES && load_iter < tiles_k; load_iter++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                if (warpid() == NUM_CONSUMER_WARPS && laneid() == 0) {
                    tma::expect_bytes(inputs_arrived[input_ring], sizeof(input_block));
                    for (int i = 0; i < M_BLOCK; i++)
                        tma::load_async(input_smem[input_ring].a[i], globals.A,
                                        {block_m + i, load_iter}, inputs_arrived[input_ring]);
                    for (int i = 0; i < N_BLOCK; i++)
                        tma::load_async(input_smem[input_ring].b[i], globals.B,
                                        {block_n_base + i, load_iter}, inputs_arrived[input_ring]);
                }
                input_ring = ring_advance<PIPE>(input_ring);
            }

            wait(finish_finished, (task_iter % 2) ^ 1);

            for (; load_iter < tiles_k; load_iter++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                if (warpid() == NUM_CONSUMER_WARPS && laneid() == 0) {
                    tma::expect_bytes(inputs_arrived[input_ring], sizeof(input_block));
                    for (int i = 0; i < M_BLOCK; i++)
                        tma::load_async(input_smem[input_ring].a[i], globals.A,
                                        {block_m + i, load_iter}, inputs_arrived[input_ring]);
                    for (int i = 0; i < N_BLOCK; i++)
                        tma::load_async(input_smem[input_ring].b[i], globals.B,
                                        {block_n_base + i, load_iter}, inputs_arrived[input_ring]);
                }
                input_ring = ring_advance<PIPE>(input_ring);
            }
            group<NUM_PRODUCER_WARPS>::sync(13);
        }
    } else {
        // ====== CONSUMER WARP GROUPS ======
        warpgroup::increase_registers<232>();
        everyone::tma::cluster::sync();

        using wide_rt = rt_fl<16, 64 * N_BLOCK>;
        using tall_st = st_bf<64 * N_BLOCK, 64>;
        rt_fl<16, 64> accum[N_BLOCK];
        int input_ring = 0;

        for (int task_iter = 0; true; task_iter++) {
            int ctile = (int)blockIdx.y + task_iter * num_clusters;
            if (ctile >= total_cluster_tiles) break;

            // Same tile scheduling
            int super_m = SUPER_M / CLUSTER_M;
            int super_rows = (cluster_tiles_m / super_m) * super_m;
            int final_rows = cluster_tiles_m - super_rows;
            int super_repeat = super_m * tiles_n;
            int cluster_m, block_n;
            if (ctile < super_rows * tiles_n) {
                cluster_m = super_m * (ctile / super_repeat) + ctile % super_m;
                block_n = (ctile % super_repeat) / super_m;
            } else {
                int rid = ctile - super_rows * tiles_n;
                cluster_m = super_rows + (rid % final_rows);
                block_n = rid / final_rows;
            }
            int block_m = (cluster_m * CLUSTER_M + cta_rank) * M_BLOCK + warpgroup::groupid();
            int block_n_base = block_n * N_BLOCK;

            for (int n = 0; n < N_BLOCK; n++)
                kittens::warp::zero(accum[n]);

            for (int ki = 0; ki < tiles_k; ki++) {
                wait(inputs_arrived[input_ring], get_phasebit<0>(semaphore_bitfield, input_ring));
                update_phasebit<0>(semaphore_bitfield, input_ring);

                warpgroup::mma_ABt(
                    reinterpret_cast<wide_rt&>(accum),
                    input_smem[input_ring].a[warpgroup::groupid()],
                    reinterpret_cast<tall_st&>(input_smem[input_ring].b)
                );
                warpgroup::mma_async_wait();
                if (warp::laneid() == 0) arrive(inputs_finished[input_ring]);
                input_ring = ring_advance<PIPE>(input_ring);
            }

            group<NUM_CONSUMER_WARPS>::sync(14);
            for (int n = 0; n < N_BLOCK; n++)
                warpgroup::store(finish_smem->c[warpgroup::groupid()][n], accum[n]);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(globals.C, finish_smem->c[warpgroup::groupid()][i],
                                     {block_m, block_n_base + i});
                }
                tma::store_async_read_wait();
            }
            if (warp::laneid() == 0) arrive(finish_finished);
            group<NUM_CONSUMER_WARPS>::sync(14);
        }
    }
    everyone::tma::cluster::sync();
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

extern "C" int tk_gemm_4608x27648x3072_globals_size() {
    return (int)sizeof(cluster_globals);
}

extern "C" void tk_gemm_4608x27648x3072_make_globals(
    void* out_buf, void* d_A, void* d_Bt, void* d_C, int M, int N, int K
) {
    global_layout Ag{(bf16*)d_A,  nullptr, nullptr, (size_t)M, (size_t)K};
    global_layout Bg{(bf16*)d_Bt, nullptr, nullptr, (size_t)N, (size_t)K};
    global_layout Cg{(bf16*)d_C,  nullptr, nullptr, (size_t)M, (size_t)N};
    cluster_globals G{Ag, Bg, Cg};
    memcpy(out_buf, &G, sizeof(G));
}

extern "C" void tk_gemm_4608x27648x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = CLUSTER_M;  // 2
    *y = 66;          // clusters
    *z = 1;
}

extern "C" int tk_gemm_4608x27648x3072_block_dim() {
    return NUM_THREADS_K;  // 384
}

extern "C" int tk_gemm_4608x27648x3072_shmem_bytes() {
    return MAX_SHARED_MEMORY - 1024;
}
