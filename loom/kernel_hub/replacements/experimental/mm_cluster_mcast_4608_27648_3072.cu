// mm_tk_abt_4608_27648_3072.cu — All-consumer cluster GEMM for 4608×27648×3072
//
// C = A(M,K) @ Bt(N,K)^T   (B stored row-major as N×K, matching F.linear)
//
// Architecture: all-consumer with 2×1 cluster and TMA multicast for B.
//   - 3 warpgroups × 128 threads = 384 total, ALL doing WGMMA
//   - Thread 0 of WG 0 handles TMA loads (brief divergence per K-step)
//   - CTA 0 multicasts B to both CTAs; CTA 1 loads only A
//   - Natural CTA sync through wait(inputs_arrived) — no explicit barrier.cluster between tiles
//
// Tile:      192M × 192N × 64K (M_BLOCK=3, N_BLOCK=3)
// Grid:      dim3(2, 66) = 132 CTAs = 66 clusters
// Pipeline:  4 stages
// Regs:      168/thread (96 accum + 72 overhead)
// Shmem:     ~200KB (4 × 48KB input + 72KB finish overlapping stage 3)

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

using namespace kittens;
using namespace kittens::prototype;

// ---- Tile configuration ----
static constexpr int M_BLOCK = 3, N_BLOCK = 3;
static constexpr int PIPE = 4;
static constexpr int SUPER_M = 12;
static constexpr int CLUSTER_M = 2;
static constexpr int NUM_WARPS = 12;     // 3 WGs × 4 warps
static constexpr int NUM_THREADS_K = NUM_WARPS * 32;  // 384

// ---- Shared memory types ----
using base_tile   = st_bf<64, 64>;
using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;

struct cluster_globals {
    global_layout A, B, C;
};

struct input_block {
    base_tile a[M_BLOCK];  // 3 × 8KB = 24KB
    base_tile b[N_BLOCK];  // 3 × 8KB = 24KB
};
// sizeof(input_block) = 48KB

struct finish_block {
    base_tile c[M_BLOCK][N_BLOCK];  // 9 × 8KB = 72KB
};

// ---- WGMMA types ----
using wide_rt = rt_fl<16, 64 * N_BLOCK>;   // rt_fl<16, 192>
using tall_st = st_bf<64 * N_BLOCK, 64>;   // st_bf<192, 64>

// ---- TMA load helper (called by thread 0 only) ----
__device__ __forceinline__ void issue_tma_loads(
    input_block *input_smem, int q,
    const cluster_globals &globals,
    int block_m_base, int block_n, int k_iter,
    int cta_rank, uint16_t cluster_mask,
    semaphore &bar
) {
    // CTA-local expect: 1 arrival + set expected TX bytes
    tma::expect_bytes(bar, sizeof(input_block));

    // Load A tiles (CTA-local TMA, each CTA loads its own M rows)
    for (int i = 0; i < M_BLOCK; i++)
        tma::load_async(input_smem[q].a[i], globals.A,
                        {block_m_base + i, k_iter}, bar);

    // Load B tiles: both CTAs load locally (no multicast baseline)
    for (int i = 0; i < N_BLOCK; i++)
        tma::load_async(input_smem[q].b[i], globals.B,
                         {block_n + i, k_iter}, bar);
}

// ---- Kernel ----
__global__ __launch_bounds__(NUM_THREADS_K, 1)
__cluster_dims__(CLUSTER_M, 1, 1)
void gemm_allconsumer(const __grid_constant__ cluster_globals globals) {
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]);

    // Allocate input pipeline ring
    input_block (&input_smem)[PIPE] = alloc.allocate<input_block, PIPE>();

    // Finish block at end of shmem (may overlap with input stage 3)
    constexpr int FINISH_OFFSET = (MAX_SHARED_MEMORY - 1024) - sizeof(finish_block);
    finish_block *finish_smem = reinterpret_cast<finish_block*>(
        ((reinterpret_cast<uint64_t>(&__shm[0]) + FINISH_OFFSET) / 1024) * 1024);

    // Compute SAFE_STAGES (input stages that don't overlap finish block)
    constexpr int SAFE_STAGES_RAW = (FINISH_OFFSET - 1024) / (int)sizeof(input_block);
    constexpr int SAFE_STAGES = SAFE_STAGES_RAW < PIPE ? SAFE_STAGES_RAW : PIPE;

    // Barriers
    __shared__ semaphore inputs_arrived[PIPE], inputs_finished[PIPE], finish_finished;
    uint32_t semaphore_bitfield = 0xFFFF0000;  // finished phases start as 1, arrived as 0

    // Cluster info
    int cta_rank = cluster_ctarank();       // 0 or 1
    int num_clusters = gridDim.y;           // 66
    uint16_t cluster_mask = (1 << CLUSTER_M) - 1;  // 0b11

    // Shape info
    int tiles_m = globals.C.rows() / (M_BLOCK * 64);
    int tiles_n = globals.C.cols() / (N_BLOCK * 64);
    int tiles_k = globals.A.cols() / 64;
    int cluster_tiles_m = tiles_m / CLUSTER_M;
    int total_cluster_tiles = cluster_tiles_m * tiles_n;

    // ---- Init barriers (single thread) ----
    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE; i++) {
            init_semaphore(inputs_arrived[i], 1, 0);   // count=1: thread 0's expect_bytes arrival
            init_semaphore(inputs_finished[i], NUM_WARPS, 0);  // count=12: all warps arrive
        }
        init_semaphore(finish_finished, NUM_WARPS, 0);  // count=12
    }
    everyone::tma::cluster::sync();  // all threads, same PC, ensures init visible

    // ---- Accumulators (in registers, all threads) ----
    rt_fl<16, 64> accum[N_BLOCK];

    // Separate phase tracking for producer (thread 0) vs consumer (all threads)
    // Producer tracks: inputs_finished phases (only thread 0 waits on these)
    // Consumer tracks: inputs_arrived phases (all threads wait on these)
    uint32_t producer_bitfield = 0xFFFF0000;  // inputs_finished: half=1, starts at 1
    uint32_t consumer_bitfield = 0x00000000;  // inputs_arrived: half=0, starts at 0
    int input_ring = 0;  // producer ring position

    // ---- Persistent tile loop ----
    for (int task_iter = 0; true; task_iter++) {
        int ctile = (int)blockIdx.y + task_iter * num_clusters;
        if (ctile >= total_cluster_tiles) break;

        // Tile scheduling: SUPER_M column-major grouping for L2 locality
        int super_m = SUPER_M / CLUSTER_M;  // 6
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
        int block_m_base = (cluster_m * CLUSTER_M + cta_rank) * M_BLOCK;

        // Zero accumulators
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++)
            kittens::warp::zero(accum[n]);

        // ---- Pipeline prefill (THREAD 0 ONLY waits on inputs_finished) ----
        if (threadIdx.x == 0) {
            int load_iter = 0;

            // Phase 1: fill SAFE_STAGES (don't overlap finish block)
            for (; load_iter < SAFE_STAGES && load_iter < tiles_k; load_iter++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(producer_bitfield, input_ring));
                update_phasebit<1>(producer_bitfield, input_ring);
                issue_tma_loads(input_smem, input_ring, globals,
                               block_m_base, block_n * N_BLOCK, load_iter,
                               cta_rank, cluster_mask, inputs_arrived[input_ring]);
                input_ring = ring_advance<PIPE>(input_ring);
            }

            // Wait for previous tile's finish phase (finish block now safe)
            wait(finish_finished, (task_iter % 2) ^ 1);

            // Phase 2: fill remaining stages (may overlap finish block)
            for (; load_iter < PIPE && load_iter < tiles_k; load_iter++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(producer_bitfield, input_ring));
                update_phasebit<1>(producer_bitfield, input_ring);
                issue_tma_loads(input_smem, input_ring, globals,
                               block_m_base, block_n * N_BLOCK, load_iter,
                               cta_rank, cluster_mask, inputs_arrived[input_ring]);
                input_ring = ring_advance<PIPE>(input_ring);
            }
        }
        __syncthreads();  // all threads converge after prefill

        // ---- Main K-loop (ALL threads participate) ----
        int compute_ring = 0;
        for (int ki = 0; ki < tiles_k; ki++) {
            int q = compute_ring;

            // ALL threads: wait for current K-step data (cluster-scope wait)
            tma::cluster::wait(inputs_arrived[q], get_phasebit<0>(consumer_bitfield, q));
            update_phasebit<0>(consumer_bitfield, q);

            // ALL threads: WGMMA (fence + mma + commit + wait are .sync.aligned)
            warpgroup::mma_ABt(
                reinterpret_cast<wide_rt&>(accum),
                input_smem[q].a[warpgroup::groupid()],
                reinterpret_cast<tall_st&>(input_smem[q].b)
            );
            warpgroup::mma_async_wait();

            // ALL warps: signal done consuming this stage
            if (warp::laneid() == 0) arrive(inputs_finished[q]);

            // Thread 0: prefetch for ki + PIPE (AFTER signaling done)
            if (ki + PIPE < tiles_k) {
                if (threadIdx.x == 0) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(producer_bitfield, input_ring));
                    update_phasebit<1>(producer_bitfield, input_ring);
                    issue_tma_loads(input_smem, input_ring, globals,
                                   block_m_base, block_n * N_BLOCK, ki + PIPE,
                                   cta_rank, cluster_mask, inputs_arrived[input_ring]);
                    input_ring = ring_advance<PIPE>(input_ring);
                }
            }

            compute_ring = ring_advance<PIPE>(compute_ring);
        }

        // ---- Finish phase: store output ----
        group<NUM_WARPS>::sync(14);  // all 12 warps sync before writing finish block

        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++)
            warpgroup::store(finish_smem->c[warpgroup::groupid()][n], accum[n]);
        warpgroup::sync(warpgroup::groupid() + 4);

        if (warpgroup::laneid() == 0) {
            int coord_x = block_m_base + warpgroup::groupid();
            int coord_y = block_n * N_BLOCK;
            for (int i = 0; i < N_BLOCK; i++) {
                tma::store_async(globals.C, finish_smem->c[warpgroup::groupid()][i],
                                 {coord_x, coord_y + i});
            }
            tma::store_async_read_wait();
        }

        if (warp::laneid() == 0) arrive(finish_finished);
        group<NUM_WARPS>::sync(14);  // ensure finish complete before next tile
    }

    // Final cluster sync
    everyone::tma::cluster::sync();
}

// ---------------------------------------------------------------------------
// Host-side helpers for TKCC launcher and standalone benchmark
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
