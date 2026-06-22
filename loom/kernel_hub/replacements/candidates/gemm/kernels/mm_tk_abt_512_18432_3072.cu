// mm_tk_abt_512_18432_3072.cu — Hand-tuned TK GEMM for M=512, N=18432, K=3072
//
// Bypasses LCF framework overhead. Uses TK primitives directly with a
// hand-written pipeline loop where all SMEM addresses resolve at compile time.
//
// Config: M_BLOCK=2, N_BLOCK=3, SUPER_M=4, 5-stage pipeline
//   Tile: 128×192 (matching cuBLAS nvjet configuration)
//   Grid: 132 persistent CTAs (384 total tiles: 4M × 96N)
//   Threads: 384 (8 consumer + 4 producer warps)
//   Pipeline: 5 input stages, finish block overlaps with stage 4
//   SMEM: ~227KB (input ring 200KB, finish 48KB overlapped)

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"
#include "tma_desc_meta.cuh"

using namespace kittens;
using namespace kittens::prototype;

static constexpr int M_BLOCK = 2, N_BLOCK = 3, SUPER_M = 4;
static constexpr int PIPE = 4;
static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;  // 8
static constexpr int NUM_PRODUCER_WARPS = 4;
static constexpr int NUM_THREADS_K = (NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * 32;  // 384

using base_tile    = st_bf<64, 64>;
using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;

struct gemm_globals { global_layout A, B, C; };

struct input_block  { base_tile a[M_BLOCK], b[N_BLOCK]; };
struct finish_block { base_tile c[M_BLOCK][N_BLOCK]; };

__global__ __launch_bounds__(NUM_THREADS_K, 1)
void tuned_512x18432x3072_kernel(const __grid_constant__ gemm_globals globals) {
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]);
    input_block (&inputs)[PIPE] = alloc.allocate<input_block, PIPE>();

    // Place finish_block at fixed offset near end of SMEM (overlaps with last input stage)
    constexpr int FINISH_OFFSET = (MAX_SHARED_MEMORY - 1024) - sizeof(finish_block);
    finish_block *finish = reinterpret_cast<finish_block*>(
        ((reinterpret_cast<uint64_t>(&__shm[0]) + FINISH_OFFSET) / 1024) * 1024);

    // Stages that don't overlap with finish block → safe to prefill
    constexpr int SAFE_STAGES = (FINISH_OFFSET - 1024) / (int)sizeof(input_block);
    constexpr int SAFE = SAFE_STAGES < PIPE ? SAFE_STAGES : PIPE;

    __shared__ semaphore arrived[PIPE], finished[PIPE], finish_done;
    uint32_t sem_bits = 0xFFFF0000;

    int tiles_m = globals.C.rows() / (M_BLOCK * 64);  // 4
    int tiles_n = globals.C.cols() / (N_BLOCK * 64);  // 96
    int tiles_k = globals.A.cols() / 64;               // 48
    int total_tiles = tiles_m * tiles_n;               // 384

    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE; i++) {
            init_semaphore(arrived[i], 1, 0);
            init_semaphore(finished[i], NUM_CONSUMER_WARPS, 0);
        }
        init_semaphore(finish_done, NUM_CONSUMER_WARPS, 0);
    }
    everyone::sync(15);

    if (warpid() >= NUM_CONSUMER_WARPS) {
        // ====== PRODUCER ======
        warpgroup::decrease_registers<40>();
        int ring = 0;
        for (int task = 0; true; task++) {
            int tid = task * gridDim.x + blockIdx.x;
            if (tid >= total_tiles) break;

            // SUPER_M tile scheduling (SUPER_M=4 matches tiles_m=4 exactly)
            int sr = (tiles_m / SUPER_M) * SUPER_M;  // 4
            int fr = tiles_m - sr;                     // 0
            int sp = SUPER_M * tiles_n;                // 384
            int bm, bn;
            if (tid < sr * tiles_n) {
                bm = SUPER_M * (tid / sp) + tid % SUPER_M;
                bn = (tid % sp) / SUPER_M;
            } else {
                int r = tid - sr * tiles_n;
                bm = sr + (r % fr);
                bn = r / fr;
            }
            int cm = bm * M_BLOCK, cn = bn * N_BLOCK;

            // PIPE=4 with SAFE=4: no overlap between input ring and finish block.
            // No finish_done wait needed — next tile's K-loop (~80us) provides
            // ample margin for any pending TMA store to complete.
            for (int li = 0; li < tiles_k; li++) {
                wait(finished[ring], get_phasebit<1>(sem_bits, ring));
                update_phasebit<1>(sem_bits, ring);
                if (warpgroup::laneid() == 0) {
                    tma::expect(arrived[ring], inputs[ring]);
                    for (int i = 0; i < M_BLOCK; i++)
                        tma::load_async(inputs[ring].a[i], globals.A, {cm+i, li}, arrived[ring]);
                    for (int i = 0; i < N_BLOCK; i++)
                        tma::load_async(inputs[ring].b[i], globals.B, {cn+i, li}, arrived[ring]);
                }
                ring = (ring + 1) & (PIPE - 1);
            }
            group<NUM_PRODUCER_WARPS>::sync(13);
        }
    } else {
        // ====== CONSUMER ======
        warpgroup::increase_registers<232>();

        using wide_rt = rt_fl<16, 64 * N_BLOCK>;
        using tall_st = st_bf<64 * N_BLOCK, 64>;
        rt_fl<16, 64> accum[N_BLOCK];

        for (int task = 0; true; task++) {
            int tid = task * gridDim.x + blockIdx.x;
            if (tid >= total_tiles) break;

            // Same SUPER_M scheduling
            int sr = (tiles_m / SUPER_M) * SUPER_M;
            int fr = tiles_m - sr;
            int sp = SUPER_M * tiles_n;
            int bm, bn;
            if (tid < sr * tiles_n) {
                bm = SUPER_M * (tid / sp) + tid % SUPER_M;
                bn = (tid % sp) / SUPER_M;
            } else {
                int r = tid - sr * tiles_n;
                bm = sr + (r % fr);
                bn = r / fr;
            }
            int cm = bm * M_BLOCK + warpgroup::groupid();
            int cn = bn * N_BLOCK;

            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) kittens::warp::zero(accum[n]);

            // Main K-loop: direct ring buffer, no LCF overhead
            int ring = 0;
            for (int ki = 0; ki < tiles_k; ki++) {
                wait(arrived[ring], get_phasebit<0>(sem_bits, ring));
                update_phasebit<0>(sem_bits, ring);
                warpgroup::mma_ABt(
                    reinterpret_cast<wide_rt&>(accum),
                    inputs[ring].a[warpgroup::groupid()],
                    reinterpret_cast<tall_st&>(inputs[ring].b)
                );
                warpgroup::mma_async_wait();
                if (warp::laneid() == 0) arrive(finished[ring]);
                ring = (ring + 1) & (PIPE - 1);
            }

            // Finish: store accumulators to shared, TMA to global
            // Wait for any previous tile's TMA store to finish reading finish block
            tma::store_async_read_wait();
            group<NUM_CONSUMER_WARPS>::sync(14);
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++)
                warpgroup::store(finish->c[warpgroup::groupid()][n], accum[n]);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(globals.C, finish->c[warpgroup::groupid()][i], {cm, cn+i});
                }
                // No store_async_read_wait: next tile's K-loop (~80us) provides
                // ample time for TMA store to complete before finish block reuse.
            }
            group<NUM_CONSUMER_WARPS>::sync(14);
        }
    }
}

// Host helpers
extern "C" int tk_gemm_512x18432x3072_globals_size() { return (int)sizeof(gemm_globals); }
extern "C" void tk_gemm_512x18432x3072_make_globals(
    void* out, void* dA, void* dBt, void* dC, int M, int N, int K
) {
    global_layout Ag{(bf16*)dA, nullptr, nullptr, (size_t)M, (size_t)K};
    global_layout Bg{(bf16*)dBt, nullptr, nullptr, (size_t)N, (size_t)K};
    global_layout Cg{(bf16*)dC, nullptr, nullptr, (size_t)M, (size_t)N};
    gemm_globals G{Ag, Bg, Cg};
    memcpy(out, &G, sizeof(G));
}
extern "C" void tk_gemm_512x18432x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = 132; *y = 1; *z = 1;
}
extern "C" int tk_gemm_512x18432x3072_block_dim() { return NUM_THREADS_K; }
extern "C" int tk_gemm_512x18432x3072_shmem_bytes() { return MAX_SHARED_MEMORY - 1024; }
extern "C" int tk_gemm_512x18432x3072_num_tma_descriptors() { return 3; }
extern "C" void tk_gemm_512x18432x3072_describe_tma_descriptors(
    void* dA, void* dBt, void* dC, int M, int N, int K, void* out_meta
) {
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)dA, 2, 1, 1, M, K, 64, 64);
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)dBt, 2, 1, 1, N, K, 64, 64);
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)dC, 2, 1, 1, M, N, 64, 64);
}
