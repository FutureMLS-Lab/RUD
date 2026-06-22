// mm_tk_abt_512x3072x7680.cu — Hand-tuned bf16 ABt GEMM for M=512, N=3072, K=7680
//
// C[m,n] = sum_k A[m,k] * B[n,k]   (B row-major N×K, matching F.linear weight)
//
// Config:
//   Tile:   128M × 96N × 64K
//   Grid:   128 blocks (1D, L2-friendly swizzle: M-tiles contiguous)
//   Threads: 384 (3 WGs: 1 producer, 2 consumers)
//   Pipeline depth: 6 stages
//   TMA loads: A[128×64] + B[96×64] per stage
//   WGMMA: 64×96×16 per consumer WG, 4 per K-step → 8 total per stage
//   K-iterations: 7680/64 = 120, rounds: 120/6 = 20

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda.h>
#include "tma_desc_meta.cuh"

using bf16 = __nv_bfloat16;
namespace cde = cuda::device::experimental;

static constexpr int GM = 512, GN = 3072, GK = 7680;
static constexpr int BM = 128, BN = 96, BK = 64;
static constexpr int QSIZE = 6;
static constexpr int NUM_THREADS = 384;
static constexpr int WGMMA_M = 64, WGMMA_K = 16;
static constexpr int TILES_M = GM / BM, TILES_N = GN / BN, TILES_K = GK / BK;
static constexpr int NUM_ROUNDS = TILES_K / QSIZE;  // 120/6 = 20

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ constexpr uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__
void wgmma_64x96x16_desc(float d[6][8], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,"
        " %40,%41,%42,%43,%44,%45,%46,%47},"
        " %48, %49,"
        " 1, 1, 1, 0, 0;\n"
        : "+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),
          "+f"(d[0][4]),"+f"(d[0][5]),"+f"(d[0][6]),"+f"(d[0][7]),
          "+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),
          "+f"(d[1][4]),"+f"(d[1][5]),"+f"(d[1][6]),"+f"(d[1][7]),
          "+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),
          "+f"(d[2][4]),"+f"(d[2][5]),"+f"(d[2][6]),"+f"(d[2][7]),
          "+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),
          "+f"(d[3][4]),"+f"(d[3][5]),"+f"(d[3][6]),"+f"(d[3][7]),
          "+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),
          "+f"(d[4][4]),"+f"(d[4][5]),"+f"(d[4][6]),"+f"(d[4][7]),
          "+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),
          "+f"(d[5][4]),"+f"(d[5][5]),"+f"(d[5][6]),"+f"(d[5][7])
        : "l"(dA), "l"(dB));
}

struct SMem {
    alignas(128) bf16 A[BM * BK * QSIZE];
    alignas(128) bf16 B[BK * BN * QSIZE];
    alignas(128) bf16 C_out[2][64][96];
};

struct Globals512x3072x7680 {
    CUtensorMap tmaA;
    CUtensorMap tmaB;
    CUtensorMap tmaC;
    bf16*       d_C;
    int         tiles_k;
};

__global__
__launch_bounds__(NUM_THREADS)
void gemm_512x3072x7680_kernel(const __grid_constant__ Globals512x3072x7680 G) {
    const CUtensorMap &tensorMapA = G.tmaA;
    const CUtensorMap &tensorMapB = G.tmaB;
    const CUtensorMap &tensorMapC = G.tmaC;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    bf16 *sA = s.A, *sB = s.B;

    __shared__ __align__(8) uint64_t full_bar[QSIZE];
    __shared__ __align__(8) uint64_t empty_bar[QSIZE];

    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;

    int linear_id = blockIdx.x;
    int block_m = linear_id % TILES_M;
    int block_n = linear_id / TILES_M;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&full_bar[i])), "r"((uint32_t)1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_bar[i])), "r"((uint32_t)257));
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();

    if (wg_idx == 0) {
        if (tid == 0) {
            uint32_t full_addrs[QSIZE], empty_addrs[QSIZE];
            uint32_t sA_addrs[QSIZE], sB_addrs[QSIZE];
            for (int i = 0; i < QSIZE; i++) {
                full_addrs[i]  = (uint32_t)__cvta_generic_to_shared(&full_bar[i]);
                empty_addrs[i] = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
                sA_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sA[i * BK * BM]);
                sB_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sB[i * BK * BN]);
            }
            constexpr uint32_t total_bytes = (BK * BM + BK * BN) * sizeof(bf16);
            int32_t a_coord1 = (int32_t)(block_m * BM);
            int32_t b_coord1 = (int32_t)(block_n * BN);

            uint32_t phase = 0;
            for (int round = 0; round < NUM_ROUNDS; round++) {
                #pragma unroll
                for (int q = 0; q < QSIZE; q++) {
                    int32_t k_coord = (int32_t)((round * QSIZE + q) * BK);

                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(empty_addrs[q]));
                    asm volatile(
                        "{\n.reg .pred P;\nLAB_WAIT_EMPTY_%=:\n"
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                        "@!P bra LAB_WAIT_EMPTY_%=;\n}\n"
                        :: "r"(empty_addrs[q]), "r"(phase));

                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                        :: "r"(full_addrs[q]), "r"(total_bytes));
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                        :: "r"(sA_addrs[q]), "l"(&tensorMapA),
                           "r"(k_coord), "r"(a_coord1), "r"(full_addrs[q]) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                        :: "r"(sB_addrs[q]), "l"(&tensorMapB),
                           "r"(k_coord), "r"(b_coord1), "r"(full_addrs[q]) : "memory");
                }
                phase ^= 1;
            }
        }
    } else {
        int m_offset = (wg_idx - 1) * WGMMA_M;

        for (int i = 0; i < QSIZE; i++) {
            uint32_t ea = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(ea));
        }

        float d[6][8];
        memset(d, 0, sizeof(d));

        uint32_t full_addrs[QSIZE], empty_addrs[QSIZE];
        for (int i = 0; i < QSIZE; i++) {
            full_addrs[i]  = (uint32_t)__cvta_generic_to_shared(&full_bar[i]);
            empty_addrs[i] = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
        }

        constexpr uint64_t DESC_CONST_BITS =
            (matrix_descriptor_encode((uint64_t)16) << 16) |
            (matrix_descriptor_encode((uint64_t)1024) << 32) |
            (1ULL << 62);
        constexpr uint64_t K_STEP = (WGMMA_K * sizeof(bf16)) >> 4;
        constexpr uint64_t Q_A = (BK * BM * (int)sizeof(bf16)) >> 4;
        constexpr uint64_t Q_B = (BK * BN * (int)sizeof(bf16)) >> 4;

        uint64_t baseA = DESC_CONST_BITS | matrix_descriptor_encode(
            (uint32_t)__cvta_generic_to_shared(&sA[m_offset * BK]));
        uint64_t baseB = DESC_CONST_BITS | matrix_descriptor_encode(
            (uint32_t)__cvta_generic_to_shared(&sB[0]));

        uint32_t phase = 0;
        for (int round = 0; round < NUM_ROUNDS; round++) {
            #pragma unroll
            for (int q = 0; q < QSIZE; q++) {
                asm volatile(
                    "{\n.reg .pred P;\nLAB_WAIT_FULL_%=:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                    "@!P bra LAB_WAIT_FULL_%=;\n}\n"
                    :: "r"(full_addrs[q]), "r"(phase));

                uint64_t dA = baseA + q * Q_A;
                uint64_t dB = baseB + q * Q_B;
                wgmma_fence();
                wgmma_64x96x16_desc(d, dA,            dB);
                wgmma_64x96x16_desc(d, dA + K_STEP,   dB + K_STEP);
                wgmma_64x96x16_desc(d, dA + 2*K_STEP, dB + 2*K_STEP);
                wgmma_64x96x16_desc(d, dA + 3*K_STEP, dB + 3*K_STEP);
                wgmma_commit();
                wgmma_wait<0>();

                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(empty_addrs[q]));
            }
            phase ^= 1;
        }

        int c_buf = wg_idx - 1;
        int row = warp * 16 + lane / 4;
        #pragma unroll
        for (int w = 0; w < 6; w++) {
            int col = w * 16 + 2 * (lane % 4);
            s.C_out[c_buf][row    ][col    ] = __float2bfloat16(d[w][0]);
            s.C_out[c_buf][row    ][col + 1] = __float2bfloat16(d[w][1]);
            s.C_out[c_buf][row + 8][col    ] = __float2bfloat16(d[w][2]);
            s.C_out[c_buf][row + 8][col + 1] = __float2bfloat16(d[w][3]);
            s.C_out[c_buf][row    ][col + 8] = __float2bfloat16(d[w][4]);
            s.C_out[c_buf][row    ][col + 9] = __float2bfloat16(d[w][5]);
            s.C_out[c_buf][row + 8][col + 8] = __float2bfloat16(d[w][6]);
            s.C_out[c_buf][row + 8][col + 9] = __float2bfloat16(d[w][7]);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC, block_n * BN, block_m * BM,      &s.C_out[0][0][0]);
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC, block_n * BN, block_m * BM + 64, &s.C_out[1][0][0]);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static void create_tensor_map_load(CUtensorMap *tma_map, bf16* gmem_ptr,
                                    int tile_major, int tile_minor,
                                    int blocks_height, int blocks_width) {
    uint64_t shape[5]   = {(uint64_t)tile_minor * blocks_width, (uint64_t)tile_major * blocks_height, 1, 1, 1};
    uint64_t stride[5]  = {sizeof(bf16), sizeof(bf16) * tile_minor * blocks_width, 0, 0, 0};
    uint32_t box[5]     = {(uint32_t)tile_minor, (uint32_t)tile_major, 1, 1, 1};
    uint32_t bstride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)gmem_ptr,
        shape, stride + 1, box, bstride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static void create_tensor_map_store(CUtensorMap *tma_map, bf16* gmem_ptr,
                                     int total_rows, int total_cols,
                                     int tile_rows, int tile_cols) {
    uint64_t shape[5]   = {(uint64_t)total_cols, (uint64_t)total_rows, 1, 1, 1};
    uint64_t stride[5]  = {sizeof(bf16), sizeof(bf16) * total_cols, 0, 0, 0};
    uint32_t box[5]     = {(uint32_t)tile_cols, (uint32_t)tile_rows, 1, 1, 1};
    uint32_t bstride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)gmem_ptr,
        shape, stride + 1, box, bstride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int gemm_512x3072x7680_globals_size() {
    return (int)sizeof(Globals512x3072x7680);
}

extern "C" void gemm_512x3072x7680_make_globals(
    void* out_buf,
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K
) {
    Globals512x3072x7680 G;
    int tiles_m = M / BM;
    int tiles_n = N / BN;
    int tiles_k = K / BK;

    create_tensor_map_load(&G.tmaA, (bf16*)d_A, BM, BK, tiles_m, tiles_k);
    create_tensor_map_load(&G.tmaB, (bf16*)d_Bt, BN, BK, tiles_n, tiles_k);
    create_tensor_map_store(&G.tmaC, (bf16*)d_C, M, N, 64, BN);
    G.d_C = (bf16*)d_C;
    G.tiles_k = tiles_k;
    memcpy(out_buf, &G, sizeof(G));
}

extern "C" void gemm_512x3072x7680_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = (M / BM) * (N / BN);
    *y = 1;
    *z = 1;
}

extern "C" int gemm_512x3072x7680_block_dim() {
    return NUM_THREADS;
}

extern "C" int gemm_512x3072x7680_shmem_bytes() {
    return (int)sizeof(SMem);
}

extern "C" int gemm_512x3072x7680_num_tma_descriptors() { return 3; }

extern "C" void gemm_512x3072x7680_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K,
    void* out_meta
) {
    // Descriptor 0: A[M, K] bf16, load tile [BM=128, BK=64]
    _fill_tma_desc_meta_2d((char*)out_meta + 0*96, (uint64_t)d_A, 2, M, K, BM, BK);
    // Descriptor 1: Bt[N, K] bf16, load tile [BN=96, BK=64]
    _fill_tma_desc_meta_2d((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, N, K, BN, BK);
    // Descriptor 2: C[M, N] bf16, store tile [64, BN=96]
    _fill_tma_desc_meta_2d((char*)out_meta + 2*96, (uint64_t)d_C, 2, M, N, 64, BN);
}
