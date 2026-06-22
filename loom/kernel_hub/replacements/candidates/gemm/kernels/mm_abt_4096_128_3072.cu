// mm_abt_4096_128_3072.cu — Hand-tuned bf16 ABt GEMM for M=4096, N=128, K=3072
//
// C[m,n] = sum_k A[m,k] * B[n,k]   (F.linear: A=[4096,3072], W=[128,3072])
//
// Memory-bound: A=25.2MB dominates, W=0.8MB fits L2, C=1.0MB fits L2.
// BW floor 8.06 us. cuBLAS ~21.6 us (37% BW util).
//
// Config:
//   Tile:   128M × 64N × 64K
//   Grid:   64 blocks (32 M-tiles × 2 N-tiles), M-major for W L2 reuse
//   Threads: 384 (3 WGs: 1 producer, 2 consumers each doing m64n64)
//   Pipeline: 6 stages, 48 K-steps, 8 rounds
//   L2 promotion for W (only 0.8MB, shared by all blocks)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda.h>
#include "tma_desc_meta.cuh"

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;
namespace cde = cuda::device::experimental;

static constexpr int BM = 128, BN = 64, BK = 64;
static constexpr int QSIZE = 6;
static constexpr int NUM_THREADS = 384;
static constexpr int WGMMA_M = 64, WGMMA_K = 16;
static constexpr int TILES_K = 3072 / BK;        // 48
static constexpr int NUM_ROUNDS = TILES_K / QSIZE; // 8

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ constexpr uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__
void wgmma_m64n64k16(float d[4][8], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31},"
        " %32, %33,"
        " 1, 1, 1, 0, 0;\n"
        : "+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),
          "+f"(d[0][4]),"+f"(d[0][5]),"+f"(d[0][6]),"+f"(d[0][7]),
          "+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),
          "+f"(d[1][4]),"+f"(d[1][5]),"+f"(d[1][6]),"+f"(d[1][7]),
          "+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),
          "+f"(d[2][4]),"+f"(d[2][5]),"+f"(d[2][6]),"+f"(d[2][7]),
          "+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),
          "+f"(d[3][4]),"+f"(d[3][5]),"+f"(d[3][6]),"+f"(d[3][7])
        : "l"(dA), "l"(dB));
}

struct SMem {
    alignas(128) bf16 A[QSIZE * BM * BK];   // 6*128*64*2 = 98304
    alignas(128) bf16 B[QSIZE * BN * BK];   // 6*64*64*2  = 49152
    alignas(128) bf16 C_out[2][64][64];      // 2*64*64*2  = 16384
};
// Total: 163840 = 160KB

struct Globals4096x128x3072 {
    CUtensorMap tmaA;
    CUtensorMap tmaB;
    CUtensorMap tmaC;
    bf16*       d_C;
    int         N;
};

__global__
__launch_bounds__(NUM_THREADS)
void gemm_4096x128x3072_kernel(const __grid_constant__ Globals4096x128x3072 G) {
    const CUtensorMap &tensorMapA = G.tmaA;
    const CUtensorMap &tensorMapB = G.tmaB;
    const CUtensorMap &tensorMapC = G.tmaC;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    bf16 *sA = s.A, *sB = s.B;

    __shared__ __align__(8) uint64_t full_bar[QSIZE];
    __shared__ __align__(8) uint64_t empty_bar[QSIZE];

    const int wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    const int lane = tid % 32;
    const int warp = tid / 32;

    constexpr int TILES_M = 4096 / BM;  // 32
    constexpr int TILES_N = 128 / BN;   // 2
    const int linear_id = blockIdx.x;
    const int block_m = linear_id % TILES_M;
    const int block_n = linear_id / TILES_M;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&full_bar[i])), "r"(1u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_bar[i])), "r"(257u));
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();

    if (wg_idx == 0) {
        // ---- Producer WG ----
        if (tid == 0) {
            uint32_t full_addrs[QSIZE], empty_addrs[QSIZE];
            uint32_t sA_addrs[QSIZE], sB_addrs[QSIZE];
            for (int i = 0; i < QSIZE; i++) {
                full_addrs[i]  = (uint32_t)__cvta_generic_to_shared(&full_bar[i]);
                empty_addrs[i] = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
                sA_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sA[i * BK * BM]);
                sB_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sB[i * BK * BN]);
            }
            constexpr uint32_t total_bytes = (BK * BM + BK * BN) * (uint32_t)sizeof(bf16);
            const int32_t a_coord1 = (int32_t)(block_m * BM);
            const int32_t b_coord1 = (int32_t)(block_n * BN);

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
        // ---- Consumer WGs (wg_idx 1 and 2) ----
        const int m_offset = (wg_idx - 1) * WGMMA_M;

        for (int i = 0; i < QSIZE; i++) {
            uint32_t ea = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(ea));
        }

        float d[4][8];
        memset(d, 0, sizeof(d));

        uint32_t full_addrs[QSIZE], empty_addrs[QSIZE];
        for (int i = 0; i < QSIZE; i++) {
            full_addrs[i]  = (uint32_t)__cvta_generic_to_shared(&full_bar[i]);
            empty_addrs[i] = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
        }

        constexpr uint64_t DESC =
            (matrix_descriptor_encode(16ULL) << 16) |
            (matrix_descriptor_encode(1024ULL) << 32) |
            (1ULL << 62);
        constexpr uint64_t KS = (WGMMA_K * sizeof(bf16)) >> 4;
        constexpr uint64_t Q_A = (BK * BM * (int)sizeof(bf16)) >> 4;
        constexpr uint64_t Q_B = (BK * BN * (int)sizeof(bf16)) >> 4;

        const uint64_t bA = DESC | matrix_descriptor_encode(
            (uint32_t)__cvta_generic_to_shared(&sA[m_offset * BK]));
        const uint64_t bB = DESC | matrix_descriptor_encode(
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

                uint64_t dA = bA + (uint64_t)q * Q_A;
                uint64_t dB = bB + (uint64_t)q * Q_B;
                wgmma_fence();
                wgmma_m64n64k16(d, dA,          dB);
                wgmma_m64n64k16(d, dA + KS,     dB + KS);
                wgmma_m64n64k16(d, dA + 2*KS,   dB + 2*KS);
                wgmma_m64n64k16(d, dA + 3*KS,   dB + 3*KS);
                wgmma_commit();
                wgmma_wait<0>();

                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(empty_addrs[q]));
            }
            phase ^= 1;
        }

        // Store to smem
        const int c_buf = wg_idx - 1;
        const int row = warp * 16 + lane / 4;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            const int col = w * 16 + 2 * (lane % 4);
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

    // TMA store
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
static void create_tensor_map_load(CUtensorMap *m, bf16 *p,
                                    int tile_major, int tile_minor,
                                    int blocks_h, int blocks_w,
                                    CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    uint64_t sh[5] = {(uint64_t)tile_minor*blocks_w, (uint64_t)tile_major*blocks_h, 1, 1, 1};
    uint64_t st[5] = {sizeof(bf16), sizeof(bf16)*tile_minor*blocks_w, 0, 0, 0};
    uint32_t bx[5] = {(uint32_t)tile_minor, (uint32_t)tile_major, 1, 1, 1};
    uint32_t bs[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p,
        sh, st+1, bx, bs,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static void create_tensor_map_store(CUtensorMap *m, bf16 *p,
                                     int rows, int cols, int tile_rows, int tile_cols) {
    uint64_t sh[5] = {(uint64_t)cols, (uint64_t)rows, 1, 1, 1};
    uint64_t st[5] = {sizeof(bf16), sizeof(bf16)*cols, 0, 0, 0};
    uint32_t bx[5] = {(uint32_t)tile_cols, (uint32_t)tile_rows, 1, 1, 1};
    uint32_t bs[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p,
        sh, st+1, bx, bs,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int gemm_4096x128x3072_globals_size() { return (int)sizeof(Globals4096x128x3072); }

extern "C" void gemm_4096x128x3072_make_globals(
    void *out, void *d_A, void *d_Bt, void *d_C, int M, int N, int K) {
    Globals4096x128x3072 G;
    create_tensor_map_load(&G.tmaA, (bf16*)d_A, BM, BK, M/BM, K/BK);
    create_tensor_map_load(&G.tmaB, (bf16*)d_Bt, BN, BK, N/BN, K/BK,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    create_tensor_map_store(&G.tmaC, (bf16*)d_C, M, N, 64, BN);
    G.d_C = (bf16*)d_C;
    G.N = N;
    memcpy(out, &G, sizeof(G));
}

extern "C" void gemm_4096x128x3072_grid_dims(int M, int N, int K,
                                               int *x, int *y, int *z) {
    *x = (M / BM) * (N / BN);
    *y = 1; *z = 1;
}

extern "C" int gemm_4096x128x3072_block_dim() { return NUM_THREADS; }
extern "C" int gemm_4096x128x3072_shmem_bytes() { return (int)sizeof(SMem); }
extern "C" int gemm_4096x128x3072_num_tma_descriptors() { return 3; }

extern "C" void gemm_4096x128x3072_describe_tma_descriptors(
    void *d_A, void *d_Bt, void *d_C, int M, int N, int K, void *out_meta) {
    _fill_tma_desc_meta_2d((char*)out_meta + 0*96, (uint64_t)d_A, 2, M, K, BM, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, N, K, BN, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 2*96, (uint64_t)d_C, 2, M, N, 64, BN);
}
