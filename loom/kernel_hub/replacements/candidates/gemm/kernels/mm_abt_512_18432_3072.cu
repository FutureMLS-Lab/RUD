// mm_abt_512_18432_3072_v2.cu — Persistent bf16 ABt GEMM for M=512, N=18432, K=3072
//
// v2: Persistent kernel with zero-overhead tile transitions + L2 promotion.
//
// Optimizations over v1:
//   1. Persistent 132 CTAs — no wave scheduling overhead
//   2. Zero-overhead tile transitions — empty_bar count=256, barriers cycle naturally
//   3. N-major tile ordering for L2 A reuse
//   4. L2 promotion for A (3MB, fits in L2, reused across 192 N-tiles)
//
// Config: same tile/pipeline as v1 (128×192, 4-stage)

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

static constexpr int GM = 512, GN = 18432, GK = 3072;
static constexpr int BM = 128, BN = 192, BK = 64;
static constexpr int BN_HALF = 96;
static constexpr int QSIZE = 4;
static constexpr int NUM_THREADS = 384;
static constexpr int WGMMA_M = 64, WGMMA_K = 16;
static constexpr int TILES_M = GM / BM;
static constexpr int TILES_N = GN / BN;
static constexpr int TILES_K = GK / BK;
static constexpr int NUM_ROUNDS = TILES_K / QSIZE;
static constexpr int TOTAL_TILES = TILES_M * TILES_N;
static constexpr int NUM_SMS = 132;

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
    alignas(128) bf16 B0[BK * BN_HALF * QSIZE];
    alignas(128) bf16 B1[BK * BN_HALF * QSIZE];
    alignas(128) bf16 C0_out[2][64][BN_HALF];
    alignas(128) bf16 C1_out[2][64][BN_HALF];
};

struct GlobalsV2 {
    CUtensorMap tmaA;
    CUtensorMap tmaB0;
    CUtensorMap tmaB1;
    CUtensorMap tmaC0;
    CUtensorMap tmaC1;
    bf16*       d_C;
};

__global__
__launch_bounds__(NUM_THREADS)
void gemm_512x18432x3072_kernel(const __grid_constant__ GlobalsV2 G) {
    const CUtensorMap &tensorMapA  = G.tmaA;
    const CUtensorMap &tensorMapB0 = G.tmaB0;
    const CUtensorMap &tensorMapB1 = G.tmaB1;
    const CUtensorMap &tensorMapC0 = G.tmaC0;
    const CUtensorMap &tensorMapC1 = G.tmaC1;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    bf16 *sA = s.A, *sB0 = s.B0, *sB1 = s.B1;

    __shared__ __align__(8) uint64_t full_bar[QSIZE];
    __shared__ __align__(8) uint64_t empty_bar[QSIZE];

    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&full_bar[i])), "r"((uint32_t)1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_bar[i])), "r"((uint32_t)256));
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();

    uint32_t full_addrs[QSIZE], empty_addrs[QSIZE];
    uint32_t sA_addrs[QSIZE], sB0_addrs[QSIZE], sB1_addrs[QSIZE];
    for (int i = 0; i < QSIZE; i++) {
        full_addrs[i]   = (uint32_t)__cvta_generic_to_shared(&full_bar[i]);
        empty_addrs[i]  = (uint32_t)__cvta_generic_to_shared(&empty_bar[i]);
        sA_addrs[i]     = (uint32_t)__cvta_generic_to_shared(&sA[i * BK * BM]);
        sB0_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sB0[i * BK * BN_HALF]);
        sB1_addrs[i]    = (uint32_t)__cvta_generic_to_shared(&sB1[i * BK * BN_HALF]);
    }
    constexpr uint32_t total_bytes = (BK * BM + 2 * BK * BN_HALF) * sizeof(bf16);

    constexpr uint64_t DESC_CONST_BITS =
        (matrix_descriptor_encode((uint64_t)16) << 16) |
        (matrix_descriptor_encode((uint64_t)1024) << 32) |
        (1ULL << 62);
    constexpr uint64_t K_STEP = (WGMMA_K * sizeof(bf16)) >> 4;
    constexpr uint64_t Q_A  = (BK * BM      * (int)sizeof(bf16)) >> 4;
    constexpr uint64_t Q_B  = (BK * BN_HALF * (int)sizeof(bf16)) >> 4;

    int m_offset = (wg_idx > 0) ? (wg_idx - 1) * WGMMA_M : 0;
    uint64_t baseA  = DESC_CONST_BITS | matrix_descriptor_encode(
        (uint32_t)__cvta_generic_to_shared(&sA[m_offset * BK]));
    uint64_t baseB0 = DESC_CONST_BITS | matrix_descriptor_encode(
        (uint32_t)__cvta_generic_to_shared(&sB0[0]));
    uint64_t baseB1 = DESC_CONST_BITS | matrix_descriptor_encode(
        (uint32_t)__cvta_generic_to_shared(&sB1[0]));

    for (int tile_id = blockIdx.x; tile_id < TOTAL_TILES; tile_id += gridDim.x) {
        int block_n = tile_id / TILES_M;
        int block_m = tile_id % TILES_M;

        if (wg_idx == 0) {
            if (tid == 0) {
                int32_t a_coord1  = (int32_t)(block_m * BM);
                int32_t b0_coord1 = (int32_t)(block_n * BN);
                int32_t b1_coord1 = (int32_t)(block_n * BN + BN_HALF);

                // Round 0: load without waiting
                #pragma unroll
                for (int q = 0; q < QSIZE; q++) {
                    int32_t k_coord = (int32_t)(q * BK);
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
                        :: "r"(sB0_addrs[q]), "l"(&tensorMapB0),
                           "r"(k_coord), "r"(b0_coord1), "r"(full_addrs[q]) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                        :: "r"(sB1_addrs[q]), "l"(&tensorMapB1),
                           "r"(k_coord), "r"(b1_coord1), "r"(full_addrs[q]) : "memory");
                }

                uint32_t phase = 0;
                for (int round = 1; round < NUM_ROUNDS; round++) {
                    #pragma unroll
                    for (int q = 0; q < QSIZE; q++) {
                        int32_t k_coord = (int32_t)((round * QSIZE + q) * BK);
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
                            :: "r"(sB0_addrs[q]), "l"(&tensorMapB0),
                               "r"(k_coord), "r"(b0_coord1), "r"(full_addrs[q]) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                            ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                            :: "r"(sB1_addrs[q]), "l"(&tensorMapB1),
                               "r"(k_coord), "r"(b1_coord1), "r"(full_addrs[q]) : "memory");
                    }
                    phase ^= 1;
                }
            }
        } else {
            float d0[6][8], d1[6][8];
            memset(d0, 0, sizeof(d0));
            memset(d1, 0, sizeof(d1));

            uint32_t phase = 0;
            for (int round = 0; round < NUM_ROUNDS; round++) {
                #pragma unroll
                for (int q = 0; q < QSIZE; q++) {
                    asm volatile(
                        "{\n.reg .pred P;\nLAB_WAIT_FULL_%=:\n"
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                        "@!P bra LAB_WAIT_FULL_%=;\n}\n"
                        :: "r"(full_addrs[q]), "r"(phase));

                    uint64_t dA   = baseA  + q * Q_A;
                    uint64_t dB0v = baseB0 + q * Q_B;
                    uint64_t dB1v = baseB1 + q * Q_B;

                    wgmma_fence();
                    wgmma_64x96x16_desc(d0, dA,            dB0v);
                    wgmma_64x96x16_desc(d0, dA + K_STEP,   dB0v + K_STEP);
                    wgmma_64x96x16_desc(d0, dA + 2*K_STEP, dB0v + 2*K_STEP);
                    wgmma_64x96x16_desc(d0, dA + 3*K_STEP, dB0v + 3*K_STEP);
                    wgmma_commit();
                    wgmma_64x96x16_desc(d1, dA,            dB1v);
                    wgmma_64x96x16_desc(d1, dA + K_STEP,   dB1v + K_STEP);
                    wgmma_64x96x16_desc(d1, dA + 2*K_STEP, dB1v + 2*K_STEP);
                    wgmma_64x96x16_desc(d1, dA + 3*K_STEP, dB1v + 3*K_STEP);
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
                s.C0_out[c_buf][row    ][col    ] = __float2bfloat16(d0[w][0]);
                s.C0_out[c_buf][row    ][col + 1] = __float2bfloat16(d0[w][1]);
                s.C0_out[c_buf][row + 8][col    ] = __float2bfloat16(d0[w][2]);
                s.C0_out[c_buf][row + 8][col + 1] = __float2bfloat16(d0[w][3]);
                s.C0_out[c_buf][row    ][col + 8] = __float2bfloat16(d0[w][4]);
                s.C0_out[c_buf][row    ][col + 9] = __float2bfloat16(d0[w][5]);
                s.C0_out[c_buf][row + 8][col + 8] = __float2bfloat16(d0[w][6]);
                s.C0_out[c_buf][row + 8][col + 9] = __float2bfloat16(d0[w][7]);
                s.C1_out[c_buf][row    ][col    ] = __float2bfloat16(d1[w][0]);
                s.C1_out[c_buf][row    ][col + 1] = __float2bfloat16(d1[w][1]);
                s.C1_out[c_buf][row + 8][col    ] = __float2bfloat16(d1[w][2]);
                s.C1_out[c_buf][row + 8][col + 1] = __float2bfloat16(d1[w][3]);
                s.C1_out[c_buf][row    ][col + 8] = __float2bfloat16(d1[w][4]);
                s.C1_out[c_buf][row    ][col + 9] = __float2bfloat16(d1[w][5]);
                s.C1_out[c_buf][row + 8][col + 8] = __float2bfloat16(d1[w][6]);
                s.C1_out[c_buf][row + 8][col + 9] = __float2bfloat16(d1[w][7]);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            int n_base = block_n * BN;
            int m_base = block_m * BM;
            cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC0, n_base,           m_base,      &s.C0_out[0][0][0]);
            cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC0, n_base,           m_base + 64, &s.C0_out[1][0][0]);
            cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC1, n_base + BN_HALF, m_base,      &s.C1_out[0][0][0]);
            cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC1, n_base + BN_HALF, m_base + 64, &s.C1_out[1][0][0]);
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        }
    }

    if (threadIdx.x == 0) {
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
}

// ---------------------------------------------------------------------------
// Host helpers — L2 promotion for A loads
// ---------------------------------------------------------------------------

static void create_tensor_map_load(CUtensorMap *tma_map, bf16* gmem_ptr,
                                    int tile_major, int tile_minor,
                                    int blocks_height, int blocks_width,
                                    CUtensorMapL2promotion l2promo) {
    uint64_t shape[5]   = {(uint64_t)tile_minor * blocks_width, (uint64_t)tile_major * blocks_height, 1, 1, 1};
    uint64_t stride[5]  = {sizeof(bf16), sizeof(bf16) * tile_minor * blocks_width, 0, 0, 0};
    uint32_t box[5]     = {(uint32_t)tile_minor, (uint32_t)tile_major, 1, 1, 1};
    uint32_t bstride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)gmem_ptr,
        shape, stride + 1, box, bstride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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

extern "C" int gemm_512x18432x3072_globals_size() { return (int)sizeof(GlobalsV2); }

extern "C" void gemm_512x18432x3072_make_globals(
    void* out_buf, void* d_A, void* d_Bt, void* d_C, int M, int N, int K
) {
    GlobalsV2 G;
    // A: L2_128B promotion (3MB, fits in L2, reused across 192 N-tiles)
    create_tensor_map_load(&G.tmaA, (bf16*)d_A, BM, BK, M/BM, K/BK,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    // B: no promotion (113MB, streams through)
    create_tensor_map_load(&G.tmaB0, (bf16*)d_Bt, BN_HALF, BK, N/BN_HALF, K/BK,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE);
    create_tensor_map_load(&G.tmaB1, (bf16*)d_Bt, BN_HALF, BK, N/BN_HALF, K/BK,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE);
    create_tensor_map_store(&G.tmaC0, (bf16*)d_C, M, N, 64, BN_HALF);
    create_tensor_map_store(&G.tmaC1, (bf16*)d_C, M, N, 64, BN_HALF);
    G.d_C = (bf16*)d_C;
    memcpy(out_buf, &G, sizeof(G));
}

extern "C" void gemm_512x18432x3072_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    int total = (M / BM) * (N / BN);
    *x = (total < NUM_SMS) ? total : NUM_SMS;
    *y = 1; *z = 1;
}

extern "C" int gemm_512x18432x3072_block_dim() { return NUM_THREADS; }
extern "C" int gemm_512x18432x3072_shmem_bytes() { return (int)sizeof(SMem); }
extern "C" int gemm_512x18432x3072_num_tma_descriptors() { return 5; }

extern "C" void gemm_512x18432x3072_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C, int M, int N, int K, void* out_meta
) {
    _fill_tma_desc_meta_2d((char*)out_meta + 0*96, (uint64_t)d_A, 2, M, K, BM, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, N, K, BN_HALF, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 2*96, (uint64_t)d_Bt, 2, N, K, BN_HALF, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 3*96, (uint64_t)d_C, 2, M, N, 64, BN_HALF);
    _fill_tma_desc_meta_2d((char*)out_meta + 4*96, (uint64_t)d_C, 2, M, N, 64, BN_HALF);
}
