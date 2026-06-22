// mm_abt_4096_3072_128.cu — Hand-tuned bf16 ABt GEMM for M=4096, N=3072, K=128
//
// C[m,n] = sum_k A[m,k] * B[n,k]   (B row-major N×K, matching F.linear weight)
//
// v2: Zero bank conflicts + coalesced cooperative global store
//
// Key insights:
//   1. K=128 is tiny → memory-bound. cuBLAS wastes 168 regs + 213KB SMEM on
//      deep pipeline that stays almost empty. We use no pipeline.
//   2. v1's C_out[64][128] caused 9.2-way bank conflicts (row stride 128 bf16
//      = 64 banks, 64%32=0 → all rows alias). Fix: pad to [64][136], stride
//      68 banks, 68%32=4 → zero conflicts.
//   3. After wgmma, A/B SMEM is dead → reuse for padded C buffer.
//      SMEM = max(48KB load, 17KB store) = 48KB → 4 blocks/SM.
//   4. Cooperative vectorized store: 16 threads/row × uint4 → fully coalesced
//      global writes with zero sector amplification.
//
// Config:
//   Tile:    64M × 128N × 64K
//   Grid:    1536 blocks (SUPER_M=8 swizzle)
//   Threads: 128 (1 warpgroup)
//   SMEM:    48 KB → 4 blocks/SM, 16 warps/SM (25% occupancy)

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

static constexpr int BM = 64, BN = 128, BK = 64;
static constexpr int BN_PAD = BN + 8;          // 136: stride 68 banks, coprime with 32
static constexpr int NUM_THREADS = 128;
static constexpr int WGMMA_K = 16;
static constexpr int SUPER_M = 8;

// ---------------------------------------------------------------------------
// WGMMA helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ constexpr uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__
void wgmma_m64n128k16(float d[8][8], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,"
        " %40,%41,%42,%43,%44,%45,%46,%47,"
        " %48,%49,%50,%51,%52,%53,%54,%55,"
        " %56,%57,%58,%59,%60,%61,%62,%63},"
        " %64, %65,"
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
          "+f"(d[5][4]),"+f"(d[5][5]),"+f"(d[5][6]),"+f"(d[5][7]),
          "+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),
          "+f"(d[6][4]),"+f"(d[6][5]),"+f"(d[6][6]),"+f"(d[6][7]),
          "+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]),
          "+f"(d[7][4]),"+f"(d[7][5]),"+f"(d[7][6]),"+f"(d[7][7])
        : "l"(dA), "l"(dB));
}

// ---------------------------------------------------------------------------
// Shared memory layout — A/B for load phase, C_pad for store phase (union)
// ---------------------------------------------------------------------------
struct SMem {
    union {
        struct {                                    // Load phase
            alignas(128) bf16 A[2 * BM * BK];      // 16 KB
            alignas(128) bf16 B[2 * BN * BK];      // 32 KB
        };                                          // Total: 48 KB
        alignas(128) bf16 C_pad[BM * BN_PAD];      // 64×136 = 17.4 KB (reuses A/B space)
    };
};  // sizeof = 48 KB

// ---------------------------------------------------------------------------
// Kernel globals
// ---------------------------------------------------------------------------
struct Globals4096x3072x128 {
    CUtensorMap tmaA;
    CUtensorMap tmaB;
    bf16*       d_C;
    int         N;
};

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(NUM_THREADS, 4)   // 128 threads, target 4 blocks/SM
void gemm_4096x3072x128_kernel(const __grid_constant__ Globals4096x3072x128 G) {
    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    bf16 *sA = s.A, *sB = s.B;

    __shared__ __align__(8) uint64_t load_bar;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    // Tile mapping with SUPER_M swizzle
    constexpr int TILES_N = 3072 / BN;
    const int sr = blockIdx.x / (SUPER_M * TILES_N);
    const int wi = blockIdx.x % (SUPER_M * TILES_N);
    const int block_m = sr * SUPER_M + wi % SUPER_M;
    const int block_n = wi / SUPER_M;

    // --- Init mbarrier ---
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&load_bar)),
               "r"(1u));
    }
    __syncthreads();

    // --- TMA Load Phase ---
    if (threadIdx.x == 0) {
        const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&load_bar);
        constexpr uint32_t total_bytes = (2*BM*BK + 2*BN*BK) * (uint32_t)sizeof(bf16);

        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar), "r"(total_bytes));

        const int32_t mc = (int32_t)(block_m * BM);
        const int32_t nc = (int32_t)(block_n * BN);

        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&sA[0])),        "l"(&G.tmaA), "r"(0),             "r"(mc), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&sA[BM * BK])),  "l"(&G.tmaA), "r"((int32_t)BK),  "r"(mc), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&sB[0])),        "l"(&G.tmaB), "r"(0),             "r"(nc), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&sB[BN * BK])),  "l"(&G.tmaB), "r"((int32_t)BK),  "r"(nc), "r"(bar) : "memory");
    }

    // --- Wait for all loads ---
    {
        const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&load_bar);
        asm volatile(
            "{\n.reg .pred P;\nLAB_WAIT_%=:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
            "@!P bra LAB_WAIT_%=;\n}\n"
            :: "r"(bar), "r"(0u));
    }

    // --- Compute: 8 × WGMMA m64n128k16 ---
    float d[8][8];
    memset(d, 0, sizeof(d));

    constexpr uint64_t DESC =
        (matrix_descriptor_encode(16ULL) << 16) |
        (matrix_descriptor_encode(1024ULL) << 32) |
        (1ULL << 62);
    constexpr uint64_t KS = (WGMMA_K * sizeof(bf16)) >> 4;
    constexpr uint64_t QA = (BK * BM * (int)sizeof(bf16)) >> 4;
    constexpr uint64_t QB = (BK * BN * (int)sizeof(bf16)) >> 4;

    const uint64_t bA = DESC | matrix_descriptor_encode(
        (uint32_t)__cvta_generic_to_shared(&sA[0]));
    const uint64_t bB = DESC | matrix_descriptor_encode(
        (uint32_t)__cvta_generic_to_shared(&sB[0]));

    wgmma_fence();
    wgmma_m64n128k16(d, bA,          bB);
    wgmma_m64n128k16(d, bA + KS,     bB + KS);
    wgmma_m64n128k16(d, bA + 2*KS,   bB + 2*KS);
    wgmma_m64n128k16(d, bA + 3*KS,   bB + 3*KS);
    wgmma_m64n128k16(d, bA + QA,          bB + QB);
    wgmma_m64n128k16(d, bA + QA + KS,     bB + QB + KS);
    wgmma_m64n128k16(d, bA + QA + 2*KS,   bB + QB + 2*KS);
    wgmma_m64n128k16(d, bA + QA + 3*KS,   bB + QB + 3*KS);
    wgmma_commit();
    wgmma_wait<0>();

    // --- Write accumulators → padded C_pad[64][136] (ZERO bank conflicts) ---
    // Bank = (row*68 + col/2) % 32. With row stride 68 (≡4 mod 32),
    // all 32 lanes within a warp map to distinct banks.
    bf16 *c_pad = s.C_pad;
    const int row = warp * 16 + lane / 4;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        const int col = w * 16 + 2 * (lane % 4);
        c_pad[(row    ) * BN_PAD + col    ] = __float2bfloat16(d[w][0]);
        c_pad[(row    ) * BN_PAD + col + 1] = __float2bfloat16(d[w][1]);
        c_pad[(row + 8) * BN_PAD + col    ] = __float2bfloat16(d[w][2]);
        c_pad[(row + 8) * BN_PAD + col + 1] = __float2bfloat16(d[w][3]);
        c_pad[(row    ) * BN_PAD + col + 8] = __float2bfloat16(d[w][4]);
        c_pad[(row    ) * BN_PAD + col + 9] = __float2bfloat16(d[w][5]);
        c_pad[(row + 8) * BN_PAD + col + 8] = __float2bfloat16(d[w][6]);
        c_pad[(row + 8) * BN_PAD + col + 9] = __float2bfloat16(d[w][7]);
    }
    __syncthreads();

    // --- Cooperative coalesced store: C_pad → global C ---
    // 16 threads/row, each stores uint4 (8 bf16 = 16 bytes).
    // 128 threads / 16 = 8 rows/pass, 64 rows / 8 = 8 passes.
    // Within a warp: threads 0-15 cover row i cols 0..127 → 8 full L2 sectors.
    const int tpr = 16;                           // threads per row
    const int rpp = NUM_THREADS / tpr;            // 8 rows per pass
    const int my_row_off = threadIdx.x / tpr;     // 0..7 within a pass
    const int my_chunk   = threadIdx.x % tpr;     // 0..15 column chunk

    bf16 *C_global = G.d_C;
    const int gN = G.N;

    #pragma unroll
    for (int pass = 0; pass < BM / rpp; pass++) {
        const int r = pass * rpp + my_row_off;
        const int c = my_chunk * 8;

        // Load 16 bytes from padded SMEM (skip padding columns ≥128)
        uint4 v = *reinterpret_cast<const uint4*>(&c_pad[r * BN_PAD + c]);

        // Store 16 bytes to global C (fully coalesced: 16 threads × 16B = 256B = 1 row)
        *reinterpret_cast<uint4*>(&C_global[(block_m * BM + r) * gN + block_n * BN + c]) = v;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void create_tensor_map_load(CUtensorMap *m, bf16 *p,
                                    int tile_major, int tile_minor,
                                    int blocks_h, int blocks_w) {
    uint64_t sh[5]  = {(uint64_t)tile_minor*blocks_w, (uint64_t)tile_major*blocks_h, 1, 1, 1};
    uint64_t st[5]  = {sizeof(bf16), sizeof(bf16)*tile_minor*blocks_w, 0, 0, 0};
    uint32_t bx[5]  = {(uint32_t)tile_minor, (uint32_t)tile_major, 1, 1, 1};
    uint32_t bs[5]  = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p,
        sh, st+1, bx, bs,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int gemm_4096x3072x128_globals_size() {
    return (int)sizeof(Globals4096x3072x128);
}

extern "C" void gemm_4096x3072x128_make_globals(
    void *out, void *d_A, void *d_Bt, void *d_C,
    int M, int N, int K
) {
    Globals4096x3072x128 G;
    create_tensor_map_load(&G.tmaA, (bf16*)d_A, BM, BK, M/BM, K/BK);
    create_tensor_map_load(&G.tmaB, (bf16*)d_Bt, BN, BK, N/BN, K/BK);
    G.d_C = (bf16*)d_C;
    G.N = N;
    memcpy(out, &G, sizeof(G));
}

extern "C" void gemm_4096x3072x128_grid_dims(int M, int N, int K,
                                                int *x, int *y, int *z) {
    *x = (M / BM) * (N / BN);
    *y = 1;
    *z = 1;
}

extern "C" int gemm_4096x3072x128_block_dim() { return NUM_THREADS; }

extern "C" int gemm_4096x3072x128_shmem_bytes() { return (int)sizeof(SMem); }

extern "C" int gemm_4096x3072x128_num_tma_descriptors() { return 2; }

extern "C" void gemm_4096x3072x128_describe_tma_descriptors(
    void *d_A, void *d_Bt, void *d_C,
    int M, int N, int K,
    void *out_meta
) {
    _fill_tma_desc_meta_2d((char*)out_meta + 0*96, (uint64_t)d_A,  2, M, K, BM, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, N, K, BN, BK);
}
