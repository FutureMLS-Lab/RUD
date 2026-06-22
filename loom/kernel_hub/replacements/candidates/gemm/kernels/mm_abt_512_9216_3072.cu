// mm_abt_512_9216_3072.cu — Hand-tuned bf16 ABt GEMM for M=512, N=9216, K=3072
//
// Persistent grid-stride: 132 blocks (1/SM), each processes ~3 tiles.
// 128M×96N×64K tiles, 6-stage pipeline, wgmma_wait<0>.
// 384 tiles (4×96) → 97% SM utilization over 3 waves.

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

static constexpr int BM = 128, BN = 96, BK = 64;
static constexpr int QSIZE = 6;
static constexpr int NUM_THREADS = 384;
static constexpr int WGMMA_M = 64, WGMMA_K = 16;
static constexpr int NUM_SMS = 132;

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ constexpr uint64_t matrix_descriptor_encode(uint64_t x) { return (x & 0x3FFFF) >> 4; }

__device__ __forceinline__
void wgmma_64x96x16_desc(float d[6][8], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47},"
        " %48,%49,1,1,1,0,0;\n"
        :"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[0][4]),"+f"(d[0][5]),"+f"(d[0][6]),"+f"(d[0][7]),
         "+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[1][4]),"+f"(d[1][5]),"+f"(d[1][6]),"+f"(d[1][7]),
         "+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[2][4]),"+f"(d[2][5]),"+f"(d[2][6]),"+f"(d[2][7]),
         "+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[3][4]),"+f"(d[3][5]),"+f"(d[3][6]),"+f"(d[3][7]),
         "+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[4][4]),"+f"(d[4][5]),"+f"(d[4][6]),"+f"(d[4][7]),
         "+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[5][4]),"+f"(d[5][5]),"+f"(d[5][6]),"+f"(d[5][7])
        :"l"(dA),"l"(dB));
}

struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
    alignas(128) bf16 C_out[2][64][96];
};

struct Globals512x9216x3072 {
    CUtensorMap tmaA, tmaB, tmaC;
    bf16* d_C; int tiles_k, total_tiles, tiles_m;
};

__global__ __launch_bounds__(NUM_THREADS)
void gemm_512x9216x3072_kernel(const __grid_constant__ Globals512x9216x3072 G) {
    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    bf16 *sA = s.A, *sB = s.B;
    __shared__ __align__(8) uint64_t full_bar[QSIZE], empty_bar[QSIZE];

    int wg_idx = threadIdx.x / 128, tid = threadIdx.x % 128;
    int lane = tid % 32, warp = tid / 32;
    int tiles_k = G.tiles_k, num_rounds = tiles_k / QSIZE;
    int total_tiles = G.total_tiles, tiles_m = G.tiles_m;

    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        int block_m = tile_id % tiles_m, block_n = tile_id / tiles_m;

        if (threadIdx.x == 0) {
            for (int i = 0; i < QSIZE; i++) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;\n"::"r"((uint32_t)__cvta_generic_to_shared(&full_bar[i])),"r"(1u));
                asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;\n"::"r"((uint32_t)__cvta_generic_to_shared(&empty_bar[i])),"r"(3u));
            }
            asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
        }
        __syncthreads();

        if (wg_idx == 0) {
            if (tid == 0) {
                uint32_t fa[QSIZE],ea[QSIZE],sAa[QSIZE],sBa[QSIZE];
                for (int i=0;i<QSIZE;i++){fa[i]=(uint32_t)__cvta_generic_to_shared(&full_bar[i]);ea[i]=(uint32_t)__cvta_generic_to_shared(&empty_bar[i]);sAa[i]=(uint32_t)__cvta_generic_to_shared(&sA[i*BK*BM]);sBa[i]=(uint32_t)__cvta_generic_to_shared(&sB[i*BK*BN]);}
                constexpr uint32_t tb = (BK*BM+BK*BN)*sizeof(bf16);
                int32_t ac=(int32_t)(block_m*BM), bc=(int32_t)(block_n*BN);
                uint32_t ph=0;
                for (int r=0;r<num_rounds;r++){
                    #pragma unroll
                    for (int q=0;q<QSIZE;q++){
                        int32_t kc=(int32_t)((r*QSIZE+q)*BK);
                        asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[q]));
                        asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(ea[q]),"r"(ph));
                        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;\n"::"r"(fa[q]),"r"(tb));
                        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(sAa[q]),"l"(&G.tmaA),"r"(kc),"r"(ac),"r"(fa[q]):"memory");
                        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(sBa[q]),"l"(&G.tmaB),"r"(kc),"r"(bc),"r"(fa[q]):"memory");
                    }
                    ph^=1;
                }
            }
        } else {
            int mo=(wg_idx-1)*WGMMA_M;
            if (warp == 0 && lane == 0) {
                for (int i=0;i<QSIZE;i++){uint32_t ea=(uint32_t)__cvta_generic_to_shared(&empty_bar[i]);asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea));}
            }
            float d[6][8]; memset(d,0,sizeof(d));
            uint32_t fa[QSIZE],ea[QSIZE];
            for (int i=0;i<QSIZE;i++){fa[i]=(uint32_t)__cvta_generic_to_shared(&full_bar[i]);ea[i]=(uint32_t)__cvta_generic_to_shared(&empty_bar[i]);}
            constexpr uint64_t DC=(matrix_descriptor_encode(16ULL)<<16)|(matrix_descriptor_encode(1024ULL)<<32)|(1ULL<<62);
            constexpr uint64_t KS=(WGMMA_K*sizeof(bf16))>>4, QA=(BK*BM*(int)sizeof(bf16))>>4, QB=(BK*BN*(int)sizeof(bf16))>>4;
            uint64_t bA=DC|matrix_descriptor_encode((uint32_t)__cvta_generic_to_shared(&sA[mo*BK]));
            uint64_t bB=DC|matrix_descriptor_encode((uint32_t)__cvta_generic_to_shared(&sB[0]));
            // ===== Pipelined wgmma_wait<1> with unrolled inner loop =====
            // Prolog: round 0, q=0 — no previous group to drain
            asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(fa[0]),"r"(0u));
            wgmma_fence();
            wgmma_64x96x16_desc(d,bA,bB); wgmma_64x96x16_desc(d,bA+KS,bB+KS);
            wgmma_64x96x16_desc(d,bA+2*KS,bB+2*KS); wgmma_64x96x16_desc(d,bA+3*KS,bB+3*KS);
            wgmma_commit();

            // Round 0, q=1..QSIZE-1 — pipelined, phase=0
            #pragma unroll
            for (int q = 1; q < QSIZE; q++) {
                wgmma_wait<1>();
                if (warp == 0 && lane == 0)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[q-1]));
                asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(fa[q]),"r"(0u));
                uint64_t dA=bA+q*QA, dB=bB+q*QB;
                wgmma_fence();
                wgmma_64x96x16_desc(d,dA,dB); wgmma_64x96x16_desc(d,dA+KS,dB+KS);
                wgmma_64x96x16_desc(d,dA+2*KS,dB+2*KS); wgmma_64x96x16_desc(d,dA+3*KS,dB+3*KS);
                wgmma_commit();
            }

            // Rounds 1..num_rounds-1 — fully pipelined, inner loop unrolled
            for (int r = 1; r < num_rounds; r++) {
                uint32_t ph = r & 1;
                #pragma unroll
                for (int q = 0; q < QSIZE; q++) {
                    wgmma_wait<1>();
                    if (warp == 0 && lane == 0)
                        asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[q==0?QSIZE-1:q-1]));
                    asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(fa[q]),"r"(ph));
                    uint64_t dA=bA+q*QA, dB=bB+q*QB;
                    wgmma_fence();
                    wgmma_64x96x16_desc(d,dA,dB); wgmma_64x96x16_desc(d,dA+KS,dB+KS);
                    wgmma_64x96x16_desc(d,dA+2*KS,dB+2*KS); wgmma_64x96x16_desc(d,dA+3*KS,dB+3*KS);
                    wgmma_commit();
                }
            }

            // Epilog: drain last group
            wgmma_wait<0>();
            if (warp == 0 && lane == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[(tiles_k-1)%QSIZE]));
            int cb=wg_idx-1, row=warp*16+lane/4;
            #pragma unroll
            for (int w=0;w<6;w++){int col=w*16+2*(lane%4);
                s.C_out[cb][row][col]=__float2bfloat16(d[w][0]);s.C_out[cb][row][col+1]=__float2bfloat16(d[w][1]);
                s.C_out[cb][row+8][col]=__float2bfloat16(d[w][2]);s.C_out[cb][row+8][col+1]=__float2bfloat16(d[w][3]);
                s.C_out[cb][row][col+8]=__float2bfloat16(d[w][4]);s.C_out[cb][row][col+9]=__float2bfloat16(d[w][5]);
                s.C_out[cb][row+8][col+8]=__float2bfloat16(d[w][6]);s.C_out[cb][row+8][col+9]=__float2bfloat16(d[w][7]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
            cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC,block_n*BN,block_m*BM,&s.C_out[0][0][0]);
            cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC,block_n*BN,block_m*BM+64,&s.C_out[1][0][0]);
            asm volatile("cp.async.bulk.commit_group;\n":::"memory");
            asm volatile("cp.async.bulk.wait_group 0;\n":::"memory");
        }
        __syncthreads();
    }
}

static void mk_load(CUtensorMap*m,bf16*p,int tmaj,int tmin,int bh,int bw){uint64_t sh[5]={tmin*(uint64_t)bw,tmaj*(uint64_t)bh,1,1,1};uint64_t st[5]={sizeof(bf16),sizeof(bf16)*tmin*bw,0,0,0};uint32_t bx[5]={(uint32_t)tmin,(uint32_t)tmaj,1,1,1},bs[5]={1,1,1,1,1};cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,sh,st+1,bx,bs,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);}
static void mk_store(CUtensorMap*m,bf16*p,int tr,int tc,int ttr,int ttc){uint64_t sh[5]={(uint64_t)tc,(uint64_t)tr,1,1,1};uint64_t st[5]={sizeof(bf16),sizeof(bf16)*tc,0,0,0};uint32_t bx[5]={(uint32_t)ttc,(uint32_t)ttr,1,1,1},bs[5]={1,1,1,1,1};cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,sh,st+1,bx,bs,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);}

extern "C" int gemm_512x9216x3072_globals_size(){return sizeof(Globals512x9216x3072);}
extern "C" void gemm_512x9216x3072_make_globals(void*out,void*dA,void*dB,void*dC,int M,int N,int K){
    Globals512x9216x3072 G; mk_load(&G.tmaA,(bf16*)dA,BM,BK,M/BM,K/BK); mk_load(&G.tmaB,(bf16*)dB,BN,BK,N/BN,K/BK);
    mk_store(&G.tmaC,(bf16*)dC,M,N,64,BN); G.d_C=(bf16*)dC; G.tiles_k=K/BK; G.total_tiles=(M/BM)*(N/BN); G.tiles_m=M/BM; memcpy(out,&G,sizeof(G));}
extern "C" void gemm_512x9216x3072_grid_dims(int M,int N,int K,int*x,int*y,int*z){int t=(M/BM)*(N/BN);*x=t<NUM_SMS?t:NUM_SMS;*y=1;*z=1;}
extern "C" int gemm_512x9216x3072_block_dim(){return NUM_THREADS;}
extern "C" int gemm_512x9216x3072_shmem_bytes(){return sizeof(SMem);}
extern "C" int gemm_512x9216x3072_num_tma_descriptors(){return 3;}
extern "C" void gemm_512x9216x3072_describe_tma_descriptors(void*dA,void*dB,void*dC,int M,int N,int K,void*out){
    _fill_tma_desc_meta_2d((char*)out+0*96,(uint64_t)dA,2,M,K,BM,BK);
    _fill_tma_desc_meta_2d((char*)out+1*96,(uint64_t)dB,2,N,K,BN,BK);
    _fill_tma_desc_meta_2d((char*)out+2*96,(uint64_t)dC,2,M,N,64,BN);}
