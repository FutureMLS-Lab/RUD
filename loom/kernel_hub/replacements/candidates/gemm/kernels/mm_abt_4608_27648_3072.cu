// mm_abt_4608_27648_3072.cu — Hand-tuned bf16 ABt GEMM for 4608×27648×3072
//
// C = A(M,K) @ Bt(N,K)^T   (B stored row-major as N×K, matching F.linear)
//
// Tile:     BM=128, BN=192 (two 96-wide halves), BK=64
// Pipeline: 4 stages, 12 rounds
// Threads:  384 (3 WGs: 1 producer, 2 consumers)
// Grid:     5184 blocks, SUPER_M=12 swizzle
// SMEM:     ~208 KB
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

static constexpr int GM=4608,GN=27648,GK=3072;
static constexpr int BM=128,BN=192,BK=64,BN_HALF=96;
static constexpr int QSIZE=4, SUPER_M=12;
static constexpr int NUM_THREADS=384;
static constexpr int WGMMA_M=64,WGMMA_K=16;
static constexpr int TILES_M=GM/BM,TILES_N=GN/BN,TILES_K=GK/BK;
static constexpr int NUM_ROUNDS=TILES_K/QSIZE;

__device__ __forceinline__ void wgmma_fence(){asm volatile("wgmma.fence.sync.aligned;\n":::"memory");}
__device__ __forceinline__ void wgmma_commit(){asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");}
template<int N>__device__ __forceinline__ void wgmma_wait(){asm volatile("wgmma.wait_group.sync.aligned %0;\n"::"n"(N):"memory");}
__device__ __forceinline__ constexpr uint64_t de(uint64_t x){return(x&0x3FFFF)>>4;}

__device__ __forceinline__
void wgmma96(float d[6][8],uint64_t dA,uint64_t dB){
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
    alignas(128) bf16 B0[BK*BN_HALF*QSIZE];
    alignas(128) bf16 B1[BK*BN_HALF*QSIZE];
    alignas(128) bf16 C0[2][64][BN_HALF];
    alignas(128) bf16 C1[2][64][BN_HALF];
};
struct KG{CUtensorMap tmaA,tmaB0,tmaB1,tmaC0,tmaC1;bf16*d_C;int tiles_k;};

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_4608_27648_3072_kernel(const __grid_constant__ KG G){
    extern __shared__ __align__(128) uint8_t smem[];
    SMem&s=*reinterpret_cast<SMem*>(smem);
    bf16*sA=s.A,*sB0=s.B0,*sB1=s.B1;
    __shared__ __align__(8) uint64_t fb[QSIZE],eb[QSIZE];
    int wg=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
    int lid=blockIdx.x;
    int sr=lid/(SUPER_M*TILES_N), wi=lid%(SUPER_M*TILES_N);
    int block_m=sr*SUPER_M+wi%SUPER_M, block_n=wi/SUPER_M;

    if(threadIdx.x==0){for(int i=0;i<QSIZE;i++){
        asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;\n"::"r"((uint32_t)__cvta_generic_to_shared(&fb[i])),"r"(1u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;\n"::"r"((uint32_t)__cvta_generic_to_shared(&eb[i])),"r"(257u));
    }asm volatile("fence.proxy.async.shared::cta;\n":::"memory");}
    __syncthreads();

    if(wg==0){if(tid==0){
        uint32_t fa[QSIZE],ea[QSIZE],sAa[QSIZE],sB0a[QSIZE],sB1a[QSIZE];
        for(int i=0;i<QSIZE;i++){fa[i]=(uint32_t)__cvta_generic_to_shared(&fb[i]);ea[i]=(uint32_t)__cvta_generic_to_shared(&eb[i]);
            sAa[i]=(uint32_t)__cvta_generic_to_shared(&sA[i*BK*BM]);sB0a[i]=(uint32_t)__cvta_generic_to_shared(&sB0[i*BK*BN_HALF]);sB1a[i]=(uint32_t)__cvta_generic_to_shared(&sB1[i*BK*BN_HALF]);}
        constexpr uint32_t tot=(BK*BM+2*BK*BN_HALF)*sizeof(bf16);
        int32_t ac=block_m*BM,b0c=block_n*BN,b1c=block_n*BN+BN_HALF;
        uint32_t ph=0;
        for(int r=0;r<NUM_ROUNDS;r++){
            #pragma unroll
            for(int q=0;q<QSIZE;q++){int32_t kc=(r*QSIZE+q)*BK;
                asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[q]));
                asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(ea[q]),"r"(ph));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;\n"::"r"(fa[q]),"r"(tot));
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(sAa[q]),"l"(&G.tmaA),"r"(kc),"r"(ac),"r"(fa[q]):"memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(sB0a[q]),"l"(&G.tmaB0),"r"(kc),"r"(b0c),"r"(fa[q]):"memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(sB1a[q]),"l"(&G.tmaB1),"r"(kc),"r"(b1c),"r"(fa[q]):"memory");
            }ph^=1;}
    }}else{
        int mo=(wg-1)*WGMMA_M;
        for(int i=0;i<QSIZE;i++){uint32_t ea=(uint32_t)__cvta_generic_to_shared(&eb[i]);asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea));}
        float d0[6][8],d1[6][8];memset(d0,0,sizeof(d0));memset(d1,0,sizeof(d1));
        uint32_t fa[QSIZE],ea[QSIZE];
        for(int i=0;i<QSIZE;i++){fa[i]=(uint32_t)__cvta_generic_to_shared(&fb[i]);ea[i]=(uint32_t)__cvta_generic_to_shared(&eb[i]);}
        constexpr uint64_t DC=(de(16ULL)<<16)|(de(1024ULL)<<32)|(1ULL<<62);
        constexpr uint64_t KS=(WGMMA_K*sizeof(bf16))>>4,QA=(BK*BM*(int)sizeof(bf16))>>4,QB=(BK*BN_HALF*(int)sizeof(bf16))>>4;
        uint64_t bA=DC|de((uint32_t)__cvta_generic_to_shared(&sA[mo*BK]));
        uint64_t bB0=DC|de((uint32_t)__cvta_generic_to_shared(&sB0[0]));
        uint64_t bB1=DC|de((uint32_t)__cvta_generic_to_shared(&sB1[0]));
        uint32_t ph=0;
        for(int r=0;r<NUM_ROUNDS;r++){
            #pragma unroll
            for(int q=0;q<QSIZE;q++){
                asm volatile("{\n.reg .pred P;\nL_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra L_%=;\n}\n"::"r"(fa[q]),"r"(ph));
                uint64_t dA=bA+q*QA,dB0v=bB0+q*QB,dB1v=bB1+q*QB;
                wgmma_fence();
                wgmma96(d0,dA,dB0v);wgmma96(d0,dA+KS,dB0v+KS);wgmma96(d0,dA+2*KS,dB0v+2*KS);wgmma96(d0,dA+3*KS,dB0v+3*KS);
                wgmma_commit();
                wgmma96(d1,dA,dB1v);wgmma96(d1,dA+KS,dB1v+KS);wgmma96(d1,dA+2*KS,dB1v+2*KS);wgmma96(d1,dA+3*KS,dB1v+3*KS);
                wgmma_commit();wgmma_wait<0>();
                asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];\n"::"r"(ea[q]));
            }ph^=1;}
        int cb=wg-1,row=warp*16+lane/4;
        #pragma unroll
        for(int w=0;w<6;w++){int col=w*16+2*(lane%4);
            s.C0[cb][row][col]=__float2bfloat16(d0[w][0]);s.C0[cb][row][col+1]=__float2bfloat16(d0[w][1]);
            s.C0[cb][row+8][col]=__float2bfloat16(d0[w][2]);s.C0[cb][row+8][col+1]=__float2bfloat16(d0[w][3]);
            s.C0[cb][row][col+8]=__float2bfloat16(d0[w][4]);s.C0[cb][row][col+9]=__float2bfloat16(d0[w][5]);
            s.C0[cb][row+8][col+8]=__float2bfloat16(d0[w][6]);s.C0[cb][row+8][col+9]=__float2bfloat16(d0[w][7]);
            s.C1[cb][row][col]=__float2bfloat16(d1[w][0]);s.C1[cb][row][col+1]=__float2bfloat16(d1[w][1]);
            s.C1[cb][row+8][col]=__float2bfloat16(d1[w][2]);s.C1[cb][row+8][col+1]=__float2bfloat16(d1[w][3]);
            s.C1[cb][row][col+8]=__float2bfloat16(d1[w][4]);s.C1[cb][row][col+9]=__float2bfloat16(d1[w][5]);
            s.C1[cb][row+8][col+8]=__float2bfloat16(d1[w][6]);s.C1[cb][row+8][col+9]=__float2bfloat16(d1[w][7]);
        }
    }
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
        int nb=block_n*BN,mb=block_m*BM;
        cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC0,nb,mb,&s.C0[0][0][0]);
        cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC0,nb,mb+64,&s.C0[1][0][0]);
        cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC1,nb+BN_HALF,mb,&s.C1[0][0][0]);
        cde::cp_async_bulk_tensor_2d_shared_to_global(&G.tmaC1,nb+BN_HALF,mb+64,&s.C1[1][0][0]);
        asm volatile("cp.async.bulk.commit_group;\n":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;\n":::"memory");
    }
}
static void mk_load(CUtensorMap*m,bf16*p,int tmaj,int tmin,int bh,int bw){
    uint64_t sh[5]={tmin*(uint64_t)bw,tmaj*(uint64_t)bh,1,1,1};uint64_t st[5]={sizeof(bf16),sizeof(bf16)*tmin*bw,0,0,0};
    uint32_t bx[5]={(uint32_t)tmin,(uint32_t)tmaj,1,1,1},bs[5]={1,1,1,1,1};
    cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,sh,st+1,bx,bs,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
static void mk_store(CUtensorMap*m,bf16*p,int tr,int tc,int ttr,int ttc){
    uint64_t sh[5]={(uint64_t)tc,(uint64_t)tr,1,1,1};uint64_t st[5]={sizeof(bf16),sizeof(bf16)*tc,0,0,0};
    uint32_t bx[5]={(uint32_t)ttc,(uint32_t)ttr,1,1,1},bs[5]={1,1,1,1,1};
    cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,sh,st+1,bx,bs,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
extern "C" int gemm_4608x27648x3072_globals_size(){return sizeof(KG);}
extern "C" void gemm_4608x27648x3072_make_globals(void*out,void*dA,void*dB,void*dC,int M,int N,int K){
    KG G;mk_load(&G.tmaA,(bf16*)dA,BM,BK,M/BM,K/BK);mk_load(&G.tmaB0,(bf16*)dB,BN_HALF,BK,N/BN_HALF,K/BK);
    mk_load(&G.tmaB1,(bf16*)dB,BN_HALF,BK,N/BN_HALF,K/BK);mk_store(&G.tmaC0,(bf16*)dC,M,N,64,BN_HALF);
    mk_store(&G.tmaC1,(bf16*)dC,M,N,64,BN_HALF);G.d_C=(bf16*)dC;G.tiles_k=K/BK;memcpy(out,&G,sizeof(G));
}
extern "C" void gemm_4608x27648x3072_grid_dims(int M,int N,int K,int*x,int*y,int*z){*x=(M/BM)*(N/BN);*y=1;*z=1;}
extern "C" int gemm_4608x27648x3072_block_dim(){return NUM_THREADS;}
extern "C" int gemm_4608x27648x3072_shmem_bytes(){return sizeof(SMem);}

extern "C" int gemm_4608x27648x3072_num_tma_descriptors(){return 5;}
extern "C" void gemm_4608x27648x3072_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C,
    int M, int N, int K,
    void* out_meta
){
    _fill_tma_desc_meta_2d((char*)out_meta + 0*96, (uint64_t)d_A, 2, M, K, BM, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, N, K, BN_HALF, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 2*96, (uint64_t)d_Bt, 2, N, K, BN_HALF, BK);
    _fill_tma_desc_meta_2d((char*)out_meta + 3*96, (uint64_t)d_C, 2, M, N, 64, BN_HALF);
    _fill_tma_desc_meta_2d((char*)out_meta + 4*96, (uint64_t)d_C, 2, M, N, 64, BN_HALF);
}
