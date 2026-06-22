// mm_tk_3x3_4608_27648_3072.cu — TK 192×192 persistent GEMM for 4608×27648×3072
//
// C = A(M,K) @ Bt(N,K)^T   (B stored row-major as N×K, matching F.linear)
//
// Key insight: cuBLAS uses 192×192 tiles for this shape.  Our previous TK kernels
// all used 128×256 (M_BLOCK=2, N_BLOCK=4).  This kernel uses M_BLOCK=3, N_BLOCK=3
// to match cuBLAS's tile geometry, with TK's auto-computed register allocation.
//
// Tile:      192M × 192N × 64K  (M_BLOCK=3, N_BLOCK=3)
// Threads:   512 (3 consumer WGs + 1 producer WG)
// Pipeline:  4 stages
// Grid:      132 persistent CTAs
// SUPER_M:   12
// Regs:      consumer=152/thread (auto via consumer_registers<3>), producer=24

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"
#include "tma_desc_meta.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct mm3x3_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};

template<int _M_BLOCK=3, int _N_BLOCK=3, int _SUPER_M=12>
struct matmul_3x3 {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = mm3x3_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERSISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64);
        int Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M;
        int final_rows = Rblocks - super_rows;
        int super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                                  (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { args.num_iters = -1; return; }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<M_BLOCK>();
            for (int n = 0; n < N_BLOCK; n++)
                kittens::warp::zero(args.state.accum[n]);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using wide_rt = rt_fl<16, 64*N_BLOCK>;
            using tall_st = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(
                reinterpret_cast<wide_rt&>(args.state.accum),
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<tall_st&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            for (int n = 0; n < N_BLOCK; n++)
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                     {args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait();
                }
            }
            for (int n = 0; n < N_BLOCK; n++)
                kittens::warp::zero(args.state.accum[n]);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

using mmt = matmul_3x3<3, 3, 12>;
using global_layout = typename mmt::layout::global_layout;
using globals = typename mmt::layout::globals;

extern "C" int tk_3x3_globals_size() { return (int)sizeof(globals); }

extern "C" void tk_3x3_make_globals(
    void* out_buf, void* d_A, void* d_Bt, void* d_C, int M, int N, int K
) {
    global_layout Ag {(bf16*)d_A,  nullptr, nullptr, (size_t)M, (size_t)K};
    global_layout Btg{(bf16*)d_Bt, nullptr, nullptr, (size_t)N, (size_t)K};
    global_layout Cg {(bf16*)d_C,  nullptr, nullptr, (size_t)M, (size_t)N};
    globals G{Ag, Btg, Cg};
    memcpy(out_buf, &G, sizeof(globals));
}

extern "C" void tk_3x3_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = mmt::grid(0, 0, 0).x;
    *y = 1;
    *z = 1;
}

extern "C" int tk_3x3_block_dim() {
    return (int)kittens::prototype::detail::NUM_THREADS_v<mmt>;
}

extern "C" int tk_3x3_shmem_bytes() {
    return MAX_SHARED_MEMORY - 1024;
}

extern "C" int tk_3x3_num_tma_descriptors() { return 3; }
extern "C" void tk_3x3_describe_tma_descriptors(
    void* d_A, void* d_Bt, void* d_C, int M, int N, int K, void* out_meta
) {
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)d_A, 2, 1, 1, M, K, 64, 64);
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)d_Bt, 2, 1, 1, N, K, 64, 64);
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)d_C, 2, 1, 1, M, N, 64, 64);
}

// Force instantiation of the LCF kernel
template __global__
    __launch_bounds__(kittens::prototype::detail::NUM_THREADS_v<mmt>,
                      kittens::prototype::detail::NUM_BLOCKS_v<mmt>)
    void lcf::kernel<mmt>(const __grid_constant__ typename mmt::layout::globals);
