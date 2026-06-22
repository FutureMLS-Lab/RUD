#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"
#include "tma_desc_meta.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// AB matmul_template: C = A(M,K) @ B(K,N)
// A is stored row-major as (M, K), B is stored row-major as (K, N).
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=8>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
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
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<wide_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait();
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

using mmt = matmul_template<2, 4, 8>;

// ---------------------------------------------------------------------------
// Host-side helpers for cubin-based driver API launch.
// ---------------------------------------------------------------------------

using global_layout = typename mmt::layout::global_layout;
using globals = typename mmt::layout::globals;

extern "C" int tk_gemm_ab_globals_size() {
    return (int)sizeof(globals);
}

extern "C" void tk_gemm_ab_make_globals(
    void* out_buf,
    void* d_A, void* d_B, void* d_C,
    int M, int N, int K
) {
    global_layout Ag{(bf16*)d_A, nullptr, nullptr, (size_t)M, (size_t)K};
    global_layout Bg{(bf16*)d_B, nullptr, nullptr, (size_t)K, (size_t)N};
    global_layout Cg{(bf16*)d_C, nullptr, nullptr, (size_t)M, (size_t)N};
    globals G{Ag, Bg, Cg};
    memcpy(out_buf, &G, sizeof(globals));
}

extern "C" void tk_gemm_ab_grid_dims(int M, int N, int K, int* x, int* y, int* z) {
    *x = mmt::grid(0, 0, 0).x;
    *y = 1;
    *z = 1;
}

extern "C" int tk_gemm_ab_block_dim() {
    return (int)kittens::prototype::detail::NUM_THREADS_v<mmt>;
}

extern "C" int tk_gemm_ab_shmem_bytes() {
    return MAX_SHARED_MEMORY - 1024;
}

extern "C" int tk_gemm_ab_num_tma_descriptors() { return 3; }

extern "C" void tk_gemm_ab_describe_tma_descriptors(
    void* d_A, void* d_B, void* d_C,
    int M, int N, int K,
    void* out_meta
) {
    // Descriptor 0: A[M, K] bf16, tile 64×64
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)d_A, 2, 1, 1, M, K, 64, 64);
    // Descriptor 1: B[K, N] bf16, tile 64×64
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)d_B, 2, 1, 1, K, N, 64, 64);
    // Descriptor 2: C[M, N] bf16, tile 64×64
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)d_C, 2, 1, 1, M, N, 64, 64);
}

template __global__
    __launch_bounds__(kittens::prototype::detail::NUM_THREADS_v<mmt>,
                      kittens::prototype::detail::NUM_BLOCKS_v<mmt>)
    void lcf::kernel<mmt>(const __grid_constant__ typename mmt::layout::globals);
