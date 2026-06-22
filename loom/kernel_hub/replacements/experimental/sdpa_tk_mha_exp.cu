/**
 * ThunderKittens bf16 scaled-dot-product attention forward pass — strided variant.
 *
 * Build artifacts (build/sdpa_tk_mha_exp.so, build/sdpa_tk_mha_exp.ptx, build/sdpa_tk_mha_exp.cubin)
 * are compiled automatically on first import by sdpa_tk_mha_exp.py.
 *
 * D=128 non-causal only (Flux-2).
 * Scalar order: batch, qo_heads, kv_heads, seq_len, then per-tensor [batch/head/seq] strides.
 * Accepts non-contiguous Q/K/V by encoding physical strides into TMA descriptors via
 * mha_fwd_strided_make_globals.  No .contiguous() copies required.
 */

#include "kittens.cuh"
#include <cooperative_groups.h>

constexpr int CONSUMER_WARPGROUPS = (3); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (4); 
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (2); 
};

template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N; 
    const int hr;
};

template<int D, bool is_causal>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    
    int kv_blocks   = g.N / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS; 

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {(int)blockIdx.z, (int)blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {(int)blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2; }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {(int)blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                warp::tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                warp::tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                warp::tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                warp::tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else { kv_iters = kv_blocks - 1; }

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            
            warp::copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                const int q_blk = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) + warpid; 
                      int k_blk = (kv_idx * (K::kv_height/kittens::TILE_ROW_DIM<bf16>)); 

                #pragma unroll
                for(; k_blk == (kv_iters-1)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>) || k_blk == (kv_iters)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>); k_blk+=10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { warp::neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { warp::make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }

            warp::row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                warp::mul(att_block, att_block,    1.44269504089f*0.125f); 
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                warp::mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled,       max_vec_last_scaled);
            warp::mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            warp::row_sum(norm_vec,  att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);
            warp::copy(att_block_mma, att_block); 
            warp::mul_row(o_reg, o_reg, max_vec_last_scaled); 

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {(int)blockIdx.z, (int)blockIdx.y, (seq_idx) + warpgroupid, 0};
            warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        warp::mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { warp::mul(norm_vec, norm_vec, -8.0f); }
        else                   { warp::mul(norm_vec, norm_vec, -11.313708499f); }
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {(int)blockIdx.z, (int)blockIdx.y, 0, (seq_idx) + warpgroupid};
            warp::tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        warp::tma::store_async_wait();
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers for cubin-based driver API launch.
// ---------------------------------------------------------------------------

extern "C" void mha_fwd_grid_dims(
    int batch, int qo_heads, int kv_heads, int seq_len,
    int* x, int* y, int* z
) {
    *x = seq_len / (CONSUMER_WARPGROUPS * (int)kittens::TILE_ROW_DIM<kittens::bf16> * 4);
    *y = qo_heads;
    *z = batch;
}

extern "C" int mha_fwd_block_dim() {
    return (int)(NUM_WORKERS * kittens::WARP_THREADS);
}

extern "C" int mha_fwd_shmem_bytes() {
    return (int)(kittens::MAX_SHARED_MEMORY - 1024);
}

// Force instantiation of fwd_attend_ker<128, false> so it appears in the compiled .cubin/.ptx.
template __global__
    __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
    void fwd_attend_ker<128, false>(const __grid_constant__ fwd_globals<128>);

// ---------------------------------------------------------------------------
// Strided descriptor helper + strided globals construction
// ---------------------------------------------------------------------------
//
// create_tensor_map_strided<ST> is a copy of TK's create_tensor_map<ST, axis=2>
// (swizzled-tile path) with three explicit physical stride parameters instead
// of the contiguous-stride formulas. Every other field is identical to TK's
// version (gmem_shape, smem_shape, smem_stride, tma_dim, swizzle mode).
//
// For Q/K/V from F.linear → einops.rearrange("B L (K H D) -> K B H L D", K=3):
//   Physical layout is (B, N, H, D) — logical view (B, H, N, D):
//     seq_stride_elements  = H * D  (e.g. 9216 for Flux2 H=24, D=128)
//     head_stride_elements = D      (= 128)
//     batch_stride_elements= N*H*D
//
// For contiguous (B, H, N, D):
//     seq_stride_elements  = D      (= 128)
//     head_stride_elements = N * D
//     batch_stride_elements= H*N*D
// ---------------------------------------------------------------------------

template<kittens::ducks::st::all ST>
__host__ static void create_tensor_map_strided(
    CUtensorMap*              tma_map,
    const typename ST::dtype* src,
    int batch, int depth, int rows, int cols,
    size_t seq_stride_elements,    // gmem_stride[0]/sizeof(dtype): N-row stride
    size_t head_stride_elements,   // gmem_stride[2]/sizeof(dtype): H stride
    size_t batch_stride_elements   // gmem_stride[3]/sizeof(dtype): B stride
) {
    static_assert(ST::swizzle, "create_tensor_map_strided only supports swizzled tiles");
    using dtype = typename ST::dtype;

    constexpr uint32_t tma_dim          = 5;
    constexpr int      swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    constexpr CUtensorMapDataType tma_fmt = (
        std::is_same_v<dtype, kittens::bf16> ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, kittens::half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16  :
        std::is_same_v<dtype, float>         ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32  :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapSwizzle tma_swizzle = (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        CU_TENSOR_MAP_SWIZZLE_NONE
    );

    // gmem_shape: same structure as TK's create_tensor_map<ST, axis=2>
    uint64_t gmem_shape[5] = {
        (uint64_t)swizzle_elements,
        (uint64_t)rows,                                         // N
        (uint64_t)((cols + swizzle_elements - 1) / swizzle_elements),  // D/swizzle_elems
        (uint64_t)depth,                                        // H
        (uint64_t)batch,                                        // B
    };
    // gmem_stride[0,2,3] are the three user-supplied physical strides (in bytes).
    // gmem_stride[1] = swizzle_bytes — this is a descriptor-internal constant,
    //                  never a layout stride, and is always unchanged.
    uint64_t gmem_stride[4] = {
        seq_stride_elements   * sizeof(dtype),   // N-row stride (bytes)
        (uint64_t)ST::swizzle_bytes,             // swizzle-bank stride (bytes, unchanged)
        head_stride_elements  * sizeof(dtype),   // H stride (bytes)
        batch_stride_elements * sizeof(dtype),   // B stride (bytes)
    };
    // smem_shape/stride: identical to TK's version — determined by the tile type, not global layout
    uint32_t smem_shape[5]  = {
        (uint32_t)swizzle_elements,
        (uint32_t)ST::rows,
        (uint32_t)(cols / swizzle_elements),
        1, 1,
    };
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    assert(gmem_stride[0] % 16 == 0);
    assert(gmem_stride[2] % 16 == 0);
    assert(gmem_stride[3] % 16 == 0);
    assert((reinterpret_cast<uint64_t>(src) & 0xf) == 0);

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, tma_fmt, tma_dim, (void*)src,
        gmem_shape, gmem_stride, smem_shape, smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, tma_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(result, &err_str);
        throw std::runtime_error(
            std::string("create_tensor_map_strided failed: ") + (err_str ? err_str : "?") +
            " | seq_stride="   + std::to_string(seq_stride_elements) +
            " head_stride="    + std::to_string(head_stride_elements) +
            " batch_stride="   + std::to_string(batch_stride_elements)
        );
    }
}

// ---------------------------------------------------------------------------
// mha_fwd_strided_make_globals — identical to mha_fwd_make_globals except it
// overwrites the TMA descriptors inside each gl with physically-strided ones.
//
// After constructing each gl{ptr, B, H, N, D} normally (which sets raw_ptr and
// dimension metadata correctly), we overwrite gl::tma_descs.tma_desc — the
// inline CUtensorMap stored as a public member of descriptor_dict — using
// create_tensor_map_strided. get_tma<>() is __device__-only and cannot be used
// here; direct member access is the correct host-side approach.
//
// l_vec is always contiguous (freshly allocated per call) — not overwritten.
// ---------------------------------------------------------------------------

extern "C" int mha_fwd_strided_globals_size() {
    return (int)sizeof(fwd_globals<128>);
}

extern "C" void mha_fwd_strided_make_globals(
    void* out_buf,
    void* q_ptr, void* k_ptr, void* v_ptr, void* l_ptr, void* o_ptr,
    int batch, int qo_heads, int kv_heads, int seq_len,
    long long q_seq_stride,  long long q_head_stride,  long long q_batch_stride,
    long long k_seq_stride,  long long k_head_stride,  long long k_batch_stride,
    long long v_seq_stride,  long long v_head_stride,  long long v_batch_stride,
    long long o_seq_stride,  long long o_head_stride,  long long o_batch_stride
) {
    using G        = fwd_globals<128>;
    using q_global = typename G::q_gl;
    using k_global = typename G::k_gl;
    using v_global = typename G::v_gl;
    using l_global = typename G::l_gl;
    using o_global = typename G::o_gl;

    constexpr int D  = 128;
    const     int hr = qo_heads / kv_heads;

    // Step 1: Build gl objects normally (contiguous descriptors as default).
    q_global qg{(bf16*)q_ptr, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, (unsigned)D};
    k_global kg{(bf16*)k_ptr, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, (unsigned)D};
    v_global vg{(bf16*)v_ptr, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, (unsigned)D};
    l_global lg{(float*)l_ptr, (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len};
    o_global og{(bf16*)o_ptr, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, (unsigned)D};

    // Step 2: Overwrite TMA descriptors with physical strides.
    //   tma_descs.tma_desc is a plain public CUtensorMap member of descriptor_dict —
    //   accessible on the host without const_cast or device-only methods.
    create_tensor_map_strided<G::q_tile>(
        &qg.tma_descs.tma_desc,
        (bf16*)q_ptr, batch, qo_heads, seq_len, D,
        (size_t)q_seq_stride, (size_t)q_head_stride, (size_t)q_batch_stride
    );
    create_tensor_map_strided<G::k_tile>(
        &kg.tma_descs.tma_desc,
        (bf16*)k_ptr, batch, kv_heads, seq_len, D,
        (size_t)k_seq_stride, (size_t)k_head_stride, (size_t)k_batch_stride
    );
    create_tensor_map_strided<G::v_tile>(
        &vg.tma_descs.tma_desc,
        (bf16*)v_ptr, batch, kv_heads, seq_len, D,
        (size_t)v_seq_stride, (size_t)v_head_stride, (size_t)v_batch_stride
    );
    create_tensor_map_strided<G::o_tile>(
        &og.tma_descs.tma_desc,
        (bf16*)o_ptr, batch, qo_heads, seq_len, D,
        (size_t)o_seq_stride, (size_t)o_head_stride, (size_t)o_batch_stride
    );
    // l_gl: always contiguous — left unchanged.

    G g{qg, kg, vg, lg, og, seq_len, hr};
    memcpy(out_buf, &g, sizeof(G));
}

// mha_fwd_strided_grid_dims — accepts the full 16-scalar arg list produced by
// the TKCC DriverLaunchExternalCubin codegen (4 shape ints + 12 stride long longs)
// and forwards only the shape ints to mha_fwd_grid_dims.  The stride args are
// ignored here; grid dimensions depend only on batch/head/seq shape.
extern "C" void mha_fwd_strided_grid_dims(
    int batch, int qo_heads, int kv_heads, int seq_len,
    long long q_seq_s, long long q_head_s, long long q_batch_s,
    long long k_seq_s, long long k_head_s, long long k_batch_s,
    long long v_seq_s, long long v_head_s, long long v_batch_s,
    long long o_seq_s, long long o_head_s, long long o_batch_s,
    int* x, int* y, int* z
) {
    mha_fwd_grid_dims(batch, qo_heads, kv_heads, seq_len, x, y, z);
}
