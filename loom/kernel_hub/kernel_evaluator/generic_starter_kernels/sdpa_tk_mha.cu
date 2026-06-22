/**
 * ThunderKittens bf16 scaled-dot-product attention forward pass.
 *
 * Build artifacts (build/sdpa_tk_mha.so, build/sdpa_tk_mha.ptx, build/sdpa_tk_mha.cubin)
 * are compiled automatically on first import by sdpa_tk_mha.py.
 *
 * D=128 non-causal only (Flux-2).
 * Scalar order: batch, qo_heads, kv_heads, seq_len.
 * Inputs must be contiguous in BHND layout (stride(-1)==1); sdpa_tk_mha.py calls
 * .contiguous() on Q/K/V before invoking the kernel.
 */

#include "kittens.cuh"
#include <cooperative_groups.h>
#include "tma_desc_meta.cuh"

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
// Globals construction for contiguous (BHND) inputs.
// Q/K/V must be contiguous before this is called — sdpa_tk_mha.py ensures this
// via .contiguous().  The standard TK gl constructor derives correct TMA
// descriptors from the tensor dimensions without any stride overrides.
// ---------------------------------------------------------------------------

extern "C" int mha_fwd_num_tma_descriptors() { return 5; }

extern "C" void mha_fwd_describe_tma_descriptors(
    void* q_ptr, void* k_ptr, void* v_ptr, void* l_ptr, void* o_ptr,
    int batch, int qo_heads, int kv_heads, int seq_len,
    void* out_meta
) {
    constexpr int D = 128;
    // Descriptor 0: Q[batch, qo_heads, seq_len, D=128] bf16, tile 64×128
    _fill_tma_desc_meta((char*)out_meta + 0*96, (uint64_t)q_ptr, 2, batch, qo_heads, seq_len, D, 64, 128);
    // Descriptor 1: K[batch, kv_heads, seq_len, D=128] bf16, tile 128×128
    _fill_tma_desc_meta((char*)out_meta + 1*96, (uint64_t)k_ptr, 2, batch, kv_heads, seq_len, D, 128, 128);
    // Descriptor 2: V[batch, kv_heads, seq_len, D=128] bf16, tile 128×128
    _fill_tma_desc_meta((char*)out_meta + 2*96, (uint64_t)v_ptr, 2, batch, kv_heads, seq_len, D, 128, 128);
    // Descriptor 3: L[batch, qo_heads, 1, seq_len] float, tile 64×128
    // L is a logsumexp buffer in float; rows=1 but tile_rows=64 is clamped by TK.
    // Conservative: use the actual layout for correct bounding-box computation.
    _fill_tma_desc_meta((char*)out_meta + 3*96, (uint64_t)l_ptr, 4, batch, qo_heads, 1, seq_len, 64, 128);
    // Descriptor 4: O[batch, qo_heads, seq_len, D=128] bf16, tile 64×128
    _fill_tma_desc_meta((char*)out_meta + 4*96, (uint64_t)o_ptr, 2, batch, qo_heads, seq_len, D, 64, 128);
}

extern "C" int mha_fwd_globals_size() {
    return (int)sizeof(fwd_globals<128>);
}

extern "C" void mha_fwd_make_globals(
    void* out_buf,
    void* q_ptr, void* k_ptr, void* v_ptr, void* l_ptr, void* o_ptr,
    int batch, int qo_heads, int kv_heads, int seq_len
) {
    using G = fwd_globals<128>;
    constexpr int D = 128;
    const int hr = qo_heads / kv_heads;

    typename G::q_gl qg{(bf16*)q_ptr, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, (unsigned)D};
    typename G::k_gl kg{(bf16*)k_ptr, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, (unsigned)D};
    typename G::v_gl vg{(bf16*)v_ptr, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, (unsigned)D};
    typename G::l_gl lg{(float*)l_ptr, (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len};
    typename G::o_gl og{(bf16*)o_ptr, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, (unsigned)D};

    G g{qg, kg, vg, lg, og, seq_len, hr};
    memcpy(out_buf, &g, sizeof(G));
}
