// Shared TMA descriptor metadata helpers for CTA access profiling.
//
// Metadata slot layout (96 bytes, read by profile_cta_kernel.py):
//   [0:8]    base_ptr     (u64)
//   [8:48]   strides[5]   (u64 × 5)  — byte stride per dimension, zero-padded
//   [48:68]  box_dims[5]  (u32 × 5)  — tile size per dimension, zero-padded
//   [68:72]  elem_size    (u32)
//   [72:96]  padding (zeroed)
//
// The profiler reads only strides[:ndim] and box_dims[:ndim], where ndim comes
// from the PTX instruction qualifier (e.g. cp.async.bulk.tensor.2d → ndim=2,
// .5d → ndim=5).  Use the variant matching the kernel's cuTensorMapEncodeTiled
// dimensionality.
#pragma once
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// 5D variant — TK Axis-2 swizzled encoding (mm_tk_ab, mm_tk_abt, sdpa_tk_mha)
// ---------------------------------------------------------------------------
static inline void _fill_tma_desc_meta(
    void* out, uint64_t base_ptr, int dtype_size,
    int batch, int depth, int rows, int cols,
    int tile_rows, int tile_cols
) {
    const int swizzle_bytes = 128;
    const int swizzle_elements = swizzle_bytes / dtype_size;
    uint64_t* strides = (uint64_t*)((char*)out + 8);
    uint32_t* box_dims = (uint32_t*)((char*)out + 48);
    *(uint64_t*)out = base_ptr;
    strides[0] = (uint64_t)dtype_size;
    strides[1] = (uint64_t)cols * dtype_size;
    strides[2] = (uint64_t)swizzle_bytes;
    strides[3] = (uint64_t)rows * cols * dtype_size;
    strides[4] = (uint64_t)depth * rows * cols * dtype_size;
    box_dims[0] = (uint32_t)swizzle_elements;
    box_dims[1] = (uint32_t)tile_rows;
    box_dims[2] = (uint32_t)(tile_cols / swizzle_elements);
    box_dims[3] = 1;
    box_dims[4] = 1;
    *(uint32_t*)((char*)out + 68) = (uint32_t)dtype_size;
    memset((char*)out + 72, 0, 24);
}

// ---------------------------------------------------------------------------
// 2D variant — plain cuTensorMapEncodeTiled (hand-tuned kernels: mm_abt_*)
//
// Matches the 2D encoding: dim0 = columns (elements), dim1 = rows.
// strides[0] = elem_size, strides[1] = row_stride_bytes.
// box_dims[0] = tile_cols, box_dims[1] = tile_rows.
// ---------------------------------------------------------------------------
static inline void _fill_tma_desc_meta_2d(
    void* out, uint64_t base_ptr, int dtype_size,
    int rows, int cols,
    int tile_rows, int tile_cols
) {
    uint64_t* strides = (uint64_t*)((char*)out + 8);
    uint32_t* box_dims = (uint32_t*)((char*)out + 48);
    *(uint64_t*)out = base_ptr;
    strides[0] = (uint64_t)dtype_size;
    strides[1] = (uint64_t)cols * dtype_size;
    memset(strides + 2, 0, 3 * sizeof(uint64_t));
    box_dims[0] = (uint32_t)tile_cols;
    box_dims[1] = (uint32_t)tile_rows;
    memset(box_dims + 2, 0, 3 * sizeof(uint32_t));
    *(uint32_t*)((char*)out + 68) = (uint32_t)dtype_size;
    memset((char*)out + 72, 0, 24);
}
