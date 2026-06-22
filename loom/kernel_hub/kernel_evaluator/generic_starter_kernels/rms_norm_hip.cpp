// Starter HIP kernel for the aiter.rms_norm plugin (target: hip).
// Export a pybind11 module named `candidate` with `dispatch`, called as:
//   dispatch(x, weight, out, dim_m, dim_n, epsilon)
// x:[M,N] bf16 read, weight:[N] bf16 read, out:[M,N] bf16 write (in-place).
// Math: out[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + epsilon) * weight
//
// This is a correct-but-naive baseline: one block per row, fp32 accumulation.
// It is meant to compile and pass correctness so you can iterate on speed.

#include <torch/extension.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void rms_norm_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ out,
    int dim_n,
    float epsilon
) {
    const int row = blockIdx.x;
    const at::BFloat16* x_row = x + static_cast<long long>(row) * dim_n;
    at::BFloat16* out_row = out + static_cast<long long>(row) * dim_n;

    float local_sq = 0.0f;
    for (int col = threadIdx.x; col < dim_n; col += blockDim.x) {
        const float v = static_cast<float>(x_row[col]);
        local_sq += v * v;
    }

    __shared__ float partial[kThreads];
    partial[threadIdx.x] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(dim_n) + epsilon);

    for (int col = threadIdx.x; col < dim_n; col += blockDim.x) {
        const float v = static_cast<float>(x_row[col]);
        const float w = static_cast<float>(weight[col]);
        out_row[col] = static_cast<at::BFloat16>(v * inv_rms * w);
    }
}

}  // namespace

void dispatch(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t dim_m,
    int64_t dim_n,
    double epsilon
) {
    const dim3 grid(static_cast<unsigned>(dim_m));
    const dim3 block(kThreads);
    rms_norm_kernel<<<grid, block>>>(
        x.data_ptr<at::BFloat16>(),
        weight.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(dim_n),
        static_cast<float>(epsilon)
    );
}

PYBIND11_MODULE(candidate, m) {
    m.def("dispatch", &dispatch);
}
