// Starter HIP kernel for the aiter.add_rms_norm plugin (target: hip).
// Export a pybind11 module named `candidate` with `dispatch`, called as:
//   dispatch(x, residual_in, weight, out, residual_out, dim_m, dim_n, epsilon)
// x:[M,N] bf16 read, residual_in:[M,N] bf16 read, weight:[N] bf16 read,
// out:[M,N] bf16 write, residual_out:[M,N] bf16 write.
// Math: residual_out[i, :] = x[i, :] + residual_in[i, :]
//       out[i, :] = residual_out[i, :] * rsqrt(mean(residual_out[i, :]^2) + epsilon) * weight
//
// Correct-but-naive baseline: one block per row, fp32 accumulation.
// It is meant to compile and pass correctness so you can iterate on speed.

#include <torch/extension.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void add_rms_norm_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ residual_in,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ out,
    at::BFloat16* __restrict__ residual_out,
    int dim_n,
    float epsilon
) {
    const int row = blockIdx.x;
    const long long base = static_cast<long long>(row) * dim_n;
    const at::BFloat16* x_row = x + base;
    const at::BFloat16* res_in_row = residual_in + base;
    at::BFloat16* out_row = out + base;
    at::BFloat16* res_out_row = residual_out + base;

    float local_sq = 0.0f;
    for (int col = threadIdx.x; col < dim_n; col += blockDim.x) {
        const float v = static_cast<float>(x_row[col]) + static_cast<float>(res_in_row[col]);
        res_out_row[col] = static_cast<at::BFloat16>(v);
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
        const float v = static_cast<float>(res_out_row[col]);
        const float w = static_cast<float>(weight[col]);
        out_row[col] = static_cast<at::BFloat16>(v * inv_rms * w);
    }
}

}  // namespace

void dispatch(
    torch::Tensor x,
    torch::Tensor residual_in,
    torch::Tensor weight,
    torch::Tensor out,
    torch::Tensor residual_out,
    int64_t dim_m,
    int64_t dim_n,
    double epsilon
) {
    const dim3 grid(static_cast<unsigned>(dim_m));
    const dim3 block(kThreads);
    add_rms_norm_kernel<<<grid, block>>>(
        x.data_ptr<at::BFloat16>(),
        residual_in.data_ptr<at::BFloat16>(),
        weight.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        residual_out.data_ptr<at::BFloat16>(),
        static_cast<int>(dim_n),
        static_cast<float>(epsilon)
    );
}

PYBIND11_MODULE(candidate, m) {
    m.def("dispatch", &dispatch);
}
