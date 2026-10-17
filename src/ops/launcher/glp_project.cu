// Implements: include/ninfer/ops/glp_project.h
// Launch policy: one block per column. The direction r is shared by every column, so it is
// re-read from global memory per block; E = 5120 means r is 10 KiB, which stays in L2 across
// the grid. Grid dimension is the token count, which is bounded by the prefill chunk.
#include "ops/launcher/glp_project.h"

#include "ops/common/math.h"
#include "ops/kernel/glp_project.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

void glp_project_launch(Tensor& y, const Tensor& r, const Tensor& alpha, cudaStream_t stream) {
    const std::int32_t e       = static_cast<std::int32_t>(y.ne[0]);
    const std::int32_t columns = static_cast<std::int32_t>(y.ne[1]);
    if (columns <= 0) { return; }

    const float* alpha_ptr = static_cast<const float*>(alpha.data);
    glp_project_kernel<<<columns, kGlpProjectBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y.data), static_cast<const __nv_bfloat16*>(r.data), alpha_ptr,
        e, columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
