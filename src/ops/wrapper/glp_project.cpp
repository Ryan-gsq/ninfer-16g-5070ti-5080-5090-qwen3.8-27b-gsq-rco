// Implements: include/ninfer/ops/glp_project.h
//
// Validation and finite dispatch. The shape is fixed (one residual width, one dtype pair), so
// the dispatch is a single launch; the wrapper still owns the contract predicates so that the
// launcher and kernel can assume a validated problem.
#include "ninfer/ops/glp_project.h"

#include "ops/launcher/glp_project.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_activation(const Tensor& y) {
    if (y.dtype != DType::BF16 || y.ne[0] <= 0 || y.ne[1] <= 0 || y.ne[2] != 1 || y.ne[3] != 1 ||
        !y.is_contiguous() || y.data == nullptr) {
        throw std::invalid_argument("glp_project: y must be contiguous BF16 [E,T]");
    }
    // The kernel uses 4-element BF16 loads, so E must be a multiple of 4.
    if ((y.ne[0] % 4) != 0) {
        throw std::invalid_argument("glp_project: E must be a multiple of 4");
    }
}

void require_direction(const Tensor& r, std::int32_t e) {
    if (r.dtype != DType::BF16 || r.ne[0] != e || r.ne[1] != 1 || r.ne[2] != 1 || r.ne[3] != 1 ||
        !r.is_contiguous() || r.data == nullptr) {
        throw std::invalid_argument("glp_project: r must be contiguous BF16 [E]");
    }
}

void require_alpha(const Tensor& alpha) {
    if (alpha.dtype != DType::FP32 || alpha.ne[0] != 1 || alpha.ne[1] != 1 || alpha.ne[2] != 1 ||
        alpha.ne[3] != 1 || alpha.data == nullptr) {
        throw std::invalid_argument("glp_project: alpha must be a device FP32 scalar");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void glp_project(Tensor& y, const Tensor& r, const Tensor& alpha, cudaStream_t stream) {
    require_activation(y);
    require_direction(r, y.ne[0]);
    require_alpha(alpha);
    if (overlaps(y, r) || overlaps(y, alpha)) {
        throw std::invalid_argument("glp_project: y must not alias r or alpha");
    }
    detail::glp_project_launch(y, r, alpha, stream);
}

} // namespace ninfer::ops
