#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear/t2/t2_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// The integer-A8 policies reach a plain linear for T2 weights, whose integer prefill route lives
// here rather than in a fused op; every other format's dispatcher treats them as A16.
void validate_linear_policy(LinearPolicy policy) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("linear: invalid compute policy");
    }
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (is_gguf(w.qtype)) {
        if (workspace == nullptr) {
            throw std::invalid_argument("linear: a GGUF weight needs the workspace overload");
        }
        detail::gguf_linear(x, w, out, *workspace, stream);
        return;
    }
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, stream);
        return;
    case QType::T2_G128_FP16:
        // NOTE: no PTQ1 switch here. This arm is PQ2's, unchanged, and the switch below is about
        // PTQ1_0 only -- gating PQ2 on it would disable PQ2's int8 prefill whenever the switch is
        // unset, which is its default. (I did exactly that once, with a blanket replace.)
        if (workspace != nullptr && detail::t2_a8_admits(policy) &&
            detail::t2_a8_supported(w, x.ne[1])) {
            detail::t2_a8_linear(x, w, out, *workspace, stream);
            return;
        }
        detail::t2_dispatch(x, w, out, policy, stream);
        return;
    case QType::PTQ1_G128_FP16:
        // PTQ1_0 now has an int8 route, and it is the SAME tile GEMM the T2 arm above uses, with a
        // PTQ1 schedule and a direct base-3 -> int8 decode (no 2-bit relabelling anywhere, which is
        // what the I line measured the repack costing on this rung). t2_a8_supported is the gate: it
        // admits PTQ1 only on the Prefill verdict, because that route's small-T kernel reads 2-bit
        // fields out of the code plane and would mis-decode PTQ1's five-trits-per-byte silently
        // rather than fail. Its T <= 64 band therefore still goes to t2_dispatch, which owns
        // t2_small_t_v2_ptq1 and the reference rung.
        if (workspace != nullptr && detail::t2_ptq1_fast() && detail::t2_a8_admits(policy) &&
            detail::t2_a8_supported(w, x.ne[1])) {
            detail::t2_a8_linear(x, w, out, *workspace, stream);
            return;
        }
        detail::t2_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
    default:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    if (is_gguf(qtype)) {
        const detail::GgufShape shape{qtype, output_rows, input_rows};
        return detail::gguf_project_workspace_bytes({&shape, 1}, min_tokens, max_tokens);
    }
    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::T2_G128_FP16:
        (void)detail::select_t2_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_t2_launch(output_rows, input_rows, max_tokens, policy);
        return detail::t2_a8_workspace_bytes(output_rows, input_rows, policy, max_tokens);
    case QType::PTQ1_G128_FP16:
        // PTQ1_0's prefill band rides the int8 tile GEMM, so the arena must cover the int8 activation
        // planes even though the T <= 64 band needs nothing at all. Returning 0 here would leave the
        // arena too small the moment a prefill routes through t2_a8 -- and the failure would be a
        // capture-time allocation, not a clear error. t2_a8_workspace_bytes is qtype-agnostic (the
        // planes are the same shape either way), so this is the T2 arm's number.
        (void)detail::select_t2_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_t2_launch(output_rows, input_rows, max_tokens, policy);
        return detail::t2_a8_workspace_bytes(output_rows, input_rows, policy, max_tokens);
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32:
    case QType::INT32:
    default:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninfer::ops

