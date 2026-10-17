#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv_r8_w1_k5120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_r8_w1_k6144(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_r8_w1_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_t2_gemv_r4_w1_word(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_small_t_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_small_t_v2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// The PTQ1_0 correctness-first path: correct at every T, and currently PTQ1's fallback rung.
void launch_t2_ptq1_reference(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// The same small-T kernel as launch_t2_small_t_v2 with the PTQ1_0 packing: the staging cp.asyncs
// the raw 24+2-byte groups and repacks them in shared memory (t2_ptq1_repack.cuh), so everything
// from the A fragments down is the T2 code path unchanged. Needs w.qhigh.
void launch_t2_small_t_v2_ptq1(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream);
// The widest token slice that launch accepts (its widest column tile). Published rather than
// repeated: the PTQ1 prefill slicer in t2_dispatch.cpp walks T in slices of exactly this width, and
// a hand-copied 16 there would be a second source of truth for one limit.
std::int32_t t2_small_t_v2_max_columns();

// The I line's wide-T ternary MMA kernel, ported for PTQ1_0 (ops/linear/t2/t2_ptq1_wide_t.cuh): a
// bf16 A operand on tensor cores, 64-token tiles, one weight window staged per chunk and reused
// across the whole tile. It consumes the RAW 24+2-byte groups -- no activation quantisation and no
// workspace -- and writes with an explicit row stride, so it serves ONE parent into ONE destination:
// exactly the call shape of linear()'s PTQ1 arm and of the wrappers' fallback path.
// Provenance and the two container-convention deviations are documented in its own header; its
// crossover is the I line's measured 41 tokens, and the I line itself tries the int8 route FIRST, so
// here it is an ALTERNATIVE to the int8 tile rather than a replacement -- off by default, turned on
// with NINFER_PTQ1_WIDE_T=on.
void launch_t2_ptq1_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, cudaStream_t stream);
// The rung gate (defined in t2_a8.cu): NINFER_PTQ1_WIDE_T (off unless set) and NINFER_PTQ1_WIDE_T_MIN
// (default kTernaryWideMinTokens = 41). Read once, like every other route switch in this tree.
[[nodiscard]] bool t2_ptq1_wide_t_enabled();
[[nodiscard]] std::int32_t t2_ptq1_wide_t_min_tokens();

} // namespace ninfer::ops::detail
