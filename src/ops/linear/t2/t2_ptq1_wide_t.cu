#include "ops/linear/t2/t2_ptq1_wide_t.cuh"

#include "core/device.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// The PTQ1_0 wide-T rung: the T >= kTernaryWideMinTokens kernel of
// ops/linear/t2/t2_ptq1_wide_t.cuh, launched for one whole token count (the kernel walks the token
// axis itself, in tiles of kTernaryWideTokens, so the caller does not slice).
//
// Unlike launch_t2_ptq1_reference (which reads `rows` off `out.ne[0]` and lets the grid cover
// them), this rung takes the row count from the WEIGHT -- the code plane's row pitch comes from
// w.n -- and only ever writes rows < w.n through out_row_stride. Hence the shape checks below: a
// caller whose `out` is narrower or whose pitch is shorter than w.n gets a throw instead of a
// silently clipped or out-of-bounds store.
void launch_t2_ptq1_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, cudaStream_t stream) {
    // `x.ne[0] == w.k` mirrors launch_t2_small_t_v2 (t2_small_t_v2.cu:27): the activation staging
    // indexes x as token * k + column with k == w.k, so a mismatch would read the wrong rows
    // rather than fail. `padded_shape[1] == w.k` is required for the same reason as in the
    // reference rung (t2_ptq1_reference.cu:21-24): the plane pitch is the padded width while the
    // loop bound is the real k, and silently mixing them reads the wrong rows. Every width in this
    // model (5120 / 6144 / 10240 / 17408 / 248320) is a whole number of 128-groups, so the check
    // has always held; it throws instead of mis-decoding if it ever does not.
    //
    // `% kTernaryWideChunkK` is this rung's `% 128` written as the constant it is: the kernel
    // header pins kTernaryWideChunkK == T2RowSplitStorage::kGroupK == 128 with a static_assert,
    // and the kernel consumes K in whole chunks. The qhigh alignment is the I line's
    // ptq1_repack_admits condition (ternary_rowsplit_gemm.cu:211-214) and is not cosmetic here:
    // stage_weights moves 8 bytes of a row's base-3 plane per cp.async, so a base that is not
    // 8-byte aligned is a misaligned copy.
    if (w.qtype != QType::PTQ1_G128_FP16 || w.qdata == nullptr || w.qhigh == nullptr ||
        w.scales == nullptr || (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) != 0u ||
        w.padded_shape[1] != w.k || (w.k % kTernaryWideChunkK) != 0 || x.ne[0] != w.k ||
        x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[1] < 1 || out.ne[0] < w.n ||
        out_row_stride < w.n) {
        throw std::invalid_argument(
            "t2 PTQ1 wide-t mma: unsupported shape or packing (needs a PTQ1_G128_FP16 weight "
            "with an 8-byte aligned high plane and k a whole number of 128-groups, BF16 x/out, "
            "x.ne[0] == k, at least one token, out.ne[0] >= n and out_row_stride >= n)");
    }
    // Static shared memory (29.25 KiB at the shipped constants, see the storage struct), so no
    // dynamic size, no cudaFuncSetAttribute and no opt-in -- which is what keeps
    // kTernaryWideMinBlocksPerSm reachable. One CTA covers kTernaryWideRowsPerCta rows, hence the
    // row-block grid.
    const unsigned grid =
        static_cast<unsigned>((w.n + kTernaryWideRowsPerCta - 1) / kTernaryWideRowsPerCta);
    ternary_wide_t_kernel<true><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), w.n, w.k, x.ne[1], out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
