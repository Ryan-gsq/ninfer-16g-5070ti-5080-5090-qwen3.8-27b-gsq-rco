#pragma once

// PTQ1_0 correctness-first reference GEMM.
//
// COPIED FROM the I line's ops/linear/ternary/ternary_rowsplit_gemm.cuh (110 lines), which is the
// kernel that line routed PTQ1_0 to before the fast rungs admitted it -- see the pitfall record
// "a format in the registry is not a specialised kernel" (00-总索引\接棒共识-工作须知.md §3.9:
// ternary_rowsplit_gemm.cu:87-88 sent PTQ1_0_G128 to the reference template
// <PTQ1RowSplitStorage, PTQ1SimtDecodeAtom>). Same role here: it is correct at EVERY T, so it is
// the rung that makes a PTQ1 artifact usable at all while the fast rungs are ported one by one.
//
// It is generic over <Storage, Atom> and the Atom interface it needs (load_scale + decode_one) is
// exactly what PTQ1DecodeAtom in t2_rowsplit_storage.cuh provides. It also already handles
// `high == nullptr`, which is how the two ternary packings are told apart: PTQ1_0 carries a 2-byte
// high plane, T2 carries none.
//
// ADAPTATION vs the I line (the only one): the row pitch. The I line passed `groups_per_row` and
// used it both for the plane stride and implicitly assumed K was unpadded. This container's other
// t2 kernels (t2_rowsplit_gemm_simt.cuh:176-184, t2_small_t_v2.cuh:131-136) take the row pitch from
// the PADDED width and guard the column against the real k, so the parameter is named
// `groups_per_row_padded` here and callers must pass padded_shape[1] / 128. Passing k / 128 instead
// would read the wrong rows out of the plane whenever K is padded -- silently.
//
// The I line's other note is kept because it is still a warning, not a curiosity: do NOT widen the
// token tile to amortise the per-tile reduction. kTileT = 32 measured 45.2 -> 19.0 t/s on a
// 2,536-token prompt (5,036 tokens: 19.0 t/s too) because 32 accumulators per thread plus a 16 KB
// partials tile cost resident CTAs, and this kernel is latency-bound. kTileT = 8 is the evidenced
// value; the way to make PTQ1 prefill fast is a tensor-core rung, not this.

#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One CTA owns one output row and a tile of kTileT tokens. The CTA has exactly kGroupK (128)
// threads, so thread `index` owns weight `index` inside every 128-weight group: a group is decoded
// once by the whole CTA and reused across the whole token tile, which is what keeps this from being
// 128x redundant.
//
// Activation layout is [K, T] with ne[0] = K CONTIGUOUS, i.e. the ninfer/ggml convention: element
// (column, token) lives at token * k + column. Output is [N, T] the same way: (row, token) at
// token * rows + row. Treating either as row-major ([K, T] with k*T + t) scrambles every prefill
// while T == 1 still works, because the two layouts coincide exactly at one token.
template <class Storage, class Atom, int kTileT>
__global__ __launch_bounds__(Storage::kGroupK) void t2_ptq1_reference_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t t,
    std::int32_t groups_per_row_padded, std::int32_t out_row_stride) {
    static_assert(kTileT >= 1, "the ternary GEMM needs a positive token tile");
    constexpr int kThreads = Storage::kGroupK;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) { return; }
    const int t0 = static_cast<int>(blockIdx.y) * kTileT;

    // Planes are laid out row-major over [rows, groups_per_row_padded].
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups_per_row_padded * Storage::kCodeBytesPerGroup;
    const std::uint8_t* high_row =
        high == nullptr ? nullptr
                        : high + static_cast<std::int64_t>(row) * groups_per_row_padded *
                                     Storage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales +
        static_cast<std::int64_t>(row) * groups_per_row_padded * Storage::kScaleBytesPerGroup;

    const int index = static_cast<int>(threadIdx.x);

    float accumulator[kTileT];
#pragma unroll
    for (int i = 0; i < kTileT; ++i) { accumulator[i] = 0.0f; }

    const std::int32_t groups_per_row = k / Storage::kGroupK;
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t* group_codes = code_row + group * Storage::kCodeBytesPerGroup;
        const std::uint8_t* group_high =
            high_row == nullptr ? nullptr : high_row + group * Storage::kHighBytesPerGroup;
        const float scale = Atom::load_scale(scale_row + group * Storage::kScaleBytesPerGroup);
        const float weight = Atom::decode_one(group_codes, group_high, scale, index);

        const std::int32_t column = group * Storage::kGroupK + index;
        if (column < k) {
#pragma unroll
            for (int i = 0; i < kTileT; ++i) {
                const std::int32_t token = t0 + i;
                if (token < t) {
                    accumulator[i] +=
                        weight * __bfloat162float(x[static_cast<std::int64_t>(token) * k + column]);
                }
            }
        }
    }

    __shared__ float partials[kTileT][kThreads];
#pragma unroll
    for (int i = 0; i < kTileT; ++i) { partials[i][index] = accumulator[i]; }
    __syncthreads();

    // Tree reduction over the CTA; every stage halves the active thread range.
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (index < stride) {
#pragma unroll
            for (int i = 0; i < kTileT; ++i) { partials[i][index] += partials[i][index + stride]; }
        }
        __syncthreads();
    }

    if (index == 0) {
#pragma unroll
        for (int i = 0; i < kTileT; ++i) {
            const std::int32_t token = t0 + i;
            if (token < t) {
                // out_row_stride is the PARENT's row count, so several projections can write
                // disjoint row ranges of one fused output (the GDN split parent, for instance).
                out[static_cast<std::int64_t>(token) * out_row_stride + row] =
                    __float2bfloat16_rn(partials[i][0]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
