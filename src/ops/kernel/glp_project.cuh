#pragma once

// Implements: include/ninfer/ops/glp_project.h
// One thread block owns one column (token) of the [E, T] activation. The block reduces
// r . y over E in FP32, then subtracts alpha * s * r in place. E = 5120 is the residual
// width, so a block of 256 threads walks 20 elements per thread; T is the number of
// columns (1 in decode, up to the prefill chunk).
//
// Shape assumptions (matching the wrapper predicates): E is a multiple of 4 (so the block
// can use 4-element BF16 loads), r is contiguous BF16 of length E. All pointers are
// pre-resident device addresses and the launch branches on no device data, which keeps it
// CUDA-Graph safe.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGlpProjectBlock = 256;

// y: [E, T] row-major, column c starts at c * E. r: [E]. alpha: device scalar (FP32).
__global__ __launch_bounds__(kGlpProjectBlock) void glp_project_kernel(
    __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ r,
    const float* __restrict__ alpha, std::int32_t e, std::int32_t columns) {
    const std::int32_t c = blockIdx.x;
    if (c >= columns) { return; }

    __nv_bfloat16* __restrict__ col = y + static_cast<std::int64_t>(c) * e;
    __shared__ float warp_sums[kGlpProjectBlock / kWarpSize];
    __shared__ float scale_shared;

    // Pass 1: s = r . y, FP32 accumulation with 4-wide BF16 loads.
    float partial = 0.0F;
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x) * 4; i < e;
         i += kGlpProjectBlock * 4) {
        const Bf16x4Pack yv = load_vec<Bf16x4Pack>(col + i);
        const Bf16x4Pack rv = load_vec<Bf16x4Pack>(r + i);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float2 yf = __bfloat1622float2(yv.pair[p]);
            const float2 rf = __bfloat1622float2(rv.pair[p]);
            partial += yf.x * rf.x + yf.y * rf.y;
        }
    }
    // block_reduce_sum leaves the total in thread 0; publish it through shared memory and order it
    // before pass 2, which re-reads every element this block just touched.
    const float s = block_reduce_sum<kGlpProjectBlock>(partial, warp_sums);
    if (threadIdx.x == 0) { scale_shared = (*alpha) * s; }
    __syncthreads();
    const float scale = scale_shared;

    // Pass 2: y -= alpha * s * r, FP32 subtract with a single BF16 rounding at the store.
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x) * 4; i < e;
         i += kGlpProjectBlock * 4) {
        Bf16x4Pack yv       = load_vec<Bf16x4Pack>(col + i);
        const Bf16x4Pack rv = load_vec<Bf16x4Pack>(r + i);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float2 yf = __bfloat1622float2(yv.pair[p]);
            const float2 rf = __bfloat1622float2(rv.pair[p]);
            yv.pair[p]      = __floats2bfloat162_rn(yf.x - scale * rf.x, yf.y - scale * rf.y);
        }
        store_vec(col + i, yv);
    }
}

} // namespace ninfer::ops
