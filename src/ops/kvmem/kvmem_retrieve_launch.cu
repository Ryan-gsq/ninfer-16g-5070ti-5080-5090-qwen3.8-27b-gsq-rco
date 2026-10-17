// KVMem retrieval scoring, device side. See ops/kvmem/kvmem_retrieve.h for the contract, for the
// reference provenance of Eq.10 (implementation_notes.md:549-588, kernels_cuda.cu:5340-5435), and
// for why the mean is formed HERE from an F32 sum plus a token count.
//
// SHAPE OF THE COMPUTATION
//   Grid: one CUDA block per (layer, query token) -- the same decomposition the reference kernel uses
//   (kernels_cuda.cu:5360-5363). Each block owns the FULL softmax-over-blocks for its (layer, token)
//   pair and every query head, so no cross-block reduction is needed: the grid dimension that varies
//   is exactly the dimension the score is SUMMED over (layers and query tokens), which is why the
//   reference can accumulate with atomicAdd and a constant weight head_w = 1/(L*H).
//
// WHY TWO PASSES OVER THE BLOCKS, AND WHAT IT COSTS
//   A softmax needs the max (for stability) and the sum before any mass can be weighted. The
//   reference keeps the per-block logits in dynamic shared memory and so touches each (block, head)
//   dot product once (kernels_cuda.cu:5364, 5402-5416). We do not, because n_blocks is bounded by the
//   INDEX CAPACITY rather than by a kernel-side constant, and a shared-logit buffer sized for a
//   1M-token workspace (16384 blocks x 4 B = 64 KiB) would need the opt-in large-shared-memory path
//   plus a second code path for anything larger. Instead:
//
//     phase 1: one pass computing (m, s) per head with an ONLINE softmax (rescaling as it goes)
//     phase 2: one pass computing exp(logit - m)/s and accumulating the mass into score[b]
//
//   That is two dot products per (block, head) instead of one -- a 2x cost on the dominant term.
//   This is accepted DELIBERATELY for this cut, for two reasons: (a) retrieval runs once per view
//   update (`step` or `interval`, never per token -- implementation_notes.md:1191-1230), so 2x on a
//   per-interval op is not on the decode path; (b) a verifier can separate the statistics pass from
//   the accumulation pass, which is exactly how the two mean-K bugs were found. The one-dot path
//   (shared logits, sized to the index capacity, with an explicit large-shared opt-in) is the
//   documented follow-up optimisation -- the same trade the reference itself made and later refined
//   ("the CUDA backend preserves this exact computation above the fused kernel's shared-memory cap
//   via a bounded one-dot path", qwen_executor.cpp:23750-23753).
//
// NORMALISATION: ONE SCALAR MULTIPLY, NOT A DIVIDE PER ELEMENT
//   eq.10 wants dot(q, mean) where mean = sum / count. Distributing the division over the dot gives
//   dot(q, sum) / count, so the kernel folds it into the logit scale:
//
//     logit = (scale / count_b) * dot(q, sum_b)
//
//   Identical arithmetic, no per-element division, and it is the reason the index can store a raw
//   F32 sum with no second quantisation (kvmem_retrieve.h, interface decision (a)/(b)).
//
// FP32 ARITHMETIC, FP16 QUERY STORAGE
//   The dot accumulates in FP32 from the operand's own precision (converted per element, exactly as
//   kernels_cuda.cu:5411-5414 does for FP16), the softmax statistics are FP32, and the score
//   accumulation is FP32. The reference uses __expf for speed; this uses expf, the accurate
//   intrinsic, because the cost is off the decode path and because a verifier comparing against an
//   FP64 reference should be able to attribute any residual to the algorithm rather than to a
//   fast-math approximation. That is a deliberate, documented deviation from the reference kernel.
//
// NONDETERMINISM (measured, not assumed)
//   Scores are accumulated with atomicAdd from many (layer, token) blocks, so the summation ORDER
//   varies run to run and the last ulp of a score can move. The reference has the same property.
//   The RANKING is unaffected except for exactly-equal scores, so the verifier measures the run-to-run
//   spread and reports it instead of claiming bit-exactness.

#include "ops/kvmem/kvmem_retrieve_launch.h"

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {
namespace detail {
namespace {

constexpr int kRetrieveThreads = 256;
constexpr int kRetrieveWarps   = kRetrieveThreads / kWarpSize;

// q and the index sum have different storage precisions (BF16 in the live window, F32 sum), so the
// operand conversion is explicit and per element -- the same thing kernels_cuda.cu:5411-5414 does.
// BF16 is the shape attn_mix actually hands over (Tensor ne = {256, 24, T}, ne[0] contiguous), so this
// is the production instantiation, not a convenience.
__device__ __forceinline__ float dot_q_bf16_sum(const __nv_bfloat16* __restrict__ q,
                                                const float* __restrict__ ksum, int n) {
    float s = 0.0f;
    for (int d = 0; d < n; ++d) {
        s += __bfloat162float(q[d]) * ksum[d];
    }
    return s;
}

__device__ __forceinline__ float dot_q_f16_sum(const __half* __restrict__ q,
                                               const float* __restrict__ ksum, int n) {
    float s = 0.0f;
    for (int d = 0; d < n; ++d) {
        s += __half2float(q[d]) * ksum[d];
    }
    return s;
}

__device__ __forceinline__ float dot_q_f32_sum(const float* __restrict__ q,
                                               const float* __restrict__ ksum, int n) {
    float s = 0.0f;
    for (int d = 0; d < n; ++d) {
        s += q[d] * ksum[d];
    }
    return s;
}

template <typename QT>
__device__ __forceinline__ float dot_q_sum(const QT* q, const float* ksum, int n);

template <>
__device__ __forceinline__ float dot_q_sum<__nv_bfloat16>(const __nv_bfloat16* q,
                                                          const float* ksum, int n) {
    return dot_q_bf16_sum(q, ksum, n);
}

template <>
__device__ __forceinline__ float dot_q_sum<__half>(const __half* q, const float* ksum, int n) {
    return dot_q_f16_sum(q, ksum, n);
}

template <>
__device__ __forceinline__ float dot_q_sum<float>(const float* q, const float* ksum, int n) {
    return dot_q_f32_sum(q, ksum, n);
}

// blockIdx.x = layer * n_query_tokens + query_token. Every thread of the block cooperates on every
// query head in turn, so the head loop is INSIDE the kernel (kernels_cuda.cu:5394) and phase 2 can
// reuse the per-head statistics from phase 1 out of shared memory.
template <typename QT>
__global__ void kvmem_retrieve_scores_kernel(float* __restrict__ score,
                                             const QT* __restrict__ q_multi,
                                             const float* __restrict__ kbar_sum,
                                             const std::int32_t* __restrict__ block_tokens,
                                             int n_layers, int n_query_tokens, int n_heads,
                                             int n_kv_heads, int head_dim, int q_layer_stride,
                                             int q_token_begin, int n_blocks, int kbar_layer_stride,
                                             int incl_lo, int incl_hi, float scale, float head_w) {
    const int lt = static_cast<int>(blockIdx.x);
    const int l  = lt / n_query_tokens;
    const int t  = lt % n_query_tokens;
    if (l >= n_layers) { return; }

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;

    // [0, n_heads) = per-head max, [n_heads, 2*n_heads) = per-head softmax denominator.
    extern __shared__ float s_stats[];
    __shared__ float       s_wm[kRetrieveWarps];
    __shared__ float       s_ws[kRetrieveWarps];

    // BOTH STRIDES ARE SEPARATE FROM THEIR COUNTS. q_layer_stride is in ROWS and kbar_layer_stride in
    // BLOCKS, and either may exceed the live count when the buffers are over-allocated so that a
    // growing session keeps earlier slices valid (kernels_cuda.cu:5378-5388, 23794-23798).
    // q_token_begin is the one extra term the live-window mode needs: the marked span usually starts
    // part-way into a chunk-sized live plane, so the span's rows are [q_token_begin, +n_query_tokens)
    // of that plane's token axis, and no repacking of the live tensor is required.
    const QT* q_l = q_multi + static_cast<long long>(l) * q_layer_stride * n_heads * head_dim;
    const float* kbar_l =
        kbar_sum + static_cast<long long>(l) * kbar_layer_stride * n_kv_heads * head_dim;
    const int group = n_heads / n_kv_heads;

    // ---- Phase 1: per-head (max, denominator) over the INCLUDED blocks only.
    //
    // The included range is the retrievable middle: the reference sets logit = -inf for the
    // always-kept sink/recent bands (kernels_cuda.cu:5400-5407) so their mass redistributes. Skipping
    // them is the same arithmetic and never evaluates exp(-inf - -inf), which would be NaN if a
    // caller ever passed a range with no included block at all.
    for (int qh = 0; qh < n_heads; ++qh) {
        const int kvh  = qh / group;
        const QT* qr   = q_l + (static_cast<long long>(q_token_begin + t) * n_heads + qh) * head_dim;
        float     lm   = -CUDART_INF_F;
        float     ls   = 0.0f;
        for (int b = incl_lo + tid; b < incl_hi; b += kRetrieveThreads) {
            const float* krow = kbar_l + (static_cast<long long>(b) * n_kv_heads + kvh) * head_dim;
            // The mean is formed here: sum / count, folded into the logit scale. A null count array
            // means the operand is already a mean (count == 1).
            const float invc  = block_tokens != nullptr
                                    ? 1.0f / static_cast<float>(block_tokens[b])
                                    : 1.0f;
            const float logit = (scale * invc) * dot_q_sum<QT>(qr, krow, head_dim);
            // Online softmax: keep the running max and rescale the running sum. The first finite
            // logit finds lm == -inf, where expf(-inf) == 0 makes the rescale a no-op.
            if (logit > lm) {
                ls = ls * expf(lm - logit);
                lm = logit;
            }
            ls += expf(logit - lm);
        }
        // warp_max broadcasts (shfl_xor), so every lane holds the warp's max; only lane 0 ends up
        // with the reduced sum (warp_reduce_sum is a down-shuffle). Guarding on lm == -inf keeps
        // expf(-inf - -inf) out of the reduction for a warp that owns no included block.
        const float wm = warp_max(lm);
        float       ws = (lm == -CUDART_INF_F) ? 0.0f : ls * expf(lm - wm);
        ws             = warp_reduce_sum(ws);
        if (lane == 0) {
            s_wm[warp] = wm;
            s_ws[warp] = ws;
        }
        __syncthreads();
        if (tid == 0) {
            float gm = -CUDART_INF_F;
            float gs = 0.0f;
            for (int w = 0; w < kRetrieveWarps; ++w) {
                const float m = s_wm[w];
                if (m == -CUDART_INF_F) { continue; }
                if (m > gm) {
                    gs = (gm == -CUDART_INF_F) ? 0.0f : gs * expf(gm - m);
                    gm = m;
                }
                gs += s_ws[w] * expf(m - gm);
            }
            s_stats[qh]           = gm;
            s_stats[n_heads + qh] = gs;
        }
        __syncthreads();
    }

    // ---- Phase 2: mass accumulation, ONE atomic per (block, layer, query token).
    //
    // The head loop is inside so the atomics are n_heads times fewer than one per (block, head), and
    // the store is coalesced across the block dimension. head_w carries the mean over layers AND over
    // heads, which is what turns the grid-wide sum into Eq.10's "sum_m mean_l mean_h".
    for (int b = incl_lo + tid; b < incl_hi; b += kRetrieveThreads) {
        const float invc      = block_tokens != nullptr
                                    ? 1.0f / static_cast<float>(block_tokens[b])
                                    : 1.0f;
        const float logit_w   = scale * invc;
        float       local     = 0.0f;
        for (int qh = 0; qh < n_heads; ++qh) {
            const float gm = s_stats[qh];
            const float gs = s_stats[n_heads + qh];
            // Unreachable while the band rule guarantees a non-empty middle, and defended anyway:
            // contributing nothing is the correct degenerate answer, whereas dividing by a zero
            // denominator would write inf/NaN into the score the selector sorts on.
            if (gm == -CUDART_INF_F || gs <= 0.0f) { continue; }
            const int    kvh  = qh / group;
            const QT*    qr =
                q_l + (static_cast<long long>(q_token_begin + t) * n_heads + qh) * head_dim;
            const float* krow = kbar_l + (static_cast<long long>(b) * n_kv_heads + kvh) * head_dim;
            const float  logit = logit_w * dot_q_sum<QT>(qr, krow, head_dim);
            local += expf(logit - gm) * (head_w / gs);
        }
        if (local != 0.0f) {
            atomicAdd(&score[b], local);
        }
    }
}

} // namespace

cudaError_t launch_kvmem_retrieve_scores(const KvMemRetrieveConfig& config, float* score,
                                         const void* q, const float* kbar_sum,
                                         const std::int32_t* block_tokens, float scale,
                                         KvMemRetrieveMask mask, cudaStream_t stream) {
    if (score == nullptr || q == nullptr || kbar_sum == nullptr) { return cudaErrorInvalidValue; }
    // Empty query / empty index: nothing to score, and that is a normal state (MeanKIndex::empty()).
    if (config.n_layers <= 0 || config.n_query_tokens <= 0 || config.n_blocks <= 0) {
        return cudaSuccess;
    }
    if (config.n_heads <= 0 || config.n_kv_heads <= 0 || config.n_heads % config.n_kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    const int incl_lo = mask.active ? mask.lo_end : 0;
    const int incl_hi = mask.active ? mask.hi_begin : config.n_blocks;
    // A mask that leaves nothing competing means no block receives mass at all; returning early keeps
    // the score array untouched (still all zero) rather than launching a grid that can only add 0.
    if (incl_hi <= incl_lo) { return cudaSuccess; }
    const long long grid_x =
        static_cast<long long>(config.n_layers) * static_cast<long long>(config.n_query_tokens);
    if (grid_x > 0x7fffffffLL) { return cudaErrorInvalidValue; }

    // head_w carries the mean over LAYERS and heads. It must be built from the model's TOTAL
    // full-attention layer count, not from this call's layer count: in the live-window mode the caller
    // issues one launch per layer (only one qn is alive at a time), and using 1/(1*H) there would make
    // the 16 accumulated calls 16x too large.
    const int   total_layers = config.n_layers_total > 0 ? config.n_layers_total : config.n_layers;
    const float head_w =
        1.0f / (static_cast<float>(total_layers) * static_cast<float>(config.n_heads));
    const std::size_t shmem = 2u * sizeof(float) * static_cast<std::size_t>(config.n_heads);
    const dim3        grid(static_cast<unsigned>(grid_x));

    if (config.dtype == KvMemRetrieveDtype::BF16) {
        kvmem_retrieve_scores_kernel<__nv_bfloat16><<<grid, kRetrieveThreads, shmem, stream>>>(
            score, static_cast<const __nv_bfloat16*>(q), kbar_sum, block_tokens, config.n_layers,
            config.n_query_tokens, config.n_heads, config.n_kv_heads, config.head_dim,
            config.q_layer_stride, config.q_token_begin, config.n_blocks, config.kbar_layer_stride,
            incl_lo, incl_hi, scale, head_w);
    } else if (config.dtype == KvMemRetrieveDtype::F16) {
        kvmem_retrieve_scores_kernel<__half><<<grid, kRetrieveThreads, shmem, stream>>>(
            score, static_cast<const __half*>(q), kbar_sum, block_tokens, config.n_layers,
            config.n_query_tokens, config.n_heads, config.n_kv_heads, config.head_dim,
            config.q_layer_stride, config.q_token_begin, config.n_blocks, config.kbar_layer_stride,
            incl_lo, incl_hi, scale, head_w);
    } else {
        kvmem_retrieve_scores_kernel<float><<<grid, kRetrieveThreads, shmem, stream>>>(
            score, static_cast<const float*>(q), kbar_sum, block_tokens, config.n_layers,
            config.n_query_tokens, config.n_heads, config.n_kv_heads, config.head_dim,
            config.q_layer_stride, config.q_token_begin, config.n_blocks, config.kbar_layer_stride,
            incl_lo, incl_hi, scale, head_w);
    }
    return cudaGetLastError();
}

} // namespace detail
} // namespace ninfer::ops
