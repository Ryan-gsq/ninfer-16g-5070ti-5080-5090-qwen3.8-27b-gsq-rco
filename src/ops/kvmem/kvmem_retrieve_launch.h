#pragma once

// Host-safe half of the retrieval-scoring launch. Separate from the .cu on purpose: a CUDA kernel
// header pulled into an MSVC host translation unit drags cuda_pipeline_helpers.h in and fails with
// "error C2760: syntax error: unexpected token 'volatile'" -- the same split as
// ops/kvmem/mean_k_index_launch.h and ops/linear/ternary/ternary_s8_scratch.h.

#include "ops/kvmem/kvmem_retrieve.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

// Host-side entry point: validates the geometry, derives scale and the band mask, then launches.
//
// `score` is float [n_blocks] and must be PRE-ZEROED: this op only ADDS to it, because each
// (layer, query token) pair contributes its own share of mass and the grid runs those in parallel --
// the contract the reference states outright ("score must be pre-zeroed", kernels_cuda.cu:5332).
//
// Geometry errors throw std::invalid_argument (programming error). A launch failure is RETURNED, not
// thrown, because the engine treats a failed launch as fatal while a standalone verifier has to tell
// "the launch was rejected" from "it ran and produced wrong numbers" apart.
[[nodiscard]] cudaError_t kvmem_retrieve_scores(const KvMemRetrieveConfig& config, float* score,
                                                const void* q, const float* kbar_sum,
                                                const std::int32_t* block_tokens,
                                                cudaStream_t stream);

namespace detail {

// The bare launch, for callers that have already validated and already computed the scale and mask
// (a verifier exercising a specific band configuration, for instance).
//
//   score        : float [n_blocks], pre-zeroed; ADDED to.
//   q            : FP16/FP32 [n_layers, q_layer_stride, n_heads, head_dim], the de-RoPEd query rows
//                  in the same content frame as the index. n_heads is the QUERY head count; the KV
//                  head is derived by GQA grouping (kernels_cuda.cu:5394-5395).
//   kbar_sum     : F32 [n_layers, kbar_layer_stride, n_kv_heads, head_dim], the SUM of that block's
//                  de-RoPEd keys -- NOT a mean. The mean is formed inside the kernel as
//                  sum / block_tokens[b] (see kvmem_retrieve.h for why the division is on this side
//                  of the interface). One representative per block (n_subblocks == 1).
//   block_tokens : int32 [n_blocks], the REAL token count of each block. A partial tail block must
//                  carry its true fill (mean_k_index.h:127-131), never the nominal block size. A null
//                  pointer means "every block holds exactly one token", i.e. the operand is already a
//                  mean -- which is how a pre-normalised F32 index is scored with the same kernel.
//
// Capture-safe: no allocation, no synchronisation.
[[nodiscard]] cudaError_t launch_kvmem_retrieve_scores(const KvMemRetrieveConfig& config,
                                                       float* score, const void* q,
                                                       const float* kbar_sum,
                                                       const std::int32_t* block_tokens,
                                                       float scale, KvMemRetrieveMask mask,
                                                       cudaStream_t stream);

} // namespace detail
} // namespace ninfer::ops
