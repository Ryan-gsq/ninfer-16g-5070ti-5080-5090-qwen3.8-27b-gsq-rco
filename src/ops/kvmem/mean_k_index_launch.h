#pragma once

// Host-safe half of the mean-K accumulation launch. Separate from the .cu on purpose: including a
// CUDA kernel header from an MSVC host translation unit drags cuda_pipeline_helpers.h in and fails
// with "error C2760: syntax error: unexpected token 'volatile'" -- the failure that broke the first
// build of the int8 rung (ops/linear/ternary/ternary_s8_scratch.h:3-9) and that this split avoids.

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// ACCUMULATES `full_blocks` complete blocks plus an optional partial tail block of `tail_fill` tokens
// (0 == no tail) for one layer into the F32 SUM index, and adds those tokens to the block's count.
//
// ACCUMULATE, NOT STORE -- AND THE CALLER OWNS TWO CONSEQUENCES.
//   `index` holds ordered K SUMs plus a per-block token COUNT; the mean is derived at READ time as
//   `sum / count`. That is the official contract ("ordered K sum (F32) at first write, mean-K computed
//   on read", kvmem-llama.cpp `kvmem/include/kvmem/raw_kv_store.hpp:3-8` -- read for the approach only,
//   that repository carries no licence). Consequences:
//     * The kernel does `+=` everywhere, so EARLIER CONTENT IS FOLDED INTO THE RESULT. A block's
//       slots must be ZERO before their first write; a freshly allocated arena is not guaranteed to be
//       zeroed. `MeanKIndex::zero()` exists for this and the append path calls it once.
//     * Calling twice into the SAME block CONTINUES it, which is exactly what completing a partial
//       tail block is. That is the defect this layout removes: with a stored MEAN there was no way to
//       extend a block without re-reading the tokens it already held, so "tail block indexed while
//       partial, then filled" left a stale mean behind -- a silently wrong index, not an error.
//
// TWO STRIDES, NOT ONE. `staging` is BF16 {kv_size, width} with ne[0] contiguous, so element (k, t)
// sits at t * kv_size + k: `kv_size` is the ROW STRIDE and the token count is not. An earlier version
// passed the token count for both and read 4x past the allocation whenever the two differed -- the
// standalone verifier caught that as cudaErrorIllegalAddress.
//
// `index` is F32, ONE REGION PER LAYER, each region laid out as
//     [ sums: blocks_per_layer * heads * head_dim ][ counts: blocks_per_layer ]
// so the destination layer's base is `index + dst_layer * index_layer_stride`, and its count row begins
// exactly `blocks_per_layer * heads * head_dim` elements after that base (the kernel derives it; both
// live in the same allocation so the layout cannot drift between host and device). `first_block` is the
// destination BLOCK offset within the layer's rows.
//
// `layer_stride` is the element stride between consecutive layers WITHIN `staging` (the harvest packs
// every layer into one allocation), while `index_layer_stride` is the destination's F32 element count
// per layer. They are not interchangeable.
//
// `n_layers` and `layer_index` are separate arguments ON PURPOSE. An earlier version had a single
// `layer` parameter doing both jobs: it sized the grid AND named the destination layer. That made a
// 0-based layer INDEX of 0 produce a zero-sized grid dimension (launches rejected outright, which is
// how the verifier caught it), and an index of 4 of 16 would have written the wrong layer's rows
// without any error at all. One parameter, two meanings, was the whole bug.
//
// Capture-safe: no allocation, no synchronisation. Callers must supply a stream.
//
// Returns the launch status. It deliberately does NOT throw: the engine treats a failed launch as
// fatal, but the standalone verifier has to distinguish "the launch was rejected" from "the kernel ran
// and produced wrong numbers", and an assert on the way out destroys exactly that distinction.
[[nodiscard]] cudaError_t launch_mean_k_accumulate(const void* staging, void* index,
                                                  std::int32_t n_layers, std::int32_t layer_index,
                                                  std::int32_t first_block,
                                                  std::int32_t full_blocks,
                                                  std::int32_t tail_fill, std::int32_t heads,
                                                  std::int32_t head_dim, std::int32_t kv_size,
                                                  std::int32_t block_tokens,
                                                  std::int32_t layer_stride,
                                                  std::int32_t blocks_per_layer,
                                                  std::int32_t index_layer_stride,
                                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
