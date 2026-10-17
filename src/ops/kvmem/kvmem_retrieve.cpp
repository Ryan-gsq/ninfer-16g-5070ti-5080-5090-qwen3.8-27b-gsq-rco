#include "ops/kvmem/kvmem_retrieve.h"

#include "ops/kvmem/kvmem_retrieve_launch.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// KVMem retrieval scoring, host side. See ops/kvmem/kvmem_retrieve.h for the contract, for where
// Eq.10 comes from, and for the invariant (sum_b score[b] == n_query_tokens) that makes it testable.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "kvmem_retrieve";

} // namespace

float kvmem_retrieve_scale(std::int32_t head_dim) noexcept {
    // 1/sqrt(head_dim), exactly as src/qwen_executor.cpp:23755. head_dim <= 0 is a programming error
    // that kvmem_retrieve_validate() rejects; returning 0 here keeps this function total.
    if (head_dim <= 0) { return 0.0f; }
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

KvMemRetrieveMask kvmem_retrieve_mask_bands(const KvMemRetrieveConfig& config) noexcept {
    KvMemRetrieveMask out{};
    if (config.mask_mode == KvMemRetrieveMaskMode::Never) { return out; }
    const std::int32_t n_blocks = config.n_blocks;
    if (n_blocks <= 0) { return out; }

    // The reference masks ONLY when selection is actually competitive, i.e. the history is over
    // budget (qwen_executor.cpp:23776). Below that, pick_topk_blocks returns everything in order, so
    // masking the bands could only move mass around without changing the outcome -- and it would
    // change the scores an A/B compares.
    if (config.mask_mode == KvMemRetrieveMaskMode::Official) {
        if (config.budget_blocks <= 0 || n_blocks <= config.budget_blocks) { return out; }
    }

    // Both bands are clamped to the history exactly as the reference clamps them
    // (qwen_executor.cpp:23778-23779), and the mask is dropped if no non-empty middle survives --
    // that guard is what keeps at least one finite logit in the softmax.
    const std::int32_t sink   = std::min(config.sink_blocks, n_blocks);
    const std::int32_t recent = std::min(config.recent_blocks, n_blocks);
    if (sink + recent >= n_blocks) { return out; }

    out.lo_end   = sink;
    out.hi_begin = n_blocks - recent;
    out.active   = true;
    return out;
}

void kvmem_retrieve_validate(const KvMemRetrieveConfig& config) {
    // ZERO IS A NORMAL STATE, NEGATIVE IS A BUG. A caller with no captured query rows or an empty
    // index has nothing to score -- the same rule MeanKIndex applies to an empty index -- so those
    // cases return quietly and produce an all-zero score array. Negative counts and inconsistent
    // geometry are programmer errors and are refused loudly, because silently scoring the wrong
    // layer's rows would look exactly like a working retrieval that ranks badly.
    if (config.n_layers < 0 || config.n_query_tokens < 0 || config.n_heads < 0 ||
        config.n_kv_heads < 0 || config.head_dim < 0 || config.n_blocks < 0 ||
        config.n_layers_total < 0 || config.q_layer_stride < 0 || config.kbar_layer_stride < 0 ||
        config.q_token_begin < 0 || config.budget_blocks < 0 || config.sink_blocks < 0 ||
        config.recent_blocks < 0) {
        throw std::invalid_argument(std::string(kSubject) + ": negative geometry");
    }
    // Nothing to do: no index, or no query rows. Not an error.
    if (config.n_layers == 0 || config.n_query_tokens == 0 || config.n_blocks == 0) { return; }

    if (config.n_heads <= 0 || config.n_kv_heads <= 0) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": n_heads and n_kv_heads must both be positive when there is "
                                    "work to do");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        // The GQA mapping is kvh = qh / (n_heads / n_kv_heads) (kernels_cuda.cu:5369, 5395). A
        // non-multiple would truncate that group size and silently mis-address every query head past
        // the first group, so it is refused rather than rounded.
        throw std::invalid_argument(std::string(kSubject) + ": n_heads (" +
                                    std::to_string(config.n_heads) +
                                    ") must be a multiple of n_kv_heads (" +
                                    std::to_string(config.n_kv_heads) + ") for the GQA mapping");
    }
    if (config.head_dim <= 0) {
        throw std::invalid_argument(std::string(kSubject) + ": head_dim must be positive");
    }
    // Strides may EXCEED their counts (over-allocated session buffers) but never fall below them: an
    // under-strided q or index reads a neighbouring layer's rows, which produces plausible scores for
    // the wrong history. For q the bound is the END of the span, not its length, because in the live
    // window the span sits inside a chunk-sized plane at q_token_begin (kvmem_retrieve.h, constraint 5).
    if (config.q_layer_stride < config.q_token_begin + config.n_query_tokens) {
        throw std::invalid_argument(std::string(kSubject) + ": q_layer_stride (" +
                                    std::to_string(config.q_layer_stride) +
                                    ") cannot hold the span [q_token_begin=" +
                                    std::to_string(config.q_token_begin) + ", +" +
                                    std::to_string(config.n_query_tokens) + ") rows");
    }
    if (config.kbar_layer_stride < config.n_blocks) {
        throw std::invalid_argument(std::string(kSubject) + ": kbar_layer_stride (" +
                                    std::to_string(config.kbar_layer_stride) +
                                    ") is below n_blocks (" + std::to_string(config.n_blocks) + ")");
    }
}

cudaError_t kvmem_retrieve_scores(const KvMemRetrieveConfig& config, float* score, const void* q,
                                  const float* kbar_sum, const std::int32_t* block_tokens,
                                  cudaStream_t stream) {
    // Geometry errors throw (programming error); launch failures are returned (runtime condition),
    // because the engine treats a failed launch as fatal while a verifier must be able to tell
    // "rejected" from "ran and produced wrong numbers".
    kvmem_retrieve_validate(config);
    if (config.n_layers == 0 || config.n_query_tokens == 0 || config.n_blocks == 0) {
        return cudaSuccess;
    }
    // `block_tokens` is a DEVICE array: it cannot be validated here without a synchronising copy, so
    // its contract -- one entry per block, holding that block's REAL fill -- is stated in
    // kvmem_retrieve_launch.h and asserted by the standalone verifier instead of guessed at runtime.
    return detail::launch_kvmem_retrieve_scores(config, score, q, kbar_sum, block_tokens,
                                                kvmem_retrieve_scale(config.head_dim),
                                                kvmem_retrieve_mask_bands(config), stream);
}

} // namespace ninfer::ops
