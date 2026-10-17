#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

// --- LOCAL PROTOTYPE: retrieval-mask scaffolding -----------------------------
// A device bitmap over 64-token KV pages (kPagedKVPageSize), driven by
// NINFER_PROTO_SELECT, so the block-level retrieval mask can be exercised on both
// the prefill (prompt) and decode (small-T) routes without changing any public op
// signature:
//   unset / "all"  -> nullptr   (dense: must be bit-identical to the original)
//   "even"         -> drop every other page
//   "first_half"   -> drop the upper half of the addressable pages
//   "off_1_in_4"   -> drop every 4th page
// Allocated once and cached. Throwaway scaffolding: the real port replaces this
// with a per-request selection produced by the retrieval policy.
inline const std::uint32_t* proto_selected_blocks() {
    static const std::uint32_t* bits = []() -> const std::uint32_t* {
        const char* mode = std::getenv("NINFER_PROTO_SELECT");
        if (mode == nullptr || std::strcmp(mode, "all") == 0) { return nullptr; }
        constexpr int kPages = 8192;  // covers 262144 tokens / 64
        constexpr int kWords = kPages / 32;
        std::uint32_t host[kWords];
        for (int i = 0; i < kWords; ++i) { host[i] = 0xffffffffU; }
        const auto clear_page = [&](int p) { host[p >> 5] &= ~(1U << (p & 31)); };
        if (std::strcmp(mode, "even") == 0) {
            for (int p = 1; p < kPages; p += 2) { clear_page(p); }
        } else if (std::strcmp(mode, "first_half") == 0) {
            for (int p = kPages / 2; p < kPages; ++p) { clear_page(p); }
        } else if (std::strcmp(mode, "off_1_in_4") == 0) {
            for (int p = 3; p < kPages; p += 4) { clear_page(p); }
        } else {
            return nullptr;
        }
        std::uint32_t* device_bits = nullptr;
        if (cudaMalloc(&device_bits, sizeof(host)) != cudaSuccess) { return nullptr; }
        if (cudaMemcpy(device_bits, host, sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(device_bits);
            return nullptr;
        }
        return device_bits;
    }();
    return bits;
}

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

// The widest prompt-route row block (the eight-warp fast INT8 kernel); every prompt kernel's row
// block divides it.
inline constexpr std::int32_t kPromptWaveRows = 128;

// Whether a prompt call on this storage takes the fast kernel (register-resident rows, FP16 PV per
// 64-key tile): when the caller asks for it (--fast-prefill-kernel), else as NINFER_PROMPT_FAST=0|1
// or the device profile's "attn_prompt_fast" says. Storages outside the INT8 family never do.
[[nodiscard]] bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested);

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
    // --- LOCAL PROTOTYPE: block-level retrieval mask ---------------------------
    // Optional little-endian bitmap over 64-token KV pages (kPagedKVPageSize).
    // Bit p of word (p >> 5) set means page p is attended. nullptr means every
    // page is selected, which must reproduce the original dense behaviour
    // bit-for-bit. Used by the KVMem-style host-offload prototype; not part of
    // the public op contract yet.
    const std::uint32_t* selected_blocks = nullptr;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

// A non-null gate is applied by the reduce epilogue at the store. Only the shared BF16/INT8
// reducer accepts one; the caller keeps the standalone multiply for every other storage.
void causal_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& table_rows, float scale,
                                     PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t column_begin, std::int32_t width,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     Tensor& out, cudaStream_t stream, const void* gate = nullptr);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream);

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream);

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, bool fast,
                                    cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out, bool fast,
                                              cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream);

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
