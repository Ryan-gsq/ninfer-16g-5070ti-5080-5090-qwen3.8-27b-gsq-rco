#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/rope.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream,
                          InputBasis basis = InputBasis::Primal);

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, Tensor& key, cudaStream_t stream);

// Normalize q and k and rotate them. Where the fused Op covers the geometry this is one graph node
// instead of three; everywhere else it is the three calls it replaces, which are the same
// arithmetic bit for bit. It chooses a schedule, not a result.
void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, const ops::RopeYarn& yarn, cudaStream_t stream);

// LOCAL (KVMem content scoring, 2026-10-02): tell the norm/rope stage which full-attention layer and
// which chunk width the NEXT text_qk_norm_rope call belongs to, and hand it the harvest round for the
// chunk in progress. Inside that call, between the RMS-norm of K and the in-place RoPE, the pre-RoPE
// key and query are copied into the harvest (the only place in the engine where those exist as
// materialised tensors), which is what the mean-K index is built from.
//
// `harvest` is an opaque pointer on purpose: this header is included by several translation units and
// must not drag the KVMem headers into them. Null = feature off, and then every added line is a no-op,
// so the OFF arm stays bit-identical (raw_k_shadow_harvest_for() returns null when the switch is off).
void text_set_kvmem_hook(std::int32_t layer_index, std::int32_t chunk_tokens,
                         void* harvest) noexcept;

} // namespace ninfer::models::qwen3_5::execution
