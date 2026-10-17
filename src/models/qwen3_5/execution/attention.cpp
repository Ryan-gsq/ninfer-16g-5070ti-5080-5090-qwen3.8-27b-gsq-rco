#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/rotation.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"

// LOCAL (KVMem content scoring, 2026-10-02): the pre-RoPE harvest and the query-side scoring glue.
// These are the ported KVMem component's host-safe halves; kvmem_score.h is header-only (inline), so
// nothing new has to be added to the build.
#include "ops/kvmem/kvmem_score.h"
#include "ops/kvmem/raw_k_harvest.h"
#include "ops/kvmem/raw_k_shadow.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

// The fused text form is registered for the two text head geometries with a one-dimensional
// position axis. The MRoPE path and any other geometry take the three calls it replaces.
//
// It is also bounded in width. One warp owns one head, so the fused kernel stops gaining once a
// width alone fills the machine, and past that the three separate kernels - each free to choose
// its own shape - are ahead: measured on an RTX 5090, the fused form wins by 22 to 52 % through
// 256 tokens and loses by up to 22 % at 1024. The bound sits a doubling below the crossover
// because the two geometries cross at different widths. The Op itself is valid at any width; this
// is a dispatch choice, and both branches are the same arithmetic bit for bit.
constexpr std::int32_t kFusedTextQkNormRopeMaximumTokens = 256;

bool fused_text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                             const AttentionConfig& attention, std::int32_t tokens) {
    return positions.ne[1] == 1 && tokens <= kFusedTextQkNormRopeMaximumTokens &&
           attention.head_dim == 256 && rope.rotary_dim == 64 &&
           ((attention.num_attention_heads == 16 && attention.num_key_value_heads == 2) ||
            (attention.num_attention_heads == 24 && attention.num_key_value_heads == 4));
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    const Tensor& signs = projection_signs(parameters.projection);
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return rotated_workspace_bytes(
            signs, gguf->parts.front().weight.k, last,
            ops::attn_input_proj_workspace_capacity_bytes(*gguf, first, last));
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return rotated_workspace_bytes(
            signs, weight.k, last,
            ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                          single->policy, first, last));
    }
    const auto& pair = std::get<ops::PairedProjectionWeights>(parameters.projection);
    return rotated_workspace_bytes(signs, pair.first.k, last,
                                   ops::attn_input_proj_split_workspace_capacity_bytes(
                                       pair.first.qtype, pair.first.n, pair.second.qtype,
                                       pair.second.n, pair.first.k, pair.policy, first, last));
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(x, *gguf, query, gate, key, value, workspace, stream);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(x, pair->first, pair->second, query, gate, key, value, pair->policy,
                             workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(x, single.weight, query, gate, key, value, single.policy, workspace,
                             stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, Tensor& key, cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn, query, key, stream);
}

// ---- LOCAL (KVMem content scoring, 2026-10-02) ------------------------------------------------
// The harvest hook. Mirrors the donor engine's per-layer capture (fusion:
// targets/qwen3_6/impl/runtime/text_context_impl.h:950-963 for the pre-RoPE key and :971-973 for the
// query) but carries the round state as file-static state in this translation unit, so the header
// stays free of KVMem includes and no existing signature changes.
namespace {

struct KvmemHook {
    void*         harvest = nullptr;   // ops::RawKShadowHarvest*, opaque here on purpose
    std::int32_t  layer   = -1;        // full-attention layer index for the call in progress
    std::int32_t  tokens  = 0;         // chunk width (== q_layer_stride == round tokens)
    bool          shape_noted = false; // the "this pass is not the armed round" line prints once
};

KvmemHook& kvmem_hook() {
    static KvmemHook hook;
    return hook;
}

// True while a round is armed for harvesting: the KVMem index wants this round's pre-RoPE keys, and
// only the three separate calls can hand them over (rmsnorm leaves the normalized BF16 tensor alive
// for the hook; the fused Op rotates in a single pass and materialises no pre-RoPE tensor,
// rmsnorm_rope.h:22-23 and :56-62). `tokens` is the width the round was armed with, so a pass of a
// different shape still reports armed and is then skipped by kvmem_harvest_pre_rope's shape guard.
bool kvmem_harvest_armed() {
    const KvmemHook& hook = kvmem_hook();
    return hook.harvest != nullptr && hook.tokens > 0 && hook.layer >= 0;
}

void kvmem_harvest_pre_rope(const Tensor& query, const Tensor& key, cudaStream_t stream) {
    KvmemHook& hook = kvmem_hook();
    if (hook.harvest == nullptr || hook.tokens <= 0 || hook.layer < 0) { return; }
    // SHAPE GUARD. The staging view is sized by the width the hook was armed with, and the component
    // THROWS when the source does not match it (raw_k_shadow.h: geometry mismatch is an exception by
    // design, so a wrong-shaped harvest can never be silently mis-laid-out). That exception is fatal in
    // this engine: on 2026-10-02 a verify-phase call inherited the prefill round's armed width and
    // killed the worker with `shadow [1024, 4] does not match the raw key [256, 4, 1, 1]`, surfacing as
    // HTTP 500 on turn 2. A pass that does not have this round's shape is simply not this round's
    // business; say so once instead of throwing.
    const std::int32_t key_tokens =
        static_cast<std::int32_t>(key.ne[2]) * (key.ne[3] > 0 ? static_cast<std::int32_t>(key.ne[3]) : 1);
    if (key_tokens != hook.tokens) {
        if (!hook.shape_noted) {
            hook.shape_noted = true;
            std::fprintf(stderr,
                         "[ninfer] kvmem harvest: skipped a %d-token key (round width is %d, layer %d); "
                         "this pass is not the prefill round the harvest was armed for\n",
                         key_tokens, hook.tokens, hook.layer);
        }
        return;
    }
    auto* harvest = static_cast<ops::RawKShadowHarvest*>(hook.harvest);
    // Same source object the donor feeds: `kn` is the RMS-normalised key, a real BF16 tensor in the
    // arena, still pre-RoPE at this point. Geometry {head_dim, kv_heads, tokens} is what
    // raw_k_shadow.h's contract accepts and what the index's layout expects.
    Tensor shadow = harvest->staging_for_round(hook.layer);
    ops::raw_k_shadow_copy(key, shadow, stream);
    // LOCAL FIX (B08): stamp this layer as harvested IN THIS ROUND. The harvest's own contract says
    // every slot that was NOT written must be marked, because it still holds the previous round's bytes
    // (raw_k_harvest.h:172-177) -- but nothing in the tree ever called mark_layer_harvested, so every
    // slot read as "not harvested": the raw-K arena gate (raw_k_block_arena.h:462) could never pass
    // (that arm stored nothing), and the index feed had no way to tell a fresh slot from a stale one.
    // With the stamp in place `layers_harvested() == layers()` becomes B08's positive criterion -- the
    // round covered EVERY full-attention layer, layer 0 included -- and the index can refuse a stale
    // slot instead of ranking blocks by another round's keys.
    harvest->mark_layer_harvested(hook.layer);
    // The query half: score what this chunk is ASKING with, in the same content frame. `q_layer_stride`
    // inside the scorer is a token count, and the live plane's row count is the chunk width, so both
    // arguments are `tokens`. A span longer than MAXQ used to be skipped there; since 2026-10-04 the
    // scorer segments it and sums the segments (B05 -- see the LOCAL FIX comment in kvmem_score.h), so
    // the delivered chunk width of 1024 no longer requires NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL to be
    // set for it.
    ops::detail::kvmem_score_accumulate(hook.layer, query.data, hook.tokens, hook.tokens, stream);
}

} // namespace

void text_set_kvmem_hook(std::int32_t layer_index, std::int32_t chunk_tokens,
                         void* harvest) noexcept {
    KvmemHook& hook = kvmem_hook();
    hook.harvest = harvest;
    hook.layer   = layer_index;
    hook.tokens  = chunk_tokens;
    if (harvest == nullptr) { return; }
    // The round's WIDTH has to be recorded before the first copy: staging_for_round() hands out a view
    // sized by it, and a short final chunk must not publish the tail of an earlier, wider round
    // (raw_k_harvest.h:70-81). The width is only known here -- the round is opened before the chunk,
    // where the caller knows the chunk capacity, not this chunk's real token count.
    static_cast<ops::RawKShadowHarvest*>(harvest)->set_round_width(chunk_tokens);
}

void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, const ops::RopeYarn& yarn, cudaStream_t stream) {
    require_rope_axes(positions, rope);
    // The fused Op rotates unscaled positions at the unscaled frequencies, so YaRN or position
    // interpolation keeps the three separate calls -- and so does an ARMED KVMem harvest: the fused
    // Op materialises no pre-RoPE tensor (rmsnorm_rope.h:22-23), which is why rounds that took it
    // used to be missing from the index entirely (`fused rmsnorm+rope branch cannot expose the
    // pre-RoPE key (layer 0, N tokens); this chunk is NOT in the index`). The two forms are
    // documented bit-identical (rmsnorm_rope.h:56-62), so routing an armed round through the three
    // calls changes no numerics; it only gives up the fused Op's own 22-52% edge at <= 256 tokens,
    // and only for the rounds the index is armed for.
    if (!yarn.active() && !kvmem_harvest_armed() &&
        fused_text_qk_norm_rope(positions, rope, attention, query.ne[2])) {
        ops::rmsnorm_rope(positions, q_norm_weight, k_norm_weight, query, key, normalized_query,
                          normalized_key, stream);
        return;
    }
    ops::rmsnorm(query, q_norm_weight, rms_norm_eps, true, normalized_query, stream);
    ops::rmsnorm(key, k_norm_weight, rms_norm_eps, true, normalized_key, stream);
    kvmem_harvest_pre_rope(normalized_query, normalized_key, stream);
    ops::rope(positions, dimension(rope.rotary_dim), rope.rope_theta, yarn, normalized_query,
              normalized_key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
