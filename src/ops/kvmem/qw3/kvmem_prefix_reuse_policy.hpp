// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : src\kvmem_prefix_reuse_policy.hpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream; only this attribution block was prepended.
// ===========================================================================
#pragma once

#include <cstdint>

namespace qw3::detail {

// Stage P before one complete branch-sensitive tail block.  Agent harnesses
// rewrite a small suffix of the prior live assistant turn when it becomes
// history (for example, the generated empty-think marker may disappear).  The
// exact rewrite is tokenizer/template dependent and has exceeded the old
// four-token allowance.  Keeping one full KVMem block after P ensures that a
// rewrite of at most block_tokens still leaves P no later than the next turn's
// block-aligned query replay boundary.  If that conservative boundary is
// already behind a warm resume point, retain the first aligned boundary that
// is still reachable from the appended suffix.  It may lie in the rewritten
// tail, but exact-LCP admission will reject it if that tail actually changes;
// unlike an unaligned prompt-end fallback, it remains a legal semantic replay
// origin when the tail is unchanged.
constexpr uint32_t kvmem_branch_safe_prompt_checkpoint_split(
        uint32_t prompt_end, uint32_t prefill_base,
        uint32_t block_tokens) {
    constexpr uint32_t kShortTailGuard = 4;
    if (prompt_end <= prefill_base) return 0;
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    uint32_t split = 0;
    if (prompt_end > bt) {
        const uint32_t last_boundary = ((prompt_end - 1) / bt) * bt;
        // The last boundary can itself fall inside the rewritten branch tail.
        // Retreat exactly one complete block to make it branch-safe.
        split = last_boundary >= bt ? last_boundary - bt : 0;
    } else {
        split = prompt_end > kShortTailGuard
            ? prompt_end - kShortTailGuard : 0;
    }
    // Equality is useful on a P->P continuation: the executor is already at
    // this exact safe boundary, so the caller can carry the restored checkpoint
    // forward without re-prefilling or replacing it with an unaligned prompt-end
    // fallback.
    if (split < prefill_base) {
        const uint64_t aligned =
            ((static_cast<uint64_t>(prefill_base) + bt - 1) / bt) * bt;
        return aligned < prompt_end ? static_cast<uint32_t>(aligned) : 0;
    }
    if (split >= prompt_end) return 0;
    return split;
}

// A branch-safe checkpoint can still be unusable for query-conditioned
// continuation when it falls after the aligned replay boundary but before the
// end of Q.  Such a snapshot contains only a prefix of the retrieval query:
// it is too late to replay Q from its boundary and too early to use the
// complete-query snapshot.  Clamp that interval back to the replay boundary.
//
// `prefill_base == replay_boundary` is intentional.  A warm continuation has
// already restored that state, so the caller can capture it again without
// consuming another token and keep it as the next turn's safe P checkpoint.
constexpr uint32_t kvmem_query_safe_prompt_checkpoint_split(
        uint32_t prompt_end, uint32_t prefill_base,
        uint32_t block_tokens, uint32_t query_begin,
        uint32_t query_end) {
    const uint32_t split = kvmem_branch_safe_prompt_checkpoint_split(
        prompt_end, prefill_base, block_tokens);
    if (query_end <= query_begin) return split;
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    const uint32_t replay_boundary = (query_begin / bt) * bt;
    if (split > replay_boundary && split < query_end &&
        replay_boundary >= prefill_base && replay_boundary < prompt_end) {
        return replay_boundary;
    }
    return split;
}

// Prefix reuse above the semantic budget is an incremental source-index
// operation even when the current request has reselection disabled (for
// example, an agent tool-result continuation).  Such a request must therefore
// resume only from a checkpoint whose sparse state and source-index cursor are
// both durable.  Query capture has the same requirement below the budget so a
// prepared checkpoint remains safe when the following turn crosses it.
struct KvmemPrefixReuseSafety {
    bool above_selection_budget = false;
    bool source_index_required = false;
    bool source_index_layout_incompatible = false;
    bool block_end_checkpoint = false;
    bool block_prompt_checkpoint = false;
};

constexpr KvmemPrefixReuseSafety kvmem_prefix_reuse_safety(
        uint32_t logical_prompt_tokens, uint32_t selection_budget,
        bool query_capture_active, bool per_token_source_index,
        bool end_resumable, bool end_source_index_ready,
        bool prompt_resumable, bool prompt_source_index_ready) {
    KvmemPrefixReuseSafety out;
    out.above_selection_budget =
        selection_budget > 0 && logical_prompt_tokens > selection_budget;
    out.source_index_required =
        query_capture_active || out.above_selection_budget;
    // The raw per-token index is laid out using the current request length and
    // cannot be incrementally extended after an above-budget prefix restore.
    out.source_index_layout_incompatible =
        out.above_selection_budget && per_token_source_index;
    out.block_end_checkpoint = out.source_index_layout_incompatible ||
        (out.source_index_required &&
         (!end_resumable || !end_source_index_ready));
    out.block_prompt_checkpoint = out.source_index_layout_incompatible ||
        (out.source_index_required &&
         (!prompt_resumable || !prompt_source_index_ready));
    return out;
}

// A no-query above-budget continuation still incrementally extends the
// position-invariant history index.  Preserve it when a safe warm checkpoint
// was selected; a cold request intentionally rebuilds it from token zero.
constexpr bool kvmem_preserve_warm_history_index(
        bool warm_reuse, bool history_index_capture) {
    return warm_reuse && history_index_capture;
}

// A StateSnapshot is meaningful only for the exact assembled working-set
// generation that produced it.  Token LCP/source-index readiness cannot make a
// pre-selection checkpoint safe after a mid-decode semantic selection changed
// page ordering and RoPE bake positions.
constexpr bool kvmem_checkpoint_selection_compatible(
        uint64_t checkpoint_generation, uint64_t current_generation) {
    return checkpoint_generation == current_generation;
}

// A durable query snapshot is valid only for the same public task span and
// while every prompt token used to derive it remains inside the exact LCP.  A
// normal query snapshot depends only on query_end.  A private guided rewrite
// depends on the whole prompt visible to that rewrite, so source_end is the
// prompt end of the refresh request.
constexpr bool kvmem_warm_query_snapshot_coordinates_match(
        bool stashed, uint32_t stored_begin, uint32_t stored_end,
        uint32_t stored_source_end, uint32_t requested_begin,
        uint32_t requested_end) {
    return stashed && stored_begin == requested_begin &&
        stored_end == requested_end && stored_source_end >= stored_end;
}

constexpr bool kvmem_warm_query_snapshot_reusable(
        bool stashed, uint32_t stored_begin, uint32_t stored_end,
        uint32_t stored_source_end, uint32_t requested_begin,
        uint32_t requested_end, uint32_t common_prefix) {
    return kvmem_warm_query_snapshot_coordinates_match(
               stashed, stored_begin, stored_end, stored_source_end,
               requested_begin, requested_end) &&
        stored_source_end <= common_prefix;
}

constexpr bool kvmem_warm_query_snapshot_capture_ready(
        bool guided_query, bool inherited_snapshot,
        bool scorer_ready, bool original_query_complete) {
    return (guided_query || inherited_snapshot)
        ? scorer_ready : original_query_complete;
}

// Query replay normally proves that every row of the public query span was
// captured in the current pass.  A post-query checkpoint can instead restore a
// complete private guided-query snapshot whose row count intentionally differs
// from that public span.  In that case scorer readiness is the relevant proof;
// retaining the public-span equality check would reject every short rewrite.
constexpr bool kvmem_query_replay_capture_ready(
        bool inherited_snapshot, bool scorer_ready,
        bool public_query_complete) {
    return inherited_snapshot ? scorer_ready : public_query_complete;
}

// The public query span describes which prompt tokens define the retrieval
// intent.  A private guided rewrite is a detached scorer input and can have a
// different row count.  When that snapshot is restored on a later tool turn,
// the per-layer query tensor must be wide enough for both lengths; unused rows
// are capacity only and never contribute to scoring (the scorer consumes the
// separately tracked captured-row count).
constexpr uint32_t kvmem_query_storage_rows(
        uint32_t public_query_rows, uint32_t restored_query_rows) {
    const uint32_t rows = public_query_rows > restored_query_rows
        ? public_query_rows : restored_query_rows;
    return rows == 0 ? 1 : rows;
}

// Semantic query replay restores a block/page boundary snapshot. A post-query
// M checkpoint commonly ends a few chat-template tokens past a block boundary;
// even with a durable Q snapshot it cannot be used as a replay origin. Let the
// caller fall through to the aligned P checkpoint instead.
//
// A prepare-only stable tool continuation is different: it restores the exact
// current-selection M snapshot and only appends the strict-extension suffix.
// It does not rewind/replay the query or truncate pages to an earlier replay
// boundary, so an unaligned M is both necessary and safe when a guided refresh
// ended less than one block before the turn stopped. `semantic_replay` makes
// that distinction explicit instead of treating every query-capture request as
// a replay request.
constexpr bool kvmem_post_query_replay_checkpoint_aligned(
        bool semantic_replay, uint32_t checkpoint,
        uint32_t query_boundary_ceiling, uint32_t block_tokens) {
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    return !semantic_replay || checkpoint <= query_boundary_ceiling ||
        checkpoint % bt == 0;
}

constexpr uint32_t kvmem_warm_query_snapshot_source_end(
        bool guided_query, bool inherited_snapshot,
        uint32_t prompt_tokens, uint32_t inherited_source_end,
        uint32_t query_end) {
    if (guided_query) return prompt_tokens;
    if (inherited_snapshot) return inherited_source_end;
    return query_end;
}

}  // namespace qw3::detail
