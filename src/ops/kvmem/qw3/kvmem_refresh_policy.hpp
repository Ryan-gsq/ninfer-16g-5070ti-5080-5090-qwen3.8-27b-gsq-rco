// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : src\kvmem_refresh_policy.hpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream; only this attribution block was prepended.
// ===========================================================================
#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

namespace qw3::detail {

inline bool kvmem_middecode_refresh_due(
        uint32_t decoded, uint32_t epoch_begin, uint32_t trigger_tokens,
        uint32_t refreshes, uint32_t max_refreshes) {
    return trigger_tokens > 0 && max_refreshes > 0 &&
        refreshes < max_refreshes && decoded >= epoch_begin &&
        decoded - epoch_begin >= trigger_tokens;
}

// A request can enter the initial A+B epoch with only a small amount of
// headroom left. That remaining distance applies only until the first refresh.
// Every later epoch starts at the refresh boundary and therefore uses the
// normal engine-wide trigger. A zero steady trigger keeps direct callers and
// older request construction semantics unchanged.
inline constexpr uint32_t kvmem_middecode_trigger_for_epoch(
        uint32_t first_trigger_tokens, uint32_t steady_trigger_tokens,
        uint32_t completed_refreshes) {
    return completed_refreshes > 0 && steady_trigger_tokens > 0
        ? steady_trigger_tokens
        : first_trigger_tokens;
}

// Local-position MTP only needs enough RoPE headroom for one generation
// epoch. A guided mid-decode refresh compacts/rebases the selected prefix
// before the next epoch, so using the request-wide output cap here falsely
// rejects otherwise-safe multi-epoch requests. Without a complete refresh
// policy, retain the conservative request-wide horizon.
inline constexpr uint64_t kvmem_mtp_compact_output_horizon(
        uint64_t request_max_tokens, uint32_t first_epoch_limit_tokens,
        uint32_t steady_epoch_limit_tokens,
        uint32_t middecode_trigger_tokens, uint32_t max_refreshes) {
    if (request_max_tokens == 0 || middecode_trigger_tokens == 0 ||
        max_refreshes == 0 || first_epoch_limit_tokens == 0 ||
        steady_epoch_limit_tokens == 0) {
        return request_max_tokens;
    }
    const uint64_t largest_epoch = std::max<uint64_t>(
        first_epoch_limit_tokens, steady_epoch_limit_tokens);
    return std::min(request_max_tokens, largest_epoch);
}

// A generated retrieval query is publishable only after the model emits a real
// sequence boundary (EOS, ChatML end, or the private-thinking close token).
// Punctuation and newlines are content: they commonly occur inside file names,
// symbols, code expressions, lists, and error traces. Merely consuming the
// configured token cap is not a boundary either.
inline bool kvmem_guided_query_complete(
        uint32_t query_tokens, uint32_t max_tokens,
        bool natural_termination) {
    return query_tokens > 0 && max_tokens > 0 && natural_termination;
}

struct KvmemCrossTurnRefreshDecision {
    bool reset = false;
    bool due = false;
    uint64_t delta_tokens = 0;
};

struct KvmemHarnessTurnGate {
    bool reselect = false;
    bool cross_turn_refresh = false;
};

enum class KvmemHarnessRefreshReason : uint8_t {
    None = 0,
    ExplicitForce,
    ColdAboveBudget,
    NewUserQuery,
    InitialHeadroom,
    CrossTurnGrowth,
};

inline const char *kvmem_harness_refresh_reason_name(
        KvmemHarnessRefreshReason reason) {
    switch (reason) {
    case KvmemHarnessRefreshReason::None: return "none";
    case KvmemHarnessRefreshReason::ExplicitForce: return "explicit-force";
    case KvmemHarnessRefreshReason::ColdAboveBudget: return "cold-above-budget";
    case KvmemHarnessRefreshReason::NewUserQuery: return "new-user-query";
    case KvmemHarnessRefreshReason::InitialHeadroom: return "initial-headroom";
    case KvmemHarnessRefreshReason::CrossTurnGrowth: return "cross-turn-growth";
    }
    return "unknown";
}

struct KvmemHarnessRefreshInput {
    bool known_conversation = false;
    bool same_user_query = false;
    bool selection_started = false;
    bool incoming_reselect = false;
    bool force_reselect = false;
    bool reset = false;
    uint64_t prompt_tokens = 0;
    uint64_t selection_budget = 0;
    uint64_t trigger_tokens = 0;
    KvmemCrossTurnRefreshDecision cross_turn;
};

struct KvmemHarnessRefreshDecision {
    bool reselect = false;
    bool private_query = false;
    bool headroom_grace = false;
    bool initial_headroom_grace = false;
    uint64_t middecode_tokens_until_refresh = 0;
    KvmemHarnessRefreshReason reason = KvmemHarnessRefreshReason::None;
};

inline uint64_t kvmem_saturating_add_u64(uint64_t lhs, uint64_t rhs) {
    const uint64_t limit = UINT64_MAX - lhs;
    return rhs > limit ? UINT64_MAX : lhs + rhs;
}

// Exact physical-page admission for one prefill chunk. Capacity must be
// checked against the chunk that is ABOUT to run, not the previous full-size
// chunk: the final prompt tail is often only a few tokens. Treating that tail
// as another 2K chunk caused a false keep-selected overflow at
// 194816/196608 tokens even though the remaining 31-token append needed one
// page.
inline constexpr uint32_t kvmem_prefill_chunk_pages(
        uint32_t chunk_tokens, uint32_t page_tokens) {
    return chunk_tokens == 0 || page_tokens == 0
        ? 0
        : (chunk_tokens + page_tokens - 1) / page_tokens;
}

inline constexpr bool kvmem_prefill_chunk_fits(
        uint32_t free_pages, uint32_t chunk_tokens,
        uint32_t page_tokens) {
    return free_pages >=
        kvmem_prefill_chunk_pages(chunk_tokens, page_tokens);
}

// A trajectory first observed below the selection budget owns one dense
// physical epoch.  Crossing A alone does not discard history: the first
// semantic refresh is due at A+T, where T is strictly below the reserved
// generation headroom B.  The remaining distance is also the request-local
// mid-decode trigger when a tool turn starts inside that grace interval.
inline uint64_t kvmem_initial_headroom_refresh_target(
        uint64_t selection_budget, uint64_t trigger_tokens) {
    return kvmem_saturating_add_u64(selection_budget, trigger_tokens);
}

// A request-boundary private refresh normally generates its query while the
// old A+B epoch is still resident. A single large tool callback can jump past
// the remaining B headroom in one HTTP request; in that case the only bounded
// execution is pressure ingest/index first, followed by private-query
// selection from the complete durable source.
inline bool kvmem_private_refresh_requires_pressure(
        bool private_refresh, bool selection_started,
        uint64_t cross_turn_delta_tokens, uint64_t prompt_tokens,
        uint64_t selection_budget, uint64_t generation_budget) {
    if (!private_refresh || generation_budget == 0) return false;
    const uint64_t epoch_tokens_used = selection_started
        ? cross_turn_delta_tokens
        : (prompt_tokens > selection_budget
               ? prompt_tokens - selection_budget : 0);
    return epoch_tokens_used >= generation_budget;
}

// Prefix-cache capture is a storage/checkpoint concern, not a selection
// trigger. During every A+B epoch the server marks stable continuation requests
// as keep-selected; a block-aligned P checkpoint must preserve that physical
// epoch instead of silently replacing it with an A-sized sink+recent pressure
// window. Pressure normalization is allowed only on a request that is already
// performing one of the explicit semantic refresh transitions.
inline constexpr bool kvmem_checkpoint_pressure_selection_allowed(
        bool keep_selected_prefill) {
    return !keep_selected_prefill;
}

// A private retrieval query is part of the same A+B epoch that authorized the
// refresh. Generate it against that still-live epoch first, then perform the
// one semantic selection with its captured Q rows. Letting prefix-checkpoint or
// prefill pressure collapse the window beforehand both makes the query stale-
// selected and turns one logical refresh into two physical reductions.
inline constexpr bool kvmem_refresh_prefill_keeps_epoch(
        bool headroom_grace, bool private_guided_refresh) {
    return headroom_grace || private_guided_refresh;
}

inline KvmemHarnessRefreshDecision kvmem_harness_refresh_decision(
        const KvmemHarnessRefreshInput &in,
        bool guided_refresh_enabled = true) {
    KvmemHarnessRefreshDecision out;
    if (in.force_reselect) {
        out.reselect = true;
        out.reason = KvmemHarnessRefreshReason::ExplicitForce;
        return out;
    }
    if (in.selection_budget == 0 ||
        in.prompt_tokens <= in.selection_budget) {
        return out;
    }
    if (!in.known_conversation || in.reset) {
        // A server attached to an already-long trajectory has no proof that a
        // dense A+B epoch is still resident. Preserve explicit ingest/off, but
        // make the normal auto request establish a safe semantic epoch.
        if (in.incoming_reselect) {
            out.reselect = true;
            out.reason = KvmemHarnessRefreshReason::ColdAboveBudget;
        }
        return out;
    }
    if (!in.same_user_query) {
        out.reselect = true;
        out.reason = KvmemHarnessRefreshReason::NewUserQuery;
        return out;
    }
    // With private query generation disabled, growth in a stable tool task
    // is capacity pressure, not a new retrieval query. The executor retains
    // mandatory blocks and the recent tail through its ordinary pressure
    // path; never re-score the stale user query merely because A or B filled.
    if (!guided_refresh_enabled) return out;
    if (!in.selection_started) {
        const uint64_t target = kvmem_initial_headroom_refresh_target(
            in.selection_budget, in.trigger_tokens);
        if (in.trigger_tokens == 0 || in.prompt_tokens >= target) {
            out.reselect = true;
            out.private_query = in.trigger_tokens != 0;
            out.reason = KvmemHarnessRefreshReason::InitialHeadroom;
            return out;
        }
        out.headroom_grace = true;
        out.initial_headroom_grace = true;
        out.middecode_tokens_until_refresh = target - in.prompt_tokens;
        return out;
    }
    if (in.cross_turn.due) {
        out.reselect = true;
        out.private_query = true;
        out.reason = KvmemHarnessRefreshReason::CrossTurnGrowth;
        return out;
    }
    if (in.trigger_tokens > 0 &&
        in.cross_turn.delta_tokens < in.trigger_tokens) {
        // A semantic refresh starts a fresh physical A+B epoch. Stable tool
        // callbacks retain that exact A-token selection and may consume the B
        // reserve until the next private-query threshold; they must not create
        // an unreported pressure-selection transition at a prefix checkpoint.
        out.headroom_grace = true;
        out.middecode_tokens_until_refresh =
            in.trigger_tokens - in.cross_turn.delta_tokens;
    }
    // Stable tool callbacks deliberately suppress an incoming `auto`.  Force
    // remains available above and a real new user query is handled separately.
    return out;
}

inline KvmemHarnessTurnGate kvmem_harness_turn_gate(
        bool known_task, bool incoming_reselect, bool force_reselect,
        const KvmemCrossTurnRefreshDecision &cross_turn) {
    if (force_reselect) return {true, false};
    if (!known_task) return {incoming_reselect, false};
    if (cross_turn.due) return {true, true};
    // `auto` means a semantic decision, not "reselect every tool callback".
    // A stable task below threshold keeps the existing window/query.
    return {false, false};
}

inline bool kvmem_prompt_trajectory_reset(
        uint64_t prompt_tokens, uint64_t last_seen_prompt_tokens) {
    if (last_seen_prompt_tokens == 0 ||
        prompt_tokens >= last_seen_prompt_tokens) {
        return false;
    }
    // Chat templates may rewrite/drop a few tail tokens when the previous
    // assistant turn becomes history. Treat only a material shrink as a new or
    // compacted trajectory; otherwise a harmless template delta would abandon
    // the A+B grace epoch and force an early selection just above A.
    const uint64_t tolerance = std::max<uint64_t>(
        1024, last_seen_prompt_tokens / 100);
    return last_seen_prompt_tokens - prompt_tokens > tolerance;
}

// Agent harnesses normally end one model request at every tool call. Therefore
// the configured T-token mid-decode threshold must also span those short HTTP
// requests. The
// server keeps this tiny policy state per stable real-user task; tool results
// and assistant output both count because both consume the long-trajectory KV
// budget and can materially change what should be retrieved.
inline KvmemCrossTurnRefreshDecision kvmem_cross_turn_refresh_decision(
        uint64_t prompt_tokens, uint64_t last_seen_prompt_tokens,
        uint64_t last_refresh_prompt_tokens, uint64_t trigger_tokens) {
    KvmemCrossTurnRefreshDecision out;
    out.reset = kvmem_prompt_trajectory_reset(
        prompt_tokens, last_seen_prompt_tokens);
    if (out.reset || last_refresh_prompt_tokens == 0 ||
        prompt_tokens < last_refresh_prompt_tokens) {
        return out;
    }
    out.delta_tokens = prompt_tokens - last_refresh_prompt_tokens;
    out.due = trigger_tokens > 0 && out.delta_tokens >= trigger_tokens;
    return out;
}

inline bool kvmem_has_unclosed_marker(std::string_view text,
                                      std::string_view open,
                                      std::string_view close) {
    const size_t open_pos = text.rfind(open);
    if (open_pos == std::string_view::npos) return false;
    const size_t close_pos = text.rfind(close);
    return close_pos == std::string_view::npos || close_pos < open_pos;
}

inline bool kvmem_ends_with_partial_marker(std::string_view text,
                                           std::string_view marker) {
    const size_t max_prefix = std::min(text.size(), marker.size() - 1);
    for (size_t n = 1; n <= max_prefix; ++n) {
        if (text.substr(text.size() - n) == marker.substr(0, n)) return true;
    }
    return false;
}

// A context switch is safe only between complete structured fragments. Qwen's
// tool protocol uses XML wrappers; markdown fences are included because many
// harnesses parse fenced JSON/code incrementally before deciding whether the
// assistant has finished the turn.
inline bool kvmem_middecode_refresh_text_safe(std::string_view generated) {
    if (generated.empty()) return false;
    for (std::string_view marker : {std::string_view("<tool_call>"),
                                    std::string_view("</tool_call>"),
                                    std::string_view("<function="),
                                    std::string_view("</function>"),
                                    std::string_view("```")}) {
        if (kvmem_ends_with_partial_marker(generated, marker)) return false;
    }
    if (kvmem_has_unclosed_marker(generated, "<tool_call>",
                                  "</tool_call>")) return false;
    if (kvmem_has_unclosed_marker(generated, "<function=", "</function>")) {
        return false;
    }
    size_t fences = 0;
    size_t cursor = 0;
    while ((cursor = generated.find("```", cursor)) !=
           std::string_view::npos) {
        ++fences;
        cursor += 3;
    }
    if (fences % 2 != 0) return false;
    if ((generated.size() >= 12 &&
         generated.substr(generated.size() - 12) == "</tool_call>") ||
        (generated.size() >= 3 &&
         generated.substr(generated.size() - 3) == "```")) {
        return true;
    }
    while (!generated.empty() &&
           (generated.back() == ' ' || generated.back() == '\t' ||
            generated.back() == '\r')) {
        generated.remove_suffix(1);
    }
    if (generated.empty()) return false;
    if (generated.back() == '\n') return true;
    const char last = generated.back();
    if (last == '.' || last == '!' || last == '?' || last == ';' ||
        last == ':') return true;
    return generated.size() >= 3 &&
        (generated.substr(generated.size() - 3) == "。" ||
         generated.substr(generated.size() - 3) == "！" ||
         generated.substr(generated.size() - 3) == "？");
}

inline uint32_t kvmem_middecode_recent_tokens(uint32_t select_budget) {
    return std::min<uint32_t>(16384,
        std::max<uint32_t>(4096, select_budget / 8));
}

// A structured tool/code fragment can legitimately remain open for thousands
// of tokens.  Near the end of the generation reserve we must refresh even
// without a syntactic boundary, but only after pinning the complete epoch so
// the opening marker and all emitted arguments remain visible after selection.
inline bool kvmem_middecode_emergency_refresh_due(
        uint32_t epoch_tokens, uint32_t epoch_limit,
        uint32_t guard_tokens = 512) {
    if (epoch_limit == 0) return false;
    return epoch_tokens >= epoch_limit ||
        epoch_limit - epoch_tokens <= guard_tokens;
}

// A normal refresh fires while the A+B epoch still has enough space for the
// private prompt and its generated retrieval query.  An emergency refresh is
// different: it exists precisely because an open tool/code fragment consumed
// that headroom.  Appending a private branch there can exhaust the bounded GPU
// page pool before selection has a chance to free pages.  Reuse the already
// captured original task query for this rare safety path instead.
inline constexpr bool kvmem_middecode_generate_private_query(
        bool emergency_refresh) {
    return !emergency_refresh;
}

inline uint32_t kvmem_middecode_refresh_recent_tokens(
        uint32_t select_budget, uint32_t epoch_tokens,
        bool emergency_refresh) {
    const uint32_t normal = kvmem_middecode_recent_tokens(select_budget);
    return emergency_refresh ? std::max(normal, epoch_tokens) : normal;
}

inline uint32_t kvmem_middecode_pin_from_block(
        uint32_t committed_position, uint32_t select_budget,
        uint32_t block_tokens) {
    const uint32_t bt = std::max<uint32_t>(1, block_tokens);
    const uint32_t keep = kvmem_middecode_recent_tokens(select_budget);
    const uint32_t begin = committed_position > keep
        ? committed_position - keep : 0;
    return begin / bt;
}

// Return the earliest block of a contiguous live tail that fits after the
// already mandatory (sink/system/query/policy) blocks are de-duplicated. The
// desired tail is an upper bound: a nearly-full mandatory set receives only
// the actual remaining budget instead of failing during re-selection.
inline uint32_t kvmem_bounded_tail_pin_from_block(
        uint32_t block_count, uint32_t budget_blocks,
        uint32_t desired_tail_blocks,
        const std::vector<uint32_t> &base_mandatory_blocks) {
    if (block_count == 0 || budget_blocks == 0) return 0xffffffffu;
    std::vector<uint8_t> mandatory(block_count, 0);
    uint32_t mandatory_count = 0;
    for (uint32_t id : base_mandatory_blocks) {
        if (id < block_count && !mandatory[id]) {
            mandatory[id] = 1;
            ++mandatory_count;
        }
    }
    if (mandatory_count > budget_blocks) return 0xffffffffu;
    uint32_t begin = block_count -
        std::min(desired_tail_blocks, block_count);
    uint32_t union_count = mandatory_count;
    for (uint32_t id = begin; id < block_count; ++id) {
        union_count += mandatory[id] ? 0u : 1u;
    }
    while (begin < block_count && union_count > budget_blocks) {
        union_count -= mandatory[begin] ? 0u : 1u;
        ++begin;
    }
    return begin == block_count ? 0xffffffffu : begin;
}

}  // namespace qw3::detail
