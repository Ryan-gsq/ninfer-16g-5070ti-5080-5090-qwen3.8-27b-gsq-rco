#pragma once

#include "runtime/contract/resources.h"
#include "runtime/prefix_cache/tap_planner.h"

#include "models/qwen3_5/frontend/frontend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

namespace frontend {
struct ToolCallOutputContract;
}

// LOCAL (KVMem query span, D-12, 2026-10-02): publish this request's absolute query span (the last user
// turn, in TOKEN indices of the complete rendered prompt) to the KVMem scorer. Declared HERE and
// defined in the model runtime (program/prefill.cpp) on purpose: the frontend must not grow a CUDA/ops
// include just to reach the scorer, and the publish has to happen in the REQUEST path -- publishing
// only from the prefill driver missed the suffix-only prefill of a cached-prefix turn 2, which then
// silently ran on a STALE span (measured 2026-10-02; see the fallback comment in kvmem_score.h).
// -1/-1 = unknown -> the scorer keeps its tail rule.
void kvmem_publish_query_span(std::int32_t begin, std::int32_t end) noexcept;

// LOCAL (ActKV Sec. 8.1 semantic protection, 2026-10-16): publish this request's two never-compress
// spans -- the leading system/developer instruction block, and the current user task -- in the same
// absolute TOKEN indices the query span uses. Declared here and defined in program/prefill.cpp for
// exactly the reason above: the frontend already knows both frontiers (it computes the query span
// from them) and must not grow a CUDA/ops include to reach the scorer.
//
// The current user task is passed separately from the query span on purpose even though the two are
// the same span today. They answer different questions: the query span is "what to rank against",
// the protected task span is "what must never be evicted", and upstream keeps them separate, so a
// later change to the ranking query cannot silently move the protection.
//
// Published unconditionally on every request, -1 included: a request with no instruction prefix or
// no user turn must CLEAR a stale span rather than inherit the previous request's (the same
// inheritance bug already recorded for the query span).
void kvmem_publish_protect_spans(std::int32_t instruction_begin, std::int32_t instruction_end,
                                 std::int32_t task_begin, std::int32_t task_end,
                                 std::int32_t total_tokens) noexcept;

inline constexpr std::size_t kPreparedVisionPatchFeatures = 3ULL * 2ULL * 16ULL * 16ULL;
inline constexpr std::uint64_t kRawPatchesPerVisionToken  = 4;
// Aggregate prompt capacity and one-item execution capacity are intentionally distinct. Multiple
// media items are retained by one prepared prompt but pass through the Vision tower sequentially.
inline constexpr std::uint64_t kMaximumPromptVisionTokens = 32'768;
inline constexpr std::uint64_t kMaximumPromptVisionRawPatches =
    kMaximumPromptVisionTokens * kRawPatchesPerVisionToken;
inline constexpr std::uint64_t kMaximumVisionItemTokens = 16'384;
inline constexpr std::uint64_t kMaximumVisionItemRawPatches =
    kMaximumVisionItemTokens * kRawPatchesPerVisionToken;

struct PreparedMediaPayload {
    // Exact row-major BF16 input consumed by the Vision patch projection.
    std::unique_ptr<std::uint16_t[]> patches;
    std::size_t patch_elements = 0;

    [[nodiscard]] std::span<std::uint16_t> mutable_span() noexcept {
        return {patches.get(), patch_elements};
    }

    [[nodiscard]] std::span<const std::uint16_t> span() const noexcept {
        return {patches.get(), patch_elements};
    }
};

enum class PromptModality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

enum class RewriteCheckpointKind : std::uint8_t {
    TurnClosure,
    ResponseReplay,
};

struct RewriteCheckpointSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct PromptIdentity {
    bool reusable = true;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    // Exact token frontiers at which this serialization can agree with a typed rewrite captured
    // by an earlier turn. Prefill splits at these frontiers so resumed and root execution use the
    // same GDN decomposition; they are not capture requests by themselves.
    std::vector<std::uint32_t> rewrite_execution_frontiers;
};

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kPreparedSessionKeyCapacity> bytes{};

    [[nodiscard]] std::string_view view() const noexcept { return {bytes.data(), size}; }

    [[nodiscard]] friend bool operator==(const PreparedSessionKey&,
                                         const PreparedSessionKey&) noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind       = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    std::uint32_t frontier           = 0;
    std::uint32_t input_order        = 0;

    [[nodiscard]] friend bool operator==(PreparedCacheOpportunity,
                                         PreparedCacheOpportunity) noexcept = default;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    // Controls replacement of a named SessionIndex entry, not anonymous source ownership.
    bool update_session_index = true;
};

struct PrepareStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0; // Informational; not enforced against any budget.
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

// Prompt boundary facts for hybrid prefix-cache tap placement
// (docs/maintainer/hybrid-prefix-cache-spec.md §7.1). Frontiers are exact token positions.
struct PreparedTapHints {
    std::vector<runtime::prefix_cache::TapHint> hints;
};

struct PreparedPromptData {
    // Proposal-only sources, never part of target tokens, positions or cache identity.
    std::vector<std::vector<TokenId>> ngram_sources;
    std::vector<TokenId> ngram_boundaries;
    std::vector<NgramSourceView> ngram_archive_sources;
    std::shared_ptr<const NgramSnapshot> ngram_snapshot;
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // One immutable payload per Vision item, in the same order as vision_items.
    std::vector<std::shared_ptr<const PreparedMediaPayload>> media_payloads;
    std::vector<VisionItem> vision_items;
    PromptIdentity identity;
    PreparedContextCache context_cache;
    PreparedTapHints tap_hints;
    std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output;
    bool starts_in_reasoning = false;
    PrepareStats prepare;
    // LOCAL (KVMem query span, 2026-10-02, D-12): the retrieval query span in TOKEN indices -- the LAST
    // USER TURN, per the official service-layer rule (donor default: refs kvmem-qw3-main/src/
    // qw3_server.cpp:4779-4804 "use the complete final user message"; our reserve note:
    // 02-交接与状态\KVMem-移植-交接-20260921.md:308). Filled in where the message frontiers are already
    // known in token space (frontend/frontend.cpp) and consumed by the KVMem scorer, which intersects it
    // with each prefill chunk and scores only the chunk that contains it. -1 = unknown -> the scorer
    // keeps its tail rule (_SCORE_QUERY_TAIL), which is also this field's negative control.
    // APPENDED AT THE END ON PURPOSE: a mid-struct insert shifts every later member and any TU still
    // holding the old layout reads the wrong offsets (the stale-obj startup death of 2026-10-02).
    std::int32_t query_span_begin = -1;
    std::int32_t query_span_end   = -1;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }

    // Payload slots remain indexed one-to-one with vision_items for the lifetime of the prompt.
    // Releasing host storage must not destroy that structural identity while a Vision prefill
    // session can still revisit the same item in a later Text chunk.
    void release_all_media_payloads() noexcept {
        for (auto& payload : media_payloads) { payload.reset(); }
    }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::qwen3_5
