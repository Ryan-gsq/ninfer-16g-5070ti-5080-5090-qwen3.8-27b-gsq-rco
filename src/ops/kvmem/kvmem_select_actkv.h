#pragma once

// ActKV (arXiv:2609.31395) retention policy -- host-only, pure logic.
//
// This header is the FIELD-LEVEL PORT of the two mechanisms that survived the ActKV v1.3
// measurement round (`melis2023/ninfer-kvmem-actkv`, 2026-10-15):
//
//   * Sec. 5.2  action-oriented LRFU retention  (`--kvmem-action-lrfu`)
//   * Sec. 8.1  semantic protection             (`--kvmem-protect-semantics`)
//
// The three mechanisms v1.3 IMPLEMENTED AND THEN DELETED are deliberately NOT ported, and the
// reasons are recorded here so a later reader does not re-add them:
//
//   - Sec. 5.1 action-region probe: the probe never armed (this chat template's tool_calls do
//     not fall into the marked span), so the option was removed instead of left as a no-op.
//   - Alg. 3 step 8 GQA max-over-query-heads: regressed the 40-needle / 256k recall from 39/40
//     to 28/40 and decayed progressively; a per-head max is dominated by whichever head happens
//     to peak, and that noise compounds through the LRFU accumulator.
//   - Sec. 6 confidence-driven adaptive budget: traded ~15% of round wall for a collapse from
//     40/40 to 17/40. The pool is sized from the budget at startup, so it could not save memory.
//
// WHY A NEW FILE RATHER THAN AN EDIT TO qw3/kvmem_store.hpp
//   That file is a byte-identical upstream snapshot (`PORTED from kvmem-qw3`, 2026-09-19) and its
//   attribution block promises "Changes: NONE". Editing it would break that promise and make the
//   next upstream re-pull a three-way mess. This header keeps the port in NINFER's own layer while
//   implementing exactly the same algorithm, so the two can be diffed against each other.
//
// DEFAULT-OFF AND BIT-IDENTICAL WHEN OFF
//   The LRFU state is only ever consulted when `action_lrfu` is true. With it off, the ranking
//   input is the caller's one-shot score array -- byte-for-byte what it was before this file
//   existed. Every entry point takes the state as a pointer and a null pointer means "off".

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::ops::actkv {

// ---------------------------------------------------------------------------------------------
// 1. LRFU keep score (ActKV Sec. 5.2)
// ---------------------------------------------------------------------------------------------

// Configuration, mirroring the upstream names and defaults so one command line means the same
// thing on both backends: lambda 0.50, top-p 90%.
struct LrfuConfig {
    // Exponential decay applied to the accumulated score before this round's evidence is added.
    // `lambda = 0.50` means a half-life of ONE round: the most recent hits dominate, but a block
    // hit repeatedly over several rounds outranks one hit once. Range (0, 1).
    double lambda = 0.50;
    // Per-round hit fraction. Only the top `top_frac` of this round's POSITIVELY scored blocks
    // are allowed to grow the accumulator. The truncation exists to stop noise compounding: with
    // every candidate accumulating, low-level attention chatter accumulates over rounds and
    // eventually outranks the real hits, which is exactly the "frequency-dominated eviction
    // retains obsolete entries" regime the paper warns about.
    double top_frac = 0.90;
    // Cross-round normalization of the one-shot retrieval score.
    //
    // 0 = off: the raw cosine scale, i.e. the historical behaviour, byte-identical.
    // > 0: the per-block retrieval scores become a probability mass over the POSITIVELY scored
    //      blocks, and this value is the softmax temperature -- the exchange rate between the
    //      accumulated keep score and the current round's evidence.
    //
    // Why it exists: `S = lambda*S_prev + hit*score` folds an UNBOUNDED accumulator with a score
    // in [-1, 1], so after enough rounds the memory term dominates and the current query stops
    // moving the ranking. Softmax is order-preserving, so a single round's selection is
    // unchanged; what changes is how much that one round's evidence is worth inside `S`.
    double retrieval_temperature = 0.0;
    // Two-phase commit. This backend scores FIRST and selects afterwards (`set_retrieval_scores`
    // then `pick_topk_blocks`), so accumulating at scoring time would grow a block on a score it
    // never won a slot for. With this true the round's evidence is held and only applied by
    // `commit(kept)`, i.e. after the selection exists. Upstream sets this true unconditionally
    // for this backend shape; we keep it explicit so the one-phase form stays reachable.
    bool deferred = true;
};

// One row of the diagnostic trace, used by the `*_TRACE` output and by tests.
struct LrfuRow {
    std::int32_t block = -1;
    double score = 0.0;
    double accum = 0.0;
};

// The accumulator itself. One instance per selection lane; the caller owns the lifetime.
//
// The class holds no device state, no allocator and no clock, so it is unit-testable without a
// GPU -- the same discipline `kvmem_select.h` claims for the selector.
class LrfuState {
public:
    LrfuState() = default;
    explicit LrfuState(LrfuConfig config) : config_(config) {}

    [[nodiscard]] const LrfuConfig& config() const noexcept { return config_; }
    void set_config(const LrfuConfig& config) noexcept { config_ = config; }

    // Rounds whose evidence has been applied. `0` means "no history yet" -- and that is
    // load-bearing, not cosmetic: with no history the ranking MUST fall back to this round's own
    // score, otherwise a cold start would rank every block by an all-zero accumulator.
    [[nodiscard]] std::uint64_t rounds() const noexcept { return rounds_; }

    [[nodiscard]] std::size_t size() const noexcept { return accum_.size(); }

    // Grow the table so block ids in [0, n) are addressable. Existing rows keep their values; new
    // rows start at zero. Called with the history size the caller is about to rank.
    void resize(std::size_t n) {
        if (n > accum_.size()) { accum_.resize(n, 0.0); }
    }

    void clear() noexcept {
        accum_.clear();
        rounds_ = 0;
    }

    // ---- Phase 1: hand over this round's evidence ------------------------------------------
    //
    // `scores[i]` is the score of block i. Non-finite entries (NaN / +/-inf) mean "no signal" for
    // that block and are treated as zero evidence -- they must never grow the accumulator, and a
    // NaN silently compared into the ranking would be worse than useless.
    //
    // `retrieval_temperature > 0` turns the scores into a probability mass first (see the config
    // comment). The transform is order-preserving, so it cannot change THIS round's ranking.
    //
    // With `deferred` the evidence is only held; `commit(kept)` applies it. Without it the
    // evidence is applied here and `commit` is a no-op, i.e. the one-phase form.
    void begin_round(const float* scores, std::size_t count) {
        pending_.clear();
        pending_valid_ = false;
        pending_kept_.clear();
        kept_filter_ = false;
        // A round that is announced but not committed must not leave the PREVIOUS round's
        // evidence lying around to be committed by a later call.
        if (scores == nullptr || count == 0) { return; }

        resize(count);
        pending_.resize(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            const float raw = scores[i];
            pending_[i] = std::isfinite(raw) ? static_cast<double>(raw) : 0.0;
        }
        if (config_.retrieval_temperature > 0.0) {
            normalize_temperature(pending_, config_.retrieval_temperature);
        }
        // The hit mask is computed HERE, on the raw evidence, so the truncation threshold is a
        // property of the retrieval scores themselves and not of the decayed accumulator.
        //
        // "Hit" means strictly positive: a block the retrieval did not rank above zero is not
        // evidence of USE, and letting negatives through would reward the blocks least related to
        // the query. Candidates are the scored prefix; blocks past `count` are out of play.
        hit_.assign(pending_.size(), 0);
        std::vector<std::int32_t> positives;
        positives.reserve(pending_.size());
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i] > 0.0) { positives.push_back(static_cast<std::int32_t>(i)); }
        }
        if (!positives.empty()) {
            const double frac = std::min(std::max(config_.top_frac, 0.0), 1.0);
            std::size_t keep = static_cast<std::size_t>(
                std::ceil(frac * static_cast<double>(positives.size())));
            if (keep < 1) { keep = 1; }  // a non-empty positive set always admits its best block
            if (keep > positives.size()) { keep = positives.size(); }
            if (keep < positives.size()) {
                // Descending by evidence. `nth_element` is enough for a MEMBERSHIP decision, and
                // the membership it produces is total: `pending_` values are finite and the block
                // id is a unique tie-break, so no two candidates compare equal.
                std::nth_element(positives.begin(), positives.begin() + static_cast<std::ptrdiff_t>(keep),
                                 positives.end(), [this](std::int32_t a, std::int32_t b) {
                                     const double sa = pending_[static_cast<std::size_t>(a)];
                                     const double sb = pending_[static_cast<std::size_t>(b)];
                                     if (sa != sb) { return sa > sb; }
                                     return a < b;
                                 });
                positives.resize(keep);
            }
            for (const std::int32_t id : positives) { hit_[static_cast<std::size_t>(id)] = 1; }
        }

        if (!config_.deferred) {
            // One-phase form: no selection is known yet, so the restriction cannot be applied and
            // the evidence lands on every block. That is the documented difference between the two
            // forms -- and the reason upstream always sets `deferred` for this backend shape.
            kept_filter_ = false;
            apply();
            pending_.clear();
            hit_.clear();
        } else {
            pending_valid_ = true;
        }
    }

    // ---- Phase 2: apply the held evidence to the blocks this round actually kept ------------
    //
    // This is the whole reason the two-phase form exists. `kept` is the selection the round
    // produced; only those block ids may grow. A block the retrieval scored but the budget dropped
    // must NOT grow -- otherwise it accumulates heat on a slot it never won, and after enough
    // rounds the accumulator ranks it above blocks that are actually in the window.
    //
    // A no-op when nothing is held, so a round that never scored (or a caller with the feature
    // off) cannot leak one round's evidence into the next.
    void commit(const std::vector<std::int32_t>& kept) {
        if (!config_.deferred || !pending_valid_) { return; }
        pending_kept_ = kept;
        std::sort(pending_kept_.begin(), pending_kept_.end());
        pending_kept_.erase(std::unique(pending_kept_.begin(), pending_kept_.end()),
                            pending_kept_.end());
        kept_filter_ = true;  // committed with an explicit selection, empty or not
        apply();
        pending_.clear();
        hit_.clear();
        pending_kept_.clear();
        pending_valid_ = false;
    }

    // Convenience overload for callers holding the 32-bit id type.
    void commit(const std::vector<std::uint32_t>& kept) {
        if (!config_.deferred || !pending_valid_) { return; }
        pending_kept_.clear();
        pending_kept_.reserve(kept.size());
        for (const std::uint32_t id : kept) { pending_kept_.push_back(static_cast<std::int32_t>(id)); }
        std::sort(pending_kept_.begin(), pending_kept_.end());
        pending_kept_.erase(std::unique(pending_kept_.begin(), pending_kept_.end()),
                            pending_kept_.end());
        kept_filter_ = true;  // committed with an explicit selection, empty or not
        apply();
        pending_.clear();
        hit_.clear();
        pending_kept_.clear();
        pending_valid_ = false;
    }

    // ---- Read side -------------------------------------------------------------------------

    // The ranking key for block `id`. Falls back to `this_round` when there is no history yet --
    // the cold-start rule stated above -- and never returns a non-finite value.
    [[nodiscard]] double rank(std::int32_t id, double this_round, std::size_t history) const noexcept {
        if (id < 0 || static_cast<std::size_t>(id) >= accum_.size() ||
            static_cast<std::size_t>(id) >= history) {
            return this_round;
        }
        if (rounds_ == 0) { return this_round; }
        return accum_[static_cast<std::size_t>(id)];
    }

    // Full rows for diagnostics, ascending by block id.
    [[nodiscard]] std::vector<LrfuRow> rows(std::size_t history) const {
        std::vector<LrfuRow> out;
        const std::size_t n = std::min(history, accum_.size());
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(LrfuRow{static_cast<std::int32_t>(i), 0.0, accum_[i]});
        }
        return out;
    }

    // Compact one-line summary for the `*_TRACE` diagnostics: how many rows carry heat, and what
    // the accumulator's range looks like. Reported, never obeyed.
    [[nodiscard]] std::string describe(std::size_t history) const {
        const std::size_t n = std::min(history, accum_.size());
        double lo = 0.0;
        double hi = 0.0;
        std::size_t nonzero = 0;
        bool first = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (accum_[i] != 0.0) {
                ++nonzero;
                if (first) { lo = hi = accum_[i]; first = false; }
                lo = std::min(lo, accum_[i]);
                hi = std::max(hi, accum_[i]);
            }
        }
        std::string out = "lrfu rounds=" + std::to_string(rounds_) +
                          " rows=" + std::to_string(n) +
                          " nonzero=" + std::to_string(nonzero) +
                          " lambda=" + std::to_string(config_.lambda) +
                          " top_frac=" + std::to_string(config_.top_frac);
        if (!first) {
            out += " min=" + std::to_string(lo) + " max=" + std::to_string(hi);
        }
        return out;
    }

private:
    // The single place the update rule lives:  S_i <- lambda * S_i + hit_i * score_i,
    // restricted to the ids this round kept. Called once per committed round.
    void apply() {
        const double lambda = config_.lambda;
        for (std::size_t i = 0; i < accum_.size(); ++i) {
            // Decay is applied to EVERY row, kept or not: a block absent from the window this
            // round must fade, which is the half of the rule the "kept" restriction is paired
            // with. Skipping the decay for unkept rows would make the accumulator monotone and
            // reproduce the exact "obsolete entries outlive their usefulness" failure the paper
            // warns about.
            double value = lambda * accum_[i];
            // NOTE: an EMPTY `pending_kept_` means "this round kept nothing", NOT "keep
            // everything". The distinction is carried by `kept_filter_` rather than inferred from
            // emptiness: reading an empty vector as a wildcard would let a selection that was
            // legitimately empty grow every scored block, which is the exact opposite of the
            // two-phase guarantee.
            const bool kept_here =
                !kept_filter_ ||
                std::binary_search(pending_kept_.begin(), pending_kept_.end(),
                                   static_cast<std::int32_t>(i));
            if (kept_here && i < pending_.size() && i < hit_.size() && hit_[i] != 0) {
                value += pending_[i];
            }
            accum_[i] = value;
        }
        ++rounds_;
    }

    // Turn a score row into a probability mass over its positive entries. Order-preserving, so
    // the round's own ranking is untouched; the point is the SCALE the accumulator sees.
    static void normalize_temperature(std::vector<double>& values, double temperature) {
        double max_positive = 0.0;
        bool any = false;
        for (const double value : values) {
            if (value > 0.0) {
                if (!any || value > max_positive) { max_positive = value; }
                any = true;
            }
        }
        if (!any || max_positive <= 0.0) { return; }
        double total = 0.0;
        for (double& value : values) {
            value = value > 0.0 ? std::exp((value - max_positive) / temperature) : 0.0;
            total += value;
        }
        if (total <= 0.0) { return; }
        for (double& value : values) { value /= total; }
    }

    LrfuConfig config_{};
    std::vector<double> accum_;         // per-block keep score, indexed by block id
    std::vector<double> pending_;       // this round's evidence (post-temperature)
    std::vector<std::uint8_t> hit_;     // this round's top-frac hit mask
    std::vector<std::int32_t> pending_kept_;  // the round's selection, ascending
    // Whether `pending_kept_` is a FILTER. Distinct from "is it empty" on purpose: an empty
    // selection is a real outcome (the round kept nothing) and must not be read as a wildcard.
    bool kept_filter_ = false;
    bool pending_valid_ = false;
    std::uint64_t rounds_ = 0;
};

// ---------------------------------------------------------------------------------------------
// 2. Semantic protection (ActKV Sec. 8.1)
// ---------------------------------------------------------------------------------------------

// A half-open TOKEN span in the prompt that must survive compression: the leading
// system/developer instruction block, and the current user task. Both are recorded in ABSOLUTE
// token indices, so they compose with the ring's page arithmetic without a second coordinate
// system (`kvmem_score.h` already carries `abs_query_begin/end` in this same space).
struct ProtectSpan {
    std::int32_t begin = -1;
    std::int32_t end = -1;
    [[nodiscard]] bool valid() const noexcept { return begin >= 0 && end > begin; }
};

// The pages the protection wants kept, given the two spans above and the page size.
//
// `total_tokens` is the prompt's token count. A span that covers the WHOLE prompt is not a
// semantic anchor -- pinning it would leave the window no freedom to compress at all -- so it is
// dropped, exactly as upstream's `add_range` drops it. The result is ascending and duplicate-free.
//
// `page_tokens` is passed in rather than typed (house rule: never hardcode kPagedKVPageSize).
[[nodiscard]] inline std::vector<std::uint32_t> semantic_pages(
    ProtectSpan instruction, ProtectSpan task, std::int32_t page_tokens,
    std::int32_t total_tokens) {
    std::vector<std::uint32_t> out;
    if (page_tokens <= 0 || total_tokens <= 0) { return out; }
    const auto add_range = [&](std::int32_t begin, std::int32_t end) {
        if (end <= begin || begin < 0 || begin >= total_tokens) { return; }
        if (end > total_tokens) { end = total_tokens; }
        if (end - begin >= total_tokens) { return; }  // the whole prompt is not an anchor
        const std::int32_t last = (end - 1) / page_tokens;
        for (std::int32_t page = begin / page_tokens; page <= last; ++page) {
            out.push_back(static_cast<std::uint32_t>(page));
        }
    };
    add_range(instruction.begin, instruction.end);
    add_range(task.begin, task.end);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// Security/correctness cap on the protection, from the 262k NIAH round.
//
// Protection is a RESCUE, never an up-front reservation: a user turn longer than the window
// budget puts all of its pages in the protected set, and charging those first filled the window
// with mandatory pages and cost ~9% of decode throughput. Bounding the rescue to a quarter of
// the budget measured the same recall (33/33) at no cost over disabling protection entirely.
inline constexpr double kProtectRescueFraction = 0.25;

// How many protected pages one round may rescue: a quarter of the whole-page budget, at least one
// when the budget is non-empty. Kept as a named function so the bound appears in one place.
[[nodiscard]] inline std::size_t protect_cap(std::int32_t budget_pages) {
    if (budget_pages <= 0) { return 0; }
    const double scaled = static_cast<double>(budget_pages) * kProtectRescueFraction;
    const std::size_t cap = static_cast<std::size_t>(scaled);
    return cap < 1 ? 1 : cap;
}

// The rescue itself: given the selection the retrieval produced and the protected pages, return
// the protected pages the retrieval DROPPED, best-score first, truncated to `cap`.
//
// A protected page the retrieval already keeps costs nothing -- which is the common case for the
// instruction prefix -- instead of pre-consuming a budget slot and pushing a higher-scoring page
// out. That property is what makes this "on demand" rather than a reservation.
//
// `scores` is indexed by page id and may be null (then the rescue is ranked by descending page id,
// i.e. newest first, matching the recency fallback the caller already uses when no query exists).
[[nodiscard]] inline std::vector<std::uint32_t> protect_rescue(
    const std::vector<std::uint32_t>& selected_ascending,
    const std::vector<std::uint32_t>& protected_pages, const float* scores, std::size_t score_count,
    std::size_t cap) {
    std::vector<std::uint32_t> missing;
    if (cap == 0 || protected_pages.empty()) { return missing; }
    for (const std::uint32_t page : protected_pages) {
        if (!std::binary_search(selected_ascending.begin(), selected_ascending.end(), page)) {
            missing.push_back(page);
        }
    }
    const auto score_of = [&](std::uint32_t page) {
        if (scores == nullptr || static_cast<std::size_t>(page) >= score_count) { return 0.0F; }
        const float value = scores[page];
        return std::isfinite(value) ? value : -std::numeric_limits<float>::infinity();
    };
    const auto better = [&](std::uint32_t a, std::uint32_t b) {
        const float sa = score_of(a);
        const float sb = score_of(b);
        if (sa != sb) { return sa > sb; }
        return a > b;  // newer wins on ties, the same direction the selector uses
    };
    if (missing.size() > cap) {
        std::partial_sort(missing.begin(), missing.begin() + static_cast<std::ptrdiff_t>(cap),
                          missing.end(), better);
        missing.resize(cap);
    } else {
        std::sort(missing.begin(), missing.end(), better);
    }
    std::sort(missing.begin(), missing.end());
    return missing;
}

}  // namespace ninfer::ops::actkv
