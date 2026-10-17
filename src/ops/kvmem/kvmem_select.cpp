#include "ops/kvmem/kvmem_select.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>

// See kvmem_select.h for the policy this implements and for why it is host-only.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "kvmem_select";

// Descending score, and on ties the HIGHER block id first -- i.e. the NEWER block wins.
//
// The tie DIRECTION is not a taste decision: it is the reference's (`kvmem_store.cpp:403-408`). It
// matters because equal scores are a REACHABLE state, not a corner case -- the reference degrades
// Retrieval -> H2O -> Recency when the query embedding is flat or absent (`kvmem_store.hpp:140-143`),
// and the first re-selection after a prefill has no query at all. With the direction backwards we
// would keep the OLDEST candidates and evict the NEWEST, which is the opposite of what the recency
// fallback exists for.
//
// The order is TOTAL either way: block ids are unique, so no two candidates compare equal and the
// selected SET cannot depend on the order std::nth_element happened to shuffle equal elements into.
// (An earlier revision of this comment claimed only an id-ASCENDING rule is deterministic. That
// argument was wrong -- id-descending is just as total. Determinism comes from totality, not from its
// direction.)
bool score_before(std::int32_t a, std::int32_t b, const float* scores, std::int32_t first_block) {
    const float sa = scores[a - first_block];
    const float sb = scores[b - first_block];
    if (sa > sb) { return true; }
    if (sa < sb) { return false; }
    return a > b;
}

// Guard: the scorer writes "no signal" as NaN or infinity rather than a very negative number, so a
// comparison that let those through would let a block with no score outrank every real candidate.
// Treating them as the lowest possible score is the only choice that cannot starve the budget.
bool finite_score(const float* scores, std::int32_t block, std::int32_t first_block) {
    return std::isfinite(scores[block - first_block]);
}

} // namespace

KvMemSelectResult kvmem_select_blocks(const KvMemSelectConfig& config, const float* scores,
                                      std::int32_t first_block,
                                      const KvMemActKvInputs& actkv) {
    if (config.total_blocks < 0 || config.budget_blocks < 0 || config.sink_blocks < 0 ||
        config.recent_blocks < 0 || first_block < 0) {
        throw std::invalid_argument(std::string(kSubject) + ": negative count");
    }
    if (config.total_blocks > 0 && first_block >= config.total_blocks) {
        throw std::invalid_argument(std::string(kSubject) + ": first_block is past the history");
    }
    if (config.budget_blocks > 0 && scores == nullptr) {
        throw std::invalid_argument(std::string(kSubject) + ": scores is null with a nonzero budget");
    }

    KvMemSelectResult out;
    if (config.total_blocks == 0) { return out; }
    // How many entries the caller's score array actually describes, in BLOCK ids. The selector's
    // contract is "one score per block from first_block onward", so an array shorter than that is a
    // caller bug -- but the reaction is the same one the finite-score rule already takes: the missing
    // tail is treated as "no signal" rather than read out of bounds. Both the LRFU ranking and the
    // semantic rescue read this bound, so it is computed once.
    const std::size_t scores_size = scores != nullptr
        ? static_cast<std::size_t>(config.total_blocks - first_block)
        : 0;
    if (config.budget_blocks == 0) {
        // A zero budget means UNLIMITED, not "keep nothing" (reference: `kvmem_store.cpp:362` treats
        // `budget == 0 || n <= budget` as "select everything", ignoring both the scores and the sink /
        // recent regions). That is reachable in practice, not hypothetical: the caller derives the budget
        // as `select_budget / block_tokens` with INTEGER division (`kvmem_store.hpp:378-380`), so any
        // select_budget below one block truncates to 0. Returning nothing there would silently empty the
        // window on a configuration the reference answers with the entire history.
        const std::int32_t total = config.total_blocks;
        out.kept.resize(static_cast<std::size_t>(total));
        std::iota(out.kept.begin(), out.kept.end(), 0);
        // The counters still add up to the emitted list, so the "list length == counters sum"
        // invariant holds on EVERY path. Note the third counter is named `scored_kept` while in this
        // branch nothing was scored: everything fits, so the remainder is filled without consulting
        // `scores` at all.
        out.sink_kept   = std::min(config.sink_blocks, total);
        out.recent_kept = std::min(config.recent_blocks, total - out.sink_kept);
        out.scored_kept = total - out.sink_kept - out.recent_kept;
        out.candidates  = out.scored_kept;
        return out;
    }

    // ---- 1. Always-kept regions, clamped so they can never overlap and never exceed the history.
    //
    // Sink is clamped first and recent takes what is left: a configuration whose regions together
    // exceed the history must still produce a valid, duplicate-free set, and giving the prefix
    // priority is what preserves the stable-prefix anchor the sink exists to be.
    std::int32_t sink = std::min(config.sink_blocks, config.total_blocks);
    sink              = std::min(sink, config.budget_blocks);
    std::int32_t recent = std::min(config.recent_blocks, config.total_blocks - sink);
    recent              = std::min(recent, config.budget_blocks - sink);

    std::vector<std::int32_t> kept;
    kept.reserve(static_cast<std::size_t>(config.budget_blocks));
    for (std::int32_t b = 0; b < sink; ++b) { kept.push_back(b); }
    out.sink_kept = sink;
    for (std::int32_t i = 0; i < recent; ++i) {
        kept.push_back(config.total_blocks - 1 - i);
    }
    out.recent_kept = recent;

    const std::int32_t middle_begin = sink;
    const std::int32_t middle_end   = config.total_blocks - recent;  // exclusive
    const std::int32_t remaining    = config.budget_blocks - sink - recent;
    if (remaining <= 0 || middle_end <= middle_begin) {
        // ---- 1b. Nothing scored fits: the always-kept regions ARE the answer.
        //
        // THIS BRANCH MUST HAND `kept` TO `out`. It used to `return out;` with `out.kept` never
        // assigned -- only the scored path at the bottom assigned it -- so whenever sink + recent
        // consumed the whole budget, or the middle region was empty, the function returned an EMPTY
        // window while `out.sink_kept` / `out.recent_kept` still reported those blocks as kept. The
        // caller cannot see the contradiction: an empty window is not "nothing to do", it is a
        // silently wrong answer, and the counters actively argue the blocks are there.
        //
        // Caught by cross-checking against the reference implementation's own test suite (3675 cases,
        // `E:\infer-build\exp\kvmem-select-xcheck\`): the reference returns [0..budget-1] for
        // `total_blocks=40, budget_blocks=8, sink=8, recent=0`, we returned []. Our own oracle could not
        // see it -- it was a Python mirror of this very logic AND asserted only structural invariants
        // (<= budget, ordered, unique, in range), every one of which an empty list satisfies happily.
        //
        // The sort+unique mirror the scored path below, so the emitted list stays ascending and
        // duplicate-free when the two always-kept regions overlap (sink + recent > total_blocks).
        std::sort(kept.begin(), kept.end());
        kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
        out.kept = std::move(kept);
        return out;
    }

    // ---- 2. Middle blocks by score.
    //
    // Only blocks at or after first_block have a score. Anything earlier cannot be scored by this
    // caller and is left to the always-kept regions rather than being selected on a stale or absent
    // score.
    //
    // ActKV Sec. 5.2: with `action_lrfu` on, the RANKING KEY stops being this round's retrieval
    // score and becomes the accumulated keep score. Everything else about the selection -- the
    // always-kept regions, the budget, the ascending output, the counters -- is untouched, so the
    // two policies differ in exactly one number per candidate.
    const actkv::LrfuState* lrfu = actkv.lrfu;
    // With no committed history the accumulator path must NOT be taken: every row is zero, so
    // ranking on it would keep the OLDEST blocks instead of this round's best. Cold start falls
    // back to the caller's score, which is also the whole feature's negative control.
    const bool lrfu_active = lrfu != nullptr && lrfu->rounds() > 0;
    if (!lrfu_active && scores == nullptr) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": scores is null outside the everything-fits path");
    }
    const auto rank_key = [&](std::int32_t block) -> double {
        const std::size_t index = static_cast<std::size_t>(block);
        if (index >= scores_size) { return -std::numeric_limits<double>::infinity(); }
        const double this_round = static_cast<double>(scores[block - first_block]);
        if (!std::isfinite(this_round)) { return -std::numeric_limits<double>::infinity(); }
        if (!lrfu_active) { return this_round; }
        return lrfu->rank(block, this_round, scores_size);
    };
    const std::int32_t cand_begin = std::max(middle_begin, first_block);
    const std::int32_t cand_end   = middle_end;
    if (cand_end > cand_begin) {
        std::vector<std::int32_t> cand(static_cast<std::size_t>(cand_end - cand_begin));
        std::iota(cand.begin(), cand.end(), cand_begin);
        out.candidates = static_cast<std::int32_t>(cand.size());

        const auto before = [&](std::int32_t a, std::int32_t b) {
            if (!lrfu_active) { return score_before(a, b, scores, first_block); }
            const double ka = rank_key(a);
            const double kb = rank_key(b);
            if (ka != kb) { return ka > kb; }
            return a > b;
        };
        // Full stable sort of the candidate ids, with unscored (NaN/inf) blocks demoted to the end so
        // they can never displace a scored block. The sort is TOTAL -- descending score with ascending
        // id on ties -- so the selected set is independent of element order, which is what makes the
        // plan reproducible across runs and across captured-graph replays.
        //
        // The candidate set is the middle region minus the sink and recent regions, so it is bounded
        // by the history length, and unlike the reference's std::nth_element this is O(n log n) on the
        // middle. That cost is accepted deliberately at this stage: the selection runs ONCE per view
        // update (reference section 5 update modes are `step` or `interval`, not per token), and a
        // deterministic total order is worth more here than a partial one, because a plan that is not
        // bit-reproducible cannot be compared between the two arms of an A/B run.
        const auto take = std::min<std::size_t>(static_cast<std::size_t>(remaining), cand.size());
        std::stable_sort(cand.begin(), cand.end(), [&](std::int32_t a, std::int32_t b) {
            // NOTE: the finite/not-finite demotion stays keyed on the RAW retrieval score even when
            // LRFU is ranking. "No signal this round" is a property of the round's evidence, not of
            // the accumulator; an accumulated block whose query score went missing must still be
            // demoted, and a zero-rounds LRFU state must demote exactly what the old path demoted.
            const bool fa = finite_score(scores, a, first_block);
            const bool fb = finite_score(scores, b, first_block);
            if (fa != fb) { return fa; }
            return before(a, b);
        });
        for (std::size_t i = 0; i < take; ++i) {
            kept.push_back(cand[i]);
            ++out.scored_kept;
        }
    }

    // ---- 2b. On-demand semantic protection (ActKV Sec. 8.1) -----------------------------------
    //
    // The protected pages were deliberately NOT charged as mandatory up front. Charging them first
    // let one user turn longer than the budget pin the whole window (measured upstream: mandatory ==
    // selected == every page, no room left to score), so the protection is a RESCUE for the pages
    // the retrieval would drop -- and only up to a bounded share of the budget.
    //
    // ★ THE RESCUE DISPLACES, IT DOES NOT APPEND.
    //   Upstream does not append the rescued pages on top of the selection: it folds them into the
    //   MANDATORY set and runs the selector AGAIN over a narrowed budget, so a rescued page takes a
    //   slot away from the weakest scored page. Appending here would either exceed the budget (which
    //   every downstream check rejects) or, when the selection already fills the budget, silently do
    //   nothing at all -- and "fills the budget" is the NORMAL case, so an append-style rescue would
    //   be dead code that still looks implemented.
    //
    //   The victims are the selected blocks this round RANKED LOWEST, restricted to the scored
    //   region: sink and recent are unconditional by contract and are never displaced. Only
    //   candidates already in `kept` are eligible, so a reserved slot cannot be spent on a block the
    //   retrieval never chose.
    if (actkv.protected_pages != nullptr && !actkv.protected_pages->empty()) {
        std::vector<std::int32_t> sorted_kept = kept;
        std::sort(sorted_kept.begin(), sorted_kept.end());
        sorted_kept.erase(std::unique(sorted_kept.begin(), sorted_kept.end()), sorted_kept.end());
        // The selector speaks block ids as int32 (its emitted list is int32); the protection speaks
        // page ids as uint32. They are the same numbers here (a page IS a block id in this port).
        std::vector<std::uint32_t> kept_u32;
        kept_u32.reserve(sorted_kept.size());
        for (const std::int32_t id : sorted_kept) {
            if (id >= 0) { kept_u32.push_back(static_cast<std::uint32_t>(id)); }
        }
        const std::size_t cap = actkv::protect_cap(config.budget_blocks);
        const std::vector<std::uint32_t> rescued = actkv::protect_rescue(
            kept_u32, *actkv.protected_pages, scores, scores_size, cap);
        // Which of the rescue's requests the selection has ROOM for. The cap is the request limit;
        // the room is what the caller's own window can still absorb. Taking the min keeps the
        // emitted list inside the budget the caller asked for.
        const std::size_t room = kept.size() < static_cast<std::size_t>(config.budget_blocks)
                                     ? static_cast<std::size_t>(config.budget_blocks) - kept.size()
                                     : 0;
        // Make room by evicting the weakest SCORED blocks. `mandatory` is exactly the set the rescue
        // may not touch: sink + recent (both already charged and unconditional by contract).
        std::size_t to_evict = rescued.size() > room ? rescued.size() - room : 0;
        if (to_evict > 0) {
            std::vector<std::int32_t> evictable;
            evictable.reserve(kept.size());
            for (const std::int32_t id : sorted_kept) {
                if (id >= 0 && id < sink) { continue; }                         // sink: never
                if (recent > 0 && id >= config.total_blocks - recent) { continue; }  // recent: never
                evictable.push_back(id);
            }
            // Weakest first. `rank_key` is the same key the selection ranked on, so "weakest" means
            // the same thing here as it did one paragraph above -- with LRFU on, a block that has
            // accumulated heat is NOT evicted before a block that has not.
            std::sort(evictable.begin(), evictable.end(), [&](std::int32_t a, std::int32_t b) {
                const double ka = rank_key(a);
                const double kb = rank_key(b);
                if (ka != kb) { return ka < kb; }  // ascending: weakest first
                return a < b;
            });
            if (to_evict > evictable.size()) { to_evict = evictable.size(); }
            for (std::size_t i = 0; i < to_evict; ++i) {
                const std::int32_t victim = evictable[i];
                const auto found = std::find(kept.begin(), kept.end(), victim);
                if (found == kept.end()) { continue; }
                kept.erase(found);
                if (out.scored_kept > 0) { --out.scored_kept; }
            }
        }
        std::size_t admitted = 0;
        for (const std::uint32_t page : rescued) {
            if (kept.size() >= static_cast<std::size_t>(config.budget_blocks)) { break; }
            if (std::find(kept.begin(), kept.end(), static_cast<std::int32_t>(page)) != kept.end()) {
                continue;
            }
            kept.push_back(static_cast<std::int32_t>(page));
            // A rescued page is neither sink nor recent, so it is charged to the scored region --
            // the same counter the caller uses to account for "which region spent the budget".
            ++out.scored_kept;
            ++admitted;
        }
        if (actkv.trace) {
            std::fprintf(stderr,
                         "kvmem_select: PROTECTED cap=%zu protected=%zu dropped_by_retrieval=%zu "
                         "rescued=%zu evicted=%zu budget=%d\n",
                         cap, actkv.protected_pages->size(), rescued.size(), admitted, to_evict,
                         config.budget_blocks);
        }
    }

    // ---- 3. Ascending block id: the window's inside-window positions are chronological.
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    out.kept = std::move(kept);
    return out;
}

} // namespace ninfer::ops
