#pragma once

// KVMem selection policy: which blocks survive into the assembled window.
//
// This is the piece of the policy layer that is PURE HOST LOGIC -- it takes one retrieval score per
// candidate block and returns a block-id list. It has no device state, no switch and no arena, so it
// can be reasoned about and tested without a GPU. That separation is deliberate: it is the one part
// of the policy layer that can be pinned down before the index and the scorer exist.
//
// WHAT IT IMPLEMENTS (reference: kvmem-qw3 docs/kvmem_implementation_notes.md:1191-1229, section 5)
//
//   1. ALWAYS-KEPT REGIONS (section 5.1). A sink region (a prefix, normally the first block) and a
//      recent region (a suffix near the tail) are preserved unconditionally. A zero count disables
//      that region: recent_blocks = 0 means no unconditional suffix retention, and tail blocks can
//      still be selected on merit by the scorer. The SHIPPING reference configuration uses
//      sink = 8 blocks and recent = 0 (gpu_memory_optimization.md:578-600), so "recent = 0" is not a
//      degenerate test case -- it is the configuration we expect to run.
//
//   2. REMAINING BUDGET BY SCORE (section 5.2). The remaining slots are filled from the middle blocks
//      (those outside both always-kept regions) in descending score order. Ties are broken by block id
//      DESCENDING -- the NEWER block wins, matching the reference (`kvmem_store.cpp:403-408`). The
//      direction matters because equal or missing scores are a REACHABLE state, not a corner case: the
//      reference degrades Retrieval -> H2O -> Recency when the query embedding is flat or absent
//      (`kvmem_store.hpp:140-143`), and the first re-selection after a prefill has no query at all.
//      The order is TOTAL (block ids are unique), so the plan is reproducible bit-for-bit across runs
//      and across a captured-graph replay -- and that totality, NOT its direction, is what buys the
//      determinism. An earlier revision of this header argued that only an id-ASCENDING rule could be
//      deterministic; that argument was wrong (id-descending is just as total).
//
//   3. OUTPUT ORDER (section 5.2). The returned list is emitted in ASCENDING BLOCK ID order, never in
//      score order, because the view's inside-window positions are assigned in chronological order.
//      Returning score order would silently scramble the window's meaning while still looking
//      "full", which is the kind of failure this header exists to make impossible.
//
// WHY WE DO A FULL SORT (a deliberate departure from the reference)
//   The reference uses std::nth_element because only membership of the top-K matters and a partial
//   selection is O(n). We stable-sort the whole candidate set instead, for a reason that is about the
//   JUDGE rather than the runtime: nth_element leaves the order of equal elements unspecified, so a
//   plan could differ between runs and the same binary could not be trusted to reproduce its own plan
//   across the two arms of an A/B run. The sort also makes the tie rule above explicit instead of
//   incidental. The cost is accepted because selection runs ONCE per view update (the update modes are
//   `step` or `interval`, not per token). The SELECTED ids are sorted again before they are returned,
//   which is what the output order below requires.
//
// WHY THE SCORE ARRAY IS INDEXED BY BLOCK ID
//   `scores` is a view over per-block scores in which index i describes block `first_block + i`.
//   Indexing by position-in-array instead of by block id is the mistake this signature is shaped to
//   prevent: the index is not re-based when the history grows, so a caller holding a score array that
//   starts part-way through the sequence must say so rather than silently shifting every score.

#include "ops/kvmem/kvmem_select_actkv.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::ops {

// Policy inputs. Counts are in BLOCKS, never tokens: the design record (section 3.6 R1) notes that
// expressing a budget in blocks is only safe when the block size is a parameter, because the widely
// quoted "top-1024 = 32K" identity holds at block size 32 and becomes a 2x error at block size 64.
struct KvMemSelectConfig {
    std::int32_t budget_blocks = 0;  // total blocks the window may hold
    std::int32_t sink_blocks   = 0;  // unconditional prefix
    std::int32_t recent_blocks = 0;  // unconditional suffix; 0 disables it
    std::int32_t total_blocks  = 0;  // blocks in the history (the last live block is total_blocks-1)
};

// Selection outcome. `kept` is ascending by block id. The counters exist so the caller can report
// which region spent the budget, and so a test can assert the branch it thinks it exercised.
struct KvMemSelectResult {
    std::vector<std::int32_t> kept;
    std::int32_t               sink_kept    = 0;
    std::int32_t               recent_kept  = 0;
    std::int32_t               scored_kept  = 0;
    std::int32_t               candidates   = 0;  // middle blocks considered
};

// Optional ActKV policy inputs. A NULL `actkv` on every entry point below means "feature off" and
// the ranking input is exactly the caller's `scores` array -- byte-identical to the behaviour
// before this struct existed. That is the negative control, and it is the default.
//
// When `actkv` is non-null:
//   * `lrfu` (optional, may be null) replaces the ranking key with the accumulated keep score.
//     The evidence for the round must already have been handed to the state via
//     `LrfuState::begin_round`; the caller commits the result with `LrfuState::commit(kept)` after
//     this call returns. Doing it in that order is not a style choice -- see the two-phase comment
//     in kvmem_select_actkv.h.
//   * `protected_pages` are rescued on demand: only the ones this selection DROPPED are added
//     back, best score first, at most `protect_cap(budget_blocks)` of them.
struct KvMemActKvInputs {
    const actkv::LrfuState*                         lrfu = nullptr;
    const std::vector<std::uint32_t>*               protected_pages = nullptr;
    // Optional diagnostic sink: when non-null the rescue writes one line per round. Set from the
    // caller's environment switch; never read here (single read-point discipline).
    bool                                            trace = false;
};

// Select the blocks to keep.
//
// `scores` must hold at least `total_blocks - first_block` floats, where index i is the score of
// block `first_block + i`. first_block exists so a caller holding a SUFFIX of the block scores (for
// example when an older part of the history has already been evicted) cannot accidentally have its
// scores read as if they described blocks near the start of the sequence.
//
// "NO SIGNAL" IS ENCODED AS -inf. A block whose score is not finite (NaN or +/-inf) is demoted below
// every scored block, so an unscored block can never displace a scored one. This is a deliberate
// departure from the reference, which compares raw doubles: there +inf is the HIGHEST score and NaN
// makes the comparator a non-strict weak ordering, leaving std::nth_element's result unspecified (an
// independent cross-check observed it selecting a NaN block). The scorer must therefore write -inf --
// not NaN, not +inf -- when it has no signal, and the two conventions must not be mixed.
//
// The result never exceeds budget_blocks, never contains a duplicate, and is sorted ascending.
// A budget larger than the history returns the whole history. A ZERO budget also returns the whole
// history: the reference treats 0 as unlimited (`kvmem_store.cpp:362`), and the case is reachable
// because the caller derives the budget with integer division (`select_budget / block_tokens`).
// Every path also keeps `kept.size() == sink_kept + recent_kept + scored_kept` -- a branch that
// returned an unassigned `kept` (empty window, counters still populated) is exactly the silent-wrong-
// answer bug that invariant exists to catch.
[[nodiscard]] KvMemSelectResult kvmem_select_blocks(const KvMemSelectConfig& config,
                                                     const float*             scores,
                                                     std::int32_t             first_block,
                                                     const KvMemActKvInputs&  actkv = {});

} // namespace ninfer::ops
