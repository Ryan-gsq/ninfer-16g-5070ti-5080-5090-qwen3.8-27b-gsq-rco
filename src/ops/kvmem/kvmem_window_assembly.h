#pragma once

// KVMem window assembly: the MISSING PATH between "the blocks the retriever chose" and "the page
// table the attention reads".
//
// WHY THIS FILE EXISTS
//   An engine audit (exp\kvmem-reuse\ vs exp\kvmem-enginecache, 2026-09-21) found that the engine can
//   already express a window and already owns every part of one, but has NO path that turns a SCORE
//   DRIVEN block selection into one. Two capabilities are absent, and they are the whole of what KVMem
//   adds here:
//       ① a re-RoPE consumer that re-bakes a key to its new window position (ops\kvmem\rerope), and
//       ② this one -- the selection path: selection -> placement -> page table.
//   Everything else already exists and must NOT be rebuilt (the port discipline is explicit: "do not
//   build a new host layer"; hang on the engine's existing page pool + execution table + address
//   space). So this file contains exactly the missing middle sentence:
//
//       block id -> physical page slot -> published at the slot's WINDOW position
//
// THE ONE INVARIANT THAT MAKES IT CORRECT (and the failure it prevents)
//   The window is a CONTIGUOUS PREFIX whose live positions are SEQUENTIAL -- that is the engine's
//   contract, not a preference (`include\infer\ops\softmax_attention.h:118-123`: "Each masked row has
//   a live prefix [0,Vb); its live positions are sequential"), and the main path passes the mask
//   EMPTY (`text_context_impl.h:1023`), so visibility is decided by positions alone. Therefore the
//   window's slot order is CHRONOLOGICAL (ascending block id) and is NOT the retrieval score order.
//   The reference says the same thing in its own words (`qw3\kvmem_store.hpp:128-131`: the remap list
//   is "in window order", `:469-470` "it is sorted ascending internally so window order is
//   deterministic (sink first ... recent last) ... packed contiguously from window pos 0").
//
//   Getting this wrong produces a window that is FULL, VALID and WRONG: every slot holds a genuinely
//   selected block, no counter disagrees, no error is raised, and the answer is garbage. That is
//   exactly the failure mode this project keeps paying for (the empty-window/self-consistent-counters
//   bug in `kvmem_select.cpp:106-126`, and the "publish() is a pure overwrite so a half-reassembled
//   window still looks complete" hazard in exp\kvmem-stage0). So the order is CHECKED here rather than
//   assumed by the caller, and `order_error()` exists so a judge can assert the check has teeth.
//
// A HOLE IS AN ERROR, NOT A GAP
//   If a selected block has no live page, we THROW. Dropping it is the tempting alternative and it is
//   the wrong one: the window would then be a prefix of a different sequence, so its positions would
//   still be sequential and every downstream check would pass, while the model silently attends to a
//   history it was never given. `assemble_window()` already refuses unset/stale/not-device-resident
//   handles, but it can only see the slots it is handed -- it cannot notice a slot that should have
//   been handed to it. That check belongs here, where the selection is still in hand.
//
// HOST-ONLY ON PURPOSE
//   This header includes nothing but the standard library and `kvmem_select.h`. The physical slots
//   arrive as an abstract `BlockPageSource`, so the placement can be exercised with no CUDA context
//   and no GPU at all -- the property whose absence is why this project's MeanKIndex had never been
//   executed before its own round. The judge that drives the REAL page pool is a separate driver
//   (`exp\kvmem-select-assemble\`), because driving the pool is a different kind of evidence.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ops/kvmem/kvmem_select.h"

namespace ninfer::ops {

// ---------------------------------------------------------------------------------------------
// 1. Where a block's key/value actually lives
// ---------------------------------------------------------------------------------------------

// `slot` is the pool's PHYSICAL page index (`DeviceKVPagePool::physical_index()`), which is exactly
// what `KVExecutionTablePool::publish()` consumes -- it converts each handle through that same call
// (`core\paged_kv_cache.cpp:774-789`). A negative slot means "not resident": the block exists in the
// history but has no page the attention could read, which is a hole (see above).
//
// During the port `slots_in_window_order` holds the caller's own handle indices; at the wiring point
// they are physical page indices. Nothing here depends on which, which is the point.
using BlockSlot = std::int32_t;

// ★★ TWO DIFFERENT THINGS GET CONFUSED HERE, AND THE CONFUSION IS SILENT. This header used to mix
//    them up itself, and the first judge written against it went red for the wrong reason:
//
//      - a block's POSITION in the selection (`kvmem_select` emits ascending block IDS, and their
//        position within that emitted list is NOT the id -- {0,1,2,3,5,6,7,8,12,13,14,15} has
//        position 4 at id 5); and
//      - a block's slot lookup, which is keyed by ID.
//
//    The assembly consumes the FIRST (one entry per emitted block, in order) and the caller owns the
//    SECOND (id -> slot). Handing the assembly an id-keyed table as if it were position-indexed
//    produces a perfect-looking plan built from the wrong pages -- no error, no counter disagreement.
//    So the two are separate types below and the adapter between them is explicit and named.
class BlockPageSource {
public:
    virtual ~BlockPageSource() = default;
    // `position` is the rank of the block inside the selection's own list (0 == the first block the
    // selector emitted). It is NEVER the block id.
    [[nodiscard]] virtual BlockSlot position_slot(std::uint32_t position) const = 0;
};

// The caller's real lookup: block id -> physical slot. This is the piece that does not exist in the
// engine yet (the engine's own selection is driven by session keys and prefix candidates, not by a
// score), so this type is where the new id-keyed bookkeeping will live. `first_block` lets a caller
// that only tracks a suffix of the history say where its first tracked block sits without re-basing
// every index by hand -- the same trap `kvmem_select_blocks` guards against with its `first_block`.
class BlockPageTable {
public:
    BlockPageTable() = default;
    explicit BlockPageTable(std::vector<BlockSlot> slots, std::int32_t first_block = 0)
        : slots_(std::move(slots)), first_block_(first_block) {}

    void assign(std::uint32_t block_id, BlockSlot slot) {
        const std::int64_t index = static_cast<std::int64_t>(block_id) - first_block_;
        if (index < 0) { return; }  // before the tracked window: leave it absent
        if (static_cast<std::size_t>(index) >= slots_.size()) {
            slots_.resize(static_cast<std::size_t>(index) + 1, -1);
        }
        slots_[static_cast<std::size_t>(index)] = slot;
    }

    [[nodiscard]] std::size_t  size() const noexcept { return slots_.size(); }
    [[nodiscard]] std::int32_t first_block() const noexcept { return first_block_; }

    // -1 = the block is not resident (a hole).
    [[nodiscard]] BlockSlot lookup(std::uint32_t block_id) const noexcept {
        const std::int64_t index = static_cast<std::int64_t>(block_id) - first_block_;
        if (index < 0 || static_cast<std::size_t>(index) >= slots_.size()) { return -1; }
        return slots_[static_cast<std::size_t>(index)];
    }

private:
    std::vector<BlockSlot> slots_;
    std::int32_t           first_block_ = 0;
};

// The explicit adapter: pair an id-keyed table with the selection whose ids it describes. Holding BOTH
// is what makes the translation checkable -- the table alone cannot know which positions it is being
// asked about.
class SelectedBlockPages final : public BlockPageSource {
public:
    SelectedBlockPages(const BlockPageTable& table, const KvMemSelectResult& selection)
        : table_(table), selection_(selection) {}

    [[nodiscard]] BlockSlot position_slot(std::uint32_t position) const override {
        if (static_cast<std::size_t>(position) >= selection_.kept.size()) { return -1; }
        return table_.lookup(static_cast<std::uint32_t>(selection_.kept[position]));
    }

private:
    const BlockPageTable&    table_;
    const KvMemSelectResult& selection_;
};

// ---------------------------------------------------------------------------------------------
// 1b. ★ The table the ENGINE does not have yet: block id -> where that block's pages live
// ---------------------------------------------------------------------------------------------
//
// `BlockPageTable` above is the right SHAPE but the wrong CONTENT for the wiring point: its rows are
// page indices the caller already knows, whereas the engine's real question is "which of my pages
// currently holds block b's key/value" -- and that answer changes the moment a tier moves a block.
//
// The ported policy layer already carries exactly that field (`qw3\kvmem_store.hpp:53`,
// `KvMemBlock::gpu_slot`), the sibling implementation builds the same thing as
// `slot_of_block[block_id]` + a freelist (`E:\外部资料\...\docs\modification-plan.md`), and the
// official layer's remap list already keys on block id (`KvMemRemap::block_id`). So this type adds NO
// new concept; it is the engine-side half of a mapping the policy layer has been reserving a field
// for since it was ported.
//
// WHY IT CARRIES AN EPOCH, AND WHY THAT IS NOT OPTIONAL
//   A block's page identity is (position, generation). Without the generation, the failure below is
//   INVISIBLE: stage a block out, let its page be rematerialized for another block, then read the map
//   again -- the block is still "registered", the page number is still a legal page, and the window
//   built from it is full, valid and WRONG. That is the same silent-wrong-answer family as the
//   position/id mix-up, one tier further out, and it becomes reachable the moment `stage_out` exists.
//   So the epoch is part of validity, not a diagnostic: `Slot(0)` means "the source cannot tell me"
//   and checks only the registration.
template <typename Slot>
struct BlockPageMap {
    // One block's current location. `slot == 0` is a LEGAL slot (`DeviceKVPagePool` page 0 is a real
    // page -- see the CAL measurement in exp\kvmem-select-assemble), so validity is `registered`, never
    // "the value is non-zero".
    struct Entry {
        Slot          slot = 0;
        std::uint64_t epoch = 0;
        bool          registered = false;
    };

    struct Lookup {
        Slot          slot = 0;
        bool          valid = false;
        bool          stale = false;  // registered, but the page no longer holds this block
    };

    void clear() noexcept { entries_.assign(entries_.size(), Entry{}); }

    void assign(std::uint32_t block_id, Slot slot, std::uint64_t epoch) {
        const std::int64_t index = static_cast<std::int64_t>(block_id) - first_block_;
        if (index < 0) { return; }  // before the tracked window: leave it unregistered
        if (static_cast<std::size_t>(index) >= entries_.size()) {
            entries_.resize(static_cast<std::size_t>(index) + 1);
        }
        entries_[static_cast<std::size_t>(index)] = Entry{slot, epoch, true};
    }

    // A tier moved this block away (stage-out, eviction, truncate): the row must stop answering.
    void invalidate(std::uint32_t block_id) noexcept {
        const std::int64_t index = static_cast<std::int64_t>(block_id) - first_block_;
        if (index < 0 || static_cast<std::size_t>(index) >= entries_.size()) { return; }
        entries_[static_cast<std::size_t>(index)] = Entry{};
    }

    [[nodiscard]] std::size_t  size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::int32_t first_block() const noexcept { return first_block_; }
    [[nodiscard]] std::size_t  registered_blocks() const noexcept {
        std::size_t count = 0;
        for (const Entry& entry : entries_) {
            if (entry.registered) { ++count; }
        }
        return count;
    }

    // `current_epoch(block_id)` is what the SOURCE says the block's page holds right now; the caller
    // supplies it because only the tier/address layer can answer it. A `Lookup` with `stale == true`
    // must never reach the page table: the plan would be built from a page that now belongs to
    // somebody else.
    [[nodiscard]] Lookup lookup(std::uint32_t block_id, std::uint64_t current_epoch) const noexcept {
        const std::int64_t index = static_cast<std::int64_t>(block_id) - first_block_;
        if (index < 0 || static_cast<std::size_t>(index) >= entries_.size()) { return Lookup{}; }
        const Entry& entry = entries_[static_cast<std::size_t>(index)];
        if (!entry.registered) { return Lookup{}; }
        if (current_epoch != 0 && entry.epoch != 0 && entry.epoch != current_epoch) {
            return Lookup{entry.slot, false, true};
        }
        return Lookup{entry.slot, true, false};
    }

    [[nodiscard]] Lookup lookup(std::uint32_t block_id) const noexcept {
        return lookup(block_id, 0);
    }

private:
    std::vector<Entry> entries_;
    std::int32_t       first_block_ = 0;

public:
    explicit BlockPageMap(std::int32_t first_block = 0) : first_block_(first_block) {}
};

// The id-keyed map as a position-indexed `BlockPageSource`, with the staleness rule enforced.
//
// `current_epoch` is called once per selected block, in the plan's order. Returning 0 means "the
// source cannot tell me", which disables the check for that block rather than failing it -- an
// unanswerable question must not read as a wrong answer.
template <typename Slot, typename EpochFn>
class MappedBlockPages final : public BlockPageSource {
public:
    MappedBlockPages(const BlockPageMap<Slot>& map, const KvMemSelectResult& selection,
                     EpochFn current_epoch)
        : map_(map), selection_(selection), current_epoch_(std::move(current_epoch)) {}

    [[nodiscard]] BlockSlot position_slot(std::uint32_t position) const override {
        if (static_cast<std::size_t>(position) >= selection_.kept.size()) { return -1; }
        const std::uint32_t block = static_cast<std::uint32_t>(selection_.kept[position]);
        const auto          found = map_.lookup(block, current_epoch_(block));
        if (found.stale) {
            throw std::logic_error(kStaleMessage + std::to_string(block) +
                                   " at window slot " + std::to_string(position));
        }
        return found.valid ? static_cast<BlockSlot>(found.slot) : -1;
    }

    static constexpr const char* kStaleMessage =
        "kvmem_window_assembly: block-page map is STALE for block ";

private:
    const BlockPageMap<Slot>&    map_;
    const KvMemSelectResult&     selection_;
    EpochFn                      current_epoch_;
};

// ---------------------------------------------------------------------------------------------
// 1c. Filling the holes: stage-in
// ---------------------------------------------------------------------------------------------
//
// Until now every hole was an error, and that was the honest answer -- a window assembled around a
// missing block is a prefix of a different sequence, so nothing downstream can notice. But it is not
// a FINAL answer: the engine CAN bring a block back, it just has to be asked. The engine's own restore
// transaction does exactly this three-step dance (`program_impl.h:4949-4982`, spent on checkpoints):
//
//     reserve_device_replica(logical, reservation)   -> a destination page + the "destination pinned"
//     <host-to-device copy of the replica's extent>  -> the caller performs this
//     publish_device_replica(logical)                -> the destination becomes the device replica
//
// with `abort_device_replica(logical, reservation)` as the failure path. That is the whole of stage-in
// at the address layer; what this section adds is the ORCHESTRATION (which blocks, in which order, and
// what "it worked" means) plus the refusal to assemble while any hole remains.
//
// WHY A SEAM AND NOT A DIRECT CALL. The stager lives in the target tree
// (`targets/qwen3_6/impl/runtime/logical_kv_store.h`), which an ops-layer header must not include --
// and the discipline this whole port has followed is that the placement is checkable with no GPU and
// no engine. So the copy is injected, exactly as `BlockPageSource` is. One real implementation is the
// three calls above; `StageInOutcome` also records what happened so a caller can report it rather than
// guess.
class PageRestorer {
public:
    virtual ~PageRestorer() = default;

    // Is this page currently readable by the attention?
    [[nodiscard]] virtual bool device_resident(BlockSlot slot) const = 0;

    // Make it so. Must either succeed (device_resident(slot) is true afterwards) or return false with
    // nothing half-applied -- the caller aborts rather than assembling a partial window.
    [[nodiscard]] virtual bool stage_in(BlockSlot slot) = 0;
};

struct StageInOutcome {
    std::uint32_t checked = 0;   // selected blocks examined
    std::uint32_t restored = 0;  // blocks that had to be brought back
    std::uint32_t failed = 0;    // stage_in returned false
    std::vector<std::uint32_t> restored_blocks;  // in window order, for reporting
    std::string  error;          // non-empty when the caller must NOT assemble
};

// Bring back every selected block that is not on the device, in window order. Returns an outcome; it
// never throws for a missing page (that is what the outcome is for) but does throw for a malformed
// selection, by delegating the order gate to the same check the planner uses.
//
// ORDER MATTERS FOR A REASON THAT IS NOT STYLE: blocks are staged in ASCENDING id, i.e. the order the
// window will hold them. A tier that evicts to make room for an incoming page therefore evicts in a
// predictable order, and the log of what was restored reads in window order -- which is what makes the
// record comparable with the plan.
[[nodiscard]] StageInOutcome stage_in_missing(const KvMemSelectResult& selection,
                                              const BlockPageSource&   pages,
                                              PageRestorer&            restorer);

// ---------------------------------------------------------------------------------------------
// 1d. ★ Why the composition above is NOT enough, and what replaces it
// ---------------------------------------------------------------------------------------------
//
// `stage_in_missing` assumes the page source can already name a slot for every selected block. At the
// WIRING POINT it cannot, and the reason is structural rather than an oversight:
//
//   `assemble_window()` takes a list of `LogicalKVPageHandle`, and a block whose page is not on the
//   device HAS no such handle yet. Once it does have one -- after its page is materialized -- the hole
//   is gone by definition. So there is nothing to "fill" before assembly: either the handle exists and
//   the window can be assembled, or it does not and no amount of planning will produce it.
//
// Worse, encoding that state as a single `-1` slot conflates TWO different questions -- "should this
// block be staged in?" and, after materialization, "which page is it now?" -- in one value that is
// also a legal slot number's neighbour. That is the §5.18 family (`position != block id`) one layer
// out, and it is why the composition below exists: the two questions are asked in the two places that
// can actually answer them, in the only order that works.
//
//   phase 1 (before materialization): PLACEMENT -- which slot will hold this block, or -1 for a hole
//                                     within the block's own slot domain;
//   phase 2 (after  materialization): RESOLUTION -- which logical page IS it now (identity, never a
//                                     sentinel).
class PagePlacementSource {
public:
    virtual ~PagePlacementSource() = default;
    // The slot the block at `position` will occupy, or -1 if it is not materialized. This is a READ:
    // it must not stage anything in.
    [[nodiscard]] virtual BlockSlot pending_slot(std::uint32_t position) const = 0;
};

class PageResolver {
public:
    virtual ~PageResolver() = default;
    // The logical page now holding `slot`'s key/value. Called only after stage-in succeeded.
    [[nodiscard]] virtual std::uint64_t logical_page(BlockSlot slot) const = 0;
};

// The stage-in step on its own, keyed by DESIGNED slot (phase 1's answer) rather than by page. The
// engine's implementation is the three-step restore (`reserve_device_replica` -> host-to-device copy
// -> `publish_device_replica`) against that slot's address entry.
class PageStager {
public:
    virtual ~PageStager() = default;
    [[nodiscard]] virtual bool device_resident(BlockSlot slot) const = 0;
    [[nodiscard]] virtual bool stage_in(BlockSlot slot) = 0;
};

// (The two-phase path's types -- `ResolvedBlockPages`, `PositionSlots`, `WindowPromotion` and
// `stage_assemble_and_resolve` -- are DEFINED AT THE END OF THIS HEADER: they return a
// `BlockWindowPlan`, and that struct is only declared further down.)

// The complete assembly, now including stage-in: fill the holes, then plan, then assert that the plan
// is fit to publish. Throws `std::logic_error` (with the outcome's reason) when a hole could not be
// filled, so a caller can never publish a short window by accident.
//
// DEFINED AFTER `BlockWindowPlan` BELOW -- it returns one, so both the struct and the function are
// declared at the end of this header rather than here.


// ---------------------------------------------------------------------------------------------
// 2. The assembly: what the caller must materialise, and in what order
// ---------------------------------------------------------------------------------------------

struct BlockWindowPlan {
    // Blocks in WINDOW order -- always ascending by block id, never by score. This is the only order
    // the window may be built in (see the header comment).
    std::vector<std::uint32_t> ordered_blocks;
    // One physical slot per ordered block, same index, same order. This is the list a caller turns
    // into page handles and hands to `assemble_window()` / `KVExecutionTablePool::publish()`.
    std::vector<BlockSlot>     slots_in_window_order;
    // Placement bookkeeping: `block_slot_in_plan[i]` is the position the plan gave `ordered_blocks[i]`
    // (always i today -- kept here so a future reordering policy has to fill it in rather than
    // silently reusing the index), and `block_index_in_plan[i]` is where that block sat in the
    // selection's emitted list. They are equal today and are recorded separately for exactly the
    // reason the header comment gives: position is not id, and a reader must be able to see which
    // one a number means.
    std::vector<std::uint32_t> block_slot_in_plan;
    std::vector<std::uint32_t> block_index_in_plan;

    std::int32_t  sink_kept   = 0;
    std::int32_t  recent_kept = 0;
    std::int32_t  scored_kept = 0;
    std::int32_t  budget_blocks = 0;

    std::uint32_t page_count       = 0;  // == ordered_blocks.size()
    std::uint32_t total_columns    = 0;  // == filled_columns (two names, one value: the engine calls
                                         //    it the committed frontier, this layer calls it the
                                         //    window's token count, and a reader must be able to see
                                         //    they are the same number, so both are carried)
    std::uint32_t filled_columns   = 0;  // (page_count-1)*block_tokens + tail_columns
    std::uint32_t tail_columns     = 0;  // columns in the LAST page (1..block_tokens)
    std::uint32_t window_frontier  = 0;  // == filled_columns: the compact window's W
    std::uint32_t first_source_block = 0; // block id of ordered_blocks[0] (0 when empty)
    std::uint32_t budget_tokens    = 0;  // budget_blocks * block_tokens
};

// Everything the placement needs that is NOT the selection itself.
struct BlockWindowInputs {
    std::int32_t  block_tokens = 64;  // MUST be kPagedKVPageSize, passed in rather than typed
    // Tokens actually present in the history's LAST block. A history that is not a whole number of
    // blocks ends in a partial page, and the engine requires the frontier to say so
    // (`logical_kv_store.h:1573-1574`). <= 0 means "the last block is full".
    std::int32_t  last_block_columns = 0;
};

// Turn a selection into an assembly plan. `pages` answers by POSITION inside `selection.kept` -- build
// it with `SelectedBlockPages(table, selection)` when the underlying lookup is keyed by block id.
// Throws `std::invalid_argument` on a malformed input, `std::logic_error` when a selected block has no
// live page (a hole) or when the selection is not in ascending window order.
[[nodiscard]] BlockWindowPlan assemble_block_window(const KvMemSelectConfig& config,
                                                    const KvMemSelectResult& selection,
                                                    const BlockPageSource&   pages,
                                                    const BlockWindowInputs& inputs);

// Same, deriving the selection from scores. This is the wiring-point entry: it is the direct
// replacement for "engine had no way to pick blocks by score".
[[nodiscard]] BlockWindowPlan select_and_assemble(const KvMemSelectConfig& config,
                                                  const float*             scores,
                                                  std::int32_t             first_block,
                                                  const BlockPageSource&   pages,
                                                  const BlockWindowInputs& inputs);

// Stage-in plus assembly, in one call: the entry a caller should use once stage-out exists. It fills
// every hole first (section 1c) and refuses to return a plan if one could not be filled.
struct StagedWindow {
    BlockWindowPlan plan;
    StageInOutcome  stage_in;
};

[[nodiscard]] StagedWindow select_assemble_and_stage(const KvMemSelectConfig& config,
                                                     const KvMemSelectResult& selection,
                                                     const BlockPageSource&   pages,
                                                     PageRestorer&            restorer,
                                                     const BlockWindowInputs& inputs);

// ---------------------------------------------------------------------------------------------
// 3. Self-checks (return a reason instead of throwing, so a caller can report rather than abort)
// ---------------------------------------------------------------------------------------------

// Structural invariants. A plan that fails any of these must never reach `assemble_window()`.
[[nodiscard]] bool check_block_window_plan(const BlockWindowPlan& plan, const BlockWindowInputs& inputs,
                                           std::string* error);

// The ONE order check, isolated so a judge can show it has teeth by feeding it a plan built in score
// order: an ascending violation, a duplicate, or an out-of-range id all return false. `reason` may
// be null.
[[nodiscard]] bool window_order_error(const KvMemSelectResult& selection, std::string* reason);

// ★ The tail agreement check, and the reason it exists.
//
// The engine derives its OWN committed frontier from the `tail_columns` the caller passes to
// `assemble_window()` (`logical_kv_store.h:1573-1574`), and the arm() step then asserts
// `W == frontier` (`kvmem_window.h`). If the caller passes a tail that disagrees with the plan, it
// arms with a W the page table never committed -- i.e. the attention reads PAST its own window, with
// no error anywhere. That happened in this component's own GPU judge (a 40-column tail passed for a
// 64-column plan), which is why the agreement is a named, testable call rather than a comment.
[[nodiscard]] bool tail_agrees_with_engine(const BlockWindowPlan& plan,
                                           std::uint32_t          engine_tail_columns,
                                           std::string*           error);

// Render a plan's physical order the way the page table would read it back. `slot_to_page` maps a
// slot to the integer the table stores (identity during the port; `physical_index()` at the wiring
// point), so a judge can compare a readback against the plan without hand-writing the mapping.
[[nodiscard]] std::vector<std::int32_t> expected_table_row(
    const BlockWindowPlan& plan, const std::int32_t* slot_to_page = nullptr);

// ---------------------------------------------------------------------------------------------
// 4. The two-phase path, defined here because it returns a `BlockWindowPlan` (see section 1d)
// ---------------------------------------------------------------------------------------------

// A `BlockPageSource` answered from a resolver: POSITIONS in, logical pages out.
class ResolvedBlockPages final : public BlockPageSource {
public:
    ResolvedBlockPages(const PageResolver& resolver, const BlockWindowPlan& plan)
        : resolver_(resolver), plan_(plan) {}

    [[nodiscard]] BlockSlot position_slot(std::uint32_t position) const override {
        if (static_cast<std::size_t>(position) >= plan_.slots_in_window_order.size()) { return -1; }
        const BlockSlot slot = plan_.slots_in_window_order[position];
        return slot < 0 ? -1 : static_cast<BlockSlot>(resolver_.logical_page(slot));
    }

private:
    const PageResolver&    resolver_;
    const BlockWindowPlan& plan_;
};

// The trivially position-indexed source: "the slot at position i is ordered[i]". Used once phase 1 has
// produced the list, so the plan can be built without re-asking the placement source.
class PositionSlots final : public BlockPageSource {
public:
    explicit PositionSlots(std::vector<BlockSlot> ordered) : ordered_(std::move(ordered)) {}
    [[nodiscard]] BlockSlot position_slot(std::uint32_t position) const override {
        if (static_cast<std::size_t>(position) >= ordered_.size()) { return -1; }
        return ordered_[position];
    }

private:
    std::vector<BlockSlot> ordered_;
};

struct WindowPromotion {
    StageInOutcome         stage_in;
    BlockWindowPlan        plan;
    std::vector<BlockSlot> resolved;  // one LOGICAL PAGE per window slot, resolved after materialization
};

// ★ The composition that works at the wiring point, in the only order that can (section 1d).
[[nodiscard]] WindowPromotion stage_assemble_and_resolve(const KvMemSelectConfig&   config,
                                                         const KvMemSelectResult&   selection,
                                                         const PagePlacementSource& placement,
                                                         PageStager&                stager,
                                                         const PageResolver&        resolver,
                                                         const BlockWindowInputs&   inputs);

} // namespace ninfer::ops
