#include "ops/kvmem/kvmem_window_assembly.h"

#include <algorithm>
#include <stdexcept>

// See kvmem_window_assembly.h for what this is, why the window order is chronological, and why a
// hole is an error rather than a gap.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "kvmem_window_assembly";

std::string at_block(const char* what, std::size_t index, std::uint32_t block_id) {
    return std::string(kSubject) + ": " + what + " at window slot " + std::to_string(index) +
           " (block " + std::to_string(block_id) + ")";
}

} // namespace

bool window_order_error(const KvMemSelectResult& selection, std::string* reason) {
    const std::vector<std::int32_t>& kept = selection.kept;
    for (std::size_t index = 0; index < kept.size(); ++index) {
        if (kept[index] < 0) {
            if (reason != nullptr) {
                *reason = at_block("negative block id", index, static_cast<std::uint32_t>(kept[index]));
            }
            return true;
        }
        if (index != 0 && kept[index] <= kept[index - 1]) {
            if (reason != nullptr) {
                *reason = at_block(kept[index] == kept[index - 1] ? "duplicate block id"
                                                                  : "block ids are not ascending",
                                   index, static_cast<std::uint32_t>(kept[index]));
            }
            return true;
        }
    }
    return false;
}

BlockWindowPlan assemble_block_window(const KvMemSelectConfig& config,
                                      const KvMemSelectResult& selection,
                                      const BlockPageSource&   pages,
                                      const BlockWindowInputs& inputs) {
    if (inputs.block_tokens <= 0) {
        throw std::invalid_argument(std::string(kSubject) + ": block_tokens must be positive");
    }
    if (inputs.last_block_columns < 0 || inputs.last_block_columns > inputs.block_tokens) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": last_block_columns is out of range");
    }
    if (config.total_blocks < 0 || config.budget_blocks < 0) {
        throw std::invalid_argument(std::string(kSubject) + ": negative count");
    }
    if (selection.kept.size() > static_cast<std::size_t>(config.total_blocks) &&
        config.total_blocks > 0) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": the selection is larger than the history");
    }
    if (config.budget_blocks > 0 &&
        selection.kept.size() > static_cast<std::size_t>(config.budget_blocks)) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": the selection exceeds the block budget");
    }

    // ★ The order gate. A selection that is not in ascending window order cannot be assembled into a
    // contiguous prefix without silently redefining which block sits at which position, so it is
    // refused here rather than published and discovered as a wrong answer.
    std::string order_reason;
    if (window_order_error(selection, &order_reason)) { throw std::logic_error(order_reason); }

    BlockWindowPlan plan;
    plan.sink_kept     = selection.sink_kept;
    plan.recent_kept   = selection.recent_kept;
    plan.scored_kept   = selection.scored_kept;
    plan.budget_blocks = config.budget_blocks;

    const std::size_t count = selection.kept.size();
    plan.ordered_blocks.reserve(count);
    plan.slots_in_window_order.reserve(count);
    plan.block_slot_in_plan.reserve(count);
    plan.block_index_in_plan.reserve(count);

    // The placement is copied out of the selection in kept order -- never re-sorted, never
    // de-duplicated, exactly as `assemble_window()` refuses to re-sort the pages it is handed
    // ("no internal re-sort"; exp\kvmem-stage2a A-2).
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t block = static_cast<std::uint32_t>(selection.kept[index]);
        const BlockSlot     slot  = pages.position_slot(static_cast<std::uint32_t>(index));
        if (slot < 0) {
            // A hole. Fixing it is not this layer's job: the block must be staged in from the host
            // tier first, and until it is, the honest outcome is refusal.
            throw std::logic_error(at_block("selected block has no live page (stage it in first)",
                                            index, block));
        }
        plan.ordered_blocks.push_back(block);
        plan.slots_in_window_order.push_back(slot);
        plan.block_slot_in_plan.push_back(static_cast<std::uint32_t>(index));
        plan.block_index_in_plan.push_back(static_cast<std::uint32_t>(index));
    }

    const std::uint32_t block_tokens = static_cast<std::uint32_t>(inputs.block_tokens);
    plan.page_count                  = static_cast<std::uint32_t>(count);
    plan.budget_tokens               = static_cast<std::uint32_t>(config.budget_blocks) * block_tokens;
    if (count == 0) {
        // Not an assembly. `assemble_window()` rejects an empty page list on its own; a caller that
        // gets here must keep the window disarmed rather than arm an empty one.
        plan.tail_columns    = 0;
        plan.filled_columns  = 0;
        plan.total_columns   = 0;
        plan.window_frontier = 0;
        return plan;
    }

    plan.first_source_block = plan.ordered_blocks.front();

    // The tail is the LAST page's column count. It is a property of how many tokens the history's
    // final block actually holds -- not of the plan -- so a partial history keeps its partial tail.
    const bool last_block_is_history_tail =
        config.total_blocks > 0 &&
        plan.ordered_blocks.back() == static_cast<std::uint32_t>(config.total_blocks - 1);
    plan.tail_columns = (last_block_is_history_tail && inputs.last_block_columns > 0)
                            ? static_cast<std::uint32_t>(inputs.last_block_columns)
                            : block_tokens;
    plan.total_columns =
        static_cast<std::uint32_t>(count - 1) * block_tokens + plan.tail_columns;
    plan.filled_columns  = plan.total_columns;
    plan.window_frontier = plan.total_columns;
    return plan;
}

BlockWindowPlan select_and_assemble(const KvMemSelectConfig& config, const float* scores,
                                    std::int32_t first_block, const BlockPageSource& pages,
                                    const BlockWindowInputs& inputs) {
    const KvMemSelectResult selection = kvmem_select_blocks(config, scores, first_block);
    return assemble_block_window(config, selection, pages, inputs);
}

bool check_block_window_plan(const BlockWindowPlan& plan, const BlockWindowInputs& inputs,
                             std::string* error) {
    const auto fail = [error](const std::string& what) {
        if (error != nullptr) { *error = std::string(kSubject) + ": " + what; }
        return false;
    };
    if (inputs.block_tokens <= 0) { return fail("block_tokens must be positive"); }
    if (plan.ordered_blocks.size() != plan.slots_in_window_order.size()) {
        return fail("the ordered block list and the slot list disagree in length");
    }
    if (plan.block_slot_in_plan.size() != plan.ordered_blocks.size() ||
        plan.block_index_in_plan.size() != plan.ordered_blocks.size()) {
        return fail("a placement index list disagrees with the ordered block list");
    }
    if (plan.ordered_blocks.size() > static_cast<std::size_t>(plan.budget_blocks) &&
        plan.budget_blocks != 0) {
        return fail("the window holds more blocks than the budget");
    }
    for (std::size_t index = 0; index < plan.ordered_blocks.size(); ++index) {
        if (index != 0 && plan.ordered_blocks[index] <= plan.ordered_blocks[index - 1]) {
            return fail("the ordered block list is not strictly ascending");
        }
        if (plan.slots_in_window_order[index] < 0) {
            return fail("a slot is negative, i.e. the plan carries a hole");
        }
    }
    if (plan.page_count != plan.ordered_blocks.size()) {
        return fail("page_count disagrees with the ordered block list");
    }
    if (plan.page_count == 0) {
        return plan.filled_columns == 0 && plan.window_frontier == 0 && plan.tail_columns == 0
                   ? true
                   : fail("an empty window carries non-zero geometry");
    }
    const std::uint32_t block_tokens = static_cast<std::uint32_t>(inputs.block_tokens);
    if (plan.tail_columns == 0 || plan.tail_columns > block_tokens) {
        return fail("tail_columns is out of range");
    }
    const std::uint32_t expect =
        static_cast<std::uint32_t>(plan.page_count - 1) * block_tokens + plan.tail_columns;
    if (plan.filled_columns != expect || plan.total_columns != expect ||
        plan.window_frontier != expect) {
        return fail("the three token counts disagree with (pages-1)*block + tail");
    }
    if (plan.budget_tokens != 0 && plan.filled_columns > plan.budget_tokens) {
        return fail("the window's tokens exceed the budget");
    }
    if (plan.tail_columns == 0) { return fail("the plan carries no tail at all"); }
    const std::int64_t counted =
        static_cast<std::int64_t>(plan.sink_kept) + plan.recent_kept + plan.scored_kept;
    if (counted != static_cast<std::int64_t>(plan.page_count)) {
        // The counters are the caller's own accounting of which region spent the budget; a plan whose
        // placement disagrees with them is a judge of its own failure (kvmem_select.cpp:106-126).
        return fail("sink+recent+scored does not equal the window's page count");
    }
    return true;
}

bool tail_agrees_with_engine(const BlockWindowPlan& plan, std::uint32_t engine_tail_columns,
                             std::string* error) {
    if (engine_tail_columns == plan.tail_columns) { return true; }
    if (error != nullptr) {
        *error = std::string(kSubject) + ": the engine was given a " +
                 std::to_string(engine_tail_columns) + "-column tail but the plan's last page holds " +
                 std::to_string(plan.tail_columns) +
                 " columns; arming with the plan's frontier (" +
                 std::to_string(plan.window_frontier) +
                 ") would read past the window the page table actually committed";
    }
    return false;
}

std::vector<std::int32_t> expected_table_row(const BlockWindowPlan& plan,
                                             const std::int32_t*     slot_to_page) {
    std::vector<std::int32_t> row;
    row.reserve(plan.slots_in_window_order.size());
    for (const BlockSlot slot : plan.slots_in_window_order) {
        row.push_back(slot_to_page != nullptr ? slot_to_page[slot] : slot);
    }
    return row;
}

StageInOutcome stage_in_missing(const KvMemSelectResult& selection, const BlockPageSource& pages,
                                PageRestorer& restorer) {
    StageInOutcome outcome;
    // The order gate first: a selection that is not in ascending window order has no defined window,
    // so "which block goes where" -- and therefore the restoring order -- would be undefined too.
    std::string order_reason;
    if (window_order_error(selection, &order_reason)) { throw std::logic_error(order_reason); }

    for (std::size_t position = 0; position < selection.kept.size(); ++position) {
        const BlockSlot slot = pages.position_slot(static_cast<std::uint32_t>(position));
        ++outcome.checked;
        if (slot < 0) {
            outcome.error = std::string(kSubject) + ": selected block " +
                            std::to_string(selection.kept[position]) + " has no page to stage in";
            outcome.failed = 1;
            return outcome;
        }
        if (restorer.device_resident(slot)) { continue; }
        if (!restorer.stage_in(slot)) {
            outcome.error = std::string(kSubject) + ": stage-in FAILED for block " +
                            std::to_string(selection.kept[position]) + " (slot " +
                            std::to_string(slot) + "); refusing to assemble a partial window";
            outcome.failed = 1;
            return outcome;
        }
        if (!restorer.device_resident(slot)) {
            // A stager that reports success without the page being resident is exactly the failure this
            // check exists for: the window would otherwise be built around a page nobody can read.
            outcome.error = std::string(kSubject) + ": stage-in reported success for block " +
                            std::to_string(selection.kept[position]) +
                            " but the page is still not device-resident";
            outcome.failed = 1;
            return outcome;
        }
        ++outcome.restored;
        outcome.restored_blocks.push_back(static_cast<std::uint32_t>(selection.kept[position]));
    }
    return outcome;
}

// NOTE: this overload works only where the page source can already NAME a slot per block -- i.e. where
// a hole is a missing page behind an existing slot. At the wiring point a hole has no page at all, so
// `stage_assemble_and_resolve` below is the one that applies there; this one stays for callers whose
// source is slot-keyed (and for the host judge that pins the refusal behaviour).
StagedWindow select_assemble_and_stage(const KvMemSelectConfig& config,
                                       const KvMemSelectResult& selection,
                                       const BlockPageSource&   pages, PageRestorer& restorer,
                                       const BlockWindowInputs& inputs) {
    StagedWindow staged;
    staged.stage_in = stage_in_missing(selection, pages, restorer);
    if (staged.stage_in.failed != 0) { throw std::logic_error(staged.stage_in.error); }
    staged.plan = assemble_block_window(config, selection, pages, inputs);
    return staged;
}

WindowPromotion stage_assemble_and_resolve(const KvMemSelectConfig&    config,
                                           const KvMemSelectResult&    selection,
                                           const PagePlacementSource&  placement,
                                           PageStager&                 stager,
                                           const PageResolver&         resolver,
                                           const BlockWindowInputs&    inputs) {
    // ---- 1. place. A designed slot of -1 is a hole; it is staged, not dropped. Dropping would leave a
    // window that is a prefix of a different sequence -- the failure this layer exists to make
    // impossible -- so a hole we cannot fill aborts the whole promotion instead.
    std::string order_reason;
    if (window_order_error(selection, &order_reason)) { throw std::logic_error(order_reason); }

    WindowPromotion promotion;
    promotion.stage_in.checked = static_cast<std::uint32_t>(selection.kept.size());
    std::vector<BlockSlot> designed;
    designed.reserve(selection.kept.size());
    for (std::size_t position = 0; position < selection.kept.size(); ++position) {
        const BlockSlot slot = placement.pending_slot(static_cast<std::uint32_t>(position));
        if (slot < 0) {
            // No slot at all: there is nothing to materialize INTO, and the layer above has to give this
            // block a home before the window can exist. Loud, named, and before any publication.
            throw std::logic_error(std::string(kSubject) + ": selected block " +
                                   std::to_string(selection.kept[position]) +
                                   " has no page to stage in (no slot was designed for it)");
        }
        designed.push_back(slot);
        if (stager.device_resident(slot)) { continue; }
        if (!stager.stage_in(slot)) {
            throw std::logic_error(std::string(kSubject) + ": stage-in FAILED for block " +
                                   std::to_string(selection.kept[position]) + " (slot " +
                                   std::to_string(slot) + ")");
        }
        if (!stager.device_resident(slot)) {
            throw std::logic_error(std::string(kSubject) + ": stage-in reported success for block " +
                                   std::to_string(selection.kept[position]) +
                                   " but the slot is still not device-resident");
        }
        ++promotion.stage_in.restored;
        promotion.stage_in.restored_blocks.push_back(
            static_cast<std::uint32_t>(selection.kept[position]));
    }

    // ---- 2/3. window. Publish the plan keyed by DESIGNED slot -- no `-1` survives, so nothing
    // downstream can read a sentinel as a page number.
    promotion.plan = assemble_block_window(config, selection, PositionSlots(designed), inputs);

    // ---- 4. resolve. Only now does a slot become a logical page.
    promotion.resolved.reserve(promotion.plan.slots_in_window_order.size());
    for (const BlockSlot slot : promotion.plan.slots_in_window_order) {
        promotion.resolved.push_back(static_cast<BlockSlot>(resolver.logical_page(slot)));
    }
    return promotion;
}

} // namespace ninfer::ops
