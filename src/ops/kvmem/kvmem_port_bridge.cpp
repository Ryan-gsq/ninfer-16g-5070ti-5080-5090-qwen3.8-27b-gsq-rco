#include "ops/kvmem/kvmem_port_bridge.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace ninfer::ops::kvmem_port {
namespace {

std::string block_list(const std::vector<std::uint32_t>& ids, std::size_t limit = 12) {
    std::string s;
    for (std::size_t i = 0; i < ids.size() && i < limit; ++i) {
        if (!s.empty()) { s += ","; }
        s += std::to_string(ids[i]);
    }
    if (ids.size() > limit) { s += ",...(" + std::to_string(ids.size()) + ")"; }
    return s.empty() ? std::string("<empty>") : s;
}

bool intersects(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b,
                std::uint32_t* which) {
    for (std::uint32_t x : a) {
        if (std::find(b.begin(), b.end(), x) != b.end()) { *which = x; return true; }
    }
    return false;
}

}  // namespace

Plan translate(const qw3::KvMemPlan& in, std::uint32_t /*budget_tokens*/) {
    Plan out;
    out.stage_in = in.stage_in;
    out.stage_out = in.stage_out;
    out.total_window_tokens = in.total_window_tokens;
    out.selection_overlap_blocks = in.selection_overlap_blocks;
    out.gpu_reused_blocks = in.gpu_reused_blocks;
    out.retained_position_stable = in.retained_position_stable;
    out.retained_position_moved = in.retained_position_moved;
    out.remaps.reserve(in.remaps.size());
    for (const qw3::KvMemRemap& r : in.remaps) {
        Remap m;
        m.block_id = r.block_id;
        m.n_tokens = r.n_tokens;
        m.from_base = r.from_base;
        m.to_base = r.to_base;
        m.skip = r.skip;
        m.raw_refresh = r.raw_refresh;
        out.remaps.push_back(m);
    }
    return out;
}

bool check_plan_invariants(const Plan& plan, const Geometry& geo, std::uint32_t budget_tokens,
                           std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error != nullptr) { *error = why; }
        return false;
    };

    // (1) Window order. "Blocks are packed contiguously from window pos 0" and the id list is
    // sorted ascending, so the remap list must be ascending by id AND the destination bases must
    // tile [0, total_window_tokens) without a gap. A gap here silently shifts every later block's
    // position, which is exactly the class of error that produces plausible-looking garbage.
    std::uint32_t expect_base = 0;
    for (std::size_t i = 0; i < plan.remaps.size(); ++i) {
        const Remap& r = plan.remaps[i];
        if (i > 0 && r.block_id <= plan.remaps[i - 1].block_id) {
            return fail("remaps not ascending by block id at index " + std::to_string(i) +
                        " (" + std::to_string(plan.remaps[i - 1].block_id) + " then " +
                        std::to_string(r.block_id) + ")");
        }
        if (r.n_tokens == 0 || r.n_tokens > static_cast<std::uint32_t>(geo.block_tokens)) {
            return fail("remap block " + std::to_string(r.block_id) + " has n_tokens=" +
                        std::to_string(r.n_tokens) + ", outside (0, block_tokens=" +
                        std::to_string(geo.block_tokens) + "]");
        }
        if (r.to_base != static_cast<std::int32_t>(expect_base)) {
            return fail("remap block " + std::to_string(r.block_id) + " lands at to_base=" +
                        std::to_string(r.to_base) + " but the window expects " +
                        std::to_string(expect_base) + " (blocks must tile from 0)");
        }
        if (r.from_base < 0) {
            return fail("remap block " + std::to_string(r.block_id) + " has negative from_base");
        }
        // `skip` means a valid resident K is already baked at to_base; `raw_refresh` means the
        // block is rebuilt from the raw mirror. Both at once is contradictory and would let the
        // executor skip the very refresh the drift policy demanded.
        if (r.skip && r.raw_refresh) {
            return fail("remap block " + std::to_string(r.block_id) +
                        " is both skip and raw_refresh");
        }
        expect_base += r.n_tokens;
    }

    // (2) The window token count must equal what the remaps actually carry. Upstream defines it as
    // "sum of n_tokens over selected blocks", so a disagreement means the plan and its own summary
    // describe different windows -- the "list disagrees with its own counters" failure mode.
    if (expect_base != plan.total_window_tokens) {
        return fail("total_window_tokens=" + std::to_string(plan.total_window_tokens) +
                    " but the remaps carry " + std::to_string(expect_base));
    }

    // (3) Budget. The semantic window may never exceed the configured selection budget.
    if (budget_tokens != 0 && plan.total_window_tokens > budget_tokens) {
        return fail("window " + std::to_string(plan.total_window_tokens) +
                    " exceeds the selection budget " + std::to_string(budget_tokens));
    }

    // (4) A block cannot be staged in and out in the same plan.
    std::uint32_t dup = 0;
    if (intersects(plan.stage_in, plan.stage_out, &dup)) {
        return fail("block " + std::to_string(dup) + " appears in BOTH stage_in and stage_out");
    }

    // (5) An empty remap list is legal (a plan for "nothing selected"), but a non-empty working
    // set transition with an empty window is not: it means the selection was lost in translation.
    if (plan.remaps.empty() && plan.total_window_tokens == 0 &&
        (!plan.stage_in.empty() || !plan.stage_out.empty())) {
        return fail("stage_in/out is non-empty (" + block_list(plan.stage_in) + " / " +
                    block_list(plan.stage_out) + ") while the window is empty");
    }

    if (error != nullptr) { error->clear(); }
    return true;
}

ResolvedConfig make_store_config(const ConfigInputs& in) {
    ResolvedConfig out;
    qw3::KvMemStoreConfig& cfg = out.cfg;

    if (in.geo.block_tokens <= 0) {
        throw std::invalid_argument("kvmem_port: block_tokens must come from the KV page size");
    }
    cfg.block_tokens = static_cast<std::uint32_t>(in.geo.block_tokens);
    cfg.select_budget = in.select_budget_tokens;
    cfg.gen_budget = in.gen_budget_tokens;

    // Always-kept bands. -1 keeps upstream's "auto" derivation; >= 0 is an explicit override.
    // This is where the most mis-quoted number in the port gets settled: upstream's own command
    // line passes --kvmem-sink-blocks 8, which is an EXPLICIT value, not the formula's output.
    out.resolved = qw3::resolve_kvmem_keep_allocation(cfg.block_tokens, cfg.select_budget,
                                                     in.sink_blocks, in.recent_blocks, -1, -1);
    cfg.sink_blocks = out.resolved.sink_blocks;
    cfg.recent_blocks = out.resolved.recent_blocks;

    // Drift-bounded K construction: the raw-K mirror is what makes a remap rebuildable instead of
    // repeatedly re-rotated. The thresholds are upstream's shipping values; they are written out
    // here rather than relied on as defaults so a reader can see the contract in one place.
    cfg.immutable_source_k = true;
    cfg.immutable_refresh_remaps = 8;
    cfg.immutable_refresh_abs_delta_tokens = 262144;
    cfg.immutable_max_baked_position = 262144;

    cfg.mtp_enabled = in.mtp_enabled;  // we run MTP draft=3: the MTP raw-K mirror is live

    // Tiers. We implement the first two and leave the third as an interface on purpose; a zero
    // byte budget is how that decision is expressed to the ported layer.
    cfg.cpu_tier_bytes = in.cpu_tier_bytes;
    cfg.nvme_tier_bytes = in.nvme_tier_bytes;

    // Per-block byte accounting. A block spans ALL full-attention layers (a token range has KV in
    // every one of them), so the per-block figure multiplies by the layer count. fp8 KV is one
    // byte per element, and K and V are both counted. `gpu_resident_block_bytes` is the value the
    // page pool must use in immutable-source mode, where the lower tiers hold position-free K.
    const std::uint64_t per_layer_per_block =
        static_cast<std::uint64_t>(cfg.block_tokens) *
        static_cast<std::uint64_t>(in.geo.kv_heads) *
        static_cast<std::uint64_t>(in.geo.head_dim) * 2ull /*K and V*/ * 1ull /*fp8*/;
    cfg.estimated_block_bytes =
        per_layer_per_block * static_cast<std::uint64_t>(in.geo.full_attention_layers);
    cfg.gpu_resident_block_bytes = cfg.estimated_block_bytes;
    cfg.estimated_gpu_block_capacity = static_cast<std::uint32_t>(
        cfg.gpu_resident_block_bytes != 0
            ? (in.geo.block_tokens > 0 ? (in.select_budget_tokens / cfg.block_tokens) : 0)
            : 0);
    return out;
}

const std::vector<MissingPiece>& plan_missing_for_engine() {
    static const std::vector<MissingPiece> kMissing = {
        {"page-table alias",
         "the plan names logical blocks; nothing yet maps a block id to the physical pages that "
         "hold it, so stage_in/stage_out cannot be turned into transfers."},
        {"re-RoPE consumer",
         "remaps carry from_base/to_base, but no kernel consumes them yet. The ported layer only "
         "PLANS the in-place de-rotate/re-rotate; the executor has to perform it."},
        {"raw-K mirror",
         "immutable_source_k=true assumes a CPU mirror of unrotated K exists to rebuild cold "
         "blocks and blocks past a refresh threshold. Our raw-K harvest is device-resident only "
         "today."},
        {"raw-Q for scoring",
         "retrieval scores need the de-RoPEd query; it is not captured yet, so the scorer has no "
         "input and the plan can only be driven by synthetic scores."},
        {"mean-K read semantics",
         "MeanKSource is an interface only: our index stores an FP16 mean, the sibling port "
         "stores an F32 sum + count and normalises on read. Which one the scorer sees must be "
         "decided before retrieval is wired (it changes index storage, not just plumbing)."},
        {"GDN side",
         "the ported layer covers only the 16 full-attention layers. The 48 GDN layers keep O(1) "
         "recurrent state that the selected-block window does NOT reconstruct -- the open "
         "correctness question this port exists to answer."},
        {"GC attachment",
         "KvMemGc is optional (attach_gc) and is not attached: every block is treated as live. "
         "Attaching it later changes selection via the liveness filter, so the judges must be "
         "re-run when it is."},
    };
    return kMissing;
}

}  // namespace ninfer::ops::kvmem_port
