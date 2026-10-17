#pragma once

// Bridge between the PORTED kvmem-qw3 policy layer (ops/kvmem/qw3/*, Apache-2.0, verbatim)
// and ninfer's own host-side types.
//
// WHY THIS FILE EXISTS
//   The ported layer is deliberately engine-independent: it owns block metadata, the selection,
//   the working-set diff and the remap plan, and it touches no GPU memory and calls no kernels
//   (see the header comment of qw3/kvmem_store.hpp). What it does NOT know is our geometry
//   (block = one KV page = 64 tokens, 16 full-attention + 48 GDN layers), our tier sizes, and
//   our switch surface. Everything that has to be translated lives here, in ONE place, so the
//   ported code can stay byte-identical to upstream and keep passing upstream's own tests.
//
// WHAT IS DELIBERATELY *NOT* HERE
//   No device buffers, no kernels, no arena. This translation is pure host logic, which is what
//   makes it checkable without a GPU. Wiring the translated plan into the engine (page table,
//   stage-in/out transfers, re-RoPE) is a separate, later step -- see plan_missing_for_engine().
//
// PROVENANCE OF THE *INTERFACE SHAPE* (not of any code)
//   The idea of putting an engine-agnostic hook between the policy layer and the runtime comes
//   from the sibling port `kvmem-llama.cpp` (`kvmem/include/kvmem/kvmem_backend.hpp`, 29 lines,
//   four all-no-op virtuals, commented "so unit tests stay GPU-free"). That repository ships NO
//   LICENCE, so nothing was copied from it: this file is an independent implementation of the
//   *concept*, which is why the hook below looks nothing like theirs.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qw3/kvmem_store.hpp"  // the ported, byte-identical upstream header

namespace ninfer::ops::kvmem_port {

// ---------------------------------------------------------------------------------------------
// 1. Geometry
// ---------------------------------------------------------------------------------------------

// The two engines disagree about two sizes, and both disagreements are real (not cosmetic):
//
//   block_tokens : upstream defaults to 128 "independent of the 16-token physical KV page";
//                  ours IS the KV page (kPagedKVPageSize = 64, core/paged_kv_cache.h:18).
//                  A block must stay identifiable while KVMem moves the pages that hold it, so
//                  tying it to our page is the only choice that keeps remapping well-defined --
//                  but it must be CONFIGURED, never assumed. Every "top-1024 = 32K" style identity
//                  in the docs is a block-size-32 statement and becomes a 2x error at 64.
//   layers       : upstream: "blocks/selection/tiering/remap apply ONLY to standard attention
//                  layers (1/4 of layers in Qwen3.6). DeltaNet recurrent layers store O(1) state
//                  and are untouched." Ours: 64 layers = 16 full-attention + 48 GDN
//                  (qwen3_6_27b/impl/config.h:68-69), i.e. exactly 1/4 -- the ported layer's
//                  scope note holds for our model too.
struct Geometry {
    std::int32_t block_tokens          = 64;  // == kPagedKVPageSize; MUST be set from it, not typed
    std::int32_t kv_heads              = 4;
    std::int32_t head_dim              = 256;
    std::int32_t full_attention_layers = 16;  // the ONLY layers the policy layer covers
    std::int32_t gdn_layers            = 48;  // recurrent state, no token-wise KV blocks at all
};

// ---------------------------------------------------------------------------------------------
// 2. Plan translation
// ---------------------------------------------------------------------------------------------

struct Remap {
    std::uint32_t block_id = 0;
    std::uint32_t n_tokens = 0;
    std::int32_t  from_base = 0;  // current bake position (de-rotate source)
    std::int32_t  to_base = 0;    // assigned in-window first-token position
    bool          skip = false;         // resident K already baked at to_base
    bool          raw_refresh = false;  // rebuild from the unrotated raw-K mirror
};

struct Plan {
    std::vector<std::uint32_t> stage_in;
    std::vector<std::uint32_t> stage_out;
    std::vector<Remap>         remaps;  // ascending block id == window order
    std::uint32_t total_window_tokens = 0;
    std::uint32_t selection_overlap_blocks = 0;
    std::uint32_t gpu_reused_blocks = 0;
    std::uint32_t retained_position_stable = 0;
    std::uint32_t retained_position_moved = 0;
};

// Translate the ported plan into ours. `budget_tokens` is the semantic window budget the plan was
// produced under; it is needed to check the one invariant the upstream type cannot state itself.
Plan translate(const qw3::KvMemPlan& in, std::uint32_t budget_tokens);

// Invariant check, returning false + reason instead of throwing, so a caller can decide.
//
// These exist because of a bug this project already paid for: a selector branch returned an EMPTY
// block list while its own counters still reported the blocks as kept, and every "structural"
// assertion in the old oracle (<= budget, sorted, unique, in range) was satisfied by the empty
// list. The lesson was that a judge must assert the EXPECTED SET, and that a list which disagrees
// with its own accounting is a failing judge of its own. So the checks below are as much about
// internal consistency as about bounds.
bool check_plan_invariants(const Plan& plan, const Geometry& geo, std::uint32_t budget_tokens,
                           std::string* error);

// ---------------------------------------------------------------------------------------------
// 3. Config translation
// ---------------------------------------------------------------------------------------------

struct ConfigInputs {
    Geometry     geo{};
    std::uint32_t select_budget_tokens = 229376;  // upstream's official configuration
    std::uint32_t gen_budget_tokens    = 32768;   // reserve for one turn's new tokens
    // Explicit always-kept bands. -1 means "derive from the token targets"
    // (qw3::resolve_kvmem_keep_allocation); >= 0 overrides, exactly as upstream.
    std::int64_t  sink_blocks   = -1;
    std::int64_t  recent_blocks = -1;
    // We run MTP with draft=3, so the MTP raw-K mirror is live and the config must say so:
    // upstream keeps this explicit precisely so a non-MTP run does not tier an unused cache.
    bool          mtp_enabled = true;
    std::uint64_t cpu_tier_bytes = 0;  // host tier budget (0 => caller fills it in)
    std::uint64_t nvme_tier_bytes = 0; // 0 => no third tier (our decision: interface only)
};

// Build the upstream config from our values. The returned struct also carries the RESOLVED bands
// in `resolved`, because the derivation (clamp(1%*budget,1024,2048) tokens -> blocks at OUR block
// size) is the single most mis-quoted number in this port: upstream's command line says
// `--kvmem-sink-blocks 8`, which is an EXPLICIT override, and it is NOT what the formula yields.
struct ResolvedConfig {
    qw3::KvMemStoreConfig cfg;
    qw3::KvMemKeepAllocation resolved;
};

ResolvedConfig make_store_config(const ConfigInputs& in);

// ---------------------------------------------------------------------------------------------
// 4. Engine-agnostic host hooks
// ---------------------------------------------------------------------------------------------

// The ported policy layer is host-only, but the *engine* it eventually drives is not: staging a
// block means allocating a host-side staging buffer, copying device->host or host->device, and
// synchronising. Abstracting those three operations is what lets this bridge and the policy layer
// be exercised with no engine, no CUDA context and no GPU -- the property whose absence is why our
// own MeanKIndex class had never been executed before the mean-K round.
class HostHooks {
public:
    virtual ~HostHooks() = default;
    virtual void* alloc_host(std::size_t bytes) = 0;
    virtual void  free_host(void* p) = 0;
    virtual bool  copy_device_to_host(void* dst_host, const void* src_device, std::size_t bytes) = 0;
    virtual bool  copy_host_to_device(void* dst_device, const void* src_host, std::size_t bytes) = 0;
    virtual bool  copy_device_to_device(void* dst_device, const void* src_device, std::size_t bytes) = 0;
    virtual void  synchronize() = 0;
};

// ---------------------------------------------------------------------------------------------
// 5. mean-K read bridge (interface only)
// ---------------------------------------------------------------------------------------------

// The index our engine writes today stores an FP16 MEAN per (layer, block, head, dim)
// (ops/kvmem/mean_k_index.h). The sibling port stores an ordered FP32 SUM plus a token count and
// normalises ON READ (`kvmem-llama.cpp`'s raw_kv_store.hpp: "mean-K computed on read"), which is
// what lets a block be assembled across prefill boundaries without a second quantisation of an
// already-averaged vector.
//
// This bridge does NOT change either storage. It names the single semantic the scorer is allowed
// to depend on -- "give me the mean key vector of this block" -- and lets the implementation be
// either storage. Fixing the dtype in the scorer instead is how a port ends up re-quantising a
// mean and changing the retrieval RANKING, which is the only thing the index exists to get right.
class MeanKSource {
public:
    virtual ~MeanKSource() = default;
    // Fill `out` with heads*head_dim floats, head-major then dim (matching our index layout).
    virtual bool block_mean_key(std::int32_t layer, std::uint32_t block_id, float* out,
                                std::int32_t floats) const = 0;
};

// ---------------------------------------------------------------------------------------------
// 6. What is still missing before this can drive the engine
// ---------------------------------------------------------------------------------------------

struct MissingPiece {
    const char* what;
    const char* detail;
};

// Returns a static list; the caller reports it rather than discovering it at integration time.
const std::vector<MissingPiece>& plan_missing_for_engine();

}  // namespace ninfer::ops::kvmem_port
