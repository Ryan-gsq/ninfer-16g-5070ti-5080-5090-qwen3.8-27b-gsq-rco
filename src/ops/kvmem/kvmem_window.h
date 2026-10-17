#pragma once

// KVMem window mode: the per-lane carrier that says "this lane attends to a COMPACT window, not to
// its whole ledger prefix", and the one place where that decision becomes a cache position, a query
// RoPE position and an attention envelope.
//
// GEOMETRY -- READ THIS BEFORE CHANGING AN ACCESSOR.
//
// TERMINOLOGY. Every accessor below takes `sequence_frontier`, which is what the caller reads out of
// SequenceState::execution_frontier: the index of the last K/V slot the ledger has written, i.e. the
// value today's ingress uses as the cache position and the value today's envelope is built from as
// sequence_frontier + 1. It is deliberately NOT SequenceState::ledger_frontier, which is that value
// plus one.
//
// The compact window is a SEQUENCE of W tokens. Its K/V were re-baked by stage 3 (ops/kvmem/rerope/
// kvmem_rerope.h) to absolute RoPE positions 0..W-1 inside the engine's own fp8/H codec, and the
// token being decoded right now is appended at slot W. So for an armed lane with window frontier W:
//
//   * cache position  = W. That is both the K/V write slot and the slot attention reads last; the
//                       sequence frontier (1,000,000, say) addresses a page row that the window arm
//                       never allocated.
//   * query RoPE      = W, and NOT sequence_frontier + rope_delta. The window's absolute frame is
//                       re-baked from 0, so the true sequence position and any rope_delta inherited
//                       from a shared prefix are already folded into that re-bake; adding either
//                       again would put the query that many positions away from its own tail
//                       (distance 1 becomes 1e6). arm() therefore rejects a non-zero rope_delta, and
//                       the accessor below is allowed to ignore it precisely because that check ran.
//   * envelope        = {W + 1, W + 1}. The engine's visible key set is a contiguous prefix and
//                       validate_envelope() (causal_softmax_attention.cpp:175-186) wants a 1-based
//                       visible-key COUNT with min == 0 rejected, so "the window plus the new token"
//                       is W + 1 visible keys -- the same expression as today's frontier + 1, with the
//                       effective frontier in place of the ledger one.
//   * KV extent       = W + 1 slots, for the same reason: the arm writes only slots <= W.
//
// WHAT THIS IS NOT. The envelope carries no mask, so this rung cannot express "keep the sink blocks
// and drop a hole in the middle" on its own: the hole must already be gone from the page table,
// which is what the stage-2 assembly path (logical_kv_store.h assemble_window()) is for. With
// W == the sequence frontier -- window mode armed but not yet compacting anything -- every accessor
// below returns exactly what the engine returns today, which is the safe first end-to-end run.
//
// WHY A PER-LANE CARRIER AND NOT JUST A SWITCH. A batch can mix lanes, and a lane that never opted
// in must stay bit-identical to today while its neighbours compact; a global switch could not
// express that, and a per-lane field also survives the sequence's whole lifetime (prefill, replay,
// rebuild) instead of being re-derived at every round.
//
// OFF ARM == THE ENGINE AS IT IS TODAY, BY CONSTRUCTION. The default carrier is inert
// (window_frontier_ < 0), and every accessor then returns the caller's own ledger expression
// unchanged, so the off arm cannot drift from the current numerics. The switch is read ONCE, at
// arm() time, never inside an accessor and never inside a kernel (same discipline as
// kvmem_shadow.h:54-61): the position choice is host-side ingress data, so there is no device
// branch to keep in sync.

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// NINFER_TERNARY_KVMEM_WINDOW=1 opts a lane into window mode. Default OFF, opt-in polarity
// (`value != nullptr && std::string(value) == "1"`), matching kvmem_shadow_enabled(). Unlike that
// switch this one is not read by any kernel: it only decides whether arm() may arm a lane, so a
// stray environment variable can produce at most an explicit failure, never a silent numerical
// change.
[[nodiscard]] inline bool kvmem_window_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_WINDOW");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

// The inert value: "this lane is not in window mode". Any legal compact frontier is >= 0 (a window
// of zero tokens is not a window), so a negative sentinel cannot collide with a real one.
inline constexpr std::int32_t kKvmemWindowNotArmed = -1;

struct KvmemWindowFrontier {
    [[nodiscard]] constexpr bool active() const noexcept { return window_frontier_ >= 0; }

    // Whether this arm actually COMPACTED anything: W < the sequence frontier it was armed against.
    //
    // A lane armed with W == its sequence frontier is "armed but not compacting yet" -- the geometry
    // note above spells out that every accessor is then an identity on the engine's own value. That
    // distinction is load-bearing for any caller that must refuse a lineage the assembly rewrote:
    // `active()` is TRUE for an inert arm too, so a guard built on it refuses a lane that carries no
    // compaction at all. Measured cost of that over-wide reading: the identity arm the server's own
    // warmup leaves behind made `start_sequence` refuse EVERY later request on the lane
    // ("cannot be re-admitted", HTTP 500) -- 2026-09-24, exp\mem-stage2 cells X-A/X-B/X-C.
    //
    // Consumers that refuse a compacted lineage ask THIS; consumers that only need "which frontier do
    // I address" keep using active()/frontier().
    [[nodiscard]] constexpr bool compacting() const noexcept { return compacting_; }

    // The stored compact frontier as int32, or kKvmemWindowNotArmed. Diagnostic/ledger use only:
    // call sites should go through frontier() so that the off arm and the on arm share one
    // expression.
    [[nodiscard]] constexpr std::int32_t window_frontier() const noexcept { return window_frontier_; }

    // The frontier every host path must use for the cache position, the envelope and the KV extent.
    // Off arm: the sequence frontier (execution_frontier) verbatim.
    [[nodiscard]] constexpr std::uint32_t frontier(std::uint32_t sequence_frontier) const noexcept {
        return active() ? static_cast<std::uint32_t>(window_frontier_) : sequence_frontier;
    }

    // The query token's absolute RoPE position. Off arm: sequence_frontier + rope_delta, exactly as
    // today. On arm: the compact frontier, with rope_delta guaranteed zero by arm().
    [[nodiscard]] constexpr std::int32_t rope_position(std::uint32_t sequence_frontier,
                                                       std::int32_t rope_delta) const noexcept {
        return active() ? window_frontier_
                        : static_cast<std::int32_t>(sequence_frontier) + rope_delta;
    }

    // Arm this lane, or throw. Every precondition is checked HERE, once, so that the accessors stay
    // branch-light and a misconfiguration fails loudly instead of decoding garbage:
    //   * the switch must be on -- arming is otherwise unreachable and a silent arm would turn an
    //     experiment into production numerics;
    //   * 0 <= window_frontier <= sequence_frontier -- the window cannot be larger than the history
    //     it is compacted from, and equality is the legal "armed but not compacting yet" state;
    //   * rope_delta == 0 -- see the geometry note above; a shared-prefix or multimodal offset is
    //     incompatible with a frame that stage 3 re-baked from 0.
    void arm(std::int32_t window_frontier, std::uint32_t sequence_frontier,
             std::int32_t rope_delta) {
        if (!kvmem_window_enabled()) {
            throw std::logic_error(
                "KVMem window mode was armed without NINFER_TERNARY_KVMEM_WINDOW=1");
        }
        if (window_frontier < 0 ||
            static_cast<std::uint32_t>(window_frontier) > sequence_frontier) {
            throw std::logic_error("KVMem window frontier must be within [0, sequence frontier]");
        }
        if (rope_delta != 0) {
            throw std::logic_error(
                "KVMem window mode requires rope_delta == 0 (the window frame is re-baked from 0)");
        }
        window_frontier_ = window_frontier;
        compacting_      = static_cast<std::uint32_t>(window_frontier) < sequence_frontier;
    }

    void disarm() noexcept {
        window_frontier_ = kKvmemWindowNotArmed;
        compacting_      = false;
    }

private:
    std::int32_t window_frontier_ = kKvmemWindowNotArmed;
    bool         compacting_      = false;
};

} // namespace ninfer::ops::detail
