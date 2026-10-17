// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : include\qw3\pinned_kv_tier.hpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream; only this attribution block was prepended.
// ===========================================================================
#pragma once

// CPU pinned (page-locked) KV tier — host-side slot bookkeeping.
//
// Block-sparse KV attention (see kv_block_store.hpp) keeps the whole context's
// KV resident, but a long agent session can exceed GPU memory. The CPU tier is
// a second-level cache between GPU and NVMe: cold blocks spill from GPU into a
// fixed pool of page-locked host slots (one slot holds one block's KV across
// all attention layers), and a selected block stages back to GPU on demand.
//
// This module is PURE HOST LOGIC: a fixed-count slab allocator (LIFO free list,
// mirroring GlobalKvPagePool) plus a block_id <-> slot map. It owns NO memory
// and issues NO copies — the CUDA backend supplies the cudaHostAlloc'd buffer
// and performs the actual D2H/H2D over copy_stream_. Keeping it host-only makes
// the eviction/placement math unit-testable without a GPU (see
// tests/pinned_kv_tier_test.cpp).
//
// Slot sizing is the caller's contract: slot_bytes must be >= the byte size of
// one block's KV summed across all standard-attention layers. The pool only
// tracks slot indices; byte offsets are slot_index * slot_bytes.

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qw3 {

enum class PinnedKvCachePolicy : uint8_t {
    LegacyLru = 0,
    HeatAware = 1,
};

struct PinnedKvTierConfig {
    uint64_t total_bytes = 0;  // --kv-store-cpu-bytes (0 = tier disabled)
    uint64_t slot_bytes = 0;   // bytes per block KV across all attention layers
    PinnedKvCachePolicy cache_policy = PinnedKvCachePolicy::LegacyLru;
    // A block's semantic-selection frequency halves after this many
    // reselection epochs without reuse. Zero disables decay.
    uint32_t heat_half_life_epochs = 8;
};

// Result of placing a block into the CPU tier. When the pool is full and a
// victim must be evicted to NVMe (or dropped), `evicted_block` names it
// (-1 = none evicted) so the caller can spill it onward before reusing its
// slot. v1 skeleton never evicts (returns slot=-1 when full); the eviction
// hook is reserved for the NVMe tier task.
struct PinnedSlotPlacement {
    int32_t slot = -1;            // assigned host slot index (-1 = pool full)
    int32_t evicted_block = -1;   // block displaced to make room (-1 = none)
};

struct PinnedKvCacheStats {
    uint64_t selection_epochs = 0;
    uint64_t admissions = 0;
    uint64_t admission_rejections = 0;
    uint64_t evictions = 0;
};

class PinnedKvTier {
public:
    explicit PinnedKvTier(PinnedKvTierConfig cfg) : cfg_(cfg) {
        if (cfg_.slot_bytes > 0 && cfg_.total_bytes >= cfg_.slot_bytes) {
            slot_count_ = static_cast<uint32_t>(cfg_.total_bytes / cfg_.slot_bytes);
        }
        free_slots_.reserve(slot_count_);
        // LIFO free list: hand out low indices first (push high..low).
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
    }

    const PinnedKvTierConfig &config() const { return cfg_; }
    bool enabled() const { return slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint32_t free_slots() const { return static_cast<uint32_t>(free_slots_.size()); }
    uint32_t used_slots() const { return slot_count_ - free_slots(); }
    PinnedKvCachePolicy cache_policy() const { return cfg_.cache_policy; }
    const PinnedKvCacheStats &stats() const { return stats_; }

    // Byte offset of a slot within the pinned buffer.
    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }

    // Is this block currently resident in the CPU tier? Returns its slot or -1.
    int32_t block_slot(uint32_t block_id) const {
        auto it = block_to_slot_.find(block_id);
        return it == block_to_slot_.end() ? -1 : it->second;
    }

    // Reserve a slot for `block_id` (the caller then D2H-copies the block's KV
    // into slot_offset(slot)). If the block is already resident its existing
    // slot is returned unchanged. When the pool is full, v1 returns slot=-1
    // (no eviction); the caller must keep the block on GPU. Marks the block
    // resident on success.
    PinnedSlotPlacement place_block(uint32_t block_id) {
        PinnedSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            return out;
        }
        if (free_slots_.empty()) return out;  // full: slot stays -1
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        block_to_slot_[block_id] = slot;
        touch(block_id);
        ++stats_.admissions;
        out.slot = slot;
        return out;
    }

    // Reserve a slot and evict the LRU resident block if the pool is full.
    // The caller must spill `evicted_block` onward (e.g. to NVMe) before
    // overwriting the returned slot's bytes. HeatAware mode first performs a
    // TinyLFU-style admission comparison: a colder streaming candidate bypasses
    // CPU and leaves a hotter resident in place.
    PinnedSlotPlacement place_block_evicting(uint32_t block_id) {
        PinnedSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch(block_id);
            return out;
        }
        if (!free_slots_.empty()) return place_block(block_id);
        return replace_lru(block_id);
    }

    // Reuse an already allocated resident slot even when the bookkeeping tier
    // still has unused logical slots. Immutable K shares one CPU budget with
    // demand-allocated raw-K chunks, so allocation can hit the byte budget long
    // before all logical CPU slots are materialized. This operation lets a hot
    // candidate replace a cold resident without allocating another host buffer.
    PinnedSlotPlacement place_block_replacing(uint32_t block_id) {
        PinnedSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch(block_id);
            return out;
        }
        return replace_lru(block_id);
    }

private:
    PinnedSlotPlacement replace_lru(uint32_t block_id) {
        PinnedSlotPlacement out;
        if (lru_.empty()) return out;
        const uint32_t victim = lru_.front();
        if (cfg_.cache_policy == PinnedKvCachePolicy::HeatAware &&
            effective_heat(block_id) < effective_heat(victim)) {
            ++stats_.admission_rejections;
            return out;
        }
        auto vit = block_to_slot_.find(victim);
        if (vit == block_to_slot_.end()) {
            erase_lru(victim);
            return out;
        }
        const int32_t slot = vit->second;
        block_to_slot_.erase(vit);
        erase_lru(victim);
        block_to_slot_[block_id] = slot;
        touch(block_id);
        ++stats_.admissions;
        ++stats_.evictions;
        out.slot = slot;
        out.evicted_block = static_cast<int32_t>(victim);
        return out;
    }

public:
    // Pressure-window movement must not call these methods: sequential ingest
    // should not make every streamed block appear hot.
    void begin_selection_epoch() {
        ++selection_epoch_;
        ++stats_.selection_epochs;
    }

    // Record one block selected by semantic retrieval. The metadata survives
    // CPU->GPU movement so a repeatedly selected block can win admission when
    // it is later staged out again.
    void record_selected(uint32_t block_id) {
        Heat &h = heat_[block_id];
        decay_to_current(h);
        h.consecutive =
            h.last_epoch + 1 == selection_epoch_ ? h.consecutive + 1 : 1;
        h.frequency += 1.0;
        h.last_epoch = selection_epoch_;
        if (block_slot(block_id) >= 0) touch(block_id);
    }

    // Release a block's slot back to the pool (it was staged back to GPU, or
    // the session ended). No-op if the block is not resident.
    void release_block(uint32_t block_id) {
        auto it = block_to_slot_.find(block_id);
        if (it == block_to_slot_.end()) return;
        free_slots_.push_back(it->second);
        block_to_slot_.erase(it);
        erase_lru(block_id);
    }

    // Mark a block most-recently-used (called on stage-in / access) so the LRU
    // victim picked by evict_lru_victim() is the coldest resident block.
    void touch(uint32_t block_id) {
        erase_lru(block_id);
        lru_.push_back(block_id);
    }

    // Pick (without removing) the least-recently-used resident block, for the
    // NVMe spill path to displace. Returns -1 when the tier is empty.
    int32_t lru_victim() const {
        return lru_.empty() ? -1 : static_cast<int32_t>(lru_.front());
    }

    void clear() {
        free_slots_.clear();
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
        block_to_slot_.clear();
        lru_.clear();
        heat_.clear();
        selection_epoch_ = 0;
        stats_ = PinnedKvCacheStats{};
    }

private:
    struct Heat {
        double frequency = 0.0;
        uint64_t last_epoch = 0;
        uint32_t consecutive = 0;
    };

    void decay_to_current(Heat &h) const {
        if (h.frequency <= 0.0 || selection_epoch_ <= h.last_epoch ||
            cfg_.heat_half_life_epochs == 0) {
            return;
        }
        const double elapsed =
            static_cast<double>(selection_epoch_ - h.last_epoch);
        const double halves =
            elapsed / static_cast<double>(cfg_.heat_half_life_epochs);
        h.frequency *= std::exp2(-halves);
        if (h.frequency < 0.01) h.frequency = 0.0;
    }

    double effective_heat(uint32_t block_id) const {
        const auto it = heat_.find(block_id);
        if (it == heat_.end()) return 0.0;
        Heat h = it->second;
        decay_to_current(h);
        // Consecutive turn reuse predicts near-term working-set oscillation
        // better than lifetime frequency alone.
        return h.frequency + 0.25 * static_cast<double>(h.consecutive);
    }

    void erase_lru(uint32_t block_id) {
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            if (*it == block_id) { lru_.erase(it); return; }
        }
    }

    PinnedKvTierConfig cfg_;
    uint32_t slot_count_ = 0;
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;  // front = least recently used
    std::unordered_map<uint32_t, Heat> heat_;
    uint64_t selection_epoch_ = 0;
    PinnedKvCacheStats stats_;
};

} // namespace qw3
