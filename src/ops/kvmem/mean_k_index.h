#pragma once

// KVMem mean-K index (Eq.9): the FIRST consumer of the raw-K harvest.
//
// WHAT THIS IS
//   The harvest (ops/kvmem/raw_k_harvest.h) brings each layer's PRE-RoPE key to a device staging
//   slot. This component reduces those keys into the retrieval index KVMem scores against:
//
//     sumK[layer, block, head, dim] = SUM over the tokens of that block of K_raw[token, head, dim]
//     count[layer, block]           = the number of tokens that sum REALLY covers
//     meanK                         = sumK / count            <- derived at READ time, never stored
//
//   i.e. one vector per (layer, logical block, KV head). Blocks are LOGICAL blocks the size of one
//   KV page (P = 64, core/paged_kv_cache.h:18), so a block's identity is stable across the paging
//   that KVMem later performs on it. Rows are the page's position in the sequence, not a physical
//   page index: the index describes the HISTORY, and the page table decides where that history
//   currently lives.
//
// WHY IT MUST BE INCREMENTAL (this is a correctness rule, not a performance preference)
//   kvmem-qw3 docs/kvmem_implementation_notes.md:395-409 records that the index is built during the
//   prefill, per chunk, while the keys are still in hand, and explicitly rejects the alternative of
//   scanning the paged KV after the prefill: with offloading, a block evicted mid-prefill would
//   never be reachable by a later scan, so the index would silently lose history. This component
//   therefore appends one chunk's worth of blocks per call and keeps a running block cursor; it has
//   no "rebuild from the cache" entry point on purpose.
//
// STORAGE: F32 SUMS + A TOKEN COUNT, MEAN AT READ (2026-09-21 -- THIS REPLACED A STORED FP16 MEAN)
//   Earlier this component divided by the block's fill and stored an FP16 MEAN. It now accumulates an
//   F32 SUM per (block, head, dim) and keeps a per-(layer, block) F32 COUNT; the mean is `sum / count`
//   on the read path. Three reasons, in order of importance:
//     1. A PARTIAL TAIL BLOCK CAN BE COMPLETED. The official store allows a block with fewer tokens
//        than the nominal size and grows it later (kvmem-qw3 src/kvmem_store.cpp register_append /
//        truncate_to). With a STORED mean, filling such a block afterwards kept the OLD mean -- the
//        tokens added later never entered the index and nothing reported it. With a sum, completing a
//        block is pure addition (`sum += new`, `count += n`), so the stale-mean state does not exist.
//     2. NO SECOND QUANTISATION. A stored FP16 mean rounds once and every later use inherits that
//        rounding. The official implementation requires FP16 for the production index but REVERTED an
//        FP8 index experiment precisely because re-quantising an already-averaged vector MOVES THE
//        RETRIEVAL RANKING -- the one thing the index exists to get right. Accumulation here is F32
//        throughout and the single narrowing happens at read, where the ranking decision is made.
//     3. IT MATCHES THE OFFICIAL CONTRACT: "ordered K sum (F32) at first write, mean-K computed on
//        read" (kvmem-llama.cpp kvmem/include/kvmem/raw_kv_store.hpp:3-8 -- read for the APPROACH
//        only: that repository carries NO licence, so its code must not be copied).
//
//   REGISTERED DEVIATIONS (both are consequences of the above, both are deliberate):
//     * DEVIATION 1 -- the stored medium is no longer FP16, so the index costs twice the bytes. See
//       the accounting block below. The FP16 requirement applies to what the SCORER consumes, and the
//       read path is where that narrowing now happens (exactly once per read).
//     * DEVIATION 2 -- summation order is now part of the contract: within a block the terms are added
//       by one lane in token order (<= 64 F32 adds), and across rounds the per-round partial sums are
//       added into the stored sum. That is MORE stable than the old path, which re-derived the mean
//       from scratch, and it is what makes completion-by-addition exact.
//   NOT YET MEASURED (registered, do it once the scorer is wired): whether the retrieval RANKING
//   produced from these sums matches the official MeanK (content frame, dot-product convention,
//   softmax-over-pages statistics domain). The storage change cannot be declared ranking-neutral on
//   arithmetic alone.
//
// LAYOUT AND COST (geometry: qwen3_6_27b/impl/config.h:31-33 -> kv_heads=4, head_dim=256)
//   Per layer, ONE F32 region: [ sums: blocks*kv_heads*head_dim ][ counts: blocks ], head-major then
//   dim (ne[0] contiguous), so one (block, head) row of 256 floats is a contiguous, coalesced store
//   and the count row follows the sums inside the same allocation (the layout cannot drift between
//   host and device because both live in one buffer).
//   Per layer, per block:  kv_heads(4) * head_dim(256) * 4 B = 4 KiB of sums, + 4 B of count.
//   Per layer, per token:  4 KiB / P(64) = 64 B/token.
//   Per token:             full_attention_layers(16) * 64 B = 1024 B/token of sums, + 16 * 4/64 = 1
//                          B/token of counts  => 1025 B/token ~= 1.00 KB/token.
//   The FP16-mean revision this replaced was 512 B/token, so the index DOUBLES. Stated against the
//   whole resident account: 16.4 KB/token (fp8 active KV) + 32 KB/token (raw-K shadow, fp16) + 1.0
//   KB/token (this index) = 49.4 KB/token, up from 48.9.
//   NOTE the widely quoted "~1 KB/token" is the official figure at BLOCK SIZE 32; the cost is exactly
//   LINEAR in tokens and inversely linear in block size, so at our P=64 the sums alone reproduce that
//   figure and the FP16 version was half of it. Cite the block size with the number.
//   (The comment history here has two earlier mistakes worth keeping: "16 B/token" was the PER-LAYER
//   number, and a reviewer then read the per-layer 32 KiB/chunk as the total and called it a 32x
//   error. The rule that prevents both: always say per LAYER or across-all-layers explicitly.)
//
// ZEROING (an obligation the `+=` design creates, and why it is NOT folded into append)
//   Appends ACCUMULATE, so a block's slots must be zero before their first write; stale bytes are
//   silently folded into the mean rather than detected. `zero(stream)` fills the whole index with
//   zeros on a caller-supplied stream and resets the block cursor, so a driver calls it once before
//   the first append (arena memory is not guaranteed to be zeroed) and again before reusing a live
//   index for a different sequence.
//   It is deliberately NOT done inside the append: an append that zeroed its own fresh blocks would
//   be replayed verbatim inside a captured decode graph, and a replayed memset would wipe the sums the
//   PREFILL accumulated -- the index is state that outlives the graph. Zeroing is a setup step, not a
//   per-append step. The cost of that choice is one caller obligation, which is why the standalone
//   verifier carries a negative control showing the judge CAN see an un-zeroed destination.
//
// OWNERSHIP
//   Same rule as the harvest: index slots are owned for the lifetime of the engine (they outlive
//   every TextContext, and a captured graph must never point at recycled workspace). Allocation
//   happens in the constructor, which therefore must not run inside a capture.

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/kvmem/raw_k_harvest.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Inclusive cap on indexed layers, matching the harvest's own slot array. A model with more than 64
// full-attention layers would need this raised in both places together.
inline constexpr std::size_t kMeanKMaxLayers = 64;

// Bytes of the stored index for `layers` x `blocks` blocks: F32 sums + one F32 count per (layer,
// block). Pure arithmetic, no switch and no side effects: the two A/B arms must declare the same arena
// capacity so that switching the feature on cannot move any other tensor's pointer (the discipline
// documented in ops/linear/ternary/ternary_rotation.cpp:45-64).
[[nodiscard]] inline constexpr std::size_t mean_k_index_bytes(std::int32_t layers,
                                                             std::int32_t blocks,
                                                             std::int32_t kv_heads,
                                                             std::int32_t head_dim) {
    if (layers <= 0 || blocks <= 0 || kv_heads <= 0 || head_dim <= 0) { return 0; }
    const std::size_t sums = static_cast<std::size_t>(layers) * static_cast<std::size_t>(blocks) *
                             static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(head_dim) *
                             sizeof(float);
    const std::size_t counts =
        static_cast<std::size_t>(layers) * static_cast<std::size_t>(blocks) * sizeof(float);
    return sums + counts;
}

// The accumulated index for one sequence: all layers, grown block by block as the prefill advances.
class MeanKIndex {
public:
    // `layers` full-attention layers, `heads` KV heads, `head_dim` per head, and room for
    // `capacity_blocks` logical blocks (the caller sizes this from the workspace it is willing to
    // spend, not from the model's context ceiling).
    MeanKIndex(std::int32_t layers, std::int32_t heads, std::int32_t head_dim,
               std::int32_t capacity_blocks);

    MeanKIndex(const MeanKIndex&)            = delete;
    MeanKIndex& operator=(const MeanKIndex&) = delete;
    MeanKIndex(MeanKIndex&&)                 = delete;
    MeanKIndex& operator=(MeanKIndex&&)      = delete;
    ~MeanKIndex()                            = default;

    // Append one chunk. `store` is the harvest holding the round to consume; only the layers it
    // reports as harvested are read, and only `round_width` tokens of each. `first_block` is the
    // logical block number this chunk starts at, so the caller owns the mapping from a chunk to its
    // place in the sequence and this component never has to guess it.
    //
    // CALLING TWICE WITH THE SAME `first_block` CONTINUES THAT BLOCK, which is how a partial tail
    // block gets completed: the sums and the count are added to, never replaced. Nothing has to be
    // re-read, and the block is usable (as a mean over the tokens it holds SO FAR) at any point.
    //
    // Capture-safe: the caller supplies the stream and the buffers already exist, so nothing here
    // allocates or synchronises. Issued unconditionally by the driver; the switch is read on the
    // host in the driver, and a null/absent harvest means there is nothing to append.
    void append_round(const RawKShadowHarvest& store, std::int32_t first_block,
                      cudaStream_t stream);

    // Same, for a caller that has one layer's staging in hand (the unit-test path, and the shape a
    // future per-layer consumer wants). `layer_index` is the FULL-ATTENTION index (0..layers-1),
    // never the global layer number (text_context.h:341-342).
    void append_layer(const Tensor& raw_key, std::int32_t layer_index, std::int32_t round_width,
                      std::int32_t first_block, cudaStream_t stream);

    // Zero the whole index. REQUIRED before the first append (arena memory is not guaranteed zero) and
    // before any reuse of a live index for a different sequence. Returns the memset status instead of
    // throwing, matching the launch entry point: a caller in a capture must not get an exception from a
    // stream-ordered operation.
    [[nodiscard]] cudaError_t zero(cudaStream_t stream);

    [[nodiscard]] std::int32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::int32_t heads() const noexcept { return heads_; }
    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] std::int32_t capacity_blocks() const noexcept { return capacity_blocks_; }

    // Per-layer F32 SUM tensor, [blocks, heads, head_dim], or an invalid tensor when the index is
    // empty. THE MEAN IS THE CALLER'S DIVISION: `mean = sums[b, h, d] / counts[b]`, and the single
    // narrowing to FP16 belongs on that read path. `counts[b] == 0` means the block holds nothing --
    // there is NO signal for it, and the scorer's convention for "no signal" is -inf (see
    // kvmem_select.h), not 0.
    [[nodiscard]] const Tensor& layer_sums(std::int32_t layer_index) const noexcept;
    // Per-layer F32 token count, [blocks]. The count is a property of the BLOCK (not of a head), and
    // it is the REAL fill: a partial tail block counts only the tokens it actually holds.
    [[nodiscard]] const Tensor& layer_counts(std::int32_t layer_index) const noexcept;

    [[nodiscard]] std::size_t bytes_per_layer() const noexcept { return per_layer_bytes_; }
    [[nodiscard]] std::size_t bytes_total() const noexcept { return total_bytes_; }

    // Blocks appended so far, and the fill (0..block_tokens) of the last one. The fill is kept
    // because the final chunk of a prompt is usually SHORT: the first cut measured a 511-token final
    // chunk on a 2,273-token prompt, and a tail block's mean must be divided by the tokens it really
    // holds, never by the nominal block size. This is the same defect that staging_for_round() /
    // set_round_width() were added to fix for the harvest.
    [[nodiscard]] std::int32_t blocks_written() const noexcept { return blocks_written_; }
    [[nodiscard]] std::int32_t tail_fill() const noexcept { return tail_fill_; }

    // True when there is no index at all (the switch is off, or the constructor was given a zero
    // capacity). Callers must treat "no index" as a normal state, not an error.
    [[nodiscard]] bool empty() const noexcept { return total_bytes_ == 0; }

private:
    // `source_layer_stride` is the element distance between consecutive layers inside the SOURCE
    // allocation (the harvest packs every layer into one buffer at its configured width, so this is
    // not the round width), while `index_layer_stride` is the F32 element count of one layer's
    // destination region (sums + counts).
    void append_one(const void* raw_key_bf16, std::int32_t layer_index, std::int32_t round_width,
                    std::int32_t first_block, std::int32_t source_layer_stride,
                    std::int32_t index_layer_stride, cudaStream_t stream);

    std::int32_t layers_          = 0;
    std::int32_t heads_           = 0;
    std::int32_t head_dim_        = 0;
    std::int32_t capacity_blocks_ = 0;
    // Declared BEFORE the arena on purpose: DeviceArena has no default constructor (arena.h:64-66),
    // so it must be initialised from a byte count, and that count has to be computed from members
    // that are already initialised. Member initialisers run in DECLARATION order, which is why the
    // harvest's size fields read first too (raw_k_harvest.cpp:48-51).
    std::size_t  sums_per_layer_     = 0;  // F32 elements of sums in one layer
    std::size_t  elements_per_layer_ = 0;  // F32 elements per layer, sums + counts
    std::size_t  per_layer_bytes_    = 0;
    std::size_t  total_bytes_        = 0;
    DeviceArena  arena_;
    Tensor       flat_;  // F32 [elements_per_layer, layers] view of the arena
    std::array<Tensor, kMeanKMaxLayers> views_{};   // per-layer [blocks, heads, head_dim] sums
    std::array<Tensor, kMeanKMaxLayers> counts_{};  // per-layer [blocks] token counts
    std::int32_t blocks_written_ = 0;
    std::int32_t tail_fill_      = 0;
};

// The one-line way to size an index for a harvest: one chunk produces exactly
// ceil(chunk_tokens / block_tokens) blocks, and KVMem's whole premise is that the index describes a
// workspace far larger than the resident window -- so the caller passes the block capacity it is
// willing to spend, not the chunk's count.
[[nodiscard]] inline constexpr std::int32_t mean_k_blocks_for_tokens(std::int32_t tokens,
                                                                    std::int32_t block_tokens) {
    if (tokens <= 0 || block_tokens <= 0) { return 0; }
    return (tokens + block_tokens - 1) / block_tokens;
}

// The per-sequence index this engine actually drives, installed on first use and kept for the
// process lifetime. Same shape and same contract as raw_k_shadow_harvest_for(): the caller asks for
// it every chunk, gets the SAME object back, and gets nullptr when NINFER_TERNARY_KVMEM is off (the
// default) so the OFF arm allocates nothing and stays bit-for-bit the engine as it is today.
//
// WHY A PROCESS-LIFETIME SLOT AND NOT A WORKSPACE TENSOR
//   Same reason as the harvest: the index outlives every TextContext (which is constructed on the
//   stack for one schedule recording, text_context.h:29) and a captured graph must never end up
//   pointing at recycled arena memory. The first call also performs the zero() the class requires
//   before any append, because arena memory is not guaranteed zero and a non-zero count would let an
//   unwritten block look indexed.
//
// KNOWN LIMITATION, REGISTERED: one index for the whole process means two concurrent sequences share
//   it. V1 drives a single sequence, so this is correct today; before --max-concurrency > 1 is used
//   with KVMem on, this must become one index per lane and the arena account must grow by
//   N x mean_k_index_bytes(layers, blocks, kv_heads, head_dim).
//
// `capacity_blocks` is honoured on the FIRST call only, and it fixes the allocation for the process.
[[nodiscard]] MeanKIndex* mean_k_index_for(std::int32_t layers, std::int32_t heads,
                                           std::int32_t head_dim, std::int32_t capacity_blocks);

} // namespace ninfer::ops
