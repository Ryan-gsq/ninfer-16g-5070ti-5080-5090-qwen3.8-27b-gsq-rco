#pragma once

// KVMem raw-K shadow: per-layer staging + the drain driver (segment 2 of the first cut).
//
// WHY THIS EXISTS
//   Segment 1 delivered the op (ops/kvmem/raw_k_shadow.h) but one buffer is not a harvest: the main
//   path runs FULL_ATTENTION_LAYERS() = 16 full-attention layers (config.h:68 asserts 16), and the
//   staging they write has to survive until the host can drain it. With a single buffer each layer
//   would overwrite the previous one and only the LAST layer's raw K would remain, which is not a
//   usable history. This class owns one device slot and one pinned host slot PER LAYER.
//
// WHERE EACH PIECE LIVES (measured, not assumed)
//   * PREFILL runs EAGER: text_context_impl.h:1027 calls attn_mix() per layer, and the driver loop
//     in text_prefill_impl.h:139-151 resets the workspace around each chunk. There is NO graph
//     replay to hang a drain on, so the drain happens per CHUNK, driven by the runtime.
//   * DECODE / VERIFY replay a captured graph through schedule::run_prepared() (graph_impl.h:11-21,
//     the only cudaGraphLaunch is core/decode_graph.cpp:144). A graph body cannot synchronise or
//     allocate, which is exactly why raw_k_shadow_copy is capture-safe and raw_k_shadow_drain
//     refuses while the stream is capturing. The drain therefore has to be issued by the host after
//     the replay, never from inside the captured body.
//
// ALLOCATION CONTRACT
//   Construction performs a cudaMalloc-equivalent (DeviceArena) and a pinned host allocation, so it
//   must happen while the graph is being PREPARED, never inside a capture -- the same rule
//   RawKShadowHostStore documents in raw_k_shadow.h:88-91. The buffers are owned for the lifetime
//   of the engine, so their device addresses are stable across graph capture and replay; a buffer
//   taken from the per-call workspace arena would instead be recycled (see the scope() rollbacks in
//   text_context_impl.h:1026 and causal_softmax_attention.cpp:469) and the captured graph would end
//   up pointing at memory another tensor now owns.
//
// SIZE
//   `tokens` is chosen by the runtime as max(effective_prefill_chunk, 1). It is deliberately NOT
//   derived from kCausalAttentionMaximumVisibleKeys (262144): sizing for the ceiling would reserve
//   16 x 512 MiB = 8 GiB of device memory, while the real prefill chunk is 1024
//   (layouts_impl.h:102-103 takes min(prefill_chunk, capacity); the shipping ball passes
//   --prefill-chunk 1024), i.e. 16 x 2 MiB = 32 MiB. A copy whose token count exceeds the reserved
//   size throws rather than overflowing.

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/kvmem/raw_k_shadow.h"
// 装配臂（expmem-arm）"路 1"：按块留存 raw-K 的 arena。放在这里 include，是为了让**每一个认识
// harvest 的 TU**（raw_k_harvest.cpp / text_prefill_impl.h / program_impl.h 链路）都同时看到 arena
// 定义——否则"重烘该从哪取料"会有一半 TU 看不见。
#include "ops/kvmem/raw_k_block_arena.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Per-layer raw-K staging and its pinned host mirror. One instance per sequence.
class RawKShadowHarvest {
public:
    // `layers` full-attention layers, each staging `tokens` tokens of `kv_size` BF16 values.
    RawKShadowHarvest(std::int32_t layers, std::int32_t kv_size, std::int32_t tokens);

    RawKShadowHarvest(const RawKShadowHarvest&)            = delete;
    RawKShadowHarvest& operator=(const RawKShadowHarvest&) = delete;
    RawKShadowHarvest(RawKShadowHarvest&&)                 = delete;
    RawKShadowHarvest& operator=(RawKShadowHarvest&&)      = delete;
    ~RawKShadowHarvest()                                   = default;

    // Device staging for one full-attention layer index. Always valid (the slots are allocated
    // unconditionally, independently of NINFER_TERNARY_KVMEM): the two A/B arms must share one
    // workspace PLAN, and a buffer that only existed in one arm would move every other pointer.
    [[nodiscard]] Tensor& staging(std::int32_t layer_index) noexcept;

    // The staging tensor reshaped for ONE chunk: {kv_size, round_tokens}. The slots are allocated
    // for the widest chunk the runtime promised (`tokens`, normally the effective prefill chunk),
    // but the LAST chunk of a prompt is usually SHORT -- measured on a 2,273-token prompt at
    // --context 512 the final chunk held 511 tokens, not 512 -- and raw_k_shadow_copy validates the
    // declared geometry, so handing it the full-width view would throw on every final chunk.
    // Callers pass this view to the op and then set_round_width() to record the width actually used.
    [[nodiscard]] Tensor staging_for_round(std::int32_t layer_index) const;

    // Records the width written by the round in progress. drain_all() and dump_to_file() transfer
    // ONLY these bytes per layer, so a short final chunk cannot publish the tail of an earlier,
    // longer chunk as if it belonged to this one.
    void set_round_width(std::int32_t width) noexcept;
    [[nodiscard]] std::int32_t round_width() const noexcept { return round_width_; }

    // ★ 装配臂（exp\mem-arm）：本轮的**首个 token 的绝对位置**。为什么必须记：本对象是进程级单例
    // （raw_k_harvest.h:186-208），而设备 staging 每轮被覆盖，所以"我手里的 raw-K 对应哪一段 token"
    // 只能由这里回答。只记宽度的话，重烘会把别的轮的字节当历史——静默错答案。
    // 取值点：text_prefill_impl.h 安装 harvest 之后（state.text_kv_base）。
    void set_round_origin(std::int32_t first_token) noexcept;
    [[nodiscard]] std::int32_t round_first_token() const noexcept { return round_first_token_; }

    // 轮次戳：begin_round() 每轮 +1，用来判断"这块 raw-K 是哪一轮留下的"。
    [[nodiscard]] std::uint64_t round_stamp() const noexcept { return round_stamp_; }

    // 本轮 staging 是否完整覆盖绝对位置 [first_token, first_token + tokens)。重烘的前置条件。
    [[nodiscard]] bool covers(std::int32_t first_token, std::int32_t tokens) const noexcept {
        return round_width_ > 0 && tokens > 0 && first_token >= round_first_token_ &&
               first_token + tokens <= round_first_token_ + round_width_;
    }

    // Copies the POST-RoPE staging into its own pinned host mirror, so the dump can read both halves
    // from host memory.
    //
    // This exists because writing straight from the DEVICE pointer inside std::fwrite faulted with
    // c0000005 inside ucrtbase.dll on every attempt (measured 4/4): stdio's buffered path is not a
    // safe way to touch device memory. The pre-RoPE half never had this problem because drain_all()
    // had already brought it to pinned host memory -- so the fix is to give the post-RoPE half the
    // same treatment rather than to special-case the writer.
    //
    // Same stream and same synchronise-once discipline as drain_all(); a no-op when the switch is
    // off or the stream is capturing.
    std::int32_t drain_post_rope(cudaStream_t stream);
    [[nodiscard]] const void* post_rope_host_layer(std::int32_t layer_index) const noexcept;

    [[nodiscard]] std::int32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::int32_t kv_size() const noexcept { return kv_size_; }
    [[nodiscard]] std::int32_t tokens() const noexcept { return tokens_; }

    // Device bytes reserved for one layer's staging (the per-layer size the runtime must be able to
    // satisfy), and the total for all layers.
    [[nodiscard]] std::size_t staging_bytes_per_layer() const noexcept { return per_layer_bytes_; }
    [[nodiscard]] std::size_t staging_bytes_total() const noexcept { return total_bytes_; }

    // Host half: copy every layer's staging to its pinned slot. Same-stream, so it is ordered after
    // whatever the compute stream has already issued, and it synchronises once at the end.
    //
    // Returns the number of layers drained. Returns 0 when the switch is off (the drain gates
    // itself, raw_k_shadow.cpp:97) or when the stream is capturing, and a 0 return is NOT an error.
    // Draining per graph replay / per prefill chunk is the whole point: the alternative -- a
    // synchronise per layer per pass -- is the pattern that cost the reference implementation its
    // prefill throughput (docs/retrieval-stagein-optimization.md).
    std::int32_t drain_all(cudaStream_t stream);

    // Pinned host slot for one layer, for whatever consumes the harvest next. Null before the
    // first successful drain (and always, in the OFF arm, which by contract writes nothing).
    [[nodiscard]] const void* host_layer(std::int32_t layer_index) const noexcept;

    // DIAGNOSTIC ONLY, and the reason the L3 verdict is machine-checkable instead of argued.
    //
    // Writes the pinned slots to `path` as a self-describing blob so an external oracle can verify
    // two things at once:
    //   (1) STRUCTURE  -- all 16 layers present, no padding between them, none overwritten by
    //                     another (the failure mode that a single-buffer harvest would produce, and
    //                     which is invisible in model numerics because the whole rung is a side
    //                     channel);
    //   (2) SEMANTICS  -- the harvested tensor really is the PRE-RoPE key. That is proved the only
    //                     way it can be, without trusting this code: the blob receives a SECOND
    //                     copy, taken AFTER ops::rope has rewritten the same buffer, and the oracle
    //                     applies RoPE to the pre-RoPE half by itself. If the pre-RoPE half were in
    //                     fact post-RoPE, that reconstruction would not land on the second half.
    //
    // Layout (little-endian, int32 header, raw BF16 payload):
    //   [0] magic = 0x4B564D52 ("KVMR")
    //   [1] layers                              [2] tokens
    //   [3] kv_size (== head_dim * kv_heads)    [4] rotary_dim
    //   [5] has_post_rope (0 or 1)
    //   then int32 per layer: harvested_this_round (0 or 1)   -- the oracle fails the dump if any 0
    //   then per layer: int32 offset_bytes, int32 nbytes     (pre-RoPE half)
    //   then per layer: int32 offset_bytes, int32 nbytes     (post-RoPE half, if present)
    //   then the BF16 payloads, each layer contiguous
    //
    // `positions` is optional I32 [tokens] (the RoPE positions actually used, passed through
    // verbatim so the oracle does not have to guess them); when present its bytes follow the
    // payload as int32[tokens].
    //
    // Never called unless NINFER_TERNARY_KVMEM_DUMP names a path; writes nothing when the harvest
    // is empty. Returns false when there was nothing to write, and does not throw on I/O trouble
    // (a failed diagnostic must not take the engine down) -- it reports to stderr instead.
    bool dump_to_file(const char* path, const void* post_rope_layer_base,
                      std::size_t post_rope_layer_stride, const std::int32_t* positions,
                      std::int32_t rotary_dim) const;

    // Marks one layer as NOT harvested in the current round. Any layer whose staging slot was left
    // untouched (an arm where the call site did not run, or a run where not every full-attention
    // layer was visited) MUST be marked, because a slot that was never written still holds whatever
    // the previous round left there -- or nothing at all -- and a dump that reported such a slot as
    // a harvest would be the exact kind of "plausible but wrong" evidence this whole exercise
    // exists to prevent. Cleared by begin_round().
    void begin_round() noexcept {
        for (std::size_t i = 0; i < drop_.size(); ++i) { drop_[i] = 0; }
        // 装配臂：每一轮换一个戳，否则"这块 raw-K 是哪一轮的"无法判断（见 set_round_origin）。
        ++round_stamp_;
    }
    void mark_layer_harvested(std::int32_t layer_index) noexcept {
        if (layer_index >= 0 && layer_index < layers_) {
            drop_[static_cast<std::size_t>(layer_index)] = 1;
        }
    }

    // 装配臂（路 1）用：**逐层**查"这一层本轮真的被写过吗"。整轮计数 layers_harvested() 不够用 ——
    // arena 是按层抄 staging 的，某一层没被写过就说明那一层的槽里是**上一轮的旧字节**
    // （raw_k_harvest.h:152-165 的教训），抄进 arena 就是错料。
    [[nodiscard]] bool layer_harvested(std::int32_t layer_index) const noexcept {
        return layer_index >= 0 && layer_index < layers_ &&
               drop_[static_cast<std::size_t>(layer_index)] != 0;
    }
    [[nodiscard]] std::int32_t layers_harvested() const noexcept {
        std::int32_t n = 0;
        for (std::int32_t i = 0; i < layers_; ++i) { n += drop_[static_cast<std::size_t>(i)]; }
        return n;
    }

private:
    std::int32_t layers_         = 0;
    std::int32_t kv_size_        = 0;
    std::int32_t tokens_         = 0;
    std::size_t  per_layer_bytes_ = 0;
    std::size_t  total_bytes_    = 0;
    DeviceArena  staging_arena_;          // owns all layers' device staging
    std::array<Tensor, 64> staging_{};    // per-layer views into staging_arena_
    RawKShadowHostStore host_store_;      // owns all layers' pinned host slots (pre-RoPE)
    RawKShadowHostStore post_store_;      // pinned mirror of the post-RoPE half (dump only)
    std::array<std::int32_t, 64> drop_{}; // per-layer "was harvested this round" flag
    std::int32_t round_width_ = 0;        // tokens actually written this round (<= tokens_)
    std::int32_t round_first_token_ = 0;  // 装配臂：本轮首个 token 的绝对位置（set_round_origin）
    std::uint64_t round_stamp_      = 0;  // 装配臂：轮次戳（begin_round 每轮 +1）
};

// The process-lifetime harvest, created on first use and never destroyed.
//
// WHY THIS IS PUBLIC AND WHY IT IS A SINGLETON
//   A TextContext is constructed on the stack for one schedule recording and destroyed again
//   (text_context.h:29), but the harvest owns a DeviceArena plus a pinned host allocation that must
//   NOT be recycled per call -- so the object has to outlive every TextContext and be handed to each
//   one by pointer. Every construction site therefore installs the SAME object, which is why this
//   cannot live in a per-call scope, and why it was hoisted out of the anonymous namespace in
//   text_prefill_impl.h (where only one of the six card construction sites could reach it -- the
//   reason the whole harvest was dead code until 2026-09-21).
//
// `tokens` is a PROVEN upper bound on any later round: attn_mix() asserts T <= prefill_chunk on both
// prefill entry points (text_context_impl.h:455 and :593), so passing effective_prefill_chunk
// (layouts_impl.h:102-103) sizes the slots for the widest round any caller can request.
//
// Sizing is monotonic: a later call may widen the harvest but never narrow it, because a copy whose
// token count exceeds the reserved size throws rather than silently truncating.
//
// Reads the switch. Returns nullptr when NINFER_TERNARY_KVMEM is off, which is the default, so the
// OFF arm allocates nothing and every install site reduces to storing a null pointer.
[[nodiscard]] RawKShadowHarvest* raw_k_shadow_harvest_for(std::uint32_t prefill_chunk,
                                                          std::int32_t kv_size,
                                                          std::int32_t layers);

} // namespace ninfer::ops
