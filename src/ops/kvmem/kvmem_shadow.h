#pragma once

// Host-safe half of the KVMem raw-K shadow (first cut of the KVMem port).
//
// This header is deliberately CUDA-FREE: it carries the A/B switch and the byte arithmetic, and
// nothing else. It is included by host translation units (the op's host side, the runtime call
// site in text_context_impl.h, and the op test), so it must not pull in any device-only header.
//
// Kept separate from the kernel on purpose, exactly like ops/linear/ternary/ternary_s8_scratch.h
// (:3-9) is kept separate from ternary_rowsplit_mma_s8.cuh: including a kernel header from an MSVC
// host TU drags cuda_pipeline_helpers.h in, and the measured failure is
//   "error C2760: syntax error: unexpected token 'volatile', expected 'expression'"
// That is how the first build of the int8 rung broke; the raw-K harvest copies the same split so
// it cannot repeat the mistake. The kernel for this rung lives in raw_k_shadow_launch.cu and has
// no .cuh at all (one TU uses it, so there is nothing for a host TU to include by accident).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// One BF16 element. The shadow is a RAW BF16 copy: no quantisation, no rounding, no arithmetic.
inline constexpr std::size_t kKvmemRawKBytesPerElement = 2;

// A/B switch for the KVMem raw-K shadow, in the same style as NINFER_TERNARY_S8 / _HADAMARD.
//
//   NINFER_TERNARY_KVMEM=1   -> ON : harvest the raw (pre-RoPE) K of every full-attention layer
//   NINFER_TERNARY_KVMEM=0   -> OFF
//   unset                    -> OFF
//
// Read once per process, because the choice decides whether the harvest's device->device copy is
// issued at all, i.e. what the runtime asks the stream to do on every pass.
//
// POLARITY: the OFF arm is the DEFAULT, unlike NINFER_TERNARY_S8 (whose default is on). The
// reason is that the harvest is a DIAGNOSTIC that costs real resources -- a device->device staging
// copy of [kv_size, tokens] BF16 per full-attention layer per pass (16 layers x 4 MiB at T=2048,
// per the qwen3_6_27b geometry: layers=64 -> 16 full-attention layers, kv_size=1024), plus a pinned
// host buffer of the same size, plus a stream synchronise per drain. A default-ON diagnostic would
// silently change the resource profile of the production engine after a routine rebuild, which is
// the one thing this rung must never do. With the default OFF:
//
//   OFF ARM == THE ENGINE AS IT IS TODAY: bit-for-bit identical model numerics, and zero writes by
//   this rung to anything -- the staging copy is not issued (the call site gates the launch, see
//   raw_k_shadow.h), raw_k_shadow_drain() gates itself on this switch before it copies a single
//   host byte, and the KV pages, the residual stream and the attention inputs are never touched by
//   either path. So the OFF arm costs no device traffic and no host writes.
//
// Flip it to opt-out by changing the lambda body to what ternary_s8_enabled() uses
// (`value == nullptr || std::string(value) != "0"`); nothing else depends on the polarity, but the
// op test re-derives the documented polarity from the environment, so a flip must be deliberate.
//
// WHERE THE SWITCH IS ALLOWED TO BE READ (hard constraint, same discipline as
// ternary_rotation.cpp:45-64): host code ONLY -- this predicate, and raw_k_shadow_drain()'s early
// return. Neither the harvest kernel nor the capacity function below may consult it, so that
//   * the two arms share ONE workspace PLAN: the arena reserves the shadow's bytes in both arms,
//     so switching the feature on cannot move any other tensor's pointer;
//   * the kernel is a single implementation with no branch on the switch.
// The call site (segment 2) reads it, exactly as ternary_s8_scratch.h:22 states for the int8 rung
// ("the choice decides which kernel enters the captured CUDA graph").
// ★ DEFAULT FLIPPED 2026-10-02 (owner's rule: ship with the best parameters; a good feature must not
//   be switched off by default). Unset now means: ON whenever the KVMem ring is configured
//   (NINFER_KV_WINDOW > 0). Reason, measured the hard way: this switch gates the mean-K index
//   (mean_k_index.cpp:225), the raw-K harvest (raw_k_harvest.cpp:116/147/329) and therefore the whole
//   content scorer. With it unset the RING still ran -- mask, demote, restore, `ring retrieve` lines --
//   but the scorer never executed (0 SELECT lines), so retrieval silently fell back to the lexical
//   ranking and a multi-needle low-pool request answered `300` out of the filler instead of the needle.
//   A switch that silently turns a correctness fix into the old defect is exactly the trap the D-1
//   warning was written for; this closes it at the source.
//   Explicit values still win: `=0` is the NEGATIVE CONTROL, `=1` the explicit form.
// Read once, host code only -- the discipline in the comment block above is unchanged.
[[nodiscard]] inline bool kvmem_shadow_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM");
        if (value != nullptr && value[0] != '\0') { return std::string(value) != "0"; }
        const char* window = std::getenv("NINFER_KV_WINDOW");
        if (window == nullptr || window[0] == '\0') { return false; }
        return std::strtol(window, nullptr, 10) > 0;
    }();
    return enabled;
}

// NINFER_TERNARY_KVMEM_DUMP=<ASCII path> arms the L3 dump. Read once, like the switch, and it is a
// DIAGNOSTIC-ONLY knob: it never changes which kernels enter the graph, it only decides whether the
// harvest writes a blob after it has already been drained. The path must be ASCII for the same
// reason the embedding dump's must (ops/wrapper/embedding.cpp:279-280): a native binary cannot open
// a Chinese path. Returns nullptr when unset or empty, and the call site then does no extra work.
[[nodiscard]] inline const char* kvmem_shadow_dump_path() {
    static const char* const path = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_DUMP");
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }();
    return path;
}

// Device bytes for the raw-K shadow staging buffer of ONE full-attention layer call, BF16.
//
// Geometry (authoritative: qwen3_6_27b/impl/config.h:31-33): the op's input is the normalised
// pre-RoPE key `kn`, declared {head_dim=256, kv_heads=4, tokens}; kv_size = kv_heads * head_dim =
// 1024 is the fastest axis, so the same bytes are addressed as {kv_size, tokens}.
//
// This is the size of ONE buffer, not an arena capacity statement. A caller that uses it as the
// capacity of an arena (rather than as an allocation request) must add that arena's own alignment
// slack, the way ternary_rotation_workspace_bytes() adds 512 bytes for the 256-byte arena
// alignment (ternary_rotation.cpp:47-57): the bytes below are exact, so the arena's rounding is
// not included in them.
//
// Non-positive inputs return 0 rather than throwing: a planner asks this question for shapes it is
// about to skip (the same contract as ternary_rotation_workspace_bytes / ternary_s8_codes_bytes).
[[nodiscard]] inline constexpr std::size_t kvmem_raw_k_shadow_bytes(std::int32_t kv_size,
                                                                   std::int32_t tokens) {
    if (kv_size <= 0 || tokens <= 0) { return 0; }
    return static_cast<std::size_t>(kv_size) * static_cast<std::size_t>(tokens) *
           kKvmemRawKBytesPerElement;
}

} // namespace ninfer::ops::detail
