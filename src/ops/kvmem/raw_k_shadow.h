#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// KVMem raw-K shadow: harvest the K that exists BEFORE RoPE is applied.
//
// THE POINT (why this op exists at all)
//   In the main attention path (targets/qwen3_6/impl/runtime/text_context_impl.h:844) `kn` holds
//   the RMS-normalised raw key, and two lines later (:850) ops::rope OVERWRITES that same buffer
//   in place with the rotated key. RoPE rotates only the first rotary_dim=64 of head_dim=256
//   components, but it MIXES component pairs, so the pre-RoPE values cannot be recovered
//   afterwards: after :850 the raw K is gone for good. A KVMem-style experiment that wants the
//   raw key therefore has to copy it in the two-line window between :844 and :850, which is
//   exactly one extra op and no change to any visible set.
//
//   This op is that copy and nothing else. It is a pure BF16 BIT COPY:
//     * it READS `raw_key` and writes only `shadow`;
//     * it never writes `kn`, never touches a KV page, never touches the attention inputs;
//     * it performs no arithmetic, no quantisation and no rounding, so bit-for-bit equality with
//       the source is the correct (and the required) verdict, not an approximation.
//
// NO Hadamard-basis bookkeeping is needed: the folded rotation is applied on the GEMM INPUT side
// (ops/linear/ternary/ternary_rotation.cu:82-87, /*inverse=*/0), so the projection output lives in
// the residual stream's original basis, and the only inverse-mapped tensor in the model is the
// token embedding (ternary_rotation.cu:93-94). Raw K can therefore be stored as it stands.
//
// GEOMETRY CONTRACT (checked, not assumed)
//   raw_key : BF16, contiguous, with ne[0]*ne[1] == kv_size and ne[2]*ne[3] == tokens. Both forms
//             the model actually produces are accepted: the 3-D view the call site has
//             (`results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T})`, head_dim=256,
//             kv_heads=4) and the 4-D batched view the attention call below it builds over the same
//             memory ({head_dim, kv_heads, width, batch}, tokens = width * batch).
//   shadow  : BF16, contiguous, ne = {kv_size, tokens}, ne[2] == ne[3] == 1.
//   kv_size is the fastest axis in both declarations (the elements of one token are contiguous), so
//   the copy is element-for-element in one flat order. The declared geometry is validated and a
//   mismatch throws -- a caller that hands over a shadow of the wrong shape gets an exception
//   instead of a silently mis-laid-out harvest. The layout is the classic trap here: at tokens == 1
//   every candidate layout coincides, so this check and the test's tokens>1 cases are what make a
//   transpose visible.
//
// CAPTURE CONTRACT (why the entry points look like this)
//   This op runs inside a captured CUDA graph, so raw_k_shadow_copy must be launchable while the
//   stream is capturing: it performs NO cudaStreamSynchronize and NO cudaMalloc. The shadow buffer
//   is supplied by the caller -- from the op workspace arena (whose capacity must declare
//   ops::detail::kvmem_raw_k_shadow_bytes(), see kvmem_shadow.h) or from a caller-owned
//   DeviceArena. The kernel is launched in one plain launch with caller-provided pointers.
//
//   The host half is the mirror image: raw_k_shadow_drain copies device->host, and the D2H copy
//   plus the synchronise that makes it meaningful are BOTH illegal inside a capture. It therefore
//   refuses while the stream is capturing (the same treatment as ops/wrapper/embedding.cpp:285-293)
//   and reports false; the copy then lands on the first real replay, which is the run whose numbers
//   matter anyway.
//
// SWITCH CONTRACT
//   The launch is gated at the call site; the drain gates itself. Neither the kernel nor the byte
//   arithmetic (kvmem_shadow.h) reads NINFER_TERNARY_KVMEM, because the two arms must share one
//   workspace PLAN -- reserve the shadow's bytes unconditionally, and switching the feature on
//   cannot move any other tensor's pointer. The recommended call site is
//
//     ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);            // text_context_impl.h:844
//     if (ops::detail::kvmem_shadow_enabled()) {
//         ops::raw_k_shadow_copy(kn, shadow_for_this_layer, s);         // device staging, in-graph
//     }
//     ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);  // :850 overwrites kn
//
//   ... with the drain issued later, outside the capture, on the host side. Keeping the launch
//   decision at the call site is what makes the OFF arm "the engine as it is today": no copy is
//   issued, so nothing is written at all, while the workspace plan still reserves the shadow in
//   both arms (that reservation is what keeps the two arms' pointers identical). A drain that is
//   called while the switch is off copies nothing and returns false, so a missed call-site gate
//   cannot turn into a stale harvest.

namespace ninfer::ops {

// Device -> device staging copy of the raw (pre-RoPE) key. READ-ONLY on `raw_key`; the only
// buffer written is `shadow`. Launchable inside a CUDA graph capture (no sync, no allocation).
// Throws std::invalid_argument when the geometry contract above is not met.
void raw_k_shadow_copy(const Tensor& raw_key, Tensor& shadow, cudaStream_t stream);

// Owning pinned host destination for the harvest, one per layer (or one per harvest slot).
//
// Pinned because the harvest's D2H copy is issued on the compute stream and is immediately read;
// the type is the project's existing PinnedHostBuffer (core/arena.h:93), not a new allocator.
// Construction performs a CUDA host allocation, so it is ILLEGAL during a capture: build the
// stores while the graph is being prepared, never inside the capture body.
class RawKShadowHostStore {
public:
    // `bytes` is normally ops::detail::kvmem_raw_k_shadow_bytes(kv_size, tokens).
    explicit RawKShadowHostStore(std::size_t bytes);

    RawKShadowHostStore(const RawKShadowHostStore&)            = delete;
    RawKShadowHostStore& operator=(const RawKShadowHostStore&) = delete;
    RawKShadowHostStore(RawKShadowHostStore&& other) noexcept;
    RawKShadowHostStore& operator=(RawKShadowHostStore&& other) noexcept;

    [[nodiscard]] void* data() const noexcept;
    [[nodiscard]] std::size_t size_bytes() const noexcept;

    // A byte offset into the pinned buffer, for a caller that packs SEVERAL slots into one store
    // (see ops/kvmem/raw_k_harvest.h: one pinned slot per full-attention layer). It exists so such
    // a caller can issue every layer's D2H copy and then synchronise ONCE, instead of calling
    // raw_k_shadow_drain() per layer -- that function synchronises on every call by design, and a
    // synchronise per layer per pass is the pattern the reference implementation measured as
    // costing its prefill throughput. Throws when offset + bytes exceeds the store.
    [[nodiscard]] void* slice_at(std::size_t offset, std::size_t bytes) const;

private:
    PinnedHostBuffer buffer_;
};

// Host half: device -> pinned host copy of a filled staging buffer, then a synchronise so the
// caller may read it. Returns false and copies NOTHING when NINFER_TERNARY_KVMEM is off (host-side
// gate, the same shape as folded_activation()'s early return) or when the stream is capturing (the
// copy would be illegal there, and its meaning would be wrong anyway); returns true after a
// completed copy. Callers must not read `false` as an error.
//
// Throws std::invalid_argument when the shadow geometry is unusable or when `host` is not exactly
// the size of `shadow` -- an undersized store would otherwise be a silent host overflow.
[[nodiscard]] bool raw_k_shadow_drain(const Tensor& shadow, RawKShadowHostStore& host,
                                      cudaStream_t stream);

} // namespace ninfer::ops
