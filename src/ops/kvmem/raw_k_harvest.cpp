#include "ops/kvmem/raw_k_harvest.h"

#include "core/device.h"
#include "ops/kvmem/kvmem_shadow.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// KVMem raw-K shadow: per-layer staging and the drain driver. See raw_k_harvest.h for why one
// buffer is not enough and why the drain is issued per graph replay / per prefill chunk.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "raw_k_shadow harvest";
constexpr std::size_t kMaxLayers = 64;

// The per-layer slice is a view over one flat arena allocation, so the only arithmetic that can go
// wrong silently is the stride between layers. Both factors are validated BEFORE the allocators are
// sized (member initialisers run in declaration order and the arena cannot be resized afterwards),
// and the total is computed in a way that cannot overflow.
std::size_t checked_per_layer_bytes(std::int32_t layers, std::int32_t kv_size, std::int32_t tokens) {
    if (layers <= 0 || static_cast<std::size_t>(layers) > kMaxLayers) {
        throw std::invalid_argument(std::string(kSubject) + ": layer count out of range");
    }
    const std::size_t bytes = detail::kvmem_raw_k_shadow_bytes(kv_size, tokens);
    if (bytes == 0) {
        throw std::invalid_argument(std::string(kSubject) +
                                    ": kv_size and tokens must both be positive");
    }
    if (bytes > static_cast<std::size_t>(-1) / static_cast<std::size_t>(layers)) {
        throw std::invalid_argument(std::string(kSubject) + ": total shadow size overflows");
    }
    return bytes;
}

std::size_t checked_total_bytes(std::size_t per_layer, std::int32_t layers) {
    return per_layer * static_cast<std::size_t>(layers);
}

} // namespace

RawKShadowHarvest::RawKShadowHarvest(std::int32_t layers, std::int32_t kv_size, std::int32_t tokens)
    : layers_(layers), kv_size_(kv_size), tokens_(tokens),
      per_layer_bytes_(checked_per_layer_bytes(layers, kv_size, tokens)),
      total_bytes_(checked_total_bytes(per_layer_bytes_, layers)),
      staging_arena_(total_bytes_),
      host_store_(total_bytes_),
      post_store_(total_bytes_),
      round_width_(tokens) {
    // One allocation for all layers, sliced per layer. Slicing rather than allocating `layers`
    // times keeps the arena from inserting alignment padding BETWEEN layers, which would make the
    // per-layer stride disagree with per_layer_bytes() -- the exact silent misalignment this
    // constructor checks against.
    Tensor flat = staging_arena_.alloc(DType::BF16, {kv_size_, tokens_ * layers_});
    if (flat.bytes() != total_bytes_) {
        throw std::logic_error(std::string(kSubject) +
                               ": arena allocation does not match the declared shadow size");
    }
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        staging_[static_cast<std::size_t>(layer)] = flat.slice(1, layer * tokens_, tokens_);
    }
}

Tensor& RawKShadowHarvest::staging(std::int32_t layer_index) noexcept {
    return staging_[static_cast<std::size_t>(layer_index)];
}

Tensor RawKShadowHarvest::staging_for_round(std::int32_t layer_index) const {
    const std::int32_t width = round_width_ > 0 ? round_width_ : tokens_;
    // A fresh tensor over the same memory, reshaped to this round's width. Reshaping a copy rather
    // than the stored view keeps every slot's declared shape at the full width, so the per-layer
    // stride used by drain/pack cannot change under us between rounds.
    return Tensor(staging_[static_cast<std::size_t>(layer_index)].data, DType::BF16,
                  {kv_size_, width});
}

void RawKShadowHarvest::set_round_width(std::int32_t width) noexcept {
    if (width > 0 && width <= tokens_) { round_width_ = width; }
}

// 装配臂：记录本轮 raw-K 覆盖的**首个绝对 token 位置**。只在 prefill 的安装点调用
// （text_prefill_impl.h），因为只有那里知道这一 chunk 落在序列的哪个位置（state.text_kv_base）。
// 与 set_round_width 一样是纯记录：不分配、不启动、不读开关、不碰任何张量。
void RawKShadowHarvest::set_round_origin(std::int32_t first_token) noexcept {
    round_first_token_ = first_token > 0 ? first_token : 0;
}

std::int32_t RawKShadowHarvest::drain_all(cudaStream_t stream) {
    // The switch and the capture check both live inside raw_k_shadow_drain, so the OFF arm and the
    // capture-preparation replay copy nothing here (raw_k_shadow.cpp:97-107). Calling it per layer
    // would synchronise per layer; instead every layer is issued on the same stream and a single
    // synchronise at the end makes them all meaningful. That is the difference between one barrier
    // per replay and sixteen.
    std::int32_t drained = 0;
    const std::size_t per_layer = per_layer_bytes_;
    // Transfer only the bytes this round actually wrote. A short final chunk (measured: 511 tokens
    // where the chunk width is 512) must not publish the tail of a previous, longer chunk as if it
    // belonged to this one. The HOST stride stays per_layer so slots cannot collide.
    const std::int32_t width = round_width_ > 0 ? round_width_ : tokens_;
    const std::size_t bytes_per_layer =
        static_cast<std::size_t>(kv_size_) * static_cast<std::size_t>(width) *
        detail::kKvmemRawKBytesPerElement;
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        Tensor& shadow = staging_[static_cast<std::size_t>(layer)];
        // Computed as an offset rather than via slice_at(): slice_at is bounds-checked and throws,
        // and this loop runs inside the per-replay drain path where an exception is the wrong
        // failure mode. The offset is in range by construction (layer < layers_, and the store is
        // exactly layers_ * per_layer bytes), which the constructor has already established.
        auto* destination = static_cast<std::uint8_t*>(host_store_.data()) +
                            static_cast<std::size_t>(layer) * per_layer;
        if (!detail::kvmem_shadow_enabled()) { return 0; }

        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return 0;
        }

        CUDA_CHECK(cudaMemcpyAsync(destination, shadow.data, bytes_per_layer,
                                   cudaMemcpyDeviceToHost, stream));
        ++drained;
    }
    if (drained > 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return drained;
}

const void* RawKShadowHarvest::host_layer(std::int32_t layer_index) const noexcept {
    return static_cast<const std::uint8_t*>(host_store_.data()) +
           static_cast<std::size_t>(layer_index) * per_layer_bytes_;
}

const void* RawKShadowHarvest::post_rope_host_layer(std::int32_t layer_index) const noexcept {
    return static_cast<const std::uint8_t*>(post_store_.data()) +
           static_cast<std::size_t>(layer_index) * per_layer_bytes_;
}

std::int32_t RawKShadowHarvest::drain_post_rope(cudaStream_t stream) {
    // Mirrors drain_all(), but for the post-RoPE staging. Only the bytes this round wrote are
    // transferred; the host stride stays the full per-layer stride so slots cannot collide.
    if (!detail::kvmem_shadow_enabled()) { return 0; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return 0;
    }
    const std::int32_t width = round_width_ > 0 ? round_width_ : tokens_;
    const std::size_t bytes_per_layer =
        static_cast<std::size_t>(kv_size_) * static_cast<std::size_t>(width) *
        detail::kKvmemRawKBytesPerElement;
    std::int32_t drained = 0;
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        auto* destination = static_cast<std::uint8_t*>(post_store_.data()) +
                            static_cast<std::size_t>(layer) * per_layer_bytes_;
        CUDA_CHECK(cudaMemcpyAsync(destination, staging_[static_cast<std::size_t>(layer)].data,
                                   bytes_per_layer, cudaMemcpyDeviceToHost, stream));
        ++drained;
    }
    if (drained > 0) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    return drained;
}

bool RawKShadowHarvest::dump_to_file(const char* path, const void* post_rope_layer_base,
                                     std::size_t post_rope_layer_stride,
                                     const std::int32_t* positions,
                                     std::int32_t rotary_dim) const {
    if (path == nullptr || *path == '\0' || layers_ <= 0 || per_layer_bytes_ == 0) { return false; }

    // The header is written in one go, so the index can be filled before any payload lands. Five
    // int32 header words, then two (offset, nbytes) pairs per layer.
    const bool has_post_rope = post_rope_layer_base != nullptr;
    std::FILE* file          = std::fopen(path, "wb");
    if (file == nullptr) {
        // A diagnostic that cannot open its destination must not take the engine down with it.
        std::fprintf(stderr, "raw_k_shadow: cannot open dump path %s\n", path);
        return false;
    }

    const std::int32_t magic   = 0x4B564D52;  // "KVMR"
    const std::int32_t header[6] = {magic, layers_, tokens_, kv_size_, rotary_dim,
                                    has_post_rope ? 1 : 0};
    std::fwrite(header, sizeof(std::int32_t), 6, file);

    // Per-layer "this slot was actually written this round" flags. Emitted so a slot left over from
    // an earlier round can never be mistaken for a harvest: the oracle FAILS the whole dump when any
    // layer reports 0, instead of quietly accepting stale bytes as evidence.
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        const std::int32_t flag = drop_[static_cast<std::size_t>(layer)];
        std::fwrite(&flag, sizeof(std::int32_t), 1, file);
    }

    // Index: one (offset, nbytes) pair per layer for each half. The offsets are ABSOLUTE within the
    // blob's payload region, so a gap between layers (an alignment pad, or a stride that disagrees
    // with the arithmetic) shows up as an overlap or a hole instead of silently shifting every
    // later layer -- which is exactly the class of bug this dump exists to catch.
    const std::int32_t per_layer = static_cast<std::int32_t>(per_layer_bytes_);
    const std::int32_t half      = static_cast<std::int32_t>(layers_) * per_layer;
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        const std::int32_t pair[2] = {layer * per_layer, per_layer};
        std::fwrite(pair, sizeof(std::int32_t), 2, file);
    }
    if (has_post_rope) {
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const std::int32_t pair[2] = {half + layer * per_layer, per_layer};
            std::fwrite(pair, sizeof(std::int32_t), 2, file);
        }
    }

    const std::uint8_t* host_base = static_cast<const std::uint8_t*>(host_store_.data());
    // Per-layer writes with CHECKED return values. A single 32 MiB fwrite of the two concatenated
    // halves faulted inside ucrtbase.dll (APPCRASH, c0000005) after the metadata and the whole
    // pre-RoPE half had already landed, which is the worst failure shape for a diagnostic: a
    // plausible-looking truncated blob with no error anywhere. Writing layer by layer means a
    // failure is attributable ("which layer") and is reported rather than raised as a segfault.
    //
    // Write per_layer_bytes_ (the FULL allocation width) for every layer, both halves, so the index
    // describes a uniform stride and the oracle's arithmetic is exact; bytes beyond this round's
    // width are simply not part of the round. round_width is reported separately below.
    bool write_ok = true;
    // DIAGNOSTIC ORDER SWITCH (temporary). Which half fails tells us whether the fault follows the
    // DATA or the WRITE ORDER. If post-first moves the crash to the pre-RoPE half, the second
    // fwrite of the blob is at fault regardless of contents; if post-first still dies on post, the
    // post-RoPE source pointer is the problem. Set NINFER_TERNARY_KVMEM_POST_FIRST=1.
    const bool post_first = std::getenv("NINFER_TERNARY_KVMEM_POST_FIRST") != nullptr;
    if (has_post_rope && post_first && write_ok) {
        std::fprintf(stderr, "raw_k_shadow: [post-first] base=%p stride=%zu\n", post_rope_layer_base,
                     post_rope_layer_stride);
        for (std::int32_t layer = 0; layer < layers_ && write_ok; ++layer) {
            const auto* source = static_cast<const std::uint8_t*>(post_rope_layer_base) +
                                 static_cast<std::size_t>(layer) * post_rope_layer_stride;
            const std::size_t written = std::fwrite(source, 1, per_layer_bytes_, file);
            if (written != per_layer_bytes_) {
                std::fprintf(stderr, "raw_k_shadow: [post-first] write failed at layer %d\n",
                             layer);
                write_ok = false;
            }
        }
        std::fprintf(stderr, "raw_k_shadow: [post-first] post half written\n");
    }
    // The two halves are written exactly once each: the post-first branch above already handled the
    // post-RoPE half, so these two run only in the normal (pre-first) order.
    for (std::int32_t layer = 0; layer < layers_ && write_ok && !post_first; ++layer) {
        const std::size_t written =
            std::fwrite(host_base + static_cast<std::size_t>(layer) * per_layer_bytes_, 1,
                        per_layer_bytes_, file);
        if (written != per_layer_bytes_) {
            std::fprintf(stderr, "raw_k_shadow: pre-rope write failed at layer %d (%zu of %zu)\n",
                         layer, written, per_layer_bytes_);
            write_ok = false;
        }
    }
    if (has_post_rope && write_ok && !post_first) {
        for (std::int32_t layer = 0; layer < layers_ && write_ok; ++layer) {
            const auto* source = static_cast<const std::uint8_t*>(post_rope_layer_base) +
                                 static_cast<std::size_t>(layer) * post_rope_layer_stride;
            const std::size_t written = std::fwrite(source, 1, per_layer_bytes_, file);
            if (written != per_layer_bytes_) {
                std::fprintf(stderr,
                             "raw_k_shadow: post-rope write failed at layer %d (%zu of %zu)\n",
                             layer, written, per_layer_bytes_);
                write_ok = false;
            }
        }
    }
    // Trailing metadata: the width this round actually filled, then the RoPE positions used.
    //
    // DIAGNOSTIC SWITCH (temporary): with NINFER_TERNARY_KVMEM_NO_TAIL=1 the trailing metadata is
    // skipped, which isolates whether the fault is in these writes/the close rather than the bulk
    // payload writes. The measured failure was a c0000005 inside ucrtbase after the bulk writes but
    // BEFORE the final "dumped" line, so the trailing write or fclose is the prime suspect.
    const bool skip_tail = std::getenv("NINFER_TERNARY_KVMEM_NO_TAIL") != nullptr;
    std::fprintf(stderr, "raw_k_shadow: halves done, write_ok=%d skip_tail=%d\n", write_ok ? 1 : 0,
                 skip_tail ? 1 : 0);
    if (!skip_tail) {
        const std::int32_t width_now = round_width_ > 0 ? round_width_ : tokens_;
        std::fprintf(stderr, "raw_k_shadow: writing tail round_width=%d positions=%s\n", width_now,
                     positions != nullptr ? "yes" : "no");
        // Copy the trailing metadata into LOCAL storage before writing it. Reading the caller's
        // buffer directly inside std::fwrite faulted with c0000005 on every attempt (measured 3/3:
        // crash exactly at "writing tail ... positions=yes", and EXITCODE=0 once the write was
        // skipped). The 32 MiB of payload writes are unaffected, and so is a write of the 4-byte
        // round_width alone -- so the trigger is specifically the reader reaching into the
        // caller-owned positions buffer through stdio's buffered path. Copying a few KB removes the
        // dependency entirely and costs nothing measurable.
        std::array<std::int32_t, 4096> tail_positions{};
        const bool have_positions = positions != nullptr && tokens_ > 0 &&
                                    static_cast<std::size_t>(tokens_) <= tail_positions.size();
        if (have_positions) {
            std::memcpy(tail_positions.data(), positions,
                        static_cast<std::size_t>(tokens_) * sizeof(std::int32_t));
        }
        std::fwrite(&width_now, sizeof(std::int32_t), 1, file);
        if (have_positions) {
            std::fwrite(tail_positions.data(), sizeof(std::int32_t),
                        static_cast<std::size_t>(tokens_), file);
        }
        std::fprintf(stderr, "raw_k_shadow: tail written\n");
    }
    std::fflush(file);
    std::fprintf(stderr, "raw_k_shadow: flushed, closing\n");
    std::fclose(file);
    std::fprintf(stderr, "raw_k_shadow: closed\n");

    std::fprintf(stderr,
                 "raw_k_shadow: dumped layers=%d tokens=%d round_width=%d kv_size=%d rotary=%d "
                 "post_rope=%d positions=%d -> %s\n",
                 layers_, tokens_, (round_width_ > 0 ? round_width_ : tokens_), kv_size_,
                 rotary_dim, has_post_rope ? 1 : 0, positions != nullptr ? tokens_ : 0, path);
    return true;
}

// The process-lifetime harvest. See raw_k_harvest.h for why every card construction site must
// install the SAME object, and why sizing is monotonic.
//
// A function-local static is used instead of a namespace-scope object because the type is
// deliberately non-copyable and non-movable, and because construction performs the device and pinned
// allocations -- which must not happen before the first caller needs it (or before CUDA is ready).
//
// Reads the switch, so the default (OFF) arm allocates nothing and returns nullptr; every install
// site then stores a null pointer, which is what leaves the mixer with nothing to copy into.
RawKShadowHarvest* raw_k_shadow_harvest_for(std::uint32_t prefill_chunk, std::int32_t kv_size,
                                            std::int32_t layers) {
    if (!detail::kvmem_shadow_enabled()) { return nullptr; }
    static RawKShadowHarvest* harvest = nullptr;
    static std::uint32_t sized_for    = 0;
    const std::int32_t tokens =
        (prefill_chunk == 0) ? 1 : static_cast<std::int32_t>(prefill_chunk);
    if (harvest == nullptr) {
        harvest   = new RawKShadowHarvest(layers, kv_size, tokens);
        sized_for = prefill_chunk;
        std::fprintf(stderr,
                     "raw_k_shadow: harvest installed layers=%d kv_size=%d tokens=%d (%.1f MiB "
                     "device, same pinned)\n",
                     layers, kv_size, tokens,
                     static_cast<double>(layers) * kv_size * tokens * 2 / (1024.0 * 1024.0));
        return harvest;
    }
    if (prefill_chunk > sized_for) {
        // Silently reusing an undersized harvest would make raw_k_shadow_copy throw mid-prefill;
        // silently reallocating would move device addresses a captured graph already points at.
        throw std::logic_error(std::string(kSubject) +
                               ": harvest was sized for a smaller prefill chunk");
    }
    return harvest;
}

} // namespace ninfer::ops
