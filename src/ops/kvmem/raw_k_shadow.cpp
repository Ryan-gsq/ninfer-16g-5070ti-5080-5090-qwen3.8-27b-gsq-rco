#include "ops/kvmem/raw_k_shadow.h"

#include "core/device.h"
#include "ops/kvmem/kvmem_shadow.h"
#include "ops/kvmem/raw_k_shadow_launch.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// KVMem raw-K shadow: the HOST half -- operand validation, the op entry point, the pinned host
// destination and the capture-checked drain.
//
// This translation unit is compiled by MSVC (not nvcc). It includes cuda_runtime.h through
// raw_k_shadow.h, which is legal for a host TU (the project already does this, e.g.
// ops/linear/ternary/ternary_rotation.cpp includes ternary_rotation.h -> <cuda_runtime.h>), but it
// must never include a device-ONLY header: see kvmem_shadow.h:5-14 for the measured C2760 failure
// that rule comes from.

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "raw_k_shadow";

void require_raw_key(const Tensor& raw_key) {
    if (raw_key.dtype != DType::BF16 || raw_key.data == nullptr || !raw_key.is_contiguous()) {
        throw std::invalid_argument(
            std::string(kOp) + ": raw key must be a contiguous BF16 [head_dim, kv_heads, tokens] "
                               "or [head_dim, kv_heads, width, batch] tensor");
    }
}

void require_shadow(const Tensor& shadow) {
    if (shadow.dtype != DType::BF16 || shadow.data == nullptr || !shadow.is_contiguous() ||
        shadow.ne[2] != 1 || shadow.ne[3] != 1) {
        throw std::invalid_argument(
            std::string(kOp) + ": shadow must be a contiguous BF16 [kv_size, tokens] tensor");
    }
}

// The staging buffer has to describe the SAME memory, element for element: kv_size is
// ne[0] * ne[1] of the raw key and the token count is the product of its remaining axes
// (3-D kn gives tokens = ne[2], the batched 4-D view gives tokens = ne[2] * ne[3]). Getting this
// wrong is the one way this op can produce a plausible-looking but mislaid harvest, so it is
// checked rather than trusted. (At tokens == 1 every candidate layout coincides; this check and the
// tokens>1 cases in the test are what make a transpose visible.)
void require_geometry(const Tensor& raw_key, const Tensor& shadow) {
    require_raw_key(raw_key);
    require_shadow(shadow);
    const std::int64_t raw_elements = raw_key.numel();
    const std::int64_t kv_size      = static_cast<std::int64_t>(raw_key.ne[0]) *
                                 static_cast<std::int64_t>(raw_key.ne[1]);
    const std::int64_t tokens = static_cast<std::int64_t>(raw_key.ne[2]) *
                               static_cast<std::int64_t>(raw_key.ne[3]);
    if (static_cast<std::int64_t>(shadow.ne[0]) != kv_size ||
        static_cast<std::int64_t>(shadow.ne[1]) != tokens ||
        shadow.numel() != raw_elements) {
        // Carry the actual numbers: a bare "geometry mismatch" costs a rebuild-and-guess cycle,
        // and this project has already paid that twice (see the bad_alloc diagnostics in
        // program_impl.h, which discard exactly this kind of information).
        throw std::invalid_argument(
            std::string(kOp) + ": shadow [" + std::to_string(shadow.ne[0]) + ", " +
            std::to_string(shadow.ne[1]) + "] (" + std::to_string(shadow.numel()) +
            " elems) does not match the raw key [" + std::to_string(raw_key.ne[0]) + ", " +
            std::to_string(raw_key.ne[1]) + ", " + std::to_string(raw_key.ne[2]) + ", " +
            std::to_string(raw_key.ne[3]) + "] (kv_size=" + std::to_string(kv_size) +
            ", tokens=" + std::to_string(tokens) + ", elems=" + std::to_string(raw_elements) + ")");
    }
}

std::size_t require_store_bytes(std::size_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument(std::string(kOp) + ": host store size must be positive");
    }
    return bytes;
}

} // namespace

RawKShadowHostStore::RawKShadowHostStore(std::size_t bytes) : buffer_(require_store_bytes(bytes)) {}

RawKShadowHostStore::RawKShadowHostStore(RawKShadowHostStore&& other) noexcept = default;

RawKShadowHostStore& RawKShadowHostStore::operator=(RawKShadowHostStore&& other) noexcept = default;

void* RawKShadowHostStore::data() const noexcept { return buffer_.data(); }

std::size_t RawKShadowHostStore::size_bytes() const noexcept { return buffer_.size(); }

void* RawKShadowHostStore::slice_at(std::size_t offset, std::size_t bytes) const {
    if (bytes == 0 || offset > buffer_.size() || bytes > buffer_.size() - offset) {
        throw std::invalid_argument(std::string(kOp) + ": host slot lies outside the store");
    }
    auto* base = static_cast<std::uint8_t*>(buffer_.data());
    return base + offset;
}

void raw_k_shadow_copy(const Tensor& raw_key, Tensor& shadow, cudaStream_t stream) {
    require_geometry(raw_key, shadow);
    // Pure device->device bit copy. No switch test here: the harvest is issued or not issued by the
    // call site (see raw_k_shadow.h, SWITCH CONTRACT), so both arms run this same code path and the
    // workspace plan cannot depend on the switch.
    detail::launch_raw_k_shadow_copy(raw_key.data, shadow.data, raw_key.numel(), stream);
}

bool raw_k_shadow_drain(const Tensor& shadow, RawKShadowHostStore& host, cudaStream_t stream) {
    // Host-side half of the A/B switch, the same shape as folded_activation()'s early return
    // (ops/linear/ternary/ternary_rotation.cpp:68): with the harvest switched off the op copies
    // nothing to the host, so the OFF arm performs no host write at all even if a call site forgets
    // to gate. The switch is NEVER read by the kernel or by the byte arithmetic (kvmem_shadow.h).
    if (!detail::kvmem_shadow_enabled()) { return false; }

    // Graph preparation replays this op while the stream is capturing, and both the device->host
    // copy below and the synchronize that makes it meaningful are illegal inside a capture.
    // Skipping capture means the copy lands on the first real replay instead, which is the run
    // whose numbers matter anyway. (Same treatment as ops/wrapper/embedding.cpp:285-293.)
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return false;
    }

    require_shadow(shadow);
    const std::size_t bytes = shadow.bytes();
    if (host.size_bytes() != bytes) {
        throw std::invalid_argument(
            std::string(kOp) +
            ": host store is not exactly the size of the staging buffer (would overrun the host)");
    }
    CUDA_CHECK(cudaMemcpyAsync(host.data(), shadow.data, bytes, cudaMemcpyDeviceToHost, stream));
    // The copy is only meaningful once it has completed; the caller reads the pinned bytes right
    // after this returns.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

} // namespace ninfer::ops
