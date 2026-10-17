#include "ops/kvmem/raw_k_shadow_launch.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <cstdint>

// KVMem raw-K shadow: the device half. One read-only kernel, one launch helper.
//
// WHY A RAW WORD COPY RATHER THAN A bf16-TYPED COPY
//   The shadow must be bit-for-bit equal to the pre-RoPE `kn` it copies, so the kernel is written
//   over std::uint16_t / uint4 -- the same bytes the tensor holds -- rather than over
//   __nv_bfloat16. Nothing here converts, rounds, scales or reorders anything; "the shadow is the
//   raw K" is therefore true by construction instead of by argument.
//
// WHY THERE IS NO .cuh
//   Exactly one translation unit (this one) instantiates these kernels. Keeping them in the .cu
//   means there is no device-only header for a host .cpp to include, which is the failure mode
//   recorded in ops/linear/ternary/ternary_s8_scratch.h:3-9 (MSVC: "unexpected volatile", C2760).
//
// CAPTURE SAFETY (this is a hard requirement, not a nicety)
//   The op runs inside a captured CUDA graph, so a launch-time cudaStreamSynchronize or a lazy
//   cudaMalloc would abort the capture. This file therefore contains neither: both buffers are
//   supplied by the caller (workspace arena or caller-owned arena), and the only CUDA calls are
//   the launch itself and CUDA_CHECK(cudaGetLastError()) immediately after it.

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kGridCap = 4096;

// Grid size for a grid-stride loop: at least one block, never more than kGridCap (the cap keeps a
// huge token count from asking for more blocks than the device can retire at once; the loop
// absorbs the remainder).
int shadow_grid(std::int64_t items) {
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(div_up(items, static_cast<std::int64_t>(kBlock)), kGridCap)));
}

// 16-byte path: eight BF16 values per thread per step. Used only when both addresses are 16-byte
// aligned and the element count is a multiple of 8; the caller (the op host side) never has to
// promise either.
__global__ void raw_k_shadow_copy_x8_kernel(const uint4* __restrict__ source,
                                            uint4* __restrict__ destination,
                                            const std::int64_t words) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words; index += stride) {
        destination[index] = source[index];
    }
}

// Scalar fallback: one BF16 value per thread per step, any alignment.
__global__ void raw_k_shadow_copy_kernel(const std::uint16_t* __restrict__ source,
                                         std::uint16_t* __restrict__ destination,
                                         const std::int64_t elements) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride) {
        destination[index] = source[index];
    }
}

} // namespace

void launch_raw_k_shadow_copy(const void* source, void* destination, std::int64_t elements,
                             cudaStream_t stream) {
    if (elements <= 0) { return; }
    const auto source_address      = reinterpret_cast<std::uintptr_t>(source);
    const auto destination_address = reinterpret_cast<std::uintptr_t>(destination);
    if ((elements % 8) == 0 && (source_address & 0xfu) == 0 && (destination_address & 0xfu) == 0) {
        const std::int64_t words = elements / 8;
        raw_k_shadow_copy_x8_kernel<<<shadow_grid(words), kBlock, 0, stream>>>(
            static_cast<const uint4*>(source), static_cast<uint4*>(destination), words);
    } else {
        raw_k_shadow_copy_kernel<<<shadow_grid(elements), kBlock, 0, stream>>>(
            static_cast<const std::uint16_t*>(source), static_cast<std::uint16_t*>(destination),
            elements);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
