#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Device-side launcher for the KVMem raw-K shadow (host declaration; the kernel itself is defined
// inside raw_k_shadow_launch.cu and has no .cuh, so no host translation unit can include device-only
// syntax by accident -- see kvmem_shadow.h:5-14 for the C2760 pitfall this avoids).

namespace ninfer::ops::detail {

// Bit-copy `elements` BF16 elements from `source` to `destination`. Pure copy: the kernel reads
// 16-bit words and writes the identical 16-bit words, with no conversion and no arithmetic.
//
// Capture-safe by construction: caller-provided pointers, no cudaStreamSynchronize, no cudaMalloc,
// no cudaMemcpyAsync. It only pushes one (or zero) kernel launches onto `stream`.
//
// `elements <= 0` launches nothing (a zero-sized grid is not a legal launch). The kernel performs
// no switch test of its own -- the harvest is launched or not launched by the caller, so the two
// arms of NINFER_TERNARY_KVMEM run the same kernel.
void launch_raw_k_shadow_copy(const void* source, void* destination, std::int64_t elements,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
