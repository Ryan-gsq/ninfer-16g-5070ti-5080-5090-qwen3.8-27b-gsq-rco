#include "ops/linear/t2/t2_ptq1_reference.cuh"

#include "core/device.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The I line's evidenced reference tile width. Read the note in t2_ptq1_reference.cuh before
// touching this: 32 was measured as a 2.4x REGRESSION.
constexpr int kPtq1ReferenceTileT = 8;

template <int kTileT>
void launch_reference(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    // `padded_shape[1] == k` is required rather than handled: the plane pitch below is the padded
    // width while the loop bound is the real k, and silently mixing them would read the wrong rows.
    // Every width in this model (5120 / 6144 / 10240 / 17408 / 248320) is a whole number of
    // 128-groups, so this holds; it throws instead of mis-decoding if it ever does not.
    if (w.qhigh == nullptr || w.qdata == nullptr || w.scales == nullptr ||
        w.padded_shape[1] != k || (k % PTQ1RowSplitStorage::kGroupK) != 0 || rows <= 0 ||
        cols <= 0) {
        throw std::invalid_argument("t2 PTQ1 reference: unsupported shape");
    }
    const std::int32_t padded_groups = w.padded_shape[1] / PTQ1RowSplitStorage::kGroupK;
    const dim3 grid(static_cast<unsigned>(rows),
                    static_cast<unsigned>((cols + kTileT - 1) / kTileT), 1u);
    t2_ptq1_reference_kernel<PTQ1RowSplitStorage, PTQ1DecodeAtom, kTileT>
        <<<grid, PTQ1RowSplitStorage::kGroupK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), rows, k, cols, padded_groups,
            static_cast<std::int32_t>(out.ne[0]));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_t2_ptq1_reference(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_reference<kPtq1ReferenceTileT>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
