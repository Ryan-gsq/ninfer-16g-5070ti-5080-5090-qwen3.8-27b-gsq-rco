#pragma once

// GLP (GGUF Layer Projection) steering state for the qwen3_5 text model.
//
// Holds a per-layer direction stack resident on the device plus one FP32 alpha scalar, and
// exposes the per-layer projection used by the writer sites. The stack is indexed by GLOBAL
// layer number and zero-padded for layers the vector does not cover, so an uncovered layer's
// direction is all zeros and its projection is an exact no-op.
//
// The state owns no CUDA Graph nodes; both the direction rows and alpha are ordinary device
// buffers, so a capture records stable pointers. This is deliberate: a host scalar for alpha
// would be baked into the graph and could not be changed between replays.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

/**
 * One GLP direction stack: `layers` rows of `width` BF16, plus a device FP32 alpha.
 *
 * The `alpha` tensor is the source of truth for the projection strength. Setting it to zero
 * (the constructor default) makes every projection the identity, so an engine that never calls
 * `set_alpha` behaves exactly like stock.
 */
class GlpState {
public:
    GlpState() = default;

    /**
     * Allocates the stack and uploads `directions` (row-major, `layer_count * width`, BF16).
     * `layer_count` is the model's global layer count; rows outside the vector's range are
     * zero-filled by the caller before this call.
     */
    void initialize(std::int32_t layer_count, std::int32_t width,
                    const std::vector<std::uint16_t>& directions_bits);

    /** Reads an `.npz` (dirs.npz) or GLP GGUF and produces the padded direction stack. */
    void load_from_npz(const std::filesystem::path& path, std::int32_t layer_count,
                       std::int32_t width);

    /** Uploads a new alpha (host value) into the device scalar. */
    void set_alpha(float alpha, cudaStream_t stream);

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] std::int32_t layer_count() const noexcept { return layer_count_; }
    [[nodiscard]] std::int32_t width() const noexcept { return width_; }

    /** Direction row for a global layer, or an empty tensor when the stack is disabled. */
    [[nodiscard]] Tensor direction(std::int32_t layer) const;
    [[nodiscard]] const Tensor& alpha_tensor() const noexcept { return alpha_; }

private:
    Tensor stack_;   // BF16 [width, layer_count], one direction column per layer
    Tensor alpha_;   // FP32 [1], device scalar
    std::int32_t layer_count_ = 0;
    std::int32_t width_       = 0;
    bool enabled_             = false;
};

} // namespace ninfer::models::qwen3_5::execution
