// Implements: src/models/qwen3_5/execution/glp_state.h
//
// The direction stack is stored as BF16 [width, layer_count] with one COLUMN per layer, so the
// per-layer direction `direction(layer)` is `stack_.slice(1, layer, 1)` viewed as [width] — a
// contiguous BF16 row of `width` elements that the glp_project Op can consume directly.
//
// Persistence: a small binary container produced by `tools/glp/make_glpbin.py`:
//     magic  "GLPB"           4 bytes
//     int32  layer_count
//     int32  width
//     int32  covered_layers   (rows actually present in the source vector)
//     int32  reserved = 0
//     int32  first_layer      (global layer index of the first covered row)
//     uint16 directions[layer_count * width]   BF16, row-major, zero-padded
// Everything is little-endian. Keeping this trivial avoids pulling a GGUF reader into the model
// layer and makes the padding rule explicit in the file itself.
#include "models/qwen3_5/execution/glp_state.h"

#include "core/device.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr char kGlpMagic[4] = {'G', 'L', 'P', 'B'};

[[nodiscard]] std::uint16_t f32_to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest even on the 16 dropped bits.
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

} // namespace

void GlpState::initialize(std::int32_t layer_count, std::int32_t width,
                          const std::vector<std::uint16_t>& directions_bits) {
    if (layer_count <= 0 || width <= 0) {
        throw std::invalid_argument("GLP: layer_count and width must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(layer_count) * width;
    if (directions_bits.size() != expected) {
        throw std::invalid_argument("GLP: direction payload size mismatch");
    }
    layer_count_ = layer_count;
    width_       = width;

    Tensor staged(nullptr, DType::BF16, {width, layer_count});
    CUDA_CHECK(cudaMalloc(&staged.data, staged.bytes()));
    stack_ = staged;
    CUDA_CHECK(cudaMemcpy(stack_.data, directions_bits.data(), stack_.bytes(),
                          cudaMemcpyHostToDevice));

    Tensor alpha(nullptr, DType::FP32, {1});
    CUDA_CHECK(cudaMalloc(&alpha.data, alpha.bytes()));
    const float zero = 0.0F;
    CUDA_CHECK(cudaMemcpy(alpha.data, &zero, sizeof(zero), cudaMemcpyHostToDevice));
    alpha_ = alpha;

    enabled_ = true;
}

void GlpState::load_from_npz(const std::filesystem::path& path, std::int32_t layer_count,
                             std::int32_t width) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("GLP: cannot open " + path.string()); }

    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::memcmp(magic, kGlpMagic, 4) != 0) {
        throw std::runtime_error("GLP: bad container magic in " + path.string());
    }
    std::int32_t header[4] = {};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in) { throw std::runtime_error("GLP: truncated header in " + path.string()); }

    const std::int32_t file_layers  = header[0];
    const std::int32_t file_width   = header[1];
    const std::int32_t covered      = header[2];
    const std::int32_t first_layer  = header[3];
    if (file_width != width || file_layers != layer_count) {
        throw std::runtime_error("GLP: container geometry does not match the model");
    }
    if (covered < 0 || first_layer < 0 || first_layer + covered > layer_count) {
        throw std::runtime_error("GLP: covered layer range is out of bounds");
    }

    // Read the covered rows and place them at their global layer offsets in the zero-padded stack.
    std::vector<std::uint16_t> payload(static_cast<std::size_t>(covered) * width);
    in.read(reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(payload.size() * sizeof(std::uint16_t)));
    if (!in) { throw std::runtime_error("GLP: truncated payload in " + path.string()); }

    std::vector<std::uint16_t> padded(static_cast<std::size_t>(layer_count) * width, 0);
    std::memcpy(padded.data() + static_cast<std::size_t>(first_layer) * width, payload.data(),
                payload.size() * sizeof(std::uint16_t));

    initialize(layer_count, width, padded);
}

void GlpState::set_alpha(float alpha, cudaStream_t stream) {
    if (!enabled_) { return; }
    CUDA_CHECK(cudaMemcpyAsync(alpha_.data, &alpha, sizeof(alpha), cudaMemcpyHostToDevice, stream));
}

Tensor GlpState::direction(std::int32_t layer) const {
    if (!enabled_ || layer < 0 || layer >= layer_count_) { return Tensor{}; }
    // Column `layer` of the [width, layer_count] stack, as a contiguous [width] row.
    return stack_.slice(1, layer, 1).view({width_});
}

} // namespace ninfer::models::qwen3_5::execution
