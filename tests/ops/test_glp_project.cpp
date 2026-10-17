// Contract test for ops::glp_project.
//
// Oracle: the reference evaluates the documented formula in FP64 from the represented BF16
// inputs, with the reduction done in double precision. The device accumulates in FP32 and
// rounds once to BF16 on store, so the acceptance criterion carries a relative term sized to
// the BF16 output rounding plus the FP32 reduction error.
//
// Cases: identity (alpha = 0), exact removal (alpha = 1, s -> 0), a zero direction (no-op),
// and a range of token counts including decode (T = 1) and a small width that exercises the
// 4-wide load tail.
#include "ninfer/ops/glp_project.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// BF16 store rounding (~2^-8 relative) dominates; the FP32 reduction over E = 5120 is far
// smaller. The absolute term covers near-zero results where the relative term vanishes.
constexpr PointwiseCriterion glp_bf16_criterion() {
    return {/*absolute*/ 2e-2, /*relative*/ 1.2e-2};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> glp_oracle(const std::vector<float>& y, const std::vector<float>& r, float alpha,
                               std::int32_t e, std::int32_t columns) {
    std::vector<double> out(y.size());
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base = static_cast<std::size_t>(c) * e;
        double s               = 0.0;
        for (std::int32_t i = 0; i < e; ++i) {
            s += static_cast<double>(r[i]) * static_cast<double>(y[base + i]);
        }
        const double scale = static_cast<double>(alpha) * s;
        for (std::int32_t i = 0; i < e; ++i) {
            out[base + i] = static_cast<double>(y[base + i]) - scale * static_cast<double>(r[i]);
        }
    }
    return out;
}

// Runs one case. `alpha` is passed at full precision; the oracle uses the same float value.
int run_case(const char* label, std::int32_t e, std::int32_t columns, float alpha, bool zero_dir,
             std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(e) * columns;
    std::vector<float> y(count), r(e);
    fill_uniform(y, seed, -6.0f, 6.0f);
    fill_uniform(r, seed + 1, -1.0f, 1.0f);
    round_to_bf16(y);
    round_to_bf16(r);
    if (zero_dir) {
        for (auto& value : r) { value = 0.0f; }
    } else {
        // Normalise in double, then round to BF16 so the device sees exactly this direction.
        double norm = 0.0;
        for (double value : r) { norm += value * value; }
        norm = std::sqrt(norm);
        for (auto& value : r) { value = static_cast<float>(value / norm); }
        round_to_bf16(r);
    }

    const std::vector<double> expected = glp_oracle(y, r, alpha, e, columns);
    const auto y_bits                  = encode_bf16(y);
    const auto r_bits                  = encode_bf16(r);

    GuardedDeviceBuffer device_y(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_r(e * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_alpha(sizeof(float));
    device_y.copy_from_host(y_bits.data(), device_y.bytes());
    device_r.copy_from_host(r_bits.data(), device_r.bytes());
    device_alpha.copy_from_host(&alpha, sizeof(float));

    Tensor y_tensor(device_y.data(), DType::BF16, {e, columns});
    Tensor r_tensor(device_r.data(), DType::BF16, {e});
    Tensor alpha_tensor(device_alpha.data(), DType::FP32, {1});
    ops::glp_project(y_tensor, r_tensor, alpha_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_y.data(), count), expected,
                                    glp_bf16_criterion());
    // r and alpha are read-only.
    failures += verify_exact("glp r unchanged", from_device<std::uint16_t>(device_r.data(), e), r_bits);
    failures += verify_exact("glp alpha unchanged", from_device<float>(device_alpha.data(), 1),
                             std::vector<float>{alpha});
    failures += device_y.verify_guards(label);
    failures += device_r.verify_guards("glp r");
    failures += device_alpha.verify_guards("glp alpha");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // The real profile is E = 5120; smaller widths exercise the tail of the 4-wide loads.
    failures += run_case("glp alpha=0 identity [5120,1]", 5120, 1, 0.0f, false, 101u);
    failures += run_case("glp alpha=1 decode [5120,1]", 5120, 1, 1.0f, false, 102u);
    failures += run_case("glp alpha=0.8 prefill [5120,64]", 5120, 64, 0.8f, false, 103u);
    failures += run_case("glp alpha=1 chunk [5120,333]", 5120, 333, 1.0f, false, 104u);
    failures += run_case("glp zero direction no-op [5120,7]", 5120, 7, 1.0f, true, 105u);
    failures += run_case("glp small E tail [64,3]", 64, 3, 1.0f, false, 106u);
    std::cout << (failures ? "FAIL" : "OK") << " glp_project\n";
    return failures ? 1 : 0;
}
