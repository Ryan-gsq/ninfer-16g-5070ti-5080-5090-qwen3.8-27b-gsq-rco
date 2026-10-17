#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/glp_project.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"

namespace ninfer::models::qwen3_5::execution {

// Per-writer GLP steering handle. When `active` is false the writer sites take the ordinary
// fused path and are bit-for-bit stock. `direction` is the BF16 [width] row for this layer and
// `alpha` the FP32 device scalar; both must be non-null when `active`.
struct GlpSteering {
    bool active             = false;
    Tensor direction;       // BF16 [width]
    const Tensor* alpha     = nullptr;

    [[nodiscard]] bool applies() const noexcept {
        return active && direction.data != nullptr && alpha != nullptr;
    }
};

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear(x, p.weight, output, p.policy, workspace, stream);
}

// `wide_verification` keeps the neural path's residual activation precision in a wide
// single-row copy verification (see residual_projection_policy).
//
// GLP: when `glp.applies()`, the writer output is materialised first, the direction is removed
// from it (ops::glp_project), and only then is it accumulated into the residual. This matches
// the reference steering semantics, where the projection acts on the writer output rather than
// on the accumulated residual stream. The fused linear_add path cannot express this because it
// never exposes the unaccumulated output.
inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream,
                        InputBasis basis = InputBasis::Primal, bool wide_verification = false,
                        const GlpSteering& glp = GlpSteering{}) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    if (!glp.applies()) {
        ops::linear_add(x, p.weight, residual, residual_projection_policy(p, wide_verification),
                        workspace, stream);
        return;
    }
    Tensor projected = workspace.alloc(DType::BF16, {p.weight.n, x.ne[1]});
    ops::linear(x, p.weight, projected, p.policy, workspace, stream);
    ops::glp_project(projected, glp.direction, *glp.alpha, stream);
    ops::residual_add(projected, residual, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_swiglu(x, p.weight, output, p.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
