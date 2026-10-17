#pragma once

#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"

namespace ninfer::models::qwen3_5::execution {

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
// `glp` reserves scratch for the GLP path, which materialises the down-projection output
// separately instead of accumulating it inside a fused linear_add: pass the same value used
// when the layer runs, or the GLP branch overruns the arena.
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false, bool wide_verification = false,
                                              bool glp = false);
// `wide_verification` keeps the neural path's residual activation precision in a wide single-row
// copy verification (see residual_projection_policy). `glp` applies the layer's GLP direction to
// the down-projection output before it accumulates into the residual.
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false, bool verify = false, InputBasis basis = InputBasis::Primal,
         bool wide_verification = false, const GlpSteering& glp = GlpSteering{});
// The sign vector of a rotated Dense FFN's input projection, or null.
[[nodiscard]] const Tensor* ffn_input_signs(const FfnParameters& parameters);

} // namespace ninfer::models::qwen3_5::execution
