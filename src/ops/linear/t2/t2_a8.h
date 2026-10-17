#pragma once

// Integer-activation routes for T2G128 row-split weights (the codes are exact int8 {-1, 0, +1};
// the activations are quantised to int8 with one scale per token and 64- or 128-wide group): the
// small-T kernel of t2_small_t_i8.cuh for T in [kT2I8SmallMinTokens, kT2I8SmallMaxTokens], and
// above it, from t2_a8_min_tokens() up, the tile GEMM of t2_prefill_i8.cuh, which pads T to a
// multiple of 64 internally (NINFER_T2_A8_TILE=off: the shared int8 GEMM of
// ops/common/rowsplit_a8_mma.cuh with the ternary codec).

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kT2A8MinTokens = 64;
// The tuning table's boundary for the ternary int8 rung (small_t <= 32, s8 from 33): 落点表 §27.
inline constexpr std::int32_t kPtq1TileMinTokens = 33;
// The widths the small-T kernel takes by default; NINFER_T2_I8_SMALL=off|lo,hi overrides them.
// Against the 128x64 prefill tile the small-T launches stop paying from 96 columns on the RTX 3090
// (a GDN layer's four projections: 790 vs 918 us at 96, 1091 vs 1840 us at 192; even at 64).
inline constexpr std::int32_t kT2I8SmallMinTokens = 1;
inline constexpr std::int32_t kT2I8SmallMaxTokens = 64;
// kT2A8MinTokens, or NINFER_T2_A8_MIN when it is set (benchmark A/B of the crossover).
[[nodiscard]] std::int32_t t2_a8_min_tokens();
// PTQ1_0's own tile floor: from this width up, a PTQ1_0 weight takes the int8 tile instead of the
// small-T band, whatever `route()` says for the T2 packing. The two packings do NOT share a
// boundary: T2's small-T kernel reads its codes directly, while PTQ1's repacks raw base-3 bytes
// into 2-bit codes and needs ceil(T/16) SLICES of that band, against ONE tile pass (T padded to 64
// columns). The tuning table's own boundary for the ternary s8 rung is T >= 33 (small_t <= 32) --
// 落点表 §27. Override: NINFER_PTQ1_TILE_MIN, clamped to [1, kT2I8SmallMaxTokens]. This never moves
// T2's routing, which is what keeps PQ2's bands unchanged.
[[nodiscard]] std::int32_t t2_ptq1_tile_min_tokens();

[[nodiscard]] bool t2_a8_admits(LinearPolicy policy);
// Diagnostic (NINFER_T2_A8_PAIR=off): keep the integer route but take the fused pair off it, so the
// four split parts run as separate linear() calls that still reach the same tile. Added 2026-10-01 to
// separate "the fused pair is corrupt" from "every tile call is" on the score path.
[[nodiscard]] bool t2_a8_pair_enabled();
[[nodiscard]] bool t2_a8_shape_supported(std::int32_t output_rows, std::int32_t input_rows);
[[nodiscard]] bool t2_a8_supported(const Weight& w, std::int32_t tokens);
// Activation planes for T in [1, max_tokens]; zero when no admitted call can take the route.
[[nodiscard]] std::size_t t2_a8_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t max_tokens);

// Bytes of one quantised activation plane pair for T up to max_tokens.
[[nodiscard]] std::size_t t2_a8_activation_bytes(std::int32_t input_rows, std::int32_t max_tokens);

// Records into `layout` exactly the planes t2_a8_quantize allocates for `tokens`; false when no
// integer route takes that width. `ptq1_tile` must match the weight the caller will project with
// (see t2_ptq1_tile_min_tokens): the two ways of taking a width allocate different planes.
bool t2_a8_layout_activations(WorkspaceLayoutBuilder& layout, std::int32_t input_rows,
                              std::int32_t tokens, bool ptq1_tile = false);

// x quantised once for several T2 GEMMs over the same activations; the planes live in the caller's
// workspace scope.
struct T2A8Activations {
    const std::int8_t* codes;
    const __half* scales;
    std::int32_t tokens;
    std::int32_t input_rows;
};

[[nodiscard]] T2A8Activations t2_a8_quantize(const Tensor& x, WorkspaceArena& workspace,
                                             cudaStream_t stream, bool ptq1_tile = false);
// One pass over a parent weight: rows [0, split) go to first[first_offset + row, t], the rest to
// second[second_offset + row - split, t].
void t2_a8_project_split(const T2A8Activations& x, const Weight& w, Tensor& first,
                         std::int32_t first_offset, std::int32_t split, Tensor& second,
                         std::int32_t second_offset, cudaStream_t stream);

struct T2A8Split {
    const Weight& weight;
    Tensor& first;
    std::int32_t first_offset;
    std::int32_t split;
    Tensor& second;
    std::int32_t second_offset;
};

// Both parents of a split projection over the same activations: one launch on the small-T route,
// one per parent on the prefill route.
void t2_a8_project_split_pair(const T2A8Activations& x, const T2A8Split& a, const T2A8Split& b,
                              cudaStream_t stream);

void t2_a8_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream);
void t2_a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                      cudaStream_t stream);

} // namespace ninfer::ops::detail
