#pragma once

// GLP (GGUF Layer Projection) runtime steering.
//
// This Op removes a learned per-layer direction from a writer output before it is accumulated
// into the residual stream. It implements the "project" mode of the GLP control-vector format:
// for a unit direction r and strength alpha,
//
//     P(y) = y - alpha * r * (r^T y)
//
// which is the component of y along r, scaled by alpha, subtracted from y.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: glp_project (in-place directional projection of a BF16 activation)
 *
 * Math / indexing:
 *   For each column t of y [E, T]:
 *     s_t = sum_e r[e] * y[e, t]
 *     y'[e, t] = y[e, t] - alpha * s_t * r[e]
 *   r is the unit direction for one layer; alpha = 0 makes this the identity.
 *   When r is all zeros (a layer outside the vector's range), the result is exactly y.
 *
 * Logical shapes:
 *   y is contiguous BF16 [E, T], E = 5120 (hidden), T any positive value (decode or prefill).
 *   r is BF16 [E] (one direction row from the direction stack).
 *   alpha is a device scalar of BF16 or FP32, held in device memory (never a host scalar), so a
 *   CUDA Graph capture records a stable pointer rather than a constant operand.
 *
 * Supported domain:
 *   BF16 activations and direction. E is the residual width; T is 1..max_context.
 *
 * Numeric:
 *   s_t accumulates in FP32 from BF16 inputs; the subtraction is computed in FP32 and rounded
 *   once to BF16. alpha is read as FP32. Reduction order is unspecified; every implementation
 *   must match an FP64 reference of the stated formula within the Op's acceptance tolerance.
 *
 * Effects:
 *   Updates y in place. r and alpha are read-only and must not alias y.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Ordinary stream-ordered launch. Safe inside CUDA Graph capture: no host-side branching on
 *   device data, no hidden allocation, all operands are pre-resident device pointers.
 */
void glp_project(Tensor& y, const Tensor& r, const Tensor& alpha, cudaStream_t stream);

} // namespace ninfer::ops
