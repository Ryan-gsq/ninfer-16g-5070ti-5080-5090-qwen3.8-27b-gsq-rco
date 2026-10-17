#pragma once

// ninfer::ops::detail - private launch prototype for glp_project. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void glp_project_launch(Tensor& y, const Tensor& r, const Tensor& alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
