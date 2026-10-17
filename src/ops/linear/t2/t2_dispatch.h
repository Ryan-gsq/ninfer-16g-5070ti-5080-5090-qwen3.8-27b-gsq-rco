#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

// NINFER_TERNARY_PTQ1_FAST, published so that every PTQ1 fast path asks the SAME switch. linear.cpp
// needs it because PTQ1_0's prefill band is entered one layer above t2_dispatch (the int8 route
// wants a workspace), and a fast path that ignores the switch would silently move the "off" arm off
// the reference rung -- which is exactly the promise the switch exists to keep.
[[nodiscard]] bool t2_ptq1_fast();

} // namespace ninfer::ops::detail
