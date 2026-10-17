#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class Q8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream);
void embed_gather_q4_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream);
void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream);
const char* q8_embed_route_name(Q8EmbedRoute route);
// Prism ternary tables (group 128). T2_G128_FP16 carries no high plane, PTQ1_0 carries its qh in
// one. Only PTQ1_0 is launched from here: T2's token table goes through the fused rotated-gather
// kernel, and a plain (unrotated) T2 table has no route registered today. Adding one would first
// need a per-weight decode_one on T2DecodeAtom, which currently exposes the word-shaped
// decode_sixteen / decode_quad instead.
void embed_gather_ptq1_launch(const Tensor& ids, const Weight& table, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
