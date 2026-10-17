// KVMem mean-K index accumulation, device side. See ops/kvmem/mean_k_index.h for the contract.
//
// WHY THE ACCESS PATTERN IS STRIDED, AND WHY THAT IS FINE
//   The harvest staging is a Tensor {kv_size, width} with ne[0] contiguous (set_contiguous_strides:
//   nb[0] == dtype size), so element (k, t) sits at t * kv_size + k. A block is a run of BLOCK_TOKENS
//   CONSECUTIVE tokens, so for a fixed (head, dim) its 64 inputs are 64 addresses spaced kv_size
//   elements apart -- a strided gather, not a contiguous read. The parallelisation is chosen around
//   that: a warp owns 32 (head, dim) CELLS of one head, one cell per lane, so at every step its 32
//   lanes read lane-contiguous addresses (lane c reads dim_base + c at a fixed token) and only the
//   token step strides. Every transaction stays fully coalesced while the reduction over strided
//   tokens is per-lane -- and because each lane OWNS its cell outright, this kernel has NO cross-lane
//   reduction anywhere. (An earlier revision appeared to need one; see the store. The invented
//   reduction was not merely useless, it was destructive, and that is where the defect lived.)
//
// WHY F32 SUMS AND A READ-TIME MEAN (2026-09-21 change)
//   The kernel used to divide by the fill and store an FP16 MEAN. It now ACCUMULATES an F32 sum and
//   adds the tokens to the block's count; the mean is `sum / count`, computed at read time. Three
//   reasons, in order of importance:
//     1. A PARTIAL TAIL BLOCK CAN BE COMPLETED. The official store allows a block with fewer tokens
//        than the nominal size and grows it later (register_append / truncate_to semantics in
//        kvmem-qw3 src/kvmem_store.cpp). With a stored mean, filling a block afterwards would keep the
//        OLD mean -- the tokens added later would never appear in the index, silently. With a sum,
//        completion is pure addition: `sum += new`, `count += n`.
//     2. NO SECOND QUANTISATION. A stored FP16 mean rounds once and then every later use inherits that
//        rounding; the official implementation reverted an FP8 index experiment precisely because
//        re-quantising an already-averaged vector MOVES THE RETRIEVAL RANKING. Sums stay exact-ish
//        (F32) and the single narrowing happens on the read path, where the ranking decision is made.
//     3. IT MATCHES THE OFFICIAL CONTRACT. "ordered K sum (F32) at first write, mean-K computed on
//        read" (kvmem-llama.cpp kvmem/include/kvmem/raw_kv_store.hpp:3-8, read for the approach only --
//        that repository carries no licence).
//   COST, STATED PLAINLY: F32 sums are twice the bytes of FP16 means. See the accounting block in
//   mean_k_index.h. This is a deliberate, registered deviation, not an oversight.
//
// WHY `+=` AND NOT `=` (the obligation this creates)
//   Every store below accumulates. A block's slots must therefore be ZERO when first written; stale
//   bytes are silently folded into the mean rather than being detected. MeanKIndex::zero() performs
//   that initialisation, and the standalone verifier has a negative control that shows the judge can
//   see an un-zeroed destination (it must, or the obligation would be unverifiable).

#include "ops/common/warp.cuh"
#include "ops/kvmem/raw_k_shadow.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace detail {

namespace {

// One warp per 32 dims of one head: a (head, block) row of 256 dims is covered by 4 warps x 2 passes.
// Each lane owns its (head, dim) cell and accumulates that cell ALONE -- there is nothing to share.
constexpr int kMeanKThreads = 128;

// NOTE ON THE TWO STRIDES. `kv_size` is the ROW STRIDE of the staging tensor and the number of token
// columns is not; they are NOT interchangeable. The staging tensor is {kv_size, width} with ne[0]
// contiguous, so element (k, t) sits at t * kv_size + k. An earlier version used the token count for
// both, which read 4x past the end of the allocation whenever width != kv_size: the verifier caught it
// as cudaErrorIllegalAddress.
__global__ void mean_k_accumulate_kernel(const __nv_bfloat16* __restrict__ staging,
                                         float* __restrict__ index, int n_layers, int layer_index,
                                         int first_block, int full_blocks, int tail_fill, int heads,
                                         int head_dim, int kv_size, int block_tokens,
                                         int layer_stride, int blocks_per_layer,
                                         int index_layer_stride) {
    // blockIdx.x = local block within this round, blockIdx.y = head, blockIdx.z = SOURCE layer.
    const int local_block = blockIdx.x;
    const int head        = blockIdx.y;
    const int l           = blockIdx.z;
    if (l >= n_layers || local_block > full_blocks) { return; }

    // A partial final block contributes the tokens it really holds -- and, unlike the FP16-mean
    // revision this replaces, nothing here divides by anything: `fill` is added to the block's count
    // and the division happens at read time. Dividing by the nominal block size instead is the failure
    // this whole code path exists to avoid (the harvest once measured a 511-token final chunk on a
    // 2,273-token prompt).
    const int fill = (local_block < full_blocks) ? block_tokens : tail_fill;
    if (fill <= 0) { return; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int dims_per_warp = kWarpSize;

    const long long token_base = static_cast<long long>(local_block) * block_tokens;
    const __nv_bfloat16* layer_base = staging + static_cast<long long>(l) * layer_stride;

    // DESTINATION LAYER. The index is [n_layers][ sums | counts ] with the LAYER axis outermost
    // (mean_k_index.h: one sums view and one counts view per layer), so a layer's region begins at
    // layer * index_layer_stride and its count row sits exactly `blocks_per_layer * heads * head_dim`
    // elements into that region.
    //
    // The layer term was MISSING from the revision before last, which addressed the destination as
    // `(layer_index + first_block + local_block)`: layer_index > 0 wrote into layer 0's rows, and
    // because the layer index contributed to the address only as a BLOCK number, layer 1 at block b
    // landed on top of layer 0's block b+1. `l` was used to read but not to write, so with n_layers > 1
    // every source layer computed the SAME destination cells and the survivor depended on scheduling.
    // POS-5 / POS-6 measure both halves of that.
    const int dst_layer = layer_index + l;
    float* layer_out = index + static_cast<long long>(dst_layer) * index_layer_stride;
    const long long sums_per_layer =
        static_cast<long long>(blocks_per_layer) * heads * head_dim;
    float* layer_counts = layer_out + sums_per_layer;
    const long long dst_block = first_block + local_block;

    for (int dim_base = warp * dims_per_warp; dim_base < head_dim;
         dim_base += (kMeanKThreads / kWarpSize) * dims_per_warp) {
        // ONE LANE, ONE OUTPUT CELL, NO SHARED STATE. Each lane accumulates its own (head, dim) cell
        // over the block's tokens. Nothing here couples two lanes, so a cross-lane reduction cannot
        // combine anything -- there is no value to combine.
        //
        // The revision before last ran `warp_reduce_sum` over 32 lanes holding 32 INDEPENDENT values and
        // then guarded the store with `lane == 0`. Two consequences, both silent: 32 outputs collapsed
        // into 1 slot per warp per pass (4 warps x 2 passes = 8 slots of a 256-dim row written, measured
        // as "block0 slots written = 32 of 1024"), and the value stored was the SUM of 32 means rather
        // than one mean (measured got[0] = 15.7734 against want[0] = 0.569591 -- about 32 x 0.5, which
        // is what a sum of 32 values around +0.5 looks like).
        //
        // A comment that "explains" why a wrong reduction is necessary is worth less than a coverage
        // count: aggregate error statistics cannot see a write that never happened.
        const int dim = dim_base + lane;
        if (dim >= head_dim) { continue; }  // 256 % 32 == 0 today; kept honest for a head_dim that is not
        long long off = token_base * kv_size + static_cast<long long>(head) * head_dim + dim;
        float sum = 0.0f;
        for (int t = 0; t < fill; ++t) {
            sum += __bfloat162float(layer_base[off]);
            off += kv_size;
        }
        // += ACCUMULATES over repeated calls into the same block. The destination must be zero before
        // its first write; that obligation is documented in the header and has its own control.
        layer_out[(dst_block * heads + head) * head_dim + dim] += sum;
    }

    // THE BLOCK'S TOKEN COUNT IS A PROPERTY OF THE BLOCK, so exactly ONE thread adds it: thread 0 of
    // the head-0 block of this (block, source layer) pair. Letting every head add it would count the
    // block `heads` times, and a per-lane add would count it 32 x heads times -- the failure mode is a
    // mean divided by 8x (or 256x) too much, which looks like "the values are all a bit small".
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        layer_counts[dst_block] += static_cast<float>(fill);
    }
}

} // namespace

// Returns the launch status instead of asserting on it. Two callers want different things from a
// failure: the engine treats it as fatal, while the standalone verifier has to know WHICH call failed
// and with what code -- and CUDA_CHECK's throw is what hid the distinction during bring-up.
cudaError_t launch_mean_k_accumulate(const void* staging, void* index, std::int32_t n_layers,
                                     std::int32_t layer_index, std::int32_t first_block,
                                     std::int32_t full_blocks, std::int32_t tail_fill,
                                     std::int32_t heads, std::int32_t head_dim,
                                     std::int32_t kv_size, std::int32_t block_tokens,
                                     std::int32_t layer_stride, std::int32_t blocks_per_layer,
                                     std::int32_t index_layer_stride, cudaStream_t stream) {
    const std::int32_t grid_x = full_blocks + (tail_fill > 0 ? 1 : 0);
    if (grid_x <= 0) { return cudaSuccess; }
    if (n_layers <= 0) { return cudaErrorInvalidValue; }  // a zero-sized grid is not launchable
    if (blocks_per_layer <= 0) { return cudaErrorInvalidValue; }  // guards the layer term's stride
    if (index_layer_stride <= 0) { return cudaErrorInvalidValue; }  // guards the destination stride
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(heads),
              static_cast<unsigned>(n_layers));
    // Only the sign is checkable HERE: this entry point is never given the destination's layer
    // capacity, so "layer_index + n_layers destination layers fit inside the index" cannot be enforced
    // from this side and stays a CALLER obligation. That is not a formality -- the engine's per-layer
    // append passes n_layers = layers_ while handing over ONE layer's staging, so its higher-layer
    // calls address destination layers up to layer_index + n_layers - 1. The fix belongs in the caller
    // (mean_k_index.cpp append_one: one source layer per call means n_layers == 1, or one packed call
    // with layer_index == 0 and the allocation's base pointer).
    if (layer_index < 0) { return cudaErrorInvalidValue; }
    mean_k_accumulate_kernel<<<grid, kMeanKThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(staging), static_cast<float*>(index), n_layers,
        layer_index, first_block, full_blocks, tail_fill, heads, head_dim, kv_size, block_tokens,
        layer_stride, blocks_per_layer, index_layer_stride);
    return cudaGetLastError();
}

} // namespace detail
} // namespace ninfer::ops
