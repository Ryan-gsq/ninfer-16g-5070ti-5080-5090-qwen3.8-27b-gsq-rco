#pragma once

// KVMem retrieval scoring (paper Eq.10): the query-conditioned scorer that RANKS the historical blocks.
//
// WHAT THIS IS (and where the definition comes from)
//   Given the mean-K index and the de-RoPEd query rows for the current question span, this component
//   produces ONE FLOAT SCORE PER LOGICAL BLOCK. That array is exactly the input the selection policy
//   consumes (ops/kvmem/kvmem_select.h: kvmem_select_blocks(config, scores, first_block)), so this is
//   the piece that turns the index into a decision.
//
//   The definition is taken from the reference implementation, not invented here:
//
//     docs/kvmem_implementation_notes.md:549-588   section 4.5 "Mean-K Softmax-Over-Pages Scoring"
//     src/kernels_cuda.cu:5340-5435               block_attn_score_softmax_pages_kernel
//     src/qwen_executor.cpp:23697-23792           kvmem_retrieval_score_mean_softmax (host side)
//
//   For query token m, standard-attention layer l, QUERY head h and candidate block b:
//
//     group      = n_heads / n_kv_heads                    (GQA: several query heads share one KV head)
//     kvh        = h / group
//     logit[l,m,h,b] = scale * ( q[l,m,h] . kbar[l,b,kvh] )      scale = 1/sqrt(head_dim)
//     mass [l,m,h,b] = softmax OVER BLOCKS b of logit[l,m,h,·]
//     score[b]       = sum_m mean_l mean_h mass[l,m,h,b]
//
//   where kbar[l,b,g] is the MEAN of the de-RoPEd keys of that block:
//
//     kbar[l,b,g] = (1/|b|) * sum_{t in b} deRoPE(k[l,t,g], p_t)     (implementation_notes.md:358-382)
//
//   Three properties are easy to get wrong and are therefore called out here:
//
//   1. THE SOFTMAX IS OVER BLOCKS, per (layer, query token, query head) -- not over tokens, and not
//      a plain dot-product ranking. Reference section 4.6 explains why: a block must compete for mass
//      against the other blocks, so a globally similar block does not win by being similar to
//      everything. (docs/kvmem_implementation_notes.md:590-602)
//   2. THE SCORE IS A SUM OVER QUERY TOKENS but a MEAN over layers and heads. The mean over layers
//      and heads folds into a constant weight, head_w = 1/(n_layers_total*n_heads)
//      (kernels_cuda.cu:5392-5393, 5435). The reference notes the sum over query tokens is a choice,
//      not a necessity -- every block sees the same query token count, so it cannot change the
//      ranking within one request (implementation_notes.md:575-577) -- but it does make the invariant
//      below exact.
//   3. THE ALWAYS-KEPT BANDS ARE EXCLUDED FROM THE SOFTMAX, not merely deprioritised. Blocks in the
//      sink prefix and the recent suffix get logit = -inf (kernels_cuda.cu:5400-5407), so their mass
//      redistributes onto the retrievable middle. Safe because the selector keeps those bands
//      unconditionally regardless of score (kvmem_store.cpp:381-390), so zeroing them can never drop
//      a kept block. The mask applies ONLY when the history is over budget and a non-empty middle
//      survives (qwen_executor.cpp:23776-23792) -- that guard is not cosmetic: without at least one
//      included block the max over logits is -inf and the published kernel's `exp(-inf - -inf)`
//      would produce NaN. This implementation skips excluded blocks entirely and contributes nothing
//      when no block is included, so the same input yields zero rather than NaN.
//
// ★ WHERE THE MEAN IS COMPUTED, AND WHY IT IS COMPUTED HERE (interface decision, 2026-09-21)
//   The index operand is NOT a stored mean. It is the block's **ordered F32 SUM of de-RoPEd keys
//   together with that block's REAL token count**, and the division happens inside this component:
//
//     kbar[l,b,g] = kbar_sum[l,b,g] / block_tokens[b]
//
//   Reasons, in order of weight:
//     a. It removes a second quantisation. Storing a mean means narrowing an already-averaged vector,
//        which is the change the reference REVERTED for FP8 precisely because it moves the retrieval
//        RANKING (implementation_notes.md:389-393) -- the same argument applies to FP16 once the index
//        is the only thing standing between a query and the block it must find.
//     b. It is what makes INCREMENTAL CONSTRUCTION lossless across an arbitrary number of writes:
//        sums add, means do not. A block split over two prefill chunks, or grown by a decode step, can
//        be accumulated into the same slot with no bookkeeping beyond the token count. The sibling
//        llama.cpp port states this outright: "ordered K sum (F32) at first write, mean-K computed on
//        read", with the prefill path (write_layer_mean_k) and the incremental decode path
//        (write_layer_mean_sum) kept SEPARATE
//        (refs\kvmem\kvmem-llama.cpp-master\kvmem\include\kvmem\raw_kv_store.hpp:3-8, :60-63, :78-79).
//     c. It keeps the per-block token count meaningful at the point of use, so a PARTIAL tail block is
//        divided by the tokens it really holds, never by the nominal block size -- the defect the
//        mean-K step already had to fix once (mean_k_index.h:127-131).
//   The cost is one scalar multiply per (block, query head): normalisation is folded into the logit as
//   `logit = (scale / count) * dot(q, sum)`, mathematically identical to `scale * dot(q, mean)` and
//   needing no per-element division.
//   If a caller holds a pre-normalised mean (F32, or F16 to be widened), it passes count == 1.
//
//   DECISION ON RECORD: normalisation lives on THIS side of the interface, not in the caller.
//
// ★ LIVE-WINDOW CONTRACT (hard constraints, 2026-09-21; these are not preferences)
//   The raw-K/raw-Q design was re-decided: `ops::rope` cannot be made non-in-place in this engine, and
//   llama.cpp's "keep the tensor alive inside the graph" has NO counterpart here because each layer's
//   attn_mix is wrapped in a DeviceArena::Scope whose destructor rewinds the bump offset
//   (text_context_impl.h:1178, core/arena.cu:348-353) -- so the pre-RoPE key/query of layer l is dead
//   by the time layer l+1 runs. The consumer therefore reads INSIDE the window where the data is alive
//   (attn_mix: rmsnorm at :929, rope at :992, so kn/qn are live, pre-RoPE and L2-hot in between), and
//   nothing is persisted. This component is shaped around that:
//
//   1. CAPTURE-SAFE. No cudaMalloc, no cudaStreamSynchronize, no host round-trip: this is issued from
//      inside a captured region, where both are forbidden (pitfall table section 3.7). Allocation, if
//      any, belongs to the caller BEFORE capture.
//   2. IT CONSUMES LIVE POINTERS, NOT SHADOWS. The query operand is the live `qn`
//      (BF16, Tensor ne = {head_dim=256, n_heads=24, T}, ne[0] contiguous, so element (h,d) of token t
//      sits at t*n_heads*head_dim + h*head_dim + d). That is EXACTLY the addressing this scorer uses
//      -- `q_layer + (t*n_heads + h)*head_dim` -- so the live tensor needs no repacking. The index
//      operand is the persistent F32 mean-K sum. Neither is ever copied or kept alive by this
//      component; see kvmem_retrieve_launch.h for the exact strides.
//   3. IT IS CALLED PER LAYER AND ACCUMULATES. Because only ONE layer's qn is alive at a time, the
//      per-layer call is the primary mode: set n_layers = 1, pass that layer's live q plane and its
//      kbar slice, and leave n_layers_total at the model's full-attention count so head_w still
//      carries the mean over ALL layers. `score` is ADDED to, so the caller zeroes it once (before the
//      first layer, or before capture) and the 16 calls accumulate the same value the batched form
//      would produce. The multi-layer form (n_layers = n_layers_total) exists for a verifier and for a
//      future path that scores from a stored Q.
//   4. ONLY SMALL RESULTS CROSS THE CAPTURE BOUNDARY. The only output is the caller's float score
//      array (one float per block, pre-zeroed, plus the token counts it is given); the query and the
//      index never cross, and nothing here writes into either.
//   5. THE QUERY SPAN IS SIZED BY THE SPAN, NOT BY T. `n_query_tokens` is the marked span length
//      (service-level rule: the last ordinary `user` message, implementation_notes.md:448-484) and
//      `q_token_begin` is where that span starts inside the live plane's token axis. The kernel reads
//      ONLY those M rows, so a 512-token span costs 512 rows no matter how long the chunk is -- and the
//      "raw-Q is 6x the size of raw-K" accounting problem disappears, because nothing is stored.
//
// THE INVARIANT THAT MAKES THIS TESTABLE WITHOUT AN ORACLE
//   Because each (l,m,h) mass is a softmax over the included blocks, it sums to exactly 1 across
//   blocks, hence:
//
//     sum_b score[b] == n_query_tokens          (identical with or without the band mask)
//
//   This is an exact, oracle-free self-check that catches mass leaking, being counted twice, or
//   landing on the wrong block -- the failure mode aggregate error statistics cannot see. It also
//   holds under the per-layer accumulating call sequence, which is what makes that mode verifiable.
//   The standalone verifier asserts it alongside the FP64 comparison.
//
// WHAT THIS IS NOT
//   - It does not select blocks; it only scores them. Selection stays in kvmem_select.h.
//   - It does not capture the query and does not de-RoPE anything: the caller hands over qn AFTER its
//     rmsnorm and BEFORE the in-place rope, which is what "in the content frame" means here
//     (implementation_notes.md:427-446).
//   - n_subblocks (SubBlockMeanK), semantic groups, and the adaptive/DeltaNet scorers are NOT
//     implemented (kvmem_store.hpp:146-159). n_subblocks == 1 is the plain mean-K path and is
//     byte-identical to the reference's own n_subblocks==1 path (kernels_cuda.cu:5371-5376).
//
// ARITHMETIC
//   The sum is F32. The query operand is BF16 in the live window (the shape attn_mix hands over), FP16
//   if a caller ever persists captured rows the reference's way (implementation_notes.md:441-446), or
//   F32 as a diagnostic arm. The dot accumulates in FP32 from per-element conversions, exactly as
//   kernels_cuda.cu:5410-5415 does. Softmax statistics and score accumulation stay FP32 throughout.
//   Because the oracle reads the SAME narrowed operands, the residual between kernel and oracle is
//   arithmetic only, which is what makes a tight tolerance meaningful.

#include <cstdint>

namespace ninfer::ops {

// Precision of the QUERY operand only. The index operand is always F32 (sum) -- see the interface
// decision above. BF16 is what the live window gives us; F16 is what a persisted capture would use;
// F32 is the verifier's diagnostic arm.
enum class KvMemRetrieveDtype : std::uint8_t { BF16 = 0, F16 = 1, F32 = 2 };

// The always-kept bands excluded from the softmax, as an included range [lo_end, hi_begin).
// `active == false` means the whole history [0, n_blocks) competes.
struct KvMemRetrieveMask {
    std::int32_t lo_end   = 0;
    std::int32_t hi_begin = 0;
    bool         active   = false;
};

// How the caller wants the sink/recent bands treated.
//   Official -- the reference's own rule: mask only when budget > 0 AND n_blocks > budget AND a
//               non-empty middle survives (qwen_executor.cpp:23776-23792). Default, and the only mode
//               that reproduces the shipping configuration.
//   Never    -- score over every block (the reference's QW3_KVMEM_MASK_KEPT=0 arm).
//   Always   -- mask whenever a non-empty middle survives, regardless of the budget comparison.
//               Provided so the A/B can be run at any length; NOT the reference default.
enum class KvMemRetrieveMaskMode : std::uint8_t { Official = 0, Never = 1, Always = 2 };

struct KvMemRetrieveConfig {
    // Geometry of THIS CALL.
    std::int32_t n_layers       = 0;  // layers covered by this launch (1 in the live-window mode)
    std::int32_t n_query_tokens = 0;  // M: rows of the marked query span
    std::int32_t n_heads        = 0;  // QUERY heads
    std::int32_t n_kv_heads     = 0;  // index heads; n_heads must be a multiple of it (GQA group)
    std::int32_t head_dim       = 0;
    std::int32_t n_blocks       = 0;  // indexed blocks to score (1..kbar_layer_stride)

    // Total full-attention layers the score is averaged over. head_w = 1/(n_layers_total*n_heads), so
    // a per-layer call must still declare the model's full count or the accumulated score would come
    // out 16x too large. 0 means "same as n_layers" (the batched form).
    std::int32_t n_layers_total = 0;

    // Strides, each in its OWN units, and each may EXCEED the live count: the reference over-allocates
    // q rows (q_layer_stride, in rows) and index blocks (kbar_layer_stride, in blocks) so a growing
    // session keeps earlier slices valid (kernels_cuda.cu:5378-5382, 23794-23798). They are not
    // interchangeable with the counts.
    //   Live-window mode: q_layer_stride is the live plane's TOKEN count (T), which bounds the span.
    std::int32_t q_layer_stride    = 0;  // rows per layer in q    (>= q_token_begin + n_query_tokens)
    std::int32_t kbar_layer_stride = 0;  // blocks per layer in kbar (>= n_blocks)

    // Where the marked span starts inside the query operand's token axis. Nonzero when scoring
    // straight out of a live chunk-sized plane; 0 when the operand already is the span alone.
    std::int32_t q_token_begin = 0;

    // The keep allocation, used only by the mask rule (blocks, never tokens).
    std::int32_t budget_blocks = 0;
    std::int32_t sink_blocks   = 0;
    std::int32_t recent_blocks = 0;

    KvMemRetrieveMaskMode mask_mode = KvMemRetrieveMaskMode::Official;
    KvMemRetrieveDtype    dtype     = KvMemRetrieveDtype::BF16;  // QUERY dtype only
};

// scale = 1/sqrt(head_dim), exactly as src/qwen_executor.cpp:23755. Returned as float because that is
// the precision the score arithmetic runs in.
[[nodiscard]] float kvmem_retrieve_scale(std::int32_t head_dim) noexcept;

// The reference's band-mask rule, reduced to one pure function so it can be tested without a GPU.
// `budget_blocks`/`sink_blocks`/`recent_blocks` are clamped to n_blocks exactly as the reference clamps
// them (qwen_executor.cpp:23778-23779). Returns the included range; `active == false` means "score over
// the whole history".
[[nodiscard]] KvMemRetrieveMask kvmem_retrieve_mask_bands(const KvMemRetrieveConfig& config) noexcept;

// Throws std::invalid_argument when the geometry cannot be launched. Callers must treat "no query rows"
// and "no blocks" as NORMAL states (nothing to score), not as errors -- the same rule MeanKIndex
// applies to an empty index.
void kvmem_retrieve_validate(const KvMemRetrieveConfig& config);

} // namespace ninfer::ops
