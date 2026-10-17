#pragma once

// T2G128 RowSplit x BF16 WIDE-T tensor-core GEMM (the prefill / large-T rung), PTQ1_0 packing.
//
// PORTED FROM: E:\infer-build\fusion\src\src\ops\linear\ternary\ternary_rowsplit_mma_wide_t.cuh
// (617 lines, sha256 prefix 351DD6C75D83). This file is a verbatim STRUCTURAL port: same constants
// under their original names, same TernaryWideStorage layout, same staging / mma / store structure,
// same `template <bool kPtq1>` and __launch_bounds__. Every `kTernaryWide*` name is kept verbatim
// and was grep-verified to collide with nothing in this tree (the only prior mention of any of them
// is a comment in t2_dispatch.cpp:122), so the two files can be diffed symbol for symbol.
//
// The port carries exactly TWO deviations, both convention items, each marked `// PORT:` where it
// lands:
//   1. DECODE TABLE. This container stores the ternary payload as two-bit TWO'S COMPLEMENT, the I
//      line's as OFFSET-BINARY (t2_rowsplit_storage.cuh:3-6 and :63-79). A stored field therefore
//      means {0, +1, -2, -1} for s = 0,1,2,3 here -- the table t2_prefill_i8.cuh:151 states as
//      `kTable = 0xFFFE0100u` -- and NOT the I line's `(field - 1)`. Copying the original's table
//      would shift every decoded weight by one trit: no crash, no NaN, just a quieter model.
//   2. REPACK. The raw 24+2-byte PTQ1_0 row is turned into the 32 code bytes by this tree's shared
//      ops/linear/t2/t2_ptq1_repack.cuh, which already emits `(xi - 1) & 3`, instead of the I
//      line's inline copy of the same base-3 chain. The raw staging row and the seven-slot
//      destination layout are unchanged, so the helper's contract is met verbatim.
// The TERNARY_WIDE_ABLATE_LUT ablation switch is deliberately NOT ported (see the note at the LUT
// read below).
//
// WHY THIS EXISTS -- the weight-pass law (measured 2026-09-20, I line)
// The small-t kernel puts TOKENS in the outer loop and restages the weight window for every 8-token
// tile, so one forward pass reads the entire weight tensor ceil(T/8) times. Measured on the real
// engine: prefill at T=62 costs 131 ms (= 8 passes x 6.70 GiB = 438 GiB/s), and the fitted law over
// a 14x range of T is t ~= 12.31 ms * ceil(T/8) + 44 ms (<= 6% error, 544 GiB/s at the slope),
// while the official runtime on the same card reads the weights about once (pp512 = 2087 t/s
// against 471-630). The harness carries the same fingerprint: the small-t kernel measures
// 14.56 / 16.50 / 15.58 ms at T = 1 / 3 / 8 -- one pass each, flat, which a compute-bound kernel
// could not be -- and then 22.66 -> 90.00 ms from T = 16 to 64, i.e. 2 -> 8 passes, exactly 3.97x
// against the predicted 4x. The same law explains the other two oddities: 5-lane concurrency at 25%
// efficiency (11 tokens do not fit one 8-token tile, so it pays 2 passes) and why attaching MTP
// paid off immediately (a T=3 verify pass already rides one pass).
//
// STRUCTURE -- the one thing two independent sources agree on
//   * A operand (weights): staged once per K-chunk, then reused by ALL tokens.
//   * B operand (activations): double buffered in the original; SINGLE buffered here (see below).
//   * K is the outer loop; the token sub-tile is inner.
// Upstream ninfer's q4/q8 rowsplit MMA has exactly this shape (its As[BM*BK] carries no stage
// dimension while Bs[S][BN*BK] does), and Marlin states the same preference from the other side
// (activations served from L2, weights loaded asynchronously and evicted immediately so they never
// pollute it). The small-t kernel in either tree is the opposite of both, which is why it caps
// prefill at one weight pass per 8 tokens.
//
// WHY BK IS ONE GROUP (128) AND NOT 512
// The activation tile costs BK*2 bytes per token: at BK=512 that is 1 KB/token, so 48 KB of static
// shared caps the token tile at ~10 -- which is why the shipped kernel chose 8 tokens. At BK=128 it
// costs 272 B/token, which is what buys a 64-token tile. Shared budget (stages = 2, one whole group
// per staged chunk):
//   lut 256*8 = 2048 | codes 2*4*16*48 = 6144 | raw 2*4*16*32 = 4096 | act 64*136*2 = 17408
//   | scales 2*4*16*2 = 256   ->   total 29952 B = 29.25 KiB  <=  48 KB static
//   =>  3 CTAs/SM on sm_89 (3 * 29952 = 89856 B <= the 100 KB an SM has), which is what
//   kTernaryWideMinBlocksPerSm = 3 below budgets registers for.
// PORT: the original's header states 43264 B / 42.25 KB and "2 CTAs/SM" (its own launcher comment
// repeats it). That accounting is stale in the source: it counts the activation tile TWICE although
// the struct single buffers it (the struct's own comment says so), and it omits the 4 KB raw window
// entirely. No size, constant or schedule moves here -- only the number in the comment is
// corrected, because a stale shared-memory figure is what the next person sizes an occupancy
// experiment with.
//
// LAYOUT CONTRACTS (same as the small-t kernel -- violating them makes T == 1 look perfect while
// every prefill scrambles):
//   * activations are TOKEN-major: (column, token) lives at token*k + column.
//   * output is token-major too: (row, token) at token*out_row_stride + row.
//   * the code plane packs four 2-bit codes per byte, lowest k in the lowest bits, and the stored
//     field s means the value {0, +1, -2, -1} for s = 0,1,2,3 (two's complement; see the decode
//     table); the group scale is applied AFTER the mma, per row. With BK == group size there is
//     exactly one scale application per chunk, which is also why that is the natural chunk.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/t2/t2_ptq1_repack.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

inline constexpr int kTernaryWideRowsPerWarp = 16;
// 4 warps (64 rows) per CTA -- the MEASURED optimum of the configurations tried on 2026-09-20, with
// the reason recorded, because the naive reading of the profiler says otherwise and would send the
// next person down the same dead end:
//   * Nsight Compute on this build: Active Warps/Scheduler 2.72 (max 12), No Eligible 77.3%,
//     12 warp-cycles per issued instruction, "Est. Local Speedup 52.49% from occupancy",
//     DRAM at only 19.7%. It reads like a pure occupancy problem.
//   * Raising occupancy to 16 warps/SM (8 warps/CTA + a 128-register cap) made everything WORSE:
//     T=64 went 52.88 -> 70.89 ms (+34%). The register cap is what did it -- this kernel's
//     accumulators alone are acc[8][4] + tmp[8][4] = 64 floats per thread, so the cap spilled
//     (40 B spill stores / 32 B spill loads) and the local-memory traffic cost more than the four
//     extra warps bought.
//   * Occupancy here is NOT free: it has to be earned by making the kernel need fewer registers,
//     not by capping them. The way to do that is a half-tile structure (run 32 of the 64 tokens at
// a time while keeping the weight panel staged for both halves), which halves acc[] and tmp[] to
//     16 floats each and leaves the pass count at ceil(T/64). That is the open TODO. Do NOT simply
//     raise the warp count and cap registers.
inline constexpr int kTernaryWideWarpsPerCta = 4;
inline constexpr int kTernaryWideRowsPerCta  = kTernaryWideRowsPerWarp * kTernaryWideWarpsPerCta;
// Token tile. Evidence for the value: upstream's per-T rungs keep BN=32 for the T ~ 12..110 band
// and reach for 64 only in the middle band and 128 at the very top; 64 is the widest that still
// fits 3 CTAs/SM here, and this kernel exists to serve prefill, not the verify pass.
inline constexpr int kTernaryWideTokens   = 64;
// K consumed per staged chunk == one group of the container (128 codes).
inline constexpr int kTernaryWideChunkK   = T2RowSplitStorage::kGroupK;
inline constexpr int kTernaryWideKSteps   = kTernaryWideChunkK / 16;   // mma k-steps per chunk
inline constexpr int kTernaryWideCodesPerRow = kTernaryWideChunkK / 4; // four codes per byte
// 32 used bytes, padded to 48 so the four-byte words the quad reads land on distinct banks across
// the eight rows one load touches (12 words apart => 12*r mod 32 is distinct for r = 0..7, the same
// argument the small-t kernel makes for its stride).
inline constexpr int kTernaryWideRowStride = 48;
// Raw PTQ1_0 staging row: the 24 base bytes + the 2 high bytes, padded to 32. The pad is what makes
// every row start 8-byte aligned, which is the alignment a cp.async of that size needs (see
// stage_weights), and it keeps the row inside one bank-friendly power of two.
inline constexpr int kTernaryWideRawStride = 32;
inline constexpr int kTernaryWideActStride = kTernaryWideChunkK + 8;   // 16B-aligned, padded
inline constexpr int kTernaryWideStages    = 2;
inline constexpr int kTernaryWideThreads   = kTernaryWideWarpsPerCta * 32;
// Minimum resident CTAs per SM, i.e. the occupancy ptxas must budget registers for.
//
// This is the number the 2026-09-20 occupancy experiment pinned down. At BN=64 the first version of
// this kernel used 166 registers and 43.3 KB of shared, landed at 2 CTAs/SM (8 warps/SM) and
// measured 53.3 ms per weight pass -- about 31% of this card's FLOP peak, meaning it was LATENCY
// bound, not bandwidth bound (the same pass moved 6.65 GiB, i.e. an effective ~125 GiB/s against
// the ~595 GiB/s the card actually delivers on a streaming read). Holding everything else fixed and
// dropping only the token tile to 32 cut shared to 25.9 KB, raised occupancy to 12 warps/SM and
// took that same pass to 32.6 ms -- +63% per-pass efficiency from occupancy alone.
//
// So the target is 3 CTAs/SM while KEEPING BN=64 (the wide tile is what keeps the pass count at
// ceil(T/64), and at T=64 that pass count matters more than the per-pass win: BN=32 at T=64 costs 2
// x 32.6 = 65.2 ms against 53.3 ms for one 64-wide pass). Three CTAs needs shared <= ~33 KB, which
// is why the activation tile is single buffered, and registers <= 170, which is why this constant
// exists.
inline constexpr int kTernaryWideMinBlocksPerSm = 3;
// Below this the 64-wide tile wastes more than it saves; the small-t kernel (one pass up to 8
// tokens) is the right rung and stays the validated default for the verify pass. In THIS tree that
// rung is launch_t2_small_t_v2_ptq1 (T <= 16, sliced above that, see t2_dispatch.cpp:117-146).
//
// 2026-09-20 (SECOND measurement): 41, not 9. The original 9 counted only the number of passes each
// kernel needs and missed that ONE wide pass costs ~5x one small pass (53.8 ms vs 10.2 ms over the
// artifact's real shape mix). Measured with the token sweep extended to cover the 16..64 gap
// (tscale_bench.cu -> E:\infer-build\tscale_thr.txt):
//
//   T     small_t   wide_t     winner
//    9     20.29     50.69     small 2.50x
//   11     19.67     48.59     small 2.47x   <- where a multi-lane decode lands
//   16     20.84     49.24     small 2.36x
//   24     29.74     53.75     small 1.81x
//   32     40.75     54.54     small 1.34x
//   40     51.02     54.74     small 1.07x   <- last token count small_t wins
//   48     60.90     54.79     wide  1.11x
//   64     83.60     53.81     wide  1.55x
//
// small_t == 10.2 ms * ceil(T/8) and wide_t is flat at ~54 ms, so the crossover is where ceil(T/8)
// = 6, i.e. T = 41. At 9 every pass of 9..40 tokens ran the slower kernel, and that band is exactly
// where a batched pass lands (the linear op sees T = tokens-per-lane * active lanes, so 3+ lanes
// with MTP drafts sit inside it).
inline constexpr int kTernaryWideMinTokens = 41;

// Runtime override, for the same-binary A/B this threshold requires: decode speed has to be judged
// by the engine, not by the harness (the harness ranks T=1 mma 34% faster and the engine measures
// it 3.1% slower -- see the note in ternary_rowsplit_gemm.cu).
//   NINFER_TERNARY_WIDE_MIN_TOKENS=<n>   candidate threshold; unset keeps the default.
// Read once: this decides which kernel enters the captured CUDA graph, so it must not change
// between capture and replay (same rule as NINFER_TERNARY_MMA / _HADAMARD).
//
// PORT: kept verbatim, but nothing reads it in this tree yet -- the rung gate lives in
// t2_dispatch.cpp (t2_ptq1_fast / select_t2_launch) and wiring this rung into it is a change to an
// existing file, which this port is not allowed to make. Until then the helper is inert.
[[nodiscard]] inline int ternary_wide_min_tokens() {
    static const int value = [] {
        const char* text = std::getenv("NINFER_TERNARY_WIDE_MIN_TOKENS");
        if (text == nullptr) { return kTernaryWideMinTokens; }
        const int parsed = std::atoi(text);
        return parsed > 0 ? parsed : kTernaryWideMinTokens;
    }();
    return value;
}

static_assert((kTernaryWideRowStride % 16) == 0, "cp.async needs a 16-byte aligned row start");
static_assert(kTernaryWideRowStride >= kTernaryWideCodesPerRow, "the row must hold a whole chunk");
static_assert((kTernaryWideActStride % 8) == 0, "ldmatrix wants 8-element row starts");
static_assert((kTernaryWideChunkK % T2RowSplitStorage::kGroupK) == 0,
              "a chunk must be a whole number of groups");
static_assert(kTernaryWideChunkK == T2RowSplitStorage::kGroupK,
              "this kernel applies one scale per chunk, so a chunk IS one group");

// -----------------------------------------------------------------------------------------------
// THE DECODE TABLE'S SEMANTICS -- deviation 1 of the port, and the only place the two containers'
// encodings are related.
//
// A trit is stored here as the two-bit field s = (xi - 1) & 3, where xi in {0,1,2} is the base-3
// digit PTQ1_0 carries (t2_ptq1_repack.cuh:16-36). So s = 3, 0, 1 mean -1, 0, +1, and s = 2 -- the
// -2 that T2's artifact language excludes (t2_rowsplit_storage.cuh:5-6) -- is unreachable from the
// repack. The I line stored xi itself and read `field - 1`, which is the SAME value map only if the
// fields are relabelled, so the table below is what makes the two halves (repack + decode) meet.
//
// Written as a branch chain rather than an array for the same reason ptq1_pow3 is
// (t2_rowsplit_storage.cuh:87-91): no device-side array lookup, and no host variable referenced
// from device code.
__host__ __device__ constexpr int t2_wide_field_value(int field) {
    return field == 0 ? 0 : field == 1 ? 1 : field == 2 ? -2 : -1;
}

// The same four values as the int8 tile keeps them in ONE word (t2_prefill_i8.cuh:151,
// `kTable = 0xFFFE0100u`). This is the check that makes the table falsifiable: write the I line's
// `field - 1` chain above and this fires with 0x020100FF instead -- i.e. every weight one trit off,
// which is exactly the failure a reviewer cannot see in the numbers.
constexpr std::uint32_t t2_wide_table_s8_word() {
    std::uint32_t word = 0u;
    for (int field = 0; field < 4; ++field) {
        word |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(t2_wide_field_value(field)))
                << (8 * field);
    }
    return word;
}
static_assert(t2_wide_table_s8_word() == 0xFFFE0100u,
              "the decode table must read 0, +1, -2, -1 -- t2_prefill_i8.cuh's kTable language; "
              "an offset-binary table would shift every weight by one trit");

union TernaryWidePairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

struct TernaryWideStorage {
    uint2 lut[256];
    alignas(16) std::uint8_t codes[kTernaryWideStages][kTernaryWideWarpsPerCta]
                                  [kTernaryWideRowsPerWarp][kTernaryWideRowStride];
    // Raw PTQ1_0 staging window, cp.async'd and THEN repacked (see repack_ptq1): one 24-byte qs
    // plus a 2-byte qh per (warp, row), per stage. Only the kPtq1 instantiation touches it.
    // 2*4*16*32 = 4 KB. With the single-buffered activation this puts the CTA at 29.25 KB, and 3 *
    // 29.25 KB = 87.75 KB still fits the 100 KB/SM sm_89 has, so kTernaryWideMinBlocksPerSm stays
    // reachable.
    alignas(16) std::uint8_t raw[kTernaryWideStages][kTernaryWideWarpsPerCta]
                                [kTernaryWideRowsPerWarp][kTernaryWideRawStride];
    // Activation: SINGLE buffer, deliberately. This is what takes shared from 43.3 KB down to what
    // buys the third CTA per SM (see kTernaryWideMinBlocksPerSm). It costs the overlap of one 16 KB
    // activation copy per chunk, paid against ~1.3 ms of compute per chunk at BN=64 -- under 1%.
    // The weight window above keeps its double buffer, because that is the operand whose latency
    // actually needs hiding.
    alignas(16) __nv_bfloat16 act[kTernaryWideTokens][kTernaryWideActStride];
    std::uint16_t scales[kTernaryWideStages][kTernaryWideWarpsPerCta][kTernaryWideRowsPerWarp];
};

// The three sizing claims the constants above are chosen against, asserted instead of trusted:
// static shared has to fit the 48 KB a kernel gets without an opt-in (the launcher hands out no
// dynamic size and calls no cudaFuncSetAttribute), and 3 resident CTAs have to fit the SM.
static_assert(sizeof(TernaryWideStorage) <= 48 * 1024,
              "the wide-T staging must fit static-size shared memory");
static_assert(sizeof(TernaryWideStorage) * kTernaryWideMinBlocksPerSm <= 100 * 1024,
              "three CTAs per SM at this staging size need the SM's whole 100 KiB shared budget");

// kPtq1 == false: `codes` are the T2G128 code plane (32 code bytes per row per group, positional:
//                 byte b holds weights 4b..4b+3).
// kPtq1 == true : `codes` are PTQ1_0 (24 base bytes) and `high` carries its 2 extra bytes. Staging
//                 is TWO steps, so the global latency is not on the critical path: stage_weights
//                 cp.async's the raw 24+2-byte row into `raw` one chunk ahead of use, and
//                 repack_ptq1() turns it into the same in-shared code layout after cp_wait.
//                 Every line below the repack is byte for byte the code-plane path (LUT, mma,
//                 scale folding, stores). The repack is bit-exact on the TRIT
//                 (xi = ((uint8)(raw * 3^t) * 3) >> 8, verified against real weights with both
//                 verified decoders (exp\reader-candidates\verify_pq2_repack.py -> 0 / 122,880
//                 mismatches) and the convention relabelling it then applies is the one this
//                 container requires.
template <bool kPtq1>
__global__ __launch_bounds__(kTernaryWideThreads, kTernaryWideMinBlocksPerSm)
void ternary_wide_t_kernel(const __nv_bfloat16* __restrict__ x,
                           const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ high,
                           const std::uint8_t* __restrict__ scales,
                           __nv_bfloat16* __restrict__ out, std::int32_t rows,
                           std::int32_t k, std::int32_t tokens,
                           std::int32_t out_row_stride) {
    __shared__ TernaryWideStorage staging;
    auto& lut      = staging.lut;
    auto& codes_sh = staging.codes;
    auto& raw_sh   = staging.raw;
    auto& act_sh   = staging.act;
    auto& scale_sh = staging.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2; // 0..7 -- mma row within the fragment
    const int lid  = lane & 3;  // 0..3 -- mma k pair / column pair index

    // Decode table: packed byte -> four ternary weights in bf16, built from t2_wide_field_value so
    // there is ONE statement of what a stored field means. The multiply is exact: every entry is 0
    // or a power of two with a sign, so no rounding happens anywhere on this path. Read only after
    // the first __syncthreads() below.
    //
    // PORT: the I line wrote `(i & 3) - 1` (offset-binary) here. That is wrong for this container
    // -- see the table's own comment; this is deviation 1 and the one line a `diff` will show.
    for (int i = tid; i < 256; i += kTernaryWideThreads) {
        TernaryWidePairBits low;
        TernaryWidePairBits high;
        low.pair = __floats2bfloat162_rn(static_cast<float>(t2_wide_field_value(i & 3)),
                                         static_cast<float>(t2_wide_field_value((i >> 2) & 3)));
        high.pair = __floats2bfloat162_rn(static_cast<float>(t2_wide_field_value((i >> 4) & 3)),
                                          static_cast<float>(t2_wide_field_value((i >> 6) & 3)));
        uint2 entry;
        entry.x = low.bits;
        entry.y = high.bits;
        lut[i]  = entry;
    }

    const int row0 = static_cast<int>(blockIdx.x) * kTernaryWideRowsPerCta;
    if (row0 >= rows) { return; }

    constexpr int kCodeBytes =
        kPtq1 ? PTQ1RowSplitStorage::kCodeBytesPerGroup : T2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kHighBytes = kPtq1 ? PTQ1RowSplitStorage::kHighBytesPerGroup : 0;
    const std::int32_t groups_per_row = k / T2RowSplitStorage::kGroupK;
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kCodeBytes;
    const std::int64_t high_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kHighBytes;
    // BYTES, not halves: `scales` is a byte pointer here and the pitch below is bytes per row,
    // while t2_prefill_i8.cuh takes its scales as `const __half*` and therefore a pitch in halves.
    // The two conventions coexist in this tree and mixing them is a silent factor of two, not a
    // crash (landing table §28.27 records exactly that mistake in the int8 tile).
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * T2RowSplitStorage::kScaleBytesPerGroup;
    const std::int32_t chunks = k / kTernaryWideChunkK;
    const std::int64_t warp_row_base =
        static_cast<std::int64_t>(row0) + static_cast<std::int64_t>(warp) * kTernaryWideRowsPerWarp;

    // Weight staging (double buffered): one chunk = this warp's 16 x 32-byte code window plus its
    // 16 per-row group scales.
    const auto stage_weights = [&](int buf, int chunk) {
        if constexpr (kPtq1) {
            // PREFETCH ONLY. The repack is repack_ptq1() below, which runs after these copies have
            // landed. Splitting the two is the entire point of this shape: the 24+2-byte source row
            // used to be read with plain loads whose latency sat directly on top of the dependent
            // repack arithmetic (LDG -> the byte_perm chain -> shared store), so every chunk paid a
            // full global round trip before it could store a single code byte. Now the global read
            // is an asynchronous copy issued one chunk ahead (hidden behind the previous chunk's
            // mma) and the repack reads SHARED.
            //
            // COPY WIDTH: qs is 24 bytes and the per-chunk stride is exactly 24 bytes, so an odd
            // chunk starts at an address that is 8-byte but NOT 16-byte aligned. The copy is
            // therefore three 8-byte cp.async rather than the 16+8 the 24-byte size suggests -- a
            // 16-byte cp.async on an odd chunk is a misaligned copy. 8-byte alignment always holds:
            // the chunk stride is 24 (24 % 8 == 0) and so is a row (groups_per_row * 24).
            constexpr int kQsCopies     = PTQ1RowSplitStorage::kCodeBytesPerGroup / 8; // 3
            constexpr int kQsCopiesWarp = kTernaryWideRowsPerWarp * kQsCopies;
            static_assert(PTQ1RowSplitStorage::kCodeBytesPerGroup % 8 == 0,
                          "a PTQ1_0 row copy is 8-byte granular, that is what cp.async needs");
            static_assert(kTernaryWideRawStride >= PTQ1RowSplitStorage::kCodeBytesPerGroup,
                          "the raw row must hold a whole qs");
#pragma unroll
            for (int copy = lane; copy < kQsCopiesWarp; copy += 32) {
                const int local_row           = copy / kQsCopies;
                const int byte_off            = (copy % kQsCopies) * 8;
                const std::int64_t global_row = warp_row_base + local_row;
                const bool valid              = global_row < rows;
                // An invalid row zero-fills (src_bytes = 0) instead of branching to a plain shared
                // store: the repack then produces the same decoded weights the old
                // `valid ? repack : 0u` wrote (a zeroed raw byte has xi = 0, i.e. value -1, in both
                // conventions -- here as the field 3, there as the field 0), and it needs no
                // validity test at all. The fallback source is the plane base, which is always
                // mapped -- with src_bytes == 0 nothing is read from it.
                cp_async_zfill<8, Cache::ca>(
                    &raw_sh[buf][warp][local_row][byte_off],
                    valid ? codes + global_row * code_row_bytes +
                                static_cast<std::int64_t>(chunk) * kCodeBytes + byte_off
                          : codes,
                    valid ? 8 : 0);
            }
            // qh is only 2 bytes, so it is the one part that must NOT go through cp.async: the
            // smallest copy cp_async offers is 4 bytes (memory.cuh:37), which would read past the
            // end of the high plane on the last row's last group. One plain load + plain shared
            // store per row per chunk.
            if (lane < kTernaryWideRowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                const std::uint16_t qh =
                    (global_row < rows)
                        ? load_vec<std::uint16_t>(high + global_row * high_row_bytes +
                                                  static_cast<std::int64_t>(chunk) * kHighBytes)
                        : 0u;
                // 24 is even and the raw row 32 bytes, so this 2-byte store is naturally aligned.
                store_vec<std::uint16_t>(
                    reinterpret_cast<std::uint16_t*>(
                        &raw_sh[buf][warp][lane][PTQ1RowSplitStorage::kCodeBytesPerGroup]),
                    qh);
            }
        } else {
        // Codes: kTernaryWideCodesPerRow bytes per row as 16-byte copies, round-robin across the
        // lanes. Written generically (never sized by hand for one chunk width) because a hand-sized
        // loop silently stages only PART of each row at a different width and presents as a fast
        // kernel producing garbage.
        constexpr int kCopiesPerRow  = kTernaryWideCodesPerRow / 16;
        constexpr int kCopiesPerWarp = kTernaryWideRowsPerWarp * kCopiesPerRow;
#pragma unroll
        for (int copy = lane; copy < kCopiesPerWarp; copy += 32) {
            const int local_row = copy / kCopiesPerRow;
            const int byte_off  = (copy % kCopiesPerRow) * 16;
            const std::int64_t global_row = warp_row_base + local_row;
            std::uint8_t* dst             = &codes_sh[buf][warp][local_row][byte_off];
            if (global_row < rows) {
                cp_async<16, Cache::cg>(
                    dst, codes + global_row * code_row_bytes +
                             static_cast<std::int64_t>(chunk) * kTernaryWideCodesPerRow + byte_off);
            } else {
                store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
        }

        { // one scale per (row, chunk), because a chunk is exactly one group
            if (lane < kTernaryWideRowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                std::uint16_t value           = 0u;
                if (global_row < rows) {
                    value = load_vec<std::uint16_t>(
                        scales + global_row * scale_row_bytes +
                        static_cast<std::int64_t>(chunk) * T2RowSplitStorage::kScaleBytesPerGroup);
                }
                scale_sh[buf][warp][lane] = value;
            }
        }
    };

    // REPACK: raw (24-byte qs + 2-byte qh) -> the in-shared T2 code layout, so every line
    // downstream of this step is byte for byte the code-plane path. Called one chunk after
    // stage_weights issued the copies, i.e. after cp_wait, when the global latency has already been
    // absorbed by the previous chunk's mma.
    //
    // PORT (deviation 2): the arithmetic is NOT the I line's inline `code_of`/`stage_codes` pair.
    // It is this tree's shared repack, ops/linear/t2/t2_ptq1_repack.cuh -- the same base-3 /
    // byte_perm chain (PrismML's ggml_cuda_mmq_decode_ptq1_0_qs4, MIT) with the one thing this
    // container needs and the I line's must not have: every emitted code is `(xi - 1) & 3` instead
    // of `xi` (t2_ptq1_repack.cuh:53-57 for the qh tail, :86 for the qs stages). The two agree on
    // the trit, which is the part the original proved bit-exact against two independent decoders,
    // and differ only in that relabelling -- i.e. the convention boundary is in exactly one place,
    // shared with the small-t and int8 rungs.
    //
    // The row contract also matches verbatim: the staged raw row is qs[0..24) then qh[0..2) at
    // PTQ1RowSplitStorage::kCodeBytesPerGroup, stride kTernaryWideRawStride -- which is what the
    // helper's contiguous form documents at t2_ptq1_repack.cuh:38-42 -- and this is the flattened
    // warp loop t2_ptq1_repack.cuh:148-152 says the wide-t kernel wants.
    //
    // Seven jobs per row, flattened over the warp's 32 lanes; each job owns a DISJOINT set of the
    // 32 destination code bytes (the helper's own contract, unchanged from the I line):
    //   slot 0..3: qs[4g..4g+4)      at stages 0..4 -> code bytes 4t+g    (weights 0..79)
    //   slot 4..5: qs[16+4p..20+4p)  at stages 0..4 -> code bytes 20+2t+p (weights 80..119)
    //   slot 6   : qh[0..2) -> byte 30 (trits 0,0,1,1) and byte 31 (trits 2,2,3,3)
    const auto repack_ptq1 = [&](int buf) {
        if constexpr (kPtq1) {
            t2_ptq1_repack_warp(&raw_sh[buf][warp][0][0], kTernaryWideRawStride,
                                &codes_sh[buf][warp][0][0], kTernaryWideRowStride,
                                kTernaryWideRowsPerWarp, lane);
        }
    };

    // Activation staging (single buffer, whole CTA cooperates): the kTernaryWideTokens x
    // kTernaryWideChunkK tile for one chunk, 16 bytes (8 bf16) per copy. Offsets are in bf16
    // ELEMENTS: x is a __nv_bfloat16* while the code staging above is byte-based, and mixing the
    // two over-advances the source window by 2x (values plausible, NaN at the tail). Tokens past
    // the end of the sequence are zero-filled, so the mma sees a defined activation and the guarded
    // store drops the result.
    const auto stage_act = [&](int chunk, int tok_base) {
        constexpr int kCopiesPerToken = kTernaryWideChunkK / 8;
        constexpr int kActCopies      = kTernaryWideTokens * kCopiesPerToken;
        for (int copy = tid; copy < kActCopies; copy += kTernaryWideThreads) {
            const int tok        = copy / kCopiesPerToken;
            const int elem       = (copy % kCopiesPerToken) * 8;
            const int global_tok = tok_base + tok;
            const int safe_tok   = (global_tok < tokens) ? global_tok : tok_base;
            cp_async_zfill<16>(
                &act_sh[tok][elem],
                x + static_cast<std::int64_t>(safe_tok) * k +
                    static_cast<std::int64_t>(chunk) * kTernaryWideChunkK + elem,
                (global_tok < tokens) ? 16 : 0);
        }
    };

    // Walk the sequence in tiles of kTernaryWideTokens TOKENS (outer), and inside each tile walk K
    // (inner). This is the whole point of the kernel: the weight window for a chunk is staged once
    // and every token sub-tile consumes it, so the pass count is ceil(T / kTernaryWideTokens)
    // instead of ceil(T / 8).
    const int row_lo = row0 + warp * kTernaryWideRowsPerWarp + gid;
    const int row_hi = row_lo + 8;
    constexpr int kSubTiles = kTernaryWideTokens / 8;

    for (int tok_base = 0; tok_base < tokens; tok_base += kTernaryWideTokens) {
        float acc[kSubTiles][4];
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[sub][i] = 0.0f; }
        }

        // Prologue: the first chunk's weights and activations, then one full wait. From here the
        // weight window is double buffered and the activation is not (see the struct).
        stage_weights(0, 0);
        stage_act(0, tok_base);
        cp_commit();
        cp_wait<0>();
        if constexpr (kPtq1) {
            // cp.async completion is only ordered for the thread that ISSUED the copy, while the
            // repack reads raw bytes that other lanes of this warp copied -- so one warp fence
            // first. The codes_sh writes the repack makes are read by this same warp in the loop
            // below, and the __syncthreads() right after orders them for every lane.
            __syncwarp();
            repack_ptq1(0);
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int buf = chunk & 1;
            // Issue the next weight window straight away: its latency hides behind the compute
            // below, which is the whole reason this operand gets two buffers.
            if (chunk + 1 < chunks) {
                stage_weights(buf ^ 1, chunk + 1);
                cp_commit();
            }

            // One group per chunk, so the mma result for the whole chunk takes a single scale per
            // row: accumulate the k-steps in a temporary, then fold into the running accumulator.
            float tmp[kSubTiles][4];
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { tmp[sub][i] = 0.0f; }
            }

#pragma unroll
            for (int step = 0; step < kTernaryWideKSteps; ++step) {
                const int word_off = step * 4; // one k-step of 16 codes == one 4-byte word

                // A operand: rows gid and gid+8, each contributing k and k+8 pairs.
                const unsigned word_lo =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid][word_off]);
                const unsigned word_hi =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid + 8][word_off]);
                const unsigned shift_near = 8u * static_cast<unsigned>(lid >> 1);
                const unsigned shift_far  = 8u * static_cast<unsigned>((lid >> 1) + 2);

                // PORT: the original's TERNARY_WIDE_ABLATE_LUT branch (a measurement-only build
                // that replaces these four shared reads with constants and is wrong by
                // construction) is NOT ported: this tree has no such ablation switch anywhere, and
                // a switch that silently computes garbage is a trap in a file whose whole job is
                // the convention. A/B it by editing these four lines in a scratch copy instead.
                const uint2 near_lo = lut[(word_lo >> shift_near) & 0xFFu];
                const uint2 far_lo  = lut[(word_lo >> shift_far) & 0xFFu];
                const uint2 near_hi = lut[(word_hi >> shift_near) & 0xFFu];
                const uint2 far_hi  = lut[(word_hi >> shift_far) & 0xFFu];

                // A quad of lanes shares a byte; lanes with an odd lid need its high pair.
                const bool odd = (lid & 1) != 0;
                const unsigned a0 = odd ? near_lo.y : near_lo.x;
                const unsigned a1 = odd ? near_hi.y : near_hi.x;
                const unsigned a2 = odd ? far_lo.y : far_lo.x;
                const unsigned a3 = odd ? far_hi.y : far_hi.x;

#pragma unroll
                for (int sub = 0; sub < kSubTiles; ++sub) {
                    // B operand: the activation tile is already n-major (token rows, k contiguous)
                    // -- exactly the .col layout the mma wants, no transpose.
                    unsigned bf0 = 0u;
                    unsigned bf1 = 0u;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&act_sh[sub * 8 + (lane & 7)]
                                                [step * 16 + ((lane >> 3) & 1) * 8]));
                    mma_bf16(tmp[sub][0], tmp[sub][1], tmp[sub][2], tmp[sub][3], a0, a1, a2,
                             a3, bf0, bf1);
                }
            }

            const float top =
                __half2float(__ushort_as_half(scale_sh[buf][warp][gid]));
            const float bottom =
                __half2float(__ushort_as_half(scale_sh[buf][warp][gid + 8]));
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
                acc[sub][0] = fmaf(tmp[sub][0], top, acc[sub][0]);
                acc[sub][1] = fmaf(tmp[sub][1], top, acc[sub][1]);
                acc[sub][2] = fmaf(tmp[sub][2], bottom, acc[sub][2]);
                acc[sub][3] = fmaf(tmp[sub][3], bottom, acc[sub][3]);
            }

            __syncthreads(); // every read of the single activation buffer is done
            if (chunk + 1 < chunks) {
                stage_act(chunk + 1, tok_base); // reuse the activation buffer
                cp_commit();
                cp_wait<0>();                   // next chunk's weights AND activation landed
                if constexpr (kPtq1) {
                    // Repack the window this iteration prefetched (buf ^ 1 == (chunk+1) & 1) while
                    // nothing else needs this warp's lanes: its destination, codes_sh[buf ^ 1], is
                    // not read until the next iteration, and the __syncthreads() below publishes
                    // those shared writes before that happens. __syncwarp is the fence that makes
                    // this warp's own cp.async bytes readable by its other lanes (see the
                    // prologue).
                    __syncwarp();
                    repack_ptq1(buf ^ 1);
                }
                __syncthreads();
            }
        }

        // C fragment: c0/c1 are rows gid, columns 2*lid / 2*lid+1; c2/c3 are rows gid+8.
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
            const int token_a = tok_base + sub * 8 + 2 * lid;
            const int token_b = token_a + 1;
            if (row_lo < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][0]);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][1]);
                }
            }
            if (row_hi < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][2]);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][3]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
