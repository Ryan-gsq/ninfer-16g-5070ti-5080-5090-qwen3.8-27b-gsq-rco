#pragma once

// T2G128 RowSplit x int8 prefill GEMM with its own activation contract, in place of the shared
// rowsplit_a8_mma.cuh mainloop with the ternary codec. On the RTX 5060 Ti (tools/t2_a8_tile_probe.cu,
// T = 1024, L2 flushed) that one ran the text-layer shapes at 74-94 TOP/s, cuBLAS int8 at 170-184;
// this runs at 134-138 and beats cuBLAS with its conversions on every shape. Where the time went:
//
//   scale    the shared contract scales activations per (token, 64 k), so every two MMAs the int32
//            partial is converted and multiplied by two scales. Here the partial runs the whole
//            128-k weight group and the activation scale is per (token, 128 k). The conversion is a
//            float subtract: the accumulator starts at the bits of 1.5 * 2^23, so as a float it
//            reads that plus the dot.
//   tile     the shared T2 tile is 64 rows and every such block re-streams its activation band; a
//            ternary row is a quarter of an int8 one, so the band is the traffic. 128 x 64 tiles
//            with three blocks per SM won over 64 x 128 and 256 x 64.
//   stages   two stages of 128 k with one barrier each; three or four stages measure the same.
//   decode   thread tig of an MMA quad owns k in [32 tig, 32 tig + 32) of each group, so its codes
//            for a row are one 8-byte shared load and every code word decodes in nine instructions:
//            masked with 0x3333 its nibbles are PRMT selectors of the even codes as they stand,
//            shifted by two those of the odd ones. The quantiser stores activations in the k order
//            that produces (t2_prefill_permute16).
//
// Activations: codes [Tpad][K] s8, scales [K / 128][Tpad] binary16, Tpad a multiple of kColumns;
// padded columns are zero. Output rows must be a multiple of kRows.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"
#include "ops/linear/t2/t2_s8_ptq1_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct T2PrefillI8 {
    static constexpr int kGroupK    = 128; // one weight scale group and one pipeline stage
    static constexpr int kWarpsM    = 2;
    static constexpr int kWarpsN    = 2;
    static constexpr int kMTiles    = 4;
    static constexpr int kNTiles    = 4;
    static constexpr int kStages    = 2;
    static constexpr int kMinBlocks = 3;
    static constexpr int kThreads   = kWarpsM * kWarpsN * 32;
    static constexpr int kRows      = kWarpsM * kMTiles * 16;
    static constexpr int kColumns   = kWarpsN * kNTiles * 8;
    // A column's 128 bytes padded to 144 so the eight columns an MMA quad reads miss each other.
    static constexpr int kXStride    = 144;
    static constexpr int kCodeBytes  = kRows * kGroupK / 4;
    static constexpr int kXBytes     = kColumns * kXStride;
    static constexpr int kScaleBytes = kColumns * 2;
    static constexpr int kStage      = kCodeBytes + kXBytes + kScaleBytes;
    // Weight scales arrive eight groups (16 bytes) per row at a time, double-buffered.
    static constexpr int kRing       = 8;
    static constexpr int kRingBytes  = kRows * kRing * 2;
    static constexpr int kSmem       = kStages * kStage + 2 * kRingBytes;
    static constexpr int kQuantThreads = 256;

    static_assert(kSmem <= 48 * 1024, "the T2 prefill tile must fit static-size shared memory");
    static_assert(kScaleBytes % 16 == 0, "activation scales are staged 16 bytes at a time");
};

// PTQ1_0 variant of the tile above. A SEPARATE schedule struct rather than a parameterised
// T2PrefillI8, so that every constant the PQ2 instantiation reads keeps its present value by
// construction -- nothing here is reachable from `kPtq1 == false`.
//
// Shape COPIED FROM THE I LINE's s8 storage (ternary_rowsplit_mma_s8.cuh): 64 rows, a single
// buffered int8 weight plane at a 144-byte stride, activations double buffered. The two changes
// against the PQ2 tile above are both forced by the plane being 4x bigger -- PTQ1 stores ONE INT8
// BYTE per weight where T2G128 packs four 2-bit codes into one byte:
//   * kCodeBytes  128*128/4 = 4,096  ->  64*144 = 9,216   (1 byte per weight, padded stride)
//   * kRows       128                ->  64               (the budget below)
// The stride is 144 and not 128 for the same reason kXStride is 144: the eight rows an MMA quad
// reads at once must miss each other's banks.
//
// BUDGET. 128 rows, like the PQ2 tile above, because that is what amortises an activation band:
// per 128-k group the weights are 128*28 B while the activations are 64*128 B, so a 64-row tile pays
// 1.63x more bytes per output element than a 128-row one (2.44 B against 1.5 B). Measured framing:
// this route's sibling reaches 3.29k tok/s prefill on this card while the 64-row version read 1,370.
//   plane 128*144 = 18,432 | raw qs 128*24 = 3,072 | stages 2*(64*144 + 128) = 18,688
//   wscale ring 2*(128*8*2) = 4,096 | qh ring 2*(128*8*2) = 4,096  ->  48,384 B
// That is 2 blocks per SM (96,768 of the SM's 102,400); three no longer fit, which is the price of
// the wider tile. Which side wins is a measurement, not an argument -- see the landing table §28.19.
struct T2PrefillI8Ptq1 {
    static constexpr int kGroupK    = 128; // one weight scale group and one pipeline stage
    // EIGHT warps over 128 rows and the same 64 columns. This is the I line's own next-step list for
    // its wide tile ("or take the CTA to 8 warps"): the route is stall-bound rather than
    // bandwidth-bound -- measured then as ~100-120 GiB/s effective against 595 available, with only
    // 8 warps/SM resident -- and more warps per block buys latency hiding without spending a byte of
    // shared. It also halves the rounds of the in-shared decode pass (128*7 items over 256 threads
    // instead of over 128) and of the plane permutation (128*32 items), both of which sit on the
    // MMA's critical path.
    static constexpr int kWarpsM    = 2;
    static constexpr int kWarpsN    = 4;
    static constexpr int kMTiles    = 4;
    static constexpr int kNTiles    = 2;
    static constexpr int kStages    = 2;
    static constexpr int kMinBlocks = 2;
    static constexpr int kThreads   = kWarpsM * kWarpsN * 32;
    static constexpr int kRows      = kWarpsM * kMTiles * 16;
    static constexpr int kColumns   = kWarpsN * kNTiles * 8;
    static constexpr int kXStride   = 144;
    // One int8 per weight, padded: the plane is SINGLE buffered (the I line measured that it has to
    // be -- double buffering it does not fit), so it is not part of the stage ring.
    static constexpr int kPlaneStride = 144;
    static constexpr int kPlaneBytes  = kRows * kPlaneStride;
    // Raw staging is ONE group per row: its 24 qs bytes. NOTE (2026-10-01, landing table §28.27):
    // the "two groups per row / 64-byte stride" variant (tuning table item #4, weight-stream
    // continuity) was tried and WITHDRAWN -- with it the weight scale was read from the wrong half of
    // the pair for half the k-groups (per-group isolation sweep: pairs at odd index read a zero
    // scale). This one-group-per-row form is the measured green one (7 shapes x t=128, rel_l2
    // 0.00689543). #4 has to be redone against exp\ptq1-group-isolate.py before it comes back.
    static constexpr int kRawStride = PTQ1RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kRawBytes  = kRows * kRawStride;
    static constexpr int kXBytes     = kColumns * kXStride;
    static constexpr int kScaleBytes = kColumns * 2;
    // Stage ring: activations and activation scales only; the weight plane lives outside it.
    static constexpr int kStage = kXBytes + kScaleBytes;
    // Weight scales and the qh tail both arrive eight groups per row at a time, double buffered --
    // the same ring mechanism the PQ2 tile uses for its scales, reused for qh because a single
    // group's 2 qh bytes can satisfy neither cp.async's minimum size nor its alignment.
    static constexpr int kRing        = 8;
    static constexpr int kRingBytes   = kRows * kRing * 2;
    static constexpr int kQhRingBytes = kRows * kRing * 2;
    // plane (single buffered) | raw qs rows | scale ring x2 | qh ring x2 | activation stage ring
    static constexpr int kSmem =
        kPlaneBytes + kRawBytes + 2 * kRingBytes + 2 * kQhRingBytes + kStages * kStage;
    static constexpr int kQuantThreads = 256;

    static_assert(kSmem <= 48 * 1024, "the PTQ1 prefill tile must fit static-size shared memory");
    static_assert(kSmem * kMinBlocks <= 100 * 1024,
                  "two blocks per SM need the SM's whole 100 KiB shared budget");
    static_assert(kScaleBytes % 16 == 0, "activation scales are staged 16 bytes at a time");
    static_assert(kPlaneStride % 16 == 0, "the int8 plane is read as uint4 per lane");
    static_assert(kRawStride % 8 == 0, "the raw qs row is copied 8 bytes at a time");
    static_assert(kRawStride >= PTQ1RowSplitStorage::kCodeBytesPerGroup,
                  "a raw qs row must hold the whole group's 24 base bytes");
};

// Where k (mod 16) of a code word lands among the sixteen activation bytes the MMA pairs it with:
// the decode yields k {0,2,4,6}, {1,3,5,7}, {8,10,12,14}, {9,11,13,15} per register.
__host__ __device__ constexpr int t2_prefill_permute16(int k) {
    return (k / 8) * 8 + (k % 2) * 4 + (k % 8) / 2;
}

// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) to four registers of s8.
__device__ __forceinline__ void t2_prefill_decode(unsigned word, unsigned* d) {
    constexpr unsigned kTable = 0xFFFE0100u; // 0, +1, -2, -1
    const unsigned even       = word & 0x33333333u;
    const unsigned odd        = (word >> 2) & 0x33333333u;
    d[0]                      = __byte_perm(kTable, 0u, even);
    d[1]                      = __byte_perm(kTable, 0u, odd);
    d[2]                      = __byte_perm(kTable, 0u, even >> 16);
    d[3]                      = __byte_perm(kTable, 0u, odd >> 16);
}

// fwd()'s 3-cycles inside one 32-byte lane block, each triple written as (x, fwd(x), fwd(fwd(x))).
// The permutation "plane[P] = weight_order[fwd(P)]" is then a rotation of these eight triples plus
// eight fixed points (0, 7, 8, 15, 16, 23, 24, 31), which is why it can be done in place with one
// temporary and no scratch buffer. Cycle structure computed, not assumed -- landing table §28.12.
__device__ constexpr int kT2Ptq1Cycles[8][3] = {{1, 2, 4},   {3, 6, 5},   {9, 10, 12}, {11, 14, 13},
                                                {17, 18, 20}, {19, 22, 21}, {25, 26, 28}, {27, 30, 29}};

// out[row, t] = sum_k W[row, k] X[k, t] for t < tokens, through epilogue(row, token, value). With
// rows_fast false the grid is (column blocks, row blocks) and runs column blocks fastest, so a
// weight row block is read once from DRAM while the column blocks that need it pass through; with
// rows_fast it is (row blocks, column blocks), for activation bands too wide to stay in L2.
// kPtq1_ switches the weight packing: false is the T2G128 code plane (four 2-bit codes per byte, 32
// bytes per 128-group), true is PTQ1_0 (24 base bytes + 2 high bytes per group, decoded STRAIGHT to
// the int8 plane the MMA reads -- no 2-bit relabelling, see t2_s8_ptq1_decode.cuh).
//
// The parameter is trailing and defaulted and every kPtq1-dependent expression below is written so
// that it reduces to the literal it replaced when the flag is false, so the PQ2 path stays the
// instantiation it has always been -- member for member and byte for byte. `w_high` is appended
// with a default for the same reason: the existing call site does not mention it.
template <std::int32_t kCols, class Epilogue, bool kPtq1 = false>
__global__ __launch_bounds__(
    std::conditional_t<kPtq1, T2PrefillI8Ptq1, T2PrefillI8>::kThreads,
    std::conditional_t<kPtq1, T2PrefillI8Ptq1, T2PrefillI8>::kMinBlocks) void t2_prefill_i8_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t t_pad,
    std::int32_t tokens, bool rows_fast, Epilogue epilogue,
    const std::uint8_t* __restrict__ w_high = nullptr) {
    using C                     = std::conditional_t<kPtq1, T2PrefillI8Ptq1, T2PrefillI8>;
    // The launch bounds come from the schedule this instantiation uses. The two schedules no longer
    // share a thread count (the PTQ1 tile runs 8 warps to the PQ2 tile's 4) and no longer share a
    // block count per SM; both are per-schedule facts, and the launcher reads them off C.
    constexpr int kGroups   = kCols / C::kGroupK;
    // The code plane's row pitch in the artifact: PQ2 packs four codes per byte, PTQ1 keeps 24 base
    // bytes per group (its 2 high bytes live in their own plane, kHighRowBytes apart).
    constexpr int kRowBytes = kPtq1 ? kGroups * PTQ1RowSplitStorage::kCodeBytesPerGroup : kCols / 4;
    // kHighRowBytes is in BYTES (w_high is a byte pointer). There is deliberately no scale row-byte
    // constant: w_scales is `const __half*`, so its pitch is kGroups halves and any byte-flavoured
    // expression for it is a factor-of-two bug (landing table §28.27).
    constexpr int kHighRowBytes = kGroups * PTQ1RowSplitStorage::kHighBytesPerGroup;
    static_assert(kCols % (C::kGroupK * C::kRing) == 0, "K must be whole scale rings");
    static_assert(!kPtq1 || T2PrefillI8Ptq1::kRawStride >= PTQ1RowSplitStorage::kCodeBytesPerGroup,
                  "a raw qs row must hold the whole group's 24 base bytes");
    __shared__ __align__(16) char smem[C::kSmem];
    // PTQ1's int8 plane and raw qs rows sit OUTSIDE the stage ring: the plane is single buffered (it
    // does not fit twice), so only the activations and their scales rotate. With kPtq1 false the two
    // offsets are 0 and this is the same `smem + kStages * kStage` the PQ2 path has always used.
    [[maybe_unused]] char* const s_plane = smem;
    [[maybe_unused]] char* const s_raw   = smem + (kPtq1 ? T2PrefillI8Ptq1::kPlaneBytes : 0);
    char* const s_ring = smem + (kPtq1 ? T2PrefillI8Ptq1::kPlaneBytes + T2PrefillI8Ptq1::kRawBytes
                                       : C::kStages * C::kStage);
    [[maybe_unused]] char* const s_qh = s_ring + 2 * C::kRingBytes;
    char* const s_stage = kPtq1 ? s_qh + 2 * T2PrefillI8Ptq1::kQhRingBytes : smem;

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_m = warp / C::kWarpsN;
    const int warp_n = warp % C::kWarpsN;
    const int col0   = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * C::kColumns;
    const int row0   = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * C::kRows;

    const auto issue = [&](int g, int buf) {
        char* const s_w = kPtq1 ? s_raw + 0 : s_stage + buf * C::kStage;
        char* const s_x = kPtq1 ? s_stage + buf * C::kStage : s_w + T2PrefillI8::kCodeBytes;
        char* const s_s = s_x + C::kXBytes;
        if constexpr (kPtq1) {
            // One group per row: its 24 qs bytes, as three 8-byte copies. ca, not cg: cp.async.cg is
            // 16-byte only (see memory.cuh) and a 24-byte row stride is not 16-byte aligned.
            for (int c = tid; c < C::kRows * (T2PrefillI8Ptq1::kRawStride / 8); c += C::kThreads) {
                constexpr int kCopiesPerRow = T2PrefillI8Ptq1::kRawStride / 8;
                const int r                 = c / kCopiesPerRow;
                const int k                 = c - r * kCopiesPerRow;
                cp_async<8>(s_raw + r * C::kRawStride + k * 8,
                            w_codes + static_cast<std::size_t>(row0 + r) * kRowBytes +
                                g * PTQ1RowSplitStorage::kCodeBytesPerGroup + k * 8);
            }
        } else {
            for (int c = tid; c < C::kRows * 2; c += C::kThreads) {
                const int r = c >> 1;
                cp_async<16, Cache::cg>(s_w + r * 32 + (c & 1) * 16,
                                        w_codes + static_cast<std::size_t>(row0 + r) * kRowBytes +
                                            g * 32 + (c & 1) * 16);
            }
        }
        for (int c = tid; c < C::kColumns * 8; c += C::kThreads) {
            const int col = c >> 3;
            cp_async<16, Cache::cg>(s_x + col * C::kXStride + (c & 7) * 16,
                                    x + static_cast<std::size_t>(col0 + col) * kCols +
                                        g * C::kGroupK + (c & 7) * 16);
        }
        for (int c = tid; c < C::kScaleBytes / 16; c += C::kThreads) {
            cp_async<16, Cache::cg>(s_s + c * 16,
                                    xs + static_cast<std::size_t>(g) * t_pad + col0 + c * 8);
        }
        // Weight scales: eight groups per 16-byte ring row, double buffered. PQ2's ring is this one;
        // PTQ1 stages its qh ring in the same pass, because a single group's 2 bytes satisfy neither
        // cp.async's minimum size nor its alignment -- that is why a 16-byte copy covers eight groups.
        //
        // ⚠️ UNIT TRAP (landing table §28.27): `w_scales` is `const __half*`, so the row pitch is
        // kGroups HALVES (kGroups*2 bytes), NOT kGroups*2 halves. Passing a byte pitch here reads the
        // scale plane at twice its real row stride -- silently, and only for rows != 0 -- which is
        // exactly what the first item-#4 build did (rel_l2 0.0069 -> 0.769). `w_high` is a byte
        // pointer, so kHighRowBytes IS bytes; the two pointers do not take the same units.
        if (g % C::kRing == 0) {
            char* const ring = s_ring + ((g / C::kRing) & 1) * C::kRingBytes;
            if constexpr (kPtq1) {
                char* const qh_ring = s_qh + ((g / C::kRing) & 1) * C::kQhRingBytes;
                for (int r = tid; r < C::kRows; r += C::kThreads) {
                    cp_async<16, Cache::cg>(ring + r * 16,
                                            w_scales +
                                                static_cast<std::size_t>(row0 + r) * kGroups + g);
                    cp_async<16, Cache::cg>(
                        qh_ring + r * 16,
                        w_high + static_cast<std::size_t>(row0 + r) * kHighRowBytes +
                            (g / C::kRing) * 16);
                }
            } else {
                for (int r = tid; r < C::kRows; r += C::kThreads) {
                    cp_async<16, Cache::cg>(ring + r * 16,
                                            w_scales +
                                                static_cast<std::size_t>(row0 + r) * kGroups + g);
                }
            }
        }
        cp_commit();
    };

    float acc[C::kMTiles][C::kNTiles][4];
#pragma unroll
    for (int m = 0; m < C::kMTiles; ++m) {
#pragma unroll
        for (int n = 0; n < C::kNTiles; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[m][n][j] = 0.0F; }
        }
    }

#pragma unroll
    for (int i = 0; i < C::kStages - 1; ++i) { issue(i, i); }

    for (int g = 0; g < kGroups; ++g) {
        cp_wait<C::kStages - 2>();
        __syncthreads();
        if constexpr (kPtq1) {
            // Turn the group's raw 24+2 bytes into the int8 plane the MMA reads. ONE pass and ONE
            // barrier: each slot scatters its four bytes straight to the plane positions the A
            // fragments read, so there is no second segment to wait for. It used to be two (decode
            // in weight order, then permute the row), which cost an extra barrier plus a full
            // read-modify-write of the plane on every group -- all of it on the MMA's critical path.
            // The scatter form is judged against that two-segment pair in _tucheck_ptq1_scatter.cu
            // (32768/32768 plane bytes identical over 256 groups, byte-flip control moves).
            //
            // This still sits BEFORE issue(next): the raw rows are single buffered, and the barrier
            // below is what makes it safe for the next group's copies to overwrite them.
            const std::uint8_t* const qh_ring =
                reinterpret_cast<const std::uint8_t*>(s_qh + ((g / C::kRing) & 1) * C::kQhRingBytes);
            char* const plane = s_plane;
            // ABLATION RESULT (landing table §28.25): with this pass skipped the tile reads
            // 3,140 tok/s against 1,740 with it -- the decode is 44% of every group's time, and
            // without it PTQ1 lands at PQ2's own level (3,650). That is the whole remaining gap, so
            // the next change has to remove this pass rather than shorten it.
            // WORD stores, not the scatter's twenty byte stores per item (landing table §28.36):
            // t2_s8_ptq1_decode_slot writes each stage as ONE 32-bit store at its slot's step --
            // five stores per item against twenty -- and emits WEIGHT order. The plane wants the fwd()
            // order, so the MMA side applies that permutation in REGISTERS when it loads its lane
            // block (eight __byte_perm per 32 bytes; the rule t2_s8_ptq1_plane_permute states and
            // _tucheck_ptq1_s8_gate.cu verified at 32768/32768). Fifteen fewer shared stores per item
            // for eight register permutes per lane block, and no barrier moves anywhere.
            for (int i = tid; i < C::kRows * 7; i += C::kThreads) {
                const int r    = i / 7;
                const int slot = i - r * 7;
                t2_s8_ptq1_decode_slot(
                    reinterpret_cast<const std::uint8_t*>(s_raw) + r * C::kRawStride,
                    qh_ring + r * 16 + (g % C::kRing) * 2,
                    reinterpret_cast<std::uint8_t*>(plane + r * C::kPlaneStride), slot);
            }
            __syncthreads();
        }
        const int next = g + C::kStages - 1;
        if (next < kGroups) {
            issue(next, next % C::kStages);
        } else {
            cp_commit();
        }

        const char* const s_w = kPtq1 ? s_raw : s_stage + (g % C::kStages) * C::kStage;
        const char* const s_x =
            kPtq1 ? s_stage + (g % C::kStages) * C::kStage : s_w + T2PrefillI8::kCodeBytes;
        const __half* const s_s = reinterpret_cast<const __half*>(s_x + C::kXBytes);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / C::kRing) & 1) * C::kRingBytes);
        const int slot = g % C::kRing;

        unsigned b[C::kNTiles][8];
        float xa[C::kNTiles][2];
#pragma unroll
        for (int n = 0; n < C::kNTiles; ++n) {
            const char* p  = s_x + ((warp_n * C::kNTiles + n) * 8 + gid) * C::kXStride + tig * 32;
            const uint4 v0 = load_vec<uint4>(p);
            const uint4 v1 = load_vec<uint4>(p + 16);
            b[n][0]        = v0.x;
            b[n][1]        = v0.y;
            b[n][2]        = v0.z;
            b[n][3]        = v0.w;
            b[n][4]        = v1.x;
            b[n][5]        = v1.y;
            b[n][6]        = v1.z;
            b[n][7]        = v1.w;
            const int c    = (warp_n * C::kNTiles + n) * 8 + tig * 2;
            xa[n][0]       = __half2float(s_s[c]);
            xa[n][1]       = __half2float(s_s[c + 1]);
        }

#pragma unroll
        for (int m = 0; m < C::kMTiles; ++m) {
            const int r0   = (warp_m * C::kMTiles + m) * 16 + gid;
            unsigned d0[8];
            unsigned d1[8];
            if constexpr (kPtq1) {
                // The shared plane holds int8 in WEIGHT order (the word-store decode above), so each
                // 32-byte lane block is permuted into the fragment order here, in registers, by the
                // word-pair rule t2_s8_ptq1_plane_permute states: out[2j] takes bytes {0,2,4,6} of the
                // pair and out[2j+1] takes {1,3,5,7}. Eight __byte_perm per block; no shared traffic.
                const uint4 p0 = load_vec<uint4>(s_plane + r0 * C::kPlaneStride + tig * 32);
                const uint4 p1 = load_vec<uint4>(s_plane + r0 * C::kPlaneStride + tig * 32 + 16);
                const uint4 p2 = load_vec<uint4>(s_plane + (r0 + 8) * C::kPlaneStride + tig * 32);
                const uint4 p3 = load_vec<uint4>(s_plane + (r0 + 8) * C::kPlaneStride + tig * 32 + 16);
                const unsigned in0[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
                const unsigned in1[8] = {p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    d0[2 * j]     = __byte_perm(in0[2 * j], in0[2 * j + 1], 0x6420u);
                    d0[2 * j + 1] = __byte_perm(in0[2 * j], in0[2 * j + 1], 0x7531u);
                    d1[2 * j]     = __byte_perm(in1[2 * j], in1[2 * j + 1], 0x6420u);
                    d1[2 * j + 1] = __byte_perm(in1[2 * j], in1[2 * j + 1], 0x7531u);
                }
            } else {
            const uint2 w0 = load_vec<uint2>(s_w + r0 * 32 + tig * 8);
            const uint2 w1 = load_vec<uint2>(s_w + (r0 + 8) * 32 + tig * 8);
            t2_prefill_decode(w0.x, d0);
            t2_prefill_decode(w0.y, d0 + 4);
            t2_prefill_decode(w1.x, d1);
            t2_prefill_decode(w1.y, d1 + 4);
            }
            const float ws0 = __half2float(ring[r0 * C::kRing + slot]);
            const float ws1 = __half2float(ring[(r0 + 8) * C::kRing + slot]);
#pragma unroll
            for (int n = 0; n < C::kNTiles; ++n) {
                constexpr int kMagicBits = 0x4B400000; // 1.5 * 2^23
                int c[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    mma_s8(c[0], c[1], c[2], c[3], d0[2 * s], d1[2 * s], d0[2 * s + 1],
                           d1[2 * s + 1], b[n][2 * s], b[n][2 * s + 1]);
                }
                constexpr float kMagic = 12582912.0F;
                acc[m][n][0] = fmaf(__int_as_float(c[0]) - kMagic, ws0 * xa[n][0], acc[m][n][0]);
                acc[m][n][1] = fmaf(__int_as_float(c[1]) - kMagic, ws0 * xa[n][1], acc[m][n][1]);
                acc[m][n][2] = fmaf(__int_as_float(c[2]) - kMagic, ws1 * xa[n][0], acc[m][n][2]);
                acc[m][n][3] = fmaf(__int_as_float(c[3]) - kMagic, ws1 * xa[n][1], acc[m][n][3]);
            }
        }
    }

#pragma unroll
    for (int n = 0; n < C::kNTiles; ++n) {
        const int c0 = col0 + (warp_n * C::kNTiles + n) * 8 + tig * 2;
#pragma unroll
        for (int m = 0; m < C::kMTiles; ++m) {
            const int r0 = row0 + (warp_m * C::kMTiles + m) * 16 + gid;
            if (c0 < tokens) {
                epilogue(r0, c0, acc[m][n][0]);
                epilogue(r0 + 8, c0, acc[m][n][2]);
            }
            if (c0 + 1 < tokens) {
                epilogue(r0, c0 + 1, acc[m][n][1]);
                epilogue(r0 + 8, c0 + 1, acc[m][n][3]);
            }
        }
    }
}

// One block per padded column, a warp per 128-k group, a lane per four k. Padded columns get zero
// codes and scales. A zero group keeps a zero scale and zero codes.
template <std::int32_t kCols>
__global__ __launch_bounds__(T2PrefillI8::kQuantThreads) void t2_prefill_i8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t tokens, std::int32_t t_pad,
    std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    using C               = T2PrefillI8;
    constexpr int kGroups = kCols / C::kGroupK;
    constexpr int kWarps  = C::kQuantThreads / 32;
    const int t           = static_cast<int>(blockIdx.x);
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    std::int8_t* const dst = codes + static_cast<std::size_t>(t) * kCols;
    if (t >= tokens) {
        for (int i = static_cast<int>(threadIdx.x); i < kCols / 16; i += C::kQuantThreads) {
            store_vec(dst + i * 16, make_uint4(0u, 0u, 0u, 0u));
        }
        for (int g = static_cast<int>(threadIdx.x); g < kGroups; g += C::kQuantThreads) {
            scales[static_cast<std::size_t>(g) * t_pad + t] = __float2half(0.0F);
        }
        return;
    }
    const __nv_bfloat16* const src = x + static_cast<std::size_t>(t) * kCols;
    // Lane l writes bytes [4 l, 4 l + 4) of the group, which t2_prefill_permute16 fills from the
    // even (lanes 4q, 4q+2) or odd (4q+1, 4q+3) codes of lanes 4q + (l & 2) and the one after.
    const int source = lane & ~1;
    const int shift  = (lane & 1) * 16;
    for (int g = warp; g < kGroups; g += kWarps) {
        const uint2 raw = load_vec<uint2>(src + g * C::kGroupK + lane * 4);
        const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float f[4];
        float amax = 0.0F;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[j] = __bfloat162float(v[j]);
            amax = fmaxf(amax, fabsf(f[j]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        const __half scale = __float2half(amax / 127.0F);
        if (lane == 0) { scales[static_cast<std::size_t>(g) * t_pad + t] = scale; }
        const float s   = __half2float(scale);
        const float inv = s > 0.0F ? 1.0F / s : 0.0F;
        unsigned q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = static_cast<unsigned>(max(-127, min(127, __float2int_rn(f[j] * inv)))) & 0xffu;
        }
        const unsigned packed = q[0] | (q[2] << 8) | (q[1] << 16) | (q[3] << 24);
        const unsigned a      = __shfl_sync(0xffffffffu, packed, source);
        const unsigned b      = __shfl_sync(0xffffffffu, packed, source + 1);
        const unsigned word   = ((a >> shift) & 0xffffu) | (((b >> shift) & 0xffffu) << 16);
        store_vec(dst + g * C::kGroupK + lane * 4, word);
    }
}

} // namespace ninfer::ops::detail






