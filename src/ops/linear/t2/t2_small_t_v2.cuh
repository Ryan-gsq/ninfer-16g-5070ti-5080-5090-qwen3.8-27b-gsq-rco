#pragma once

// T2G128 RowSplit x BF16 small-T tensor-core GEMM, second design (T = 1..16), laid out like
// q4_small_t_mma.cuh: eight warps per CTA, KWarps of them split every K slab (64 k per warp, half a
// T2 group), and the CTA covers 8 / KWarps sets of TilesPerWarp sixteen-row tiles over the same K.
// Codes and scales are staged by cp.async a few slabs deep; activations are read straight from
// global memory through L1, where the CTAs resident on an SM share them (per-CTA staging fetches
// them from L2 once per CTA, which at T = 8 moves more bytes than the codes do).
//
// K order inside a warp's 64-k slice: lane lid owns k = 16 * lid + j, j = 0..15, one 32-bit code
// word per row and sixteen consecutive activations per column. The MMA's k slots may be any
// permutation of those sixteen as long as A and B agree, so the four k16 steps take
//   step 0: slots (2lid, 2lid+1) = j {0, 2}, slots (2lid+8, 2lid+9) = j {4, 6}
//   step 1: {1, 3} / {5, 7}        step 2: {8, 10} / {12, 14}        step 3: {9, 11} / {13, 15}
// which makes the A selectors two masks of the code word (`word & 0x33333333` holds the even codes
// as prmt nibbles, `(word >> 2) & 0x33333333` the odd ones) looked up in the constant
// {0, +1, -1} bf16 byte tables, and the B fragments byte permutes of the lane's activation
// vectors. The group scale multiplies the slab's accumulator once per row.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/t2/t2_ptq1_repack.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

// kPtq1_ switches what the weight rows hold. false is the T2G128_F16S code plane: 32 two-bit code
// bytes per 128-group, read straight into the A fragments. true is PTQ1_0: 24 base-3 bytes plus 2
// tail bytes per group, which the staging REPACKS into the same 32 code bytes before any consumer
// sees them -- so every line below the staging is the same code for both packings.
//
// Why repack rather than decode base-3 straight into the A fragment (which is what the I line's
// equivalent kernel ended up doing): this kernel's A path derives all four fragment registers of a
// k16 step from ONE 32-bit word, i.e. it requires sixteen codes of one group at ONE base-3 stage.
// PTQ1_0's map does not give that -- weights 80..95 need stages 0 and 1, 96..111 stages 2 and 3, and
// 112..127 stages 4 plus the qh tail's 0..3 -- so the word cannot be built without restructuring the
// whole operand path. The I line paid that cost to save its own 2 KiB uint2 lut[256]; this A path
// has no such table (two constant registers), so there is nothing here to save.
//
// The parameter defaults to false, so every existing instantiation -- and with it the entire T2
// path -- is unchanged, byte for byte.
template <int KWarps_, int TilesPerWarp_, int ColumnTiles_, int Stages_, int MinBlocks_,
          bool kPtq1_ = false>
struct T2SmallTv2Schedule {
    static constexpr bool kPtq1              = kPtq1_;
    static constexpr int kWarps               = 8;
    static constexpr int kThreads             = kWarps * 32;
    static constexpr int kKWarps              = KWarps_;
    static constexpr int kTilesPerWarp        = TilesPerWarp_;
    static constexpr int kRowTiles            = kWarps / kKWarps * kTilesPerWarp;
    static constexpr int kRows                = 16 * kRowTiles;
    static constexpr int kColumnTiles         = ColumnTiles_;
    static constexpr int kColumns             = 8 * kColumnTiles;
    static constexpr int kSlabK               = 64 * kKWarps;
    static constexpr int kGroupsPerSlab       = kSlabK / T2RowSplitStorage::kGroupK;
    static constexpr int kCodeBytesPerRowSlab = kSlabK / T2RowSplitStorage::kCodesPerByte;
    // A 16-byte pad staggers the rows' code words across the banks for the k_split x lid reads.
    static constexpr int kCodeRowStride = kCodeBytesPerRowSlab + 16;
    static constexpr int kScaleBytesPerRowSlab =
        kGroupsPerSlab * T2RowSplitStorage::kScaleBytesPerGroup;
    static constexpr int kStages    = Stages_;
    static constexpr int kMinBlocks = MinBlocks_;

    // PTQ1_0 raw plane. The ARTIFACT keeps a row's slab as one contiguous run of 24-byte qs groups --
    // which is 16-byte granular and therefore copies as cp.async -- with the 2-byte tails in the
    // high plane, PLANE_ALIGNMENT away. So the two runs are staged separately and the repack stitches
    // them; the qh run sits after the qs run inside the row, at kRawQhOffset.
    static constexpr int kRawQhOffset     = kGroupsPerSlab * PTQ1RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kRawQhBytes      = kGroupsPerSlab * PTQ1RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kRawRowStride    = (kRawQhOffset + kRawQhBytes + 15) / 16 * 16;
    static constexpr int kRawChunksPerRow = kRawQhOffset / 16;
    static_assert(!kPtq1 || (kRawQhOffset % 16) == 0,
                  "the qs run must be whole 16-byte cp.async chunks");
    static_assert(!kPtq1 || (kRawQhBytes == 4 || kRawQhBytes == 8),
                  "the qh run is one cp.async of the slab's tail bytes");
    static_assert(!kPtq1 || kCodeBytesPerRowSlab == kGroupsPerSlab * 32,
                  "a repacked group must occupy exactly the 32 code bytes the A path reads");

    static_assert(kKWarps == 4 || kKWarps == 8, "T2 small-T v2 splits K over four or eight warps");
    static_assert(kTilesPerWarp == 1 || kTilesPerWarp == 2,
                  "T2 small-T v2 gives a warp one or two tiles");
    static_assert(kColumnTiles == 1 || kColumnTiles == 2,
                  "T2 small-T v2 covers one or two n8 tiles");
    static_assert(kStages >= 1 && kStages <= 4, "T2 small-T v2 pipeline depth must fit cp_wait");
    static_assert(kMinBlocks >= 1, "T2 small-T v2 launch-bounds occupancy must be positive");

    // The raw planes only exist for the PTQ1 packing. Making them an empty base rather than a
    // zero-length array keeps the T2 stage exactly the size and layout it has today (an empty base
    // costs nothing), so nothing that depends on the T2 staging shifts by a byte.
    struct Ptq1RawPlane {
        __align__(16) std::uint8_t raw[kRows][kRawRowStride];
    };
    struct NoRawPlane {};

    struct Stage : std::conditional_t<kPtq1, Ptq1RawPlane, NoRawPlane> {
        __align__(16) std::uint8_t codes[kRows][kCodeRowStride];
        __align__(16) std::uint8_t scales[kRows][kScaleBytesPerRowSlab];
    };

    // The K reduction reuses the stage ring once every slab has been consumed.
    static constexpr int kPartialFloats = kWarps * kTilesPerWarp * kColumnTiles * 32 * 4;

    union Shared {
        Stage stages[kStages];
        float partial[kPartialFloats];
    };

    static constexpr int kSharedBytes = static_cast<int>(sizeof(Shared));
    static_assert(kSharedBytes <= 48 * 1024,
                  "T2 small-T v2 stages exceed the static shared budget");
};

// {0, +1, -1} as bf16 (0x0000, 0x3F80, 0xBF80), one byte table per half, indexed by the code
// (the illegal field 2 decodes as 0).
constexpr std::uint32_t kT2v2LutLow  = 0x80008000u;
constexpr std::uint32_t kT2v2LutHigh = 0xBF003F00u;

// Four selector nibbles (codes) -> two bf16x2 fragment registers: codes 0, 1 of the selector in
// `first`, codes 2, 3 in `second`.
__device__ __forceinline__ void t2_v2_a_fragment(std::uint32_t selector, unsigned& first,
                                                 unsigned& second) {
    const std::uint32_t low  = __byte_perm(kT2v2LutLow, 0u, selector);
    const std::uint32_t high = __byte_perm(kT2v2LutHigh, 0u, selector);
    first                    = __byte_perm(low, high, 0x5140);
    second                   = __byte_perm(low, high, 0x7362);
}

// The selector of k16 step `step` from the even/odd masks of a lane's code word.
__device__ __forceinline__ std::uint32_t t2_v2_step_selector(std::uint32_t even, std::uint32_t odd,
                                                             int step) {
    const std::uint32_t base = (step & 1) ? odd : even;
    return (step & 2) ? (base >> 16) : base;
}

// B fragments of k16 step `step` from the lane's sixteen activations (xa = j 0..7, xb = j 8..15).
__device__ __forceinline__ void t2_v2_b_fragment(const uint4& xa, const uint4& xb, int step,
                                                 unsigned& b0, unsigned& b1) {
    const uint4& source     = (step & 2) ? xb : xa;
    const std::uint32_t sel = (step & 1) ? 0x7632u : 0x5410u;
    b0                      = __byte_perm(source.x, source.y, sel);
    b1                      = __byte_perm(source.z, source.w, sel);
}

// `high` is the PTQ1_0 high plane and is null for the T2 packing, whose rows fit entirely in the
// code plane.
template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) void t2_small_t_v2_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t cols) {
    constexpr int kTpw    = Schedule::kTilesPerWarp;
    constexpr int kNt     = Schedule::kColumnTiles;
    constexpr int kKW     = Schedule::kKWarps;
    constexpr int kStages = Schedule::kStages;
    using Stage           = typename Schedule::Stage;

    __shared__ __align__(16) typename Schedule::Shared shared;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int k_split    = warp % kKW;
    const int first_tile = warp / kKW * kTpw;
    const int row0       = static_cast<int>(blockIdx.x) * Schedule::kRows;

    const int groups_per_row = k / T2RowSplitStorage::kGroupK;
    const int slabs          = k / Schedule::kSlabK;
    // A PTQ1_0 row is two runs with two strides: the 24-byte qs groups and the 2-byte qh tails, in
    // different planes. A T2 row is one 32-byte-per-group run. (Which of the two is live is a
    // compile-time property of the schedule, so this folds.)
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) *
        (Schedule::kPtq1 ? PTQ1RowSplitStorage::kCodeBytesPerGroup
                         : T2RowSplitStorage::kCodeBytesPerGroup);
    const std::int64_t high_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PTQ1RowSplitStorage::kHighBytesPerGroup;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * T2RowSplitStorage::kScaleBytesPerGroup;
    const int code_byte     = k_split * 16 + 4 * lid; // the lane's code word inside the row's slab
    const int group_in_slab = k_split >> 1;
    const int slice_k       = k_split * 64 + 16 * lid; // the lane's first k inside the slab

    const auto issue = [&](int stage_index, int slab) {
        Stage& stage = shared.stages[stage_index];
        if constexpr (Schedule::kPtq1) {
            // Stage the RAW PTQ1_0 bytes, not codes: the row's qs run as whole 16-byte cp.async
            // chunks (which is exactly why the artifact keeps it contiguous) and its qh tail in one
            // 4- or 8-byte chunk. Nothing is decoded on the way in -- the repack below runs in shared
            // memory, after the bytes have landed.
            constexpr int kChunksPerRow    = Schedule::kRawChunksPerRow;
            constexpr int kChunks          = Schedule::kRows * kChunksPerRow;
            constexpr int kChunksPerThread = (kChunks + Schedule::kThreads - 1) / Schedule::kThreads;
#pragma unroll
            for (int i = 0; i < kChunksPerThread; ++i) {
                const int item = tid + i * Schedule::kThreads;
                if (item < kChunks) {
                    const int r = item / kChunksPerRow;
                    const int c = item - r * kChunksPerRow;
                    cp_async<16, Cache::cg>(
                        &stage.raw[r][c * 16],
                        codes + static_cast<std::int64_t>(row0 + r) * code_row_bytes +
                            static_cast<std::int64_t>(slab) * Schedule::kRawQhOffset + c * 16);
                }
            }
            if (tid < Schedule::kRows) {
                // ca, not cg: cp.async.cg is 16-byte only (see memory.cuh).
                cp_async<Schedule::kRawQhBytes>(
                    &stage.raw[tid][Schedule::kRawQhOffset],
                    high + static_cast<std::int64_t>(row0 + tid) * high_row_bytes +
                        static_cast<std::int64_t>(slab) * Schedule::kRawQhBytes);
            }
        } else {
            constexpr int kChunksPerRow    = Schedule::kCodeBytesPerRowSlab / 16;
            constexpr int kChunks          = Schedule::kRows * kChunksPerRow;
            constexpr int kChunksPerThread = (kChunks + Schedule::kThreads - 1) / Schedule::kThreads;
#pragma unroll
            for (int i = 0; i < kChunksPerThread; ++i) {
                const int item = tid + i * Schedule::kThreads;
                if (item < kChunks) {
                    const int r = item / kChunksPerRow;
                    const int c = item - r * kChunksPerRow;
                    cp_async<16, Cache::cg>(
                        &stage.codes[r][c * 16],
                        codes + static_cast<std::int64_t>(row0 + r) * code_row_bytes +
                            static_cast<std::int64_t>(slab) * Schedule::kCodeBytesPerRowSlab + c * 16);
                }
            }
        }
        if (tid < Schedule::kRows) {
            cp_async<Schedule::kScaleBytesPerRowSlab>(
                &stage.scales[tid][0],
                scales + static_cast<std::int64_t>(row0 + tid) * scale_row_bytes +
                    static_cast<std::int64_t>(slab) * Schedule::kScaleBytesPerRowSlab);
        }
    };

    float acc[kTpw][kNt][4] = {};

#pragma unroll
    for (int prefetch = 0; prefetch < kStages - 1; ++prefetch) {
        if (prefetch < slabs) { issue(prefetch, prefetch); }
        cp_commit(); // empty commits keep cp_wait<kStages - 1> exact through the tail
    }

#pragma unroll 1
    for (int slab = 0; slab < slabs; ++slab) {
        {
            const int next = slab + kStages - 1;
            if (next < slabs) { issue(next % kStages, next); }
            cp_commit();
        }
        cp_wait<kStages - 1>();
        __syncthreads();
        if constexpr (Schedule::kPtq1) {
            // Turn the staged raw bytes into the T2 code plane, in place, before anything reads it.
            // No second buffer: the seven slots of a group own disjoint destination bytes, and the
            // destination region is never staged into, so the pass is free to overwrite.
            //
            // Three divergence-free loops rather than one flat 7-way loop, because seven slots over
            // 32 lanes puts lanes with different slots in one instruction and the slot branches then
            // serialise. The I line wrote that split down and measured it; this keeps it.
            //   slots 0..3: qs[4g..4g+4)      at stages 0..4 -> code bytes 4t+g
            //   slots 4..5: qs[16+4p..20+4p)  at stages 0..4 -> code bytes 20+2t+p
            //   slot 6    : the group's qh pair -> code bytes 30 and 31
            //
            // Every slot calls the adapter, so the repack arithmetic has exactly one home (and it is
            // the home the rung gate covers: t2_ptq1_repack_slot_split vs t2_ptq1_repack_slot,
            // 65,536/65,536 groups agree). This kernel only decides which thread owns which job.
            Stage& staged = shared.stages[slab % kStages];
            constexpr int kGroupCodeBytes = T2RowSplitStorage::kCodeBytesPerGroup;
            constexpr int kGroupRawBytes  = PTQ1RowSplitStorage::kCodeBytesPerGroup;
            constexpr int kGroupHighBytes = PTQ1RowSplitStorage::kHighBytesPerGroup;
            constexpr int kUnits          = Schedule::kRows * Schedule::kGroupsPerSlab;
            // A job is one (row, group) unit crossed with one slot; unit = job / kSlots, slot =
            // job % kSlots, and the three loops differ only in how many slots they walk.
            const auto run_slot = [&](int unit, int slot) {
                const int r           = unit / Schedule::kGroupsPerSlab;
                const int g           = unit - r * Schedule::kGroupsPerSlab;
                std::uint8_t* dst     = &staged.codes[r][g * kGroupCodeBytes];
                const std::uint8_t* qs = &staged.raw[r][g * kGroupRawBytes];
                const std::uint8_t* qh = &staged.raw[r][Schedule::kRawQhOffset + g * kGroupHighBytes];
                t2_ptq1_repack_slot_split(qs, qh, dst, slot);
            };
#pragma unroll
            for (int job = tid; job < kUnits * 4; job += Schedule::kThreads) {
                run_slot(job >> 2, job & 3);
            }
#pragma unroll
            for (int job = tid; job < kUnits * 2; job += Schedule::kThreads) {
                run_slot(job >> 1, 4 + (job & 1));
            }
#pragma unroll
            for (int job = tid; job < kUnits; job += Schedule::kThreads) {
                run_slot(job, 6);
            }
        }
        __syncthreads();
        const Stage& stage = shared.stages[slab % kStages];

        // B: the lane's sixteen activations of each live column of its tiles.
        uint4 xa[kNt];
        uint4 xb[kNt];
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const int col = nt * 8 + gid;
            xa[nt]        = make_uint4(0u, 0u, 0u, 0u);
            xb[nt]        = make_uint4(0u, 0u, 0u, 0u);
            if (col < cols) {
                const __nv_bfloat16* column = x + static_cast<std::int64_t>(col) * k +
                                              static_cast<std::int64_t>(slab) * Schedule::kSlabK +
                                              slice_k;
                xa[nt]                      = load_ldg<uint4>(column);
                xb[nt]                      = load_ldg<uint4>(column + 8);
            }
        }

        // A: the even/odd selector masks of the top and bottom row of each tile.
        std::uint32_t even[kTpw][2];
        std::uint32_t odd[kTpw][2];
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row = (first_tile + t) * 16 + gid;
            const std::uint32_t top =
                *reinterpret_cast<const std::uint32_t*>(&stage.codes[tile_row][code_byte]);
            const std::uint32_t bottom =
                *reinterpret_cast<const std::uint32_t*>(&stage.codes[tile_row + 8][code_byte]);
            even[t][0] = top & 0x33333333u;
            odd[t][0]  = (top >> 2) & 0x33333333u;
            even[t][1] = bottom & 0x33333333u;
            odd[t][1]  = (bottom >> 2) & 0x33333333u;
        }

        float slab_acc[kTpw][kNt][4] = {};
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            unsigned a[kTpw][4];
#pragma unroll
            for (int t = 0; t < kTpw; ++t) {
                t2_v2_a_fragment(t2_v2_step_selector(even[t][0], odd[t][0], step), a[t][0],
                                 a[t][2]);
                t2_v2_a_fragment(t2_v2_step_selector(even[t][1], odd[t][1], step), a[t][1],
                                 a[t][3]);
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned b0;
                unsigned b1;
                t2_v2_b_fragment(xa[nt], xb[nt], step, b0, b1);
#pragma unroll
                for (int t = 0; t < kTpw; ++t) {
                    float (&c)[4] = slab_acc[t][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], a[t][0], a[t][1], a[t][2], a[t][3], b0, b1);
                }
            }
        }

#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row = (first_tile + t) * 16 + gid;
            const float top_scale =
                __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                    &stage.scales[tile_row][2 * group_in_slab])));
            const float bottom_scale =
                __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                    &stage.scales[tile_row + 8][2 * group_in_slab])));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[t][nt][0] = fmaf(slab_acc[t][nt][0], top_scale, acc[t][nt][0]);
                acc[t][nt][1] = fmaf(slab_acc[t][nt][1], top_scale, acc[t][nt][1]);
                acc[t][nt][2] = fmaf(slab_acc[t][nt][2], bottom_scale, acc[t][nt][2]);
                acc[t][nt][3] = fmaf(slab_acc[t][nt][3], bottom_scale, acc[t][nt][3]);
            }
        }
        // The next iteration's issue writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();
    __syncthreads();

    // Reduce the K partials of each tile in a fixed order: odd K warps publish, even ones fold
    // their neighbour, then K warp 0 sums the even ones.
    float* partial  = shared.partial;
    const auto slot = [&](int w, int t, int nt) {
        return partial + (((w * kTpw + t) * kNt + nt) * 32 + lane) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float (&c)[4] = acc[t][nt];
                store_vec(slot(warp, t, nt), make_float4(c[0], c[1], c[2], c[3]));
            }
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float (&c)[4]        = acc[t][nt];
                const float4 partner = load_vec<float4>(slot(warp + 1, t, nt));
                c[0] += partner.x;
                c[1] += partner.y;
                c[2] += partner.z;
                c[3] += partner.w;
                if (k_split != 0) {
                    store_vec(slot(warp, t, nt), make_float4(c[0], c[1], c[2], c[3]));
                }
            }
        }
    }
    if constexpr (kKW > 2) { __syncthreads(); }

    if (k_split == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int row = row0 + (first_tile + t) * 16 + gid;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum =
                    make_float4(acc[t][nt][0], acc[t][nt][1], acc[t][nt][2], acc[t][nt][3]);
#pragma unroll
                for (int split = 2; split < kKW; split += 2) {
                    const float4 value = load_vec<float4>(slot(warp + split, t, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col = nt * 8 + 2 * lid;
                if (col < cols) {
                    out[static_cast<std::int64_t>(col) * rows + row] = __float2bfloat16_rn(sum.x);
                    out[static_cast<std::int64_t>(col) * rows + row + 8] =
                        __float2bfloat16_rn(sum.z);
                }
                if (col + 1 < cols) {
                    out[static_cast<std::int64_t>(col + 1) * rows + row] =
                        __float2bfloat16_rn(sum.y);
                    out[static_cast<std::int64_t>(col + 1) * rows + row + 8] =
                        __float2bfloat16_rn(sum.w);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
