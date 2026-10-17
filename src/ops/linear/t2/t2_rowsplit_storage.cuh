#pragma once

// T2G128_F16S row-split storage: 2-bit two's-complement codes, four per byte (lane i in bits
// (i % 4) * 2 of byte i / 4), 128 codes and 32 code bytes per group, one binary16 scale per group,
// no high plane. Legal codes are 00 (0), 01 (+1) and 11 (-1); 10 (-2) is outside the artifact
// language and the decode atoms are defined only for valid streams.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct T2RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
    static constexpr int kCodesPerByte       = 4;
    static constexpr int kCodesPerWord       = 16;
};

// The two bit planes of a packed word: bit 0 of every field is the magnitude (|w| in {0, 1}),
// bit 1 is the sign. value = magnitude - 2 * (magnitude & sign), i.e. 00 -> 0, 01 -> +1, 11 -> -1.
struct T2DecodeAtom {
    // Sixteen packed codes -> sixteen floats scaled by the group's binary16 multiplier, in k
    // order (code j is field j of the word, little-endian bytes).
    __device__ static __forceinline__ void
    decode_sixteen(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[16]) {
        const float scale     = __half2float(__ushort_as_half(scale_bits));
        const float neg_scale = -scale;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t field = (packed >> (2 * j)) & 0x3u;
            weights[j]                = (field == 0u) ? 0.0f : ((field & 0x2u) ? neg_scale : scale);
        }
    }

    // Four packed codes of one byte -> two unscaled bf16 pairs {w0, w1}, {w2, w3}; callers fold
    // the group scale into the accumulation as the Q4 MMA atom does.
    __device__ static __forceinline__ void decode_quad(std::uint8_t packed, unsigned& pair01,
                                                       unsigned& pair23) {
        // bf16 patterns: +1 = 0x3F80, -1 = 0xBF80, 0 = 0x0000.
        const auto lane = [](std::uint32_t field) -> std::uint32_t {
            return (field & 0x1u) ? (0x3F80u | ((field & 0x2u) << 14)) : 0u;
        };
        pair01 = lane(packed & 0x3u) | (lane((packed >> 2) & 0x3u) << 16);
        pair23 = lane((packed >> 4) & 0x3u) | (lane((packed >> 6) & 0x3u) << 16);
    }

    __device__ static __forceinline__ void
    decode_quad(std::uint8_t packed, std::uint16_t scale_bits, float (&weights)[4]) {
        const float scale     = __half2float(__ushort_as_half(scale_bits));
        const float neg_scale = -scale;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t field = (static_cast<std::uint32_t>(packed) >> (2 * j)) & 0x3u;
            weights[j]                = (field == 0u) ? 0.0f : ((field & 0x2u) ? neg_scale : scale);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// PTQ1_0 -- the Prism packing, kept native.
//
// WHY IT IS HERE: the converter expands PTQ1_0 into the T2 geometry at pack time (both decode to
// the same ternary values), which costs 34/28 = 21% more bytes. For a 5.5 GiB artifact that is the
// difference between fitting an 8 GB card and not, so the engine learns the 28 B/128 packing
// instead. Ported from the I line's ops/linear/ternary/ternary_rowsplit_storage.cuh, where it ran
// the fast paths for a year; the geometry below is format-independent and is copied verbatim.
//
// Group 128 -> 24 base-3 trit bytes (five trits per byte) + 2 high trit bytes (four each) + one
// binary16 multiplier per group = 28 B/128.
//
// !! CONVENTION: this container stores ternary payload as two-bit TWO'S COMPLEMENT (T2DecodeAtom
// above). The I line's container stored the same trits OFFSET-BINARY, value = (code - 1) * d, so
// every trit -> stored-code mapping must be `(code - 1) & 3` here and not `code`. The one and only
// place that happens is t2_ptq1_repack.cuh.
// ---------------------------------------------------------------------------------------------
struct PTQ1RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 24;
    static constexpr int kHighBytesPerGroup  = 2;
    static constexpr int kScaleBytesPerGroup = 2;
};

// 3^n for n in [0,5], as the table ggml's dequantizer reads. A branch chain, so no device-side
// array lookup is needed on the hot path.
__device__ __forceinline__ constexpr std::uint8_t ptq1_pow3(int n) {
    return n <= 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : n == 4 ? 81u : 243u;
}

// Per-weight form of dequantize_row_ptq1_0 (ggml-quants.c); a literal port of the I line's
// PTQ1SimtDecodeAtom::decode_one. This returns the VALUE, so it is convention-free.
//
// The stage walk matters and is not symmetric: for a 24-byte qs the c=32 stage never runs
// (0 + 32 > 24), c=16 runs once and emits weights 0..79 reading qs[0..15], c=8 runs once and emits
// weights 80..119 reading qs[16..23]; the 2-byte qh then carries weights 120..127. That is
// 80 + 40 + 8 = 128. Each trit is recovered as ((uint16)(q * 3^n) * 3) >> 8, with the uint8
// wrap-around the C code relies on (`uint8_t q = qs[..] * pow3[n];` truncates on assignment).
struct PTQ1DecodeAtom {
    static constexpr int kGroupK = PTQ1RowSplitStorage::kGroupK;

    static __device__ __forceinline__ float load_scale(const std::uint8_t* scale_ptr) {
        return __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
    }

    // qs lives in the base plane, qh in the high plane.
    static __device__ __forceinline__ float decode_one(const std::uint8_t* codes,
                                                      const std::uint8_t* high, float scale,
                                                      int index) {
        std::uint8_t raw;
        int trit;
        if (index < 80) {
            // c = 16 stage: weights n*16 + m, byte qs[m]
            trit = index >> 4;
            raw  = codes[index & 15];
        } else if (index < 120) {
            // c = 8 stage: weights 80 + n*8 + m, byte qs[16 + m]
            const int local = index - 80;
            trit            = local >> 3;
            raw             = codes[16 + (local & 7)];
        } else {
            // qh tail: weights 120 + n*2 + h, byte qh[h]
            const int local = index - 120;
            trit            = local >> 1;
            raw             = high[local & 1];
        }
        const std::uint8_t q = static_cast<std::uint8_t>(raw * ptq1_pow3(trit));
        const int xi         = static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8);
        return static_cast<float>(xi - 1) * scale;
    }
};

} // namespace ninfer::ops::detail
