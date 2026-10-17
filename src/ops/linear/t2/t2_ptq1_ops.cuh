#pragma once

// PTQ1_0 native operand production: base-3 trits straight into bf16 A-fragment registers.
//
// COPIED VERBATIM from the I line's ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh (:164-203),
// which is the form that line reached on 2026-09-23 after trying the in-shared repack first (its
// repack is still there as the named switch `kTernaryMmaPtq1Repack = false`, not deleted). Two facts
// make it worth copying rather than re-deriving:
//
//   * a ternary weight is +-1 or 0, so its bf16 encoding is one of THREE constants. The whole
//     trit -> bf16 table therefore fits in TWO REGISTERS instead of the T2 path's uint2 lut[256],
//     which is what buys back the 2 KiB of shared the extra register pressure costs -- and with it a
//     fourth CTA per SM, which is where that line's small-T rung got its occupancy.
//   * the base-3 chain CARRIES across the eight k-steps instead of restarting per step, so a stage
//     costs ONE advance per (row, word).
//
// The I line's own comment on the lane map says it "is the verified map of verify_native_ptq1_map.py,
// not a re-derivation" -- and that script is GONE (the PTQ1 handoff records exp\reader-candidates\
// as lost). So nothing here is taken on trust: ptq1-ops-oracle.cu re-derives the contract below and
// checks these primitives against it for every byte value and every stage.
//
// CONTRACT (what the oracle checks):
//   code(b, t) = ((std::uint8_t)(b * 3^t) * 3) >> 8      in {0, 1, 2}   -- the uint8 wrap is real
//   weight(c)  = c - 1                                   in {-1, 0, +1}
// `code(b, t)` is exactly what PTQ1DecodeAtom::decode_one recovers for the same byte and stage, and
// PTQ1DecodeAtom is itself already verified bit-exact against the frozen golden decode.

#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

// The three weights as two table registers: table bytes [2c, 2c+1] = bf16 of (c - 1), so
//   code 0 -> 0xBF80 (-1), code 1 -> 0x0000 (0), code 2 -> 0x3F80 (+1).
// Code 3 is outside this format's language (a legal PTQ1_0 stream never produces it).
inline constexpr unsigned kTritPairLo = 0x0000BF80u; // table bytes: 80 BF 00 00
inline constexpr unsigned kTritPairHi = 0x00003F80u; // table bytes: 80 3F 00 00

// One PRMT builds the pair: low half = c0's weight, high half = c1's weight.
// 34 = 0x22 places table byte 2c at result byte 0 and byte 2c+1 at byte 1 for c0, and the same for
// c1 at result bytes 2 and 3. Every selector nibble stays <= 7, so two registers hold the table.
__device__ __forceinline__ unsigned t2_ptq1_trit_pair(unsigned codes) {
    return __byte_perm(kTritPairLo, kTritPairHi, 34u * codes + 0x1010u);
}

// TWO pair selectors out of ONE IMAD. 34 * codes4 + 0x10101010 is byte-wise clean -- 34 * 2 + 0x10 =
// 0x54 < 0x100, so no byte ever carries into its neighbour -- and byte k of the result is exactly the
// selector for the pair whose two codes are bytes k and k+1. __byte_perm reads only the low 16 bits
// of its selector, so the low half serves the k pair and `sel >> 16` the k+8 pair.
__device__ __forceinline__ unsigned t2_ptq1_pair_selectors(unsigned codes4) {
    return 34u * codes4 + 0x10101010u;
}

// One base-3 chain step applied to TWO raw bytes at once, laid out as v = b0 | (b1 << 16).
// w = v * 3 holds 3*b0 in [15:0] and 3*b1 in [31:16] with no cross-talk (each starts below 256 and
// peaks at 765), and the HIGH byte of each product IS the code the format wants, ((3*b) >> 8):
//   codes = __byte_perm(w, 0, 0x4431)  ->  byte 0 = code(b0), byte 1 = code(b1)
//   next  = w & 0x00FF00FF             ->  the same two values mod 256, ready for the next stage
// The caller keeps `next` in a register, so a whole group costs ONE advance per k-step.
__device__ __forceinline__ unsigned t2_ptq1_chain_step(unsigned v, unsigned& next) {
    const unsigned w = v * 3u;
    next             = w & 0x00FF00FFu;
    return __byte_perm(w, 0u, 0x4431u);
}

} // namespace ninfer::ops::detail
