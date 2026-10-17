#pragma once

// THE shared PTQ1_0 -> T2 in-shared code repack.
//
// PORTED FROM: the I line's ops/linear/ternary/ternary_ptq1_repack.cuh (117 lines, 2026-09-23),
// which carried this arithmetic for the wide-t, small-t and int8 rungs. Everything below is that
// file verbatim EXCEPT the trit->stored-code mapping (see CONVENTION). Kept as one shared helper
// for the same reason the original was: this is a byte_perm/base-3 chain, and a drift here does not
// crash -- it feeds the LUT/MMA a wrong trit and shows up as a quality regression much later.
//
// WHAT IT DOES: PTQ1_0 stores a 128-weight group as 24 base-3 bytes (five trits per byte, five
// stages of c=16/8/8...) plus 2 high bytes (the last 8 weights), while T2 stores the same group as
// 32 bytes of four 2-bit codes. The repack relabels one as the other, so every line downstream of
// it (LUT, MMA, scale folding, stores) is byte for byte the T2 path.
//
// !! CONVENTION -- the one deliberate change. The I line's container stored ternary codes
// OFFSET-BINARY: value = (code - 1) * d, so a trit mapped to the code `xi` directly. This
// container stores them as two-bit TWO'S COMPLEMENT (see T2DecodeAtom in t2_rowsplit_storage.cuh),
// so a trit must map to `(xi - 1) & 3`:
//
//     xi = 0 (trit -1) -> 0b11 = 3    xi = 1 (trit 0) -> 0b00 = 0    xi = 2 (trit +1) -> 0b01 = 1
//
// Both forms are stated in the sources this was built from, and they are NOT interchangeable:
//   * offset-binary: the on-disk reference decoder E:\ninfer\dl\bonsai_ptq1_0_dequant.py:64
//     ("(q - 1) * d"), matching the I line's PQ2SimtDecodeAtom;
//   * two's complement: upstream tools/artifact/codecs/row_split.py:65 `_pack_two_bit`
//     (`codes.to(torch.int16) & 0x03`) read back at :411/:476 by sign-extending (`unsigned - 4`).
// Note the golden fixtures under E:\视频\_bonsai\ternary-oracle\ are I-LINE ERA and therefore
// offset-binary; they validate the base-3 decode arithmetic but must NOT be used to judge the
// packed-field convention of this container.
//
// CORRECTNESS CONTRACT (unchanged in substance): the trit recovered here must equal, for every
// weight index, the direct formula the I line proved bit-exact against two independent decoders --
//   xi = ((uint8)(raw * 3^t) * 3) >> 8        (PTQ1DecodeAtom above; the old evidence was
//                                              0 / 122,880 mismatches on real weights)
// with the stored field then being `(xi - 1) & 3` in this container.
//
// LAYOUT CONTRACT: `src` is the raw row -- qs[0..24) then qh[0..2) at offset
// PTQ1RowSplitStorage::kCodeBytesPerGroup; `dst` is the T2 code row, 32 bytes of which 0..31 are
// written by the seven slots below. src needs 4-byte alignment (the two 4-byte word loads), dst
// only byte alignment. The rows themselves are independent, so callers are free to distribute the
// (row, slot) jobs over a warp in any order -- see t2_ptq1_repack_warp().

#include "ops/common/memory.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

// Scalar form: the trit code, already mapped to this container's two's-complement field. Used for
// the 2-byte qh tail.
__device__ __forceinline__ unsigned t2_ptq1_stored_code_of(std::uint8_t raw, int pow3) {
    const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
    const unsigned xi    = (static_cast<unsigned>(q) * 3u) >> 8;
    return (xi + 3u) & 0x3u; // == (xi - 1) & 3: offset-binary -> two's complement
}

// One 4-byte qs group -> FIVE code bytes (all five stages), one chain step per stage instead of
// restarting the chain per stage: stage*_codes(g4, t) re-runs stages 0..t, so five calls execute
// 1+2+3+4+5 = 15 chain steps where this executes 5, for the same five bytes. `dst_step` is the
// distance between consecutive stages' destination bytes (4 for the qs[0..16) words, 2 for
// qs[16..24)) and is a literal at every call site, so it folds.
//
// The extraction (two __byte_perm calls parking the four base-3 bytes in the even lanes so ONE
// multiply-by-3 advances all four digits) and the multiply chain itself are PrismML's
// (ggml/src/ggml-cuda/mmq-load-tiles.cuh:261 ggml_cuda_mmq_decode_ptq1_0_qs4, MIT); the pack is the
// one-multiply fold: the four codes sit in bytes 0..3 of `word` with only their low two bits
// meaningful, and * 0x01041040 lands b0 + 4*b1 + 16*b2 + 64*b3 in the top byte (each field <= 3, so
// byte 2 peaks at 252 and cannot carry into it). The bit order matches upstream `_pack_two_bit`
// (unsigned[:, 0::4] | unsigned[:, 1::4] << 2 | ...).
//
// The +0x03030303 is the convention change and nothing else: every byte holds an xi in {0,1,2}, so
// adding 3 cannot carry between bytes, and `& 3` then yields (xi - 1) & 3 in place.
__device__ __forceinline__ void t2_ptq1_emit_stages(std::uint8_t* dst, int dst_step,
                                                    unsigned group4) {
    unsigned v_lo = __byte_perm(group4, 0u, 0x4140);
    unsigned v_hi = __byte_perm(group4, 0u, 0x4342);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned w_lo = v_lo * 3u;
        const unsigned w_hi = v_hi * 3u;
        v_lo                = w_lo & 0x00FF00FFu;
        v_hi                = w_hi & 0x00FF00FFu;
        const unsigned word = __byte_perm(w_lo, w_hi, 0x7531);
        const unsigned code = (((word & 0x03030303u) + 0x03030303u)) & 0x03030303u;
        dst[t * dst_step]   = static_cast<std::uint8_t>((code * 0x01041040u) >> 24);
    }
}

// Seven jobs per row, each owning a DISJOINT set of the 32 destination code bytes:
//   slot 0..3: qs[4g..4g+4)      at stages 0..4 -> code bytes 4t+g    (weights 0..79)
//   slot 4..5: qs[16+4p..20+4p)  at stages 0..4 -> code bytes 20+2t+p (weights 80..119)
//   slot 6   : qh[0..2) -> byte 30 (trits 0,0,1,1) and byte 31 (trits 2,2,3,3)
// The three destination ranges are 0..15, 20..29 and 30..31: disjoint, and slots 4/5 stop at 29 so
// the qh slot's 30/31 are never raced by them.
__device__ __forceinline__ void t2_ptq1_repack_slot(const std::uint8_t* src, std::uint8_t* dst,
                                                    int slot) {
    if (slot < 4) {
        t2_ptq1_emit_stages(dst + slot, 4, load_vec<unsigned>(src + 4 * slot));
    } else if (slot < 6) {
        const int par = slot - 4;
        t2_ptq1_emit_stages(dst + 20 + par, 2, load_vec<unsigned>(src + 16 + 4 * par));
    } else {
        const std::uint8_t h0 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup];
        const std::uint8_t h1 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup + 1];
        dst[30] = static_cast<std::uint8_t>(t2_ptq1_stored_code_of(h0, 1) |
                                            (t2_ptq1_stored_code_of(h1, 1) << 2) |
                                            (t2_ptq1_stored_code_of(h0, 3) << 4) |
                                            (t2_ptq1_stored_code_of(h1, 3) << 6));
        dst[31] = static_cast<std::uint8_t>(t2_ptq1_stored_code_of(h0, 9) |
                                            (t2_ptq1_stored_code_of(h1, 9) << 2) |
                                            (t2_ptq1_stored_code_of(h0, 27) << 4) |
                                            (t2_ptq1_stored_code_of(h1, 27) << 6));
    }
}

// The same seven slots with the qs and qh planes addressed SEPARATELY.
//
// Why this exists: t2_ptq1_repack_slot assumes a group's raw bytes are contiguous (qs[24] then
// qh[2] at kCodeBytesPerGroup), which is how the I line's shared row staged them. An ARTIFACT does
// not look like that: row_split_k128_v1 keeps the whole slab's qs run contiguous -- so it copies as
// 16-byte-granular cp.async, which a 26-byte-per-group layout could not -- and the qh run lives in
// its own plane 256 bytes away. So the two tail bytes are not at src[24..26) and this variant takes
// them from their own pointer.
//
// The arithmetic is t2_ptq1_repack_slot's, unchanged: slots 0..5 never read the high plane, so they
// are delegated verbatim, and only the qh pair moves.
__device__ __forceinline__ void t2_ptq1_repack_slot_split(const std::uint8_t* qs,
                                                          const std::uint8_t* qh,
                                                          std::uint8_t* dst, int slot) {
    if (slot < 6) {
        t2_ptq1_repack_slot(qs, dst, slot);
        return;
    }
    const std::uint8_t h0 = qh[0];
    const std::uint8_t h1 = qh[1];
    dst[30] = static_cast<std::uint8_t>(t2_ptq1_stored_code_of(h0, 1) |
                                        (t2_ptq1_stored_code_of(h1, 1) << 2) |
                                        (t2_ptq1_stored_code_of(h0, 3) << 4) |
                                        (t2_ptq1_stored_code_of(h1, 3) << 6));
    dst[31] = static_cast<std::uint8_t>(t2_ptq1_stored_code_of(h0, 9) |
                                        (t2_ptq1_stored_code_of(h1, 9) << 2) |
                                        (t2_ptq1_stored_code_of(h0, 27) << 4) |
                                        (t2_ptq1_stored_code_of(h1, 27) << 6));
}

// The flattened form: 7 * Rows jobs walked lane-strided by a warp of 32 lanes, `row_src`/`row_dst`
// being the row-0 bases and the two strides the row pitches. This is the loop the wide-t and int8
// kernels both want; the small-t kernel walks the same seven slots in three divergence-free loops
// instead (7 slots over 32 lanes puts lanes with different `slot` in one instruction, so a single
// flat loop serialises its three branches) and therefore calls t2_ptq1_repack_slot() directly.
__device__ __forceinline__ void t2_ptq1_repack_warp(const std::uint8_t* row_src, int src_stride,
                                                    std::uint8_t* row_dst, int dst_stride,
                                                    int rows, int lane) {
    const int jobs = rows * 7;
#pragma unroll
    for (int job = lane; job < jobs; job += 32) {
        const int local_row = job / 7;
        const int slot      = job - local_row * 7;
        t2_ptq1_repack_slot(row_src + static_cast<std::ptrdiff_t>(local_row) * src_stride,
                            row_dst + static_cast<std::ptrdiff_t>(local_row) * dst_stride, slot);
    }
}

} // namespace ninfer::ops::detail
