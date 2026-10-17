#pragma once

// PTQ1_0 -> int8, DIRECT (no PQ2_0 / T2 2-bit code relabelling anywhere in between), for the s8
// prefil rung. Copied from the I line's ternary_rowsplit_mma_s8.cuh:136-164 -- the two primitives
// and the slot table are that file's, verbatim; only the names are prefixed to this tree's t2_.
//
// WHY DIRECT AND NOT THE REPACK (this is the whole point of the file). The small-T rung
// (t2_small_t_v2.cuh) repacks raw PTQ1_0 bytes into the 2-bit code plane because ITS A operand is
// read as 2-bit fields out of one 32-bit word. This rung's A operand is int8, and PTQ1_0's trit
// value IS already the int8 value we want (xi - 1, xi = (v*3)>>8 in {0,1,2}), so the repack would
// insert two hops that change nothing: base-3 bytes -> 2-bit codes -> int8. The I line measured what
// those hops cost and wrote it down:
//
//   "PTQ1_0 could not take at all until this instantiation existed -- and which the repack this
//    replaced threw away again (measured 2026-09-23: 55.37 ms/forward against PQ2_0's 39.75 at T=64,
//    i.e. the repack was this rung's first-order cost)."
//
// NOTE what does NOT appear here: any container-convention adaptation. On small-T the 2-bit field
// had to be rewritten from the I line's offset-binary to this container's two's complement
// ((xi + 3) & 3); on int8 there is no field to rewrite, because the value is the byte. So this rung
// carries ZERO convention risk -- it is the one place where the two containers agree by construction.
//
// SLOT -> DESTINATION (the I line's table, byte e of a group IS weight e):
//   slot 0..3  source qs[4g..4g+4)   -> words at 16t + 4g     (dst_step 16)  weights 16t + 4g + m
//   slot 4..5  source qs[16+4p..)    -> words at 80 + 8t + 4p (dst_step 8)   weights 80 + 8t+4p+m
//   slot 6     source qh[0..1]       -> words at 120 and 124
// Coverage is exact and disjoint: slots 0..3 write bytes 0..79, slots 4/5 write 80..119, slot 6
// writes 120..127. The four 32-byte blocks are k-steps 0..3 of the mma:
//   step 0 <- slots 0-3 stages 0,1 | step 1 <- slots 0-3 stages 2,3
//   step 2 <- slots 0-3 stage 4 + slots 4-5 stages 0,1 | step 3 <- slots 4-5 stages 2,3,4 + slot 6
//
// ARITHMETIC PROVENANCE: the byte_perm + multiply-by-3 chain is PrismML's
// (ggml/src/ggml-cuda/mmq-load-tiles.cuh:259 ggml_cuda_mmq_decode_ptq1_0_qs4, MIT); the I line
// verified a bit-faithful model of these helpers against ggml's ptq1_0_trit, against its own
// PTQ1SimtDecodeAtom::decode_one and against the reader's dequantizer (0 mismatches, 64 whole rows
// == 327,680 grid cells). This tree re-verifies them independently: see the s8 leg of
// _tucheck_ptq1_smallt_ab.cu's golden comparison.

#include <cstdint>

namespace ninfer::ops::detail {

// One 4-byte qs group -> five stages of four weights, four SIGNED int8 bytes per store. Stage t's
// product already contains stage t-1's remainder in its low byte, so five multiplies by 3 produce
// all five stages where five separate chains would need 1+2+3+4+5. __vsub4 subtracts the 1 of
// (xi - 1) per byte, so four weights leave in one instruction as four signed bytes in weight order.
__device__ __forceinline__ void t2_s8_ptq1_decode_qs4(unsigned packed, std::uint8_t* dst,
                                                      int dst_step) {
    unsigned v_lo = __byte_perm(packed, 0u, 0x4140u);
    unsigned v_hi = __byte_perm(packed, 0u, 0x4342u);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned w_lo = v_lo * 3u;
        const unsigned w_hi = v_hi * 3u;
        v_lo                = w_lo & 0x00FF00FFu;
        v_hi                = w_hi & 0x00FF00FFu;
        *reinterpret_cast<unsigned*>(dst + t * dst_step) =
            __vsub4(__byte_perm(w_lo, w_hi, 0x7531u), 0x01010101u);
    }
}

// The 2-byte qh tail: weights 120 + n*2 + h read qh[h] at stage n, so one word carries stages
// (2t, 2t+1) of both bytes: [h0@2t, h1@2t, h0@2t+1, h1@2t+1].
__device__ __forceinline__ void t2_s8_ptq1_decode_qh(const std::uint8_t* qh, std::uint8_t* dst) {
    unsigned v = static_cast<unsigned>(qh[0]) | (static_cast<unsigned>(qh[1]) << 16);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const unsigned w0 = v * 3u;
        v                 = w0 & 0x00FF00FFu;
        const unsigned w1 = v * 3u;
        v                 = w1 & 0x00FF00FFu;
        *reinterpret_cast<unsigned*>(dst + 120 + 4 * t) =
            __vsub4(__byte_perm(w0, w1, 0x7531u), 0x01010101u);
    }
}

// The seven slots as a PER-SLOT entry point, so the caller keeps the choice of which thread owns
// which (row, slot) -- and so the three slot branches stay in three divergence-free loops rather
// than one loop whose lanes disagree. That split is the I line's measured lesson (see
// t2_ptq1_repack.cuh's note); this file keeps it for the same reason.
//
// `dst` is the group's 128-byte int8 plane, one byte per weight. `qs`/`qh` are addressed SEPARATELY
// because an artifact keeps a row's qs run contiguous and its qh run in another plane 256 bytes
// away -- exactly the split t2_ptq1_repack_slot_split exists for on the small-T rung.
__device__ __forceinline__ void t2_s8_ptq1_decode_slot(const std::uint8_t* qs,
                                                       const std::uint8_t* qh, std::uint8_t* dst,
                                                       int slot) {
    if (slot < 4) {
        t2_s8_ptq1_decode_qs4(*reinterpret_cast<const unsigned*>(qs + 4 * slot), dst + 4 * slot, 16);
    } else if (slot < 6) {
        const int par = slot - 4;
        t2_s8_ptq1_decode_qs4(*reinterpret_cast<const unsigned*>(qs + 16 + 4 * par),
                              dst + 80 + 4 * par, 8);
    } else {
        t2_s8_ptq1_decode_qh(qh, dst);
    }
}

// ---- the plane's order is the decode's order (measured, not assumed) ---------------------------
//
// I first wrote a t2_s8_ptq1_permute_block here, on the reasoning that the decode emits weights in
// WEIGHT order while the plane must be in the order the t2-prefill kernel's A-fragment production
// reads -- and the judge said no: decode + permute agreed with the kernel's own fragments on only
// 14,336 of 32,768 bytes (first mismatch at tig 0, p 1).
//
// The reason is that the register order IS the plane order, byte for byte:
//     d[0] = weights {0,2,4,6}    d[1] = {1,3,5,7}    d[2] = {8,10,12,14}   d[3] = {9,11,13,15}
//     d[4] = {16,18,20,22}        d[5] = {17,19,21,23} d[6] = {24,26,28,30}  d[7] = {25,27,29,31}
// and the plane's byte p of a lane block must hold weight fwd(p) = 8*(block>>1) + (block&1) + 2*m
// (block = p >> 2, m = p & 3), which for p in a word 4w..4w+4 gives exactly d[w]'s weight set. So
// the plane is the decode output verbatim -- no shuffle step exists to get wrong.
//
// That identity is what _tucheck_ptq1_s8_gate.cu measures directly (32768/32768 plane bytes over
// 256 code rows, comparing against the kernel's own decode); see the landing table §28.9 and §28.11.

// The whole group in one call: 7 sequential slots over one (qs, qh) pair. For a caller that
// parallelises over ROWS rather than over slots (one lane per row), which is the natural shape when
// the CTA already walks a row block. Slots are visited in the same order and write the same bytes as
// t2_s8_ptq1_decode_slot, so the two forms are interchangeable by construction; the three-loop
// caller should prefer decode_slot to avoid the slot branches sharing a warp.
//
// NOTE the destination is the MMA plane, one byte per weight, laid out in weight order -- which, per
// the note above, is also the order the kernel reads. Nothing else to do to it.
__device__ __forceinline__ void t2_s8_ptq1_decode_group(const std::uint8_t* qs,
                                                        const std::uint8_t* qh, std::uint8_t* dst) {
#pragma unroll
    for (int slot = 0; slot < 7; ++slot) { t2_s8_ptq1_decode_slot(qs, qh, dst, slot); }
}

// ---- the plane's order, and the one shuffle that gets there ------------------------------------
//
// The decode above writes weights in WEIGHT order: byte e of its output is weight e, because the
// slot table is about which source bytes produce which weights. The plane the MMA reads is in the
// order the t2-prefill kernel's A-fragment production produces -- byte P of a lane's 32-byte block
// must hold weight
//     fwd(P) = 8*((P >> 2) >> 1) + ((P >> 2) & 1) + 2*(P & 3)
// (measured against the kernel's own decode: _tucheck_ptq1_s8_gate.cu check 5, 32768/32768 bytes).
// So the plane is the decode output PERMUTED by fwd, i.e. plane[P] = decoded[rev(P)] with rev the
// inverse:  rev(e) = 8*(e >> 3) + 4*((e & 7) & 1) + ((e & 7) >> 1).
//
// Written as one __byte_perm per output word, because fwd maps a whole input word pair onto a whole
// output word: output word 0 takes input bytes {0,2,4,6} of the pair (selector 0x6420) and word 1
// takes {1,3,5,7} (selector 0x7531), and the pattern repeats every two words.
//
// THREE ATTEMPTS, TWO OF THEM MINE AND WRONG -- recorded because the judge is what settled it:
//   1. 0x6420/0x7531 fed with the KERNEL's own decode output: failed 14336/32768. That decode is
//      ALREADY in plane order (its registers are the fwd order -- see the note above), so permuting
//      it again double-permutes. The constants were right; the INPUT was wrong.
//   2. From that failure I concluded no shuffle was needed and deleted the helper. Wrong: the value
//      source for the plane is t2_s8_ptq1_decode_group, which IS in weight order and does need it.
//   3. 0x5140/0x7362 (the plain rev() pattern): failed 12216/32768, because the plane is the decode
//      output gathered by fwd, not scattered by rev.
// The judge that settles all three: decode_group -> this permute -> compare, per (weight, slot),
// against the slow definition (see _tucheck_ptq1_s8_gate.cu's last check).
//
// The function permutes a WHOLE ROW (128 bytes, 32 words), not one lane's 32-byte block: fwd is
// defined per block but its pattern repeats every eight bytes, so one loop over the sixteen word
// pairs covers all four lanes. (Getting that wrong -- `j < 4`, i.e. 32 bytes -- left lanes 1..3
// untouched and is exactly what the judge caught next: it went from failing at tig 0 to failing at
// tig 1, weight 32.)
__device__ __forceinline__ void t2_s8_ptq1_plane_permute(const unsigned* in8, unsigned* out8) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        out8[2 * j]     = __byte_perm(in8[2 * j], in8[2 * j + 1], 0x6420u);
        out8[2 * j + 1] = __byte_perm(in8[2 * j], in8[2 * j + 1], 0x7531u);
    }
}

// ---- the same plane, written straight to its final positions (one barrier instead of two) ------
//
// The two functions above are the verified reference. They need a barrier BETWEEN them: the decode
// writes weights in weight order and the permutation moves bytes between slots, so nothing can be
// permuted until every slot has written -- two barriers per group, with the MMA waiting on both.
//
// Writing each word's four bytes STRAIGHT to their plane positions removes that entirely: the pass
// becomes self-contained per slot, so the kernel needs ONE barrier (after it) instead of two.
//
// The positions are constants of the format. Weight x belongs at plane byte
//     p(x) = 8*(x>>3) + 4*((x&7)&1) + ((x&7)>>1)
// (the inverse of the fwd() map in §28.9), and a (slot, stage) word covers weights
//     slots 0..3: 16t + 4*slot + m      slots 4/5: 80 + 8t + 4*(slot-4) + m      qh: 120 + 4t + m
//
// Judged in _tucheck_ptq1_scatter.cu against decode_group + plane_permute: 32768/32768 plane bytes
// identical over 256 random groups, with a byte-flip negative control.
__host__ __device__ constexpr int t2_s8_ptq1_plane_of(int x) {
    return 8 * (x >> 3) + 4 * ((x & 7) & 1) + ((x & 7) >> 1);
}

__host__ __device__ constexpr int t2_s8_ptq1_word_weight(int slot, int t) {
    return slot < 4 ? 16 * t + 4 * slot : 80 + 8 * t + 4 * (slot - 4);
}

// Store one scrambled word's four signed bytes at their plane positions.
__device__ __forceinline__ void t2_s8_ptq1_scatter_word(unsigned word, std::uint8_t* plane,
                                                        int weight_base) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        plane[t2_s8_ptq1_plane_of(weight_base + m)] =
            static_cast<std::uint8_t>((word >> (8 * m)) & 0xFFu);
    }
}

// The seven slots again, this time scattering. Same arithmetic as decode_slot, same slot table; only
// the destination addressing differs.
__device__ __forceinline__ void t2_s8_ptq1_decode_slot_scatter(const std::uint8_t* qs,
                                                               const std::uint8_t* qh,
                                                               std::uint8_t* plane, int slot) {
    if (slot < 6) {
        const int base = slot < 4 ? 4 * slot : 16 + 4 * (slot - 4);
        unsigned v_lo  = __byte_perm(*reinterpret_cast<const unsigned*>(qs + base), 0u, 0x4140u);
        unsigned v_hi  = __byte_perm(*reinterpret_cast<const unsigned*>(qs + base), 0u, 0x4342u);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const unsigned w_lo = v_lo * 3u;
            const unsigned w_hi = v_hi * 3u;
            v_lo                = w_lo & 0x00FF00FFu;
            v_hi                = w_hi & 0x00FF00FFu;
            t2_s8_ptq1_scatter_word(
                __vsub4(__byte_perm(w_lo, w_hi, 0x7531u), 0x01010101u), plane,
                t2_s8_ptq1_word_weight(slot, t));
        }
        return;
    }
    unsigned v = static_cast<unsigned>(qh[0]) | (static_cast<unsigned>(qh[1]) << 16);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const unsigned w0 = v * 3u;
        v                 = w0 & 0x00FF00FFu;
        const unsigned w1 = v * 3u;
        v                 = w1 & 0x00FF00FFu;
        t2_s8_ptq1_scatter_word(__vsub4(__byte_perm(w0, w1, 0x7531u), 0x01010101u), plane,
                                120 + 4 * t);
    }
}

} // namespace ninfer::ops::detail
