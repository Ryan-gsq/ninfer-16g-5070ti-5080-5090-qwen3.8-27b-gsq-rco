#include "ops/linear/t2/t2_dispatch.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// The ternary profile's registered problems (qwen3.8-27b-ternary.md §8): decode (T = 1) through
// the GEMV, the small-T bands through the SIMT route, everything wider through the MMA route.
T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("t2 linear: unsupported shape or T"); }

    // T = 1..16 take the small-T tensor-core kernel: at T = 1 the v2 kernel streams the
    // text-layer weights at 430..660 GB/s against the SIMT GEMV's 260..430 (RTX 3090).
    // NINFER_T2_DECODE=simt keeps the GEMV at T = 1 for comparison.
    static const bool decode_simt = [] {
        const char* value = std::getenv("NINFER_T2_DECODE");
        return value != nullptr && std::string(value) == "simt";
    }();
    // NINFER_T2_SMALLT=v1 keeps the first tensor-core kernel for T = 2..16 (benchmark A/B).
    static const bool small_t_v1 = [] {
        const char* value = std::getenv("NINFER_T2_SMALLT");
        return value != nullptr && std::string(value) == "v1";
    }();
    const auto small_t = [t]() -> T2Launch {
        if (t <= 16) { return small_t_v1 ? launch_t2_small_t_mma : launch_t2_small_t_v2; }
        if (t <= 32) { return launch_t2_mma_r64_c32; }
        if (t <= 64) { return launch_t2_mma_r64_c64; }
        return launch_t2_mma_r64_c128;
    };

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
        case 4096:
        case 6144:
        case 7168:
        case 12288:
        case 14336:
        case 16384:
        case 34816:
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k5120; }
            return small_t();
        case 131072:
        case 248320:
            if (t == 1 && decode_simt) { return launch_t2_gemv_r4_w1_word; }
            return small_t();
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k6144; }
            return small_t();
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k17408; }
            return small_t();
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("t2 linear: unsupported shape or T");
}

T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    case LinearPolicy::AllowPrefillCublas:
        return select_t2_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("t2 linear: unsupported policy");
}

// NINFER_TERNARY_PTQ1_FAST=1 selects the fast rungs for PTQ1_0 instead of the reference GEMM. The
// DEFAULT IS THE REFERENCE: that is the rung with a bit-exact judgment behind it (23,085,056 weights
// identical to the frozen golden), so the switch is what makes the fast rungs A/B-able inside one
// session -- a wrong one then costs one environment variable rather than a rebuild and a rollback.
//
// Published (declared in t2_dispatch.h) because PTQ1_0's PREFILL band is entered one layer up, in
// linear.cpp's dispatch, where the workspace lives. That arm ignores this switch for a while, and
// the symptom was subtle rather than loud: the "off" arm quietly stopped being the reference (its
// prefill read 1,110 tok/s where the reference is 44.7), so the A/B silently compared two fast
// rungs. One switch, one predicate, every fast path asks it.
// DEFAULT ON (changed 2026-10-01 on the owner's instruction: ship with the best parameters).
//
// It used to default OFF, which is the I line's A/B *arm* setting, not its shipping setting: that line
// ships NINFER_TERNARY_PTQ1_FAST=1 in its own kit, and its README says setting 0 falls back to the
// reference path "for A/B only, do not run production on it". Left OFF, every SINGLE projection (MLP,
// attention output) takes the per-weight reference rung -- correct, but 44.7 tok/s prefill against
// ~2.1k with the rungs on, with no error line and correct answers, so it reads as "the model is just
// slow". Measured in one full session on 2026-10-01 (landing table section 28.46): long-prompt prefill
// 86.9 -> 1.71k tok/s, decode 13.8 -> 111.2.
//
// `=0` still forces the reference rung, so the A/B arm this switch exists for is unchanged; only the
// UNSET default moved. The PTQ1 kit's launcher also sets the variable explicitly -- belt and braces,
// and it keeps older engines behaving the same way.
bool t2_ptq1_fast() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_PTQ1_FAST");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    // The kernels address row-major records; the panel permutation is a Q4-family device form.
    if (w.layout != QuantLayout::RowSplit || w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("t2 linear: requires a row-split weight");
    }
    // PTQ1_0 originally had exactly ONE correct rung: the reference GEMM. Every fast rung is
    // word-based -- it consumes sixteen 2-bit codes out of one 32-bit word -- while PTQ1 packs five
    // base-3 trits into a byte, so there is no 16-trit word to hand them. The small-T v2 kernel now
    // carries the in-shared repack (t2_ptq1_repack.cuh), so PTQ1 runs on tensor cores in its T <= 16
    // band and, above it, IN SLICES OF THAT SAME BAND.
    //
    // Why slices rather than a wide-T kernel: this tree's wide-T MMA kernel stages HALF a quant group
    // per k-tile (kBlockK = 64), and PTQ1's group is 24 base bytes + 2 high bytes whose repack
    // produces all 32 code bytes at once -- the repack's slot ranges span the half-group boundary
    // (slots 0..3 write bytes 0..19), so that staging shape cannot express it. The I line answered
    // this with a SEPARATE wide-T kernel of its own (ternary_rowsplit_mma_wide_t.cuh: one whole group
    // per staged chunk, kTernaryWideChunkK = 128, its own 24+2 raw window), which was never imported
    // here; and the "T >= 41" band boundary in the task book is that kernel's own measured crossover.
    //
    // Slicing costs ceil(T / 16) passes over the weight rows against the wide tunnel's one, but it
    // reuses a kernel that already carries a three-leg judgment (21 cases: two rungs identical to
    // within 0.016%, both equal to an independent float64 golden), and it takes prefill off the
    // per-weight reference rung entirely. MEASURED at T = 1,827 (exp\ptq1-prefill-probe.ps1):
    // reference 44.7 tok/s before, the sliced rung's reading after -- see 落点表 §26.
    //
    // The two ternary packings are told apart by the high plane, exactly as the I line did: PTQ1_0
    // carries 2 high bytes per group (qhigh != nullptr), T2 none (qhigh == nullptr).
    if (w.qtype == QType::PTQ1_G128_FP16) {
        if (t2_ptq1_fast() && w.qhigh != nullptr) {
            const std::int32_t cols  = x.ne[1];
            const std::int32_t slice = t2_small_t_v2_max_columns();
            if (cols <= slice) {
                launch_t2_small_t_v2_ptq1(x, w, out, stream);
            } else {
                for (std::int32_t offset = 0; offset < cols; offset += slice) {
                    const std::int32_t count = std::min(slice, cols - offset);
                    const Tensor x_slice     = x.slice(1, offset, count);
                    Tensor out_slice         = out.slice(1, offset, count);
                    launch_t2_small_t_v2_ptq1(x_slice, w, out_slice, stream);
                }
            }
            return;
        }
        launch_t2_ptq1_reference(x, w, out, stream);
        return;
    }
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail


