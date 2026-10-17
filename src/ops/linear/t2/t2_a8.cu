// T2G128 row-split weights with int8 activations. Three registered input widths (the hidden 5120,
// the attention/GDN mixer output 6144 and the MLP intermediate 17408) and any row count that is a
// whole number of 128-row blocks (64 with NINFER_T2_A8_TILE=off). Inside the small-T band (T <= 192
// by default, in launches of at most 32 columns) the ternary small-T kernel of t2_small_t_i8.cuh,
// where the A16 kernels are tensor-rate bound at every width; from t2_a8_min_tokens() up the tile
// GEMM of t2_prefill_i8.cuh, with T padded to a multiple of 64 and its own activation contract (one
// binary16 scale per token and 128-wide group). NINFER_T2_A8_TILE=off puts the shared prefill GEMM
// of rowsplit_a8_mma.cuh back on that route for A/B, with T padded to its cheapest column tile.

#include "ops/linear/t2/t2_a8.h"

#include "core/device.h"
#include "ops/common/device_route.h"
#include "ops/common/rowsplit_a8_mma.cuh"
#include "ops/linear/t2/t2_prefill_i8.cuh"
#include "ops/linear/t2/t2_small_t_i8.cuh"
// The imported I-line wide-T MMA kernel (bf16 A operand). CUDA-only header, included here and by the
// judge harness; its launcher is declared in t2_launch.h.
#include "ops/linear/t2/t2_launch.h"
#include "ops/linear/t2/t2_ptq1_wide_t.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

// NINFER_T2_A8_DEBUG=1: report each DISTINCT route decision once, so "which kernel took this load
// band" is a reading rather than an inference. The handoff flagged the absence of exactly this probe
// ("要逐档打印命中的 T/rung 还得自己加"); off by default, deduplicated by the whole line, so a 27B
// model's thousands of calls cost one comparison each and stderr gets one line per case.
bool debug_enabled() {
    static const bool enabled = std::getenv("NINFER_T2_A8_DEBUG") != nullptr;
    return enabled;
}

void debug_note(const std::string& line) {
    if (!debug_enabled()) { return; }
    static std::mutex guard;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(guard);
    if (!seen.insert(line).second) { return; }
    std::fprintf(stderr, "t2_a8 %s\n", line.c_str());
    std::fflush(stderr);
}

using Rows = a8::ContiguousRows<1>;

// The padded width a ragged T runs at. Every column tile streams the whole weight, and a wider
// tile costs less per column: on the RTX 3090 one 128-, 256- and 512-column tile of the text layers
// take 1 : 1.4 : 2.2 (ninfer_linear_bench, 2026-09-22), so 300 tokens run as one 512-column tile
// (2.2) rather than three of 128 (3.0), and 600 as three of 256 rather than five of 128.
std::int32_t padded_tokens(std::int32_t tokens) {
    struct Tile {
        std::int32_t columns;
        std::int32_t cost; // tenths of a 128-column tile
    };

    constexpr Tile kTiles[] = {{128, 10}, {256, 14}, {512, 22}};
    std::int32_t best       = 0;
    std::int32_t best_cost  = 0;
    for (const Tile& tile : kTiles) {
        const std::int32_t count = (tokens + tile.columns - 1) / tile.columns;
        const std::int32_t cost  = count * tile.cost;
        if (best == 0 || cost < best_cost) {
            best      = count * tile.columns;
            best_cost = cost;
        }
    }
    return best;
}

// NINFER_T2_A8_TILE=off keeps the shared rowsplit_a8_mma.cuh GEMM on the prefill route.
bool prefill_tile() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_T2_A8_TILE");
        return value == nullptr || std::string(value) != "off";
    }();
    return enabled;
}

// NINFER_T2_A8_PAIR=off: keep the integer route, but take the FUSED PAIR off it.
//
// WHY THIS EXISTS (2026-10-01, the PTQ1 score-path catastrophe). With the a8 route on, the PTQ1
// packing reads ~1e6 on the score path and the reading is NOT repeatable (same command twice:
// 1,130,407.776 then 1,173,051.538; the t2 packing on the same engine, corpus and caliber is
// bit-identical: 6.344101 twice). The judge meanwhile proves the tile's arithmetic correct at every
// width, for the single destination AND through the split epilogue, with a poisoned workspace and a
// determinism control -- so the difference is not in the kernels but in WHICH of the route's two call
// shapes the engine takes. The pair (attention q/k and GDN qk/vz) is the shape the single-projection
// judge leg never exercises; with this switch off, the same four parts run as separate linear() calls
// -- which still reach the SAME tile for their own width, because linear()'s PTQ1 arm admits it -- so
// the reading separates "the fused pair is the corrupt one" from "every tile call is".
// The predicate itself is a public function (t2_a8_pair_enabled, below) because the two projection
// wrappers are what consult it.

std::int32_t tile_columns(std::int32_t tokens) {
    constexpr std::int32_t kColumns = T2PrefillI8::kColumns;
    return (tokens + kColumns - 1) / kColumns * kColumns;
}

std::int32_t prefill_rows_per_block() {
    return prefill_tile() ? T2PrefillI8::kRows : Rows::kRowsPerBlock;
}

struct SmallBand {
    std::int32_t lo;
    std::int32_t hi;
};

// NINFER_T2_I8_SMALL=off|lo,hi overrides the widths the small-T kernel takes (benchmark A/B).
SmallBand small_band() {
    static const SmallBand band = [] {
        const char* value = std::getenv("NINFER_T2_I8_SMALL");
        if (value == nullptr) { return SmallBand{kT2I8SmallMinTokens, kT2I8SmallMaxTokens}; }
        const std::string text(value);
        if (text == "off") { return SmallBand{1, 0}; }
        const std::size_t comma = text.find(',');
        if (comma == std::string::npos) {
            throw std::invalid_argument("NINFER_T2_I8_SMALL must be off or lo,hi");
        }
        const std::int32_t lo = std::max(1, std::atoi(text.substr(0, comma).c_str()));
        const std::int32_t hi =
            std::min(kT2I8MaxColumns, std::atoi(text.substr(comma + 1).c_str()));
        return SmallBand{lo, hi};
    }();
    return band;
}

enum class Route : std::uint8_t { None, SmallT, Prefill };

// A device profile's "t2_i8_route" entry names the route per width: "small" (the small-T kernel,
// at most kT2I8MaxColumns columns) or "tile" (the prefill GEMM). One entry serves every input width
// because the activations are quantized once, for the route of their width, and shared by the
// projections that read them.
Route route(std::int32_t tokens) {
    const std::string_view routed = device_route_schedule("t2_i8_route", tokens);
    if (routed == "small" && tokens >= 1 && tokens <= kT2I8MaxColumns) { return Route::SmallT; }
    if (routed == "tile" && tokens >= 1) { return Route::Prefill; }
    const SmallBand band = small_band();
    if (tokens >= band.lo && tokens <= band.hi) { return Route::SmallT; }
    if (tokens >= t2_a8_min_tokens()) { return Route::Prefill; }
    return Route::None;
}

// PTQ1_0's route, which deliberately does NOT follow T2's band table or a device profile:
//   tokens >= t2_ptq1_tile_min_tokens() -> the int8 tile (prefill)
//   below that floor                   -> the small-T band (t2_dispatch owns PTQ1's rung there)
// WHY it ignores the profile: a profile is calibrated per hardware class, and its "small" verdict for
// a width would send PTQ1_0 to the T2 small-T kernel, which reads 2-bit T2 fields out of the code
// plane -- that mis-decodes PTQ1's five-trits-per-byte SILENTLY rather than failing. The floor is the
// one place PTQ1's own boundary lives, and NINFER_PTQ1_TILE_MIN is the A/B lever for it.
Route route_for(bool ptq1_tile, std::int32_t tokens) {
    if (ptq1_tile && prefill_tile()) {
        return tokens >= t2_ptq1_tile_min_tokens() ? Route::Prefill : Route::SmallT;
    }
    return route(tokens);
}

// Columns [col0, col0 + cols) of the activations, cols <= kT2I8LaunchColumns.
struct SmallColumns {
    std::int32_t col0;
    std::int32_t cols;
};

template <class Epilogue>
struct SmallParent {
    const Weight* weight;
    Epilogue epilogue;
};

template <class Schedule, class Epilogue>
void small_gemm(const T2A8Activations& x, SmallColumns span, const SmallParent<Epilogue>& first,
                const SmallParent<Epilogue>* second, cudaStream_t stream) {
    const auto blocks = [&](const Weight& w) {
        if ((w.n % Schedule::kRows) != 0 || (x.input_rows % Schedule::kSlabK) != 0 ||
            span.cols > Schedule::kColumns) {
            throw std::invalid_argument("t2 small-T i8: unsupported shape");
        }
        return w.n / Schedule::kRows;
    };
    const auto& other = second != nullptr ? *second : first;
    const T2I8Parents<Epilogue> parents{{static_cast<const std::uint8_t*>(first.weight->qdata),
                                         static_cast<const std::uint8_t*>(other.weight->qdata)},
                                        {static_cast<const std::uint8_t*>(first.weight->scales),
                                         static_cast<const std::uint8_t*>(other.weight->scales)},
                                        {first.epilogue, other.epilogue},
                                        blocks(*first.weight)};
    const std::int32_t total =
        parents.first_blocks + (second != nullptr ? blocks(*second->weight) : 0);
    t2_small_t_i8_kernel<Schedule, Epilogue>
        <<<static_cast<unsigned>(total), Schedule::kThreads, 0, stream>>>(
            x.codes, x.scales, parents, x.input_rows, span.col0, span.cols);
    CUDA_CHECK(cudaGetLastError());
}

// Measured on the RTX 3090 (ninfer_linear_bench, cold L2, 350 W): two code words per lane (256
// contiguous code bytes per row and slab) beat one by 3-15% on the wide shapes at every width, and
// a deeper cp.async ring or 64/128-row CTAs lose. Sixteen rows per CTA over 1024-k slabs win every
// width up to 16 columns (the q/k + value/z pair's 12288 rows are within 5% of 32-row CTAs
// at 9..16); four column tiles take 32 rows over 512-k slabs.
using SmallSchedule8  = T2SmallTI8Schedule<8, 1, 1, 2, 4, 2>;
using SmallSchedule16 = T2SmallTI8Schedule<8, 1, 2, 2, 4, 2>;
using SmallSchedule32 = T2SmallTI8Schedule<4, 1, 4, 2, 3, 2>;
// Alternatives a device profile can route to ("t2_i8_small/<rows>x<K>"): other row counts per CTA,
// K-warp splits, ring depths and code words per lane, for parts whose SM count or memory system
// moves the balance the RTX 3090 tables above were measured at.
using SmallScheduleR16C8W1  = T2SmallTI8Schedule<8, 1, 1, 2, 4, 1>;
using SmallScheduleR16C8S3  = T2SmallTI8Schedule<8, 1, 1, 3, 4, 2>;
using SmallScheduleR32C8    = T2SmallTI8Schedule<8, 2, 1, 2, 4, 2>;
using SmallScheduleR32C8K4  = T2SmallTI8Schedule<4, 1, 1, 2, 4, 2>;
using SmallScheduleR16C16W1 = T2SmallTI8Schedule<8, 1, 2, 2, 4, 1>;
using SmallScheduleR32C16   = T2SmallTI8Schedule<4, 1, 2, 2, 4, 2>;
using SmallScheduleR16C32   = T2SmallTI8Schedule<8, 1, 4, 2, 3, 2>;
using SmallScheduleR32C16K8 = T2SmallTI8Schedule<8, 2, 2, 2, 4, 2>;

// The span's schedule by profile name; false when the name is unknown or cannot take the span.
template <class Epilogue>
bool small_named(std::string_view schedule, const T2A8Activations& x, SmallColumns span,
                 const SmallParent<Epilogue>& first, const SmallParent<Epilogue>* second,
                 cudaStream_t stream) {
    const auto take = [&]<class Schedule>() {
        if (span.cols > Schedule::kColumns || (first.weight->n % Schedule::kRows) != 0 ||
            (second != nullptr && (second->weight->n % Schedule::kRows) != 0) ||
            (x.input_rows % Schedule::kSlabK) != 0) {
            return false;
        }
        small_gemm<Schedule>(x, span, first, second, stream);
        return true;
    };
    if (schedule == "r16c8") { return take.template operator()<SmallSchedule8>(); }
    if (schedule == "r16c16") { return take.template operator()<SmallSchedule16>(); }
    if (schedule == "r32c32") { return take.template operator()<SmallSchedule32>(); }
    if (schedule == "r16c8w1") { return take.template operator()<SmallScheduleR16C8W1>(); }
    if (schedule == "r16c8s3") { return take.template operator()<SmallScheduleR16C8S3>(); }
    if (schedule == "r32c8") { return take.template operator()<SmallScheduleR32C8>(); }
    if (schedule == "r32c8k4") { return take.template operator()<SmallScheduleR32C8K4>(); }
    if (schedule == "r16c16w1") { return take.template operator()<SmallScheduleR16C16W1>(); }
    if (schedule == "r32c16") { return take.template operator()<SmallScheduleR32C16>(); }
    if (schedule == "r16c32") { return take.template operator()<SmallScheduleR16C32>(); }
    if (schedule == "r32c16k8") { return take.template operator()<SmallScheduleR32C16K8>(); }
    return false;
}

// "t2_i8_small/4096+12288x5120" for a launch over two parents, "t2_i8_small/5120x17408" for one.
std::string small_route_key(const Weight& first, const Weight* second, std::int32_t input_rows) {
    std::string key = "t2_i8_small/" + std::to_string(first.n);
    if (second != nullptr) { key += "+" + std::to_string(second->n); }
    return key + "x" + std::to_string(input_rows);
}

// Launches of at most 32 columns each: from 33 columns the weights stream once per launch, and six
// launches (192 columns) still beat the A16 route and the prefill GEMM's 256-column tile on the
// text-layer shapes (the padded GEMM reloads its activation columns in every 64-row CTA).
template <class Epilogue>
void small_route(const T2A8Activations& x, const SmallParent<Epilogue>& first,
                 const SmallParent<Epilogue>* second, cudaStream_t stream) {
    const std::string key =
        small_route_key(*first.weight, second != nullptr ? second->weight : nullptr, x.input_rows);
    for (std::int32_t col0 = 0; col0 < x.tokens; col0 += kT2I8LaunchColumns) {
        const SmallColumns span{col0, std::min(kT2I8LaunchColumns, x.tokens - col0)};
        const std::string_view routed = device_route_schedule(key, span.cols);
        if (!routed.empty() && small_named(routed, x, span, first, second, stream)) { continue; }
        if (span.cols <= 8) {
            small_gemm<SmallSchedule8>(x, span, first, second, stream);
        } else if (span.cols <= 16) {
            small_gemm<SmallSchedule16>(x, span, first, second, stream);
        } else {
            small_gemm<SmallSchedule32>(x, span, first, second, stream);
        }
    }
}

// Rows [0, split) land in `first` and the rest in `second`, each at its own row offset inside a
// destination of its own height, so one pass over a parent feeds the planes its row ranges belong
// to.
struct SplitStoreEpilogue {
    static constexpr bool kPaired = false;
    __nv_bfloat16* first;
    std::int32_t first_rows;
    std::int32_t first_offset;
    __nv_bfloat16* second;
    std::int32_t second_rows;
    std::int32_t second_offset;
    std::int32_t split;

    __device__ void operator()(std::int32_t row, std::int32_t token, float value) const {
        if (row < split) {
            first[static_cast<std::size_t>(token) * first_rows + row + first_offset] =
                __float2bfloat16(value);
        } else {
            second[static_cast<std::size_t>(token) * second_rows + (row - split) + second_offset] =
                __float2bfloat16(value);
        }
    }
};

template <std::int32_t kCols, int NT>
void quantize(const Tensor& x, std::int32_t tokens, std::int8_t* codes, __half* scales,
              cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kCols, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t kCols, int NT, class Epilogue>
void gemm(const Weight& w, Epilogue epilogue, std::int32_t tokens, const std::int8_t* codes,
          const __half* scales, cudaStream_t stream) {
    constexpr int BN       = a8::kWarpsN * NT * 8;
    const std::size_t smem = a8::shared_bytes<a8::T2Codec, 1, NT, Rows>(kCols);
    const dim3 grid(w.n / Rows::kRowsPerBlock, padded_tokens(tokens) / BN);
    auto* kernel = a8::a8_mma_kernel<a8::T2Codec, kCols, 1, NT, Rows, Epilogue>;
    if (smem > 48 * 1024) {
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem));
        });
    }
    kernel<<<grid, a8::kThreads, smem, stream>>>(static_cast<const std::uint8_t*>(w.qdata), nullptr,
                                                 static_cast<const __half*>(w.scales), codes,
                                                 scales, tokens, Rows{0}, epilogue, 0);
    CUDA_CHECK(cudaGetLastError());
}

// The activation layout depends on the token tile, so the quantiser and every GEMM that reads its
// planes resolve the same (width, tile) pair here.
template <class F>
void dispatch(std::int32_t input_rows, std::int32_t tokens, F&& f) {
    const auto by_tile = [&]<std::int32_t kCols>() {
        switch (a8::token_tile(padded_tokens(tokens), 512)) {
        case 512:
            f.template operator()<kCols, 16>();
            return;
        case 256:
            f.template operator()<kCols, 8>();
            return;
        default:
            f.template operator()<kCols, 4>();
            return;
        }
    };
    switch (input_rows) {
    case 5120:
        by_tile.template operator()<5120>();
        return;
    case 6144:
        by_tile.template operator()<6144>();
        return;
    case 17408:
        by_tile.template operator()<17408>();
        return;
    default:
        break;
    }
    throw std::invalid_argument("t2 a8: unregistered input width");
}

template <class F>
void by_width(std::int32_t input_rows, F&& f) {
    switch (input_rows) {
    case 5120:
        f.template operator()<5120>();
        return;
    case 6144:
        f.template operator()<6144>();
        return;
    case 17408:
        f.template operator()<17408>();
        return;
    default:
        break;
    }
    throw std::invalid_argument("t2 a8: unregistered input width");
}

// NINFER_T2_A8_RASTER=rows|cols forces the tile GEMM's block order (benchmark A/B).
bool tile_rows_fast(std::int32_t output_rows, std::int32_t columns) {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_T2_A8_RASTER");
        if (value == nullptr) { return 0; }
        const std::string text(value);
        return text == "rows" ? 1 : text == "cols" ? -1 : 0;
    }();
    (void)output_rows;
    (void)columns;
    return forced > 0;
}

template <std::int32_t kCols, class Epilogue>
void tile_gemm(const T2A8Activations& x, const Weight& w, Epilogue epilogue, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_G128_FP16) {
        // PTQ1_0's tile: its own schedule (64 rows, a single-buffered int8 plane), and the raw high
        // plane handed to the kernel -- the only argument the T2 instantiation does not take.
        using C                      = T2PrefillI8Ptq1;
        const std::int32_t columns   = tile_columns(x.tokens);
        const bool rows_fast         = tile_rows_fast(w.n, columns);
        const unsigned column_blocks = static_cast<unsigned>(columns / C::kColumns);
        const unsigned row_blocks    = static_cast<unsigned>(w.n / C::kRows);
        const dim3 grid = rows_fast ? dim3(row_blocks, column_blocks) : dim3(column_blocks, row_blocks);
        if (debug_enabled()) {
            debug_note("tile-ptq1 k=" + std::to_string(w.k) + " n=" + std::to_string(w.n) +
                       " tokens=" + std::to_string(x.tokens) + " columns=" + std::to_string(columns) +
                       " rows_fast=" + std::to_string(rows_fast ? 1 : 0) +
                       " blocks=" + std::to_string(column_blocks) + "x" + std::to_string(row_blocks));
        }
        t2_prefill_i8_kernel<kCols, Epilogue, true><<<grid, C::kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const __half*>(w.scales), x.codes,
            x.scales, columns, x.tokens, rows_fast, epilogue,
            static_cast<const std::uint8_t*>(w.qhigh));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    using C                    = T2PrefillI8;
    const std::int32_t columns = tile_columns(x.tokens);
    const bool rows_fast       = tile_rows_fast(w.n, columns);
    const unsigned column_blocks = static_cast<unsigned>(columns / C::kColumns);
    const unsigned row_blocks    = static_cast<unsigned>(w.n / C::kRows);
    const dim3 grid = rows_fast ? dim3(row_blocks, column_blocks) : dim3(column_blocks, row_blocks);
    t2_prefill_i8_kernel<kCols, Epilogue><<<grid, C::kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const __half*>(w.scales), x.codes,
        x.scales, columns, x.tokens, rows_fast, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

template <class Epilogue>
void project(const T2A8Activations& x, const Weight& w, Epilogue epilogue, cudaStream_t stream) {
    if (w.k != x.input_rows || !t2_a8_supported(w, x.tokens)) {
        throw std::invalid_argument("t2 a8: unsupported profile");
    }
    if (route_for(w.qtype == QType::PTQ1_G128_FP16, x.tokens) == Route::SmallT) {
        return small_route<Epilogue>(x, {&w, epilogue}, nullptr, stream);
    }
    if (prefill_tile()) {
        by_width(w.k, [&]<std::int32_t kCols>() { tile_gemm<kCols>(x, w, epilogue, stream); });
        return;
    }
    dispatch(w.k, x.tokens, [&]<std::int32_t kCols, int NT>() {
        gemm<kCols, NT>(w, epilogue, x.tokens, x.codes, x.scales, stream);
    });
}

template <class Epilogue>
void run(const Tensor& x, const Weight& w, Epilogue epilogue, WorkspaceArena& workspace,
         cudaStream_t stream) {
    auto scope = workspace.scope();
    const bool ptq1 = w.qtype == QType::PTQ1_G128_FP16;
    project(t2_a8_quantize(x, workspace, stream, ptq1), w, epilogue, stream);
}

void require_plane(const Tensor& plane, std::int32_t tokens, const char* label) {
    if (plane.dtype != DType::BF16 || plane.data == nullptr || plane.ne[1] != tokens ||
        plane.ne[2] != 1 || plane.ne[3] != 1) {
        throw std::invalid_argument(std::string("t2 a8 split: ") + label +
                                    " must be a BF16 [rows, T] plane");
    }
}

SplitStoreEpilogue split_epilogue(const T2A8Activations& x, const T2A8Split& target) {
    require_plane(target.first, x.tokens, "first");
    require_plane(target.second, x.tokens, "second");
    const Weight& w = target.weight;
    if (target.split <= 0 || target.split > w.n || target.first_offset < 0 ||
        target.first_offset + target.split > target.first.ne[0] || target.second_offset < 0 ||
        target.second_offset + (w.n - target.split) > target.second.ne[0]) {
        throw std::invalid_argument("t2 a8 split: row ranges exceed their destinations");
    }
    return SplitStoreEpilogue{reinterpret_cast<__nv_bfloat16*>(target.first.data),
                              target.first.ne[0],
                              target.first_offset,
                              reinterpret_cast<__nv_bfloat16*>(target.second.data),
                              target.second.ne[0],
                              target.second_offset,
                              target.split};
}

} // namespace

std::int32_t t2_a8_min_tokens() {
    static const std::int32_t minimum = [] {
        const char* value = std::getenv("NINFER_T2_A8_MIN");
        return value != nullptr ? std::max(1, std::atoi(value)) : kT2A8MinTokens;
    }();
    return minimum;
}

std::int32_t t2_ptq1_tile_min_tokens() {
    static const std::int32_t floor = [] {
        const char* value = std::getenv("NINFER_PTQ1_TILE_MIN");
        const std::int32_t parsed = value != nullptr ? std::atoi(value) : kPtq1TileMinTokens;
        return std::clamp(parsed, 1, kT2I8SmallMaxTokens);
    }();
    return floor;
}

bool t2_a8_admits(LinearPolicy policy) {
    return policy == LinearPolicy::AllowA8Int || policy == LinearPolicy::AllowA8IntDecode ||
           policy == LinearPolicy::AllowPrefillCublas;
}

// NINFER_T2_A8_PAIR=off: see the note in the anonymous namespace above. Default (unset) keeps the
// fused pair on, which is the shipping behaviour.
bool t2_a8_pair_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_T2_A8_PAIR");
        return value == nullptr || std::string(value) != "off";
    }();
    return enabled;
}

// NINFER_PTQ1_WIDE_T=on: PTQ1_0's single-projection prefill rides the imported I-line wide-T MMA
// kernel instead of the int8 tile.
//
// Default OFF, and that is the whole point: with this unset the int8 route is byte-for-byte what it
// has been, so the import cannot regress anything. It is a RUNG CHOICE, not a correctness switch --
// the kernel's own container-convention port (this tree stores the payload as two's complement where
// the I line stored it offset-binary) is judged by the wide-t leg of _tucheck_ptq1_smallt_ab.cu
// before any reading from this switch is worth anything.
//
// WHY IT IS INTERESTING AT ALL: the kernel keeps the A operand in bf16 (no int8 activation
// quantisation) and streams one weight window per 64-token tile, and in the I line it sits at
// T >= 41 -- but behind the int8 route, i.e. it is that line's fallback rather than its default. That
// is the same relationship it has here.
bool t2_ptq1_wide_t_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_PTQ1_WIDE_T");
        return value != nullptr && std::string(value) != "0" && std::string(value) != "off";
    }();
    return enabled;
}

std::int32_t t2_ptq1_wide_t_min_tokens() {
    static const std::int32_t floor = [] {
        const char* value = std::getenv("NINFER_PTQ1_WIDE_T_MIN");
        const std::int32_t parsed = value != nullptr ? std::atoi(value) : kTernaryWideMinTokens;
        return parsed > 0 ? parsed : kTernaryWideMinTokens;
    }();
    return floor;
}

// Every admission the ported kernel makes that does not involve the destination, asked BEFORE anything
// is quantised: the wide-T rung reads the bf16 activations directly, so taking it also skips the
// activation planes (and their workspace) entirely. Kept in one place so the debug note and the call
// site cannot disagree; the destination's own checks live at the call site.
bool t2_ptq1_wide_t_admits(const Tensor& x, const Weight& w) {
    if (!t2_ptq1_wide_t_enabled() || w.qtype != QType::PTQ1_G128_FP16) { return false; }
    return x.dtype == DType::BF16 && x.ne[1] >= t2_ptq1_wide_t_min_tokens() &&
           w.qhigh != nullptr && (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) == 0u &&
           w.padded_shape[1] == w.k && (w.k % kTernaryWideChunkK) == 0 && x.ne[0] == w.k;
}

bool t2_a8_shape_supported(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows > 0 && output_rows % prefill_rows_per_block() == 0 &&
           (input_rows == 5120 || input_rows == 6144 || input_rows == 17408);
}

bool t2_a8_supported(const Weight& w, std::int32_t tokens) {
    // PTQ1_0 rides the same tile GEMM with its own schedule, and takes it from its own floor up
    // (t2_ptq1_tile_min_tokens), NOT from T2's band table: below that floor this route's small-T
    // kernel would read PTQ1's five-trits-per-byte as 2-bit T2 fields -- silently wrong, not a throw
    // -- so the arm requires the Prefill verdict of route_for, the tile body, and its high plane.
    const bool ptq1 = w.qtype == QType::PTQ1_G128_FP16;
    const bool packing_ok =
        w.qtype == QType::T2_G128_FP16 ||
        (ptq1 && w.qhigh != nullptr && prefill_tile() && route_for(true, tokens) == Route::Prefill);
    return packing_ok && w.layout == QuantLayout::RowSplit && w.qdata != nullptr &&
           w.scales != nullptr && t2_a8_shape_supported(w.n, w.k) &&
           route_for(ptq1, tokens) != Route::None;
}

std::size_t t2_a8_activation_bytes(std::int32_t input_rows, std::int32_t max_tokens) {
    const auto round  = [](std::size_t value) { return (value + 255) / 256 * 256; };
    std::size_t bytes = 0;
    // The tile's planes are reserved from the LOWER of the two floors: PTQ1_0's tile floor is below
    // T2's, so covering only T2's would under-reserve a PTQ1 prefill call at that width.
    const std::int32_t tile_floor = std::min(t2_a8_min_tokens(), t2_ptq1_tile_min_tokens());
    if (max_tokens >= tile_floor && prefill_tile()) {
        const std::size_t columns = static_cast<std::size_t>(tile_columns(max_tokens));
        bytes = round(columns * static_cast<std::size_t>(input_rows)) +
                round(columns * static_cast<std::size_t>(input_rows / T2PrefillI8::kGroupK) *
                      sizeof(__half));
    } else if (max_tokens >= t2_a8_min_tokens()) {
        // padded_tokens never exceeds the next multiple of 512, whatever T in [1, max_tokens].
        bytes = a8::activation_workspace_bytes(input_rows, (max_tokens + 511) / 512 * 512);
    }
    const SmallBand band = small_band();
    if (band.lo <= band.hi && max_tokens >= band.lo) {
        const std::size_t tokens = static_cast<std::size_t>(std::min(max_tokens, band.hi));
        bytes = std::max(bytes, round(tokens * static_cast<std::size_t>(input_rows)) +
                                    round(static_cast<std::size_t>(kT2I8MaxColumns) *
                                          static_cast<std::size_t>(input_rows / a8::kGroup) *
                                          sizeof(__half)));
    }
    // A device profile can route widths away from the default bands; cover what it routes.
    if (installed_device_route_profile() != nullptr) {
        std::int32_t widest_small = 0;
        std::int32_t widest_tile  = 0;
        for (std::int32_t tokens = 1; tokens <= max_tokens; ++tokens) {
            const Route taken = route(tokens);
            if (taken == Route::SmallT) { widest_small = tokens; }
            if (taken == Route::Prefill) { widest_tile = tokens; }
        }
        if (widest_small != 0) {
            bytes = std::max(bytes, round(static_cast<std::size_t>(widest_small) *
                                          static_cast<std::size_t>(input_rows)) +
                                        round(static_cast<std::size_t>(kT2I8MaxColumns) *
                                              static_cast<std::size_t>(input_rows / a8::kGroup) *
                                              sizeof(__half)));
        }
        if (widest_tile != 0 && prefill_tile()) {
            const std::size_t columns = static_cast<std::size_t>(tile_columns(widest_tile));
            bytes = std::max(bytes, round(columns * static_cast<std::size_t>(input_rows)) +
                                        round(columns *
                                              static_cast<std::size_t>(input_rows /
                                                                       T2PrefillI8::kGroupK) *
                                              sizeof(__half)));
        } else if (widest_tile != 0) {
            bytes = std::max(bytes, a8::activation_workspace_bytes(
                                        input_rows, (widest_tile + 511) / 512 * 512));
        }
    }
    return bytes;
}

std::size_t t2_a8_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t max_tokens) {
    if (!t2_a8_admits(policy) || !t2_a8_shape_supported(output_rows, input_rows)) { return 0; }
    return t2_a8_activation_bytes(input_rows, max_tokens);
}

bool t2_a8_layout_activations(WorkspaceLayoutBuilder& layout, std::int32_t input_rows,
                              std::int32_t tokens, bool ptq1_tile) {
    // The planes one route allocates for this width.
    const auto planes = [&](Route taken) {
        const bool tile = taken == Route::Prefill && prefill_tile();
        const std::size_t groups =
            static_cast<std::size_t>(input_rows) / (tile ? T2PrefillI8::kGroupK : a8::kGroup);
        const std::size_t columns = static_cast<std::size_t>(
            taken == Route::SmallT ? tokens : tile ? tile_columns(tokens) : padded_tokens(tokens));
        const std::size_t scale_columns =
            taken == Route::SmallT ? static_cast<std::size_t>(kT2I8MaxColumns) : columns;
        return std::pair<std::size_t, std::size_t>{
            columns * static_cast<std::size_t>(input_rows), scale_columns * groups * sizeof(__half)};
    };
    const Route taken = route_for(ptq1_tile, tokens);
    if (taken == Route::None) { return false; }
    auto [codes, scales] = planes(taken);
    // A caller that does not know the weight's format (the width-only planners) must not under-size:
    // PTQ1_0's tile floor is below T2's, so the same width can take either layout and the larger one
    // is the safe reservation. Over-reserving costs kilobytes; under-reserving is a silent aliasing.
    const Route other = route_for(!ptq1_tile, tokens);
    if (other != Route::None) {
        const auto [other_codes, other_scales] = planes(other);
        codes  = std::max(codes, other_codes);
        scales = std::max(scales, other_scales);
    }
    (void)layout.alloc_bytes(codes);
    (void)layout.alloc_bytes(scales);
    return true;
}

T2A8Activations t2_a8_quantize(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream,
                               bool ptq1_tile) {
    const std::int32_t input_rows = x.ne[0];
    const std::int32_t tokens     = x.ne[1];
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[2] != 1 ||
        x.ne[3] != 1 || route_for(ptq1_tile, tokens) == Route::None ||
        !t2_a8_shape_supported(T2PrefillI8::kRows, input_rows)) {
        throw std::invalid_argument("t2 a8: x must be a contiguous BF16 [K, T] of a registered K");
    }
    if (route_for(ptq1_tile, tokens) == Route::SmallT) {
        const DeviceSpan codes  = workspace.alloc_bytes(static_cast<std::size_t>(tokens) *
                                                        static_cast<std::size_t>(input_rows));
        const DeviceSpan scales = workspace.alloc_bytes(
            static_cast<std::size_t>(kT2I8MaxColumns) *
            (static_cast<std::size_t>(input_rows) / kT2I8ActivationK) * sizeof(__half));
        auto* code_data      = reinterpret_cast<std::int8_t*>(codes.data);
        auto* scale_data     = reinterpret_cast<__half*>(scales.data);
        constexpr int kWarps = 8;
        const int groups     = input_rows / kT2I8ActivationK;
        const dim3 grid(static_cast<unsigned>((groups + kWarps - 1) / kWarps),
                        static_cast<unsigned>((tokens + 7) / 8 * 8));
        t2_small_t_i8_quantize_kernel<<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), input_rows, tokens, code_data,
            scale_data);
        CUDA_CHECK(cudaGetLastError());
        return {code_data, scale_data, tokens, input_rows};
    }
    if (prefill_tile()) {
        using C                    = T2PrefillI8;
        const std::int32_t columns = tile_columns(tokens);
        const std::size_t count    = static_cast<std::size_t>(columns);
        const DeviceSpan codes = workspace.alloc_bytes(count * static_cast<std::size_t>(input_rows));
        const DeviceSpan scales = workspace.alloc_bytes(
            count * static_cast<std::size_t>(input_rows / C::kGroupK) * sizeof(__half));
        auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
        auto* scale_data = reinterpret_cast<__half*>(scales.data);
        by_width(input_rows, [&]<std::int32_t kCols>() {
            t2_prefill_i8_quantize_kernel<kCols>
                <<<static_cast<unsigned>(columns), C::kQuantThreads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, columns, code_data,
                    scale_data);
            CUDA_CHECK(cudaGetLastError());
        });
        return {code_data, scale_data, tokens, input_rows};
    }
    const std::size_t columns = static_cast<std::size_t>(padded_tokens(tokens));
    const DeviceSpan codes  = workspace.alloc_bytes(columns * static_cast<std::size_t>(input_rows));
    const DeviceSpan scales = workspace.alloc_bytes(
        columns * (static_cast<std::size_t>(input_rows) / a8::kGroup) * sizeof(__half));
    auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
    auto* scale_data = reinterpret_cast<__half*>(scales.data);
    dispatch(input_rows, tokens, [&]<std::int32_t kCols, int NT>() {
        quantize<kCols, NT>(x, tokens, code_data, scale_data, stream);
    });
    return {code_data, scale_data, tokens, input_rows};
}

void t2_a8_project_split(const T2A8Activations& x, const Weight& w, Tensor& first,
                         std::int32_t first_offset, std::int32_t split, Tensor& second,
                         std::int32_t second_offset, cudaStream_t stream) {
    project(x, w, split_epilogue(x, {w, first, first_offset, split, second, second_offset}),
            stream);
}

void t2_a8_project_split_pair(const T2A8Activations& x, const T2A8Split& a, const T2A8Split& b,
                              cudaStream_t stream) {
    const SplitStoreEpilogue first  = split_epilogue(x, a);
    const SplitStoreEpilogue second = split_epilogue(x, b);
    // Either parent can be the PTQ1_0 one (the pair is only ever admitted when BOTH are), so the
    // small/tile choice is asked the same way project() asks it.
    const bool ptq1 = a.weight.qtype == QType::PTQ1_G128_FP16 ||
                      b.weight.qtype == QType::PTQ1_G128_FP16;
    if (route_for(ptq1, x.tokens) != Route::SmallT) {
        project(x, a.weight, first, stream);
        project(x, b.weight, second, stream);
        return;
    }
    for (const Weight* w : {&a.weight, &b.weight}) {
        if (w->k != x.input_rows || !t2_a8_supported(*w, x.tokens)) {
            throw std::invalid_argument("t2 a8: unsupported profile");
        }
    }
    const SmallParent<SplitStoreEpilogue> other{&b.weight, second};
    small_route<SplitStoreEpilogue>(x, {&a.weight, first}, &other, stream);
}

void t2_a8_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    if (debug_enabled()) {
        const Route taken = route_for(w.qtype == QType::PTQ1_G128_FP16, x.ne[1]);
        debug_note("linear qtype=" + std::to_string(static_cast<int>(w.qtype)) +
                   " k=" + std::to_string(w.k) + " n=" + std::to_string(w.n) +
                   " tokens=" + std::to_string(x.ne[1]) + " route=" +
                   (taken == Route::Prefill ? "tile" : taken == Route::SmallT ? "small" : "none") +
                   " out=[" + std::to_string(out.ne[0]) + "," + std::to_string(out.ne[1]) + "] nb=[" +
                   std::to_string(out.nb[0]) + "," + std::to_string(out.nb[1]) + "]");
    }
    // NINFER_PTQ1_WIDE_T=on: the alternative rung (see the gate above). Taken only when the ported
    // kernel's own shape rules hold AND the destination is a bf16 plane whose token row stride can be
    // expressed for it -- the kernel stores at `token * out_row_stride + row`, which is exactly what
    // the int8 route's StoreEpilogue{out, w.n, 0} assumes, so a destination that is correct for one is
    // correct for the other. Nothing is quantised on this path, so the workspace is not touched at all;
    // the layout planner still reserves the activation planes for this width, which is over-reserving
    // rather than under-reserving, and safe.
    if (t2_ptq1_wide_t_admits(x, w) && out.dtype == DType::BF16 && out.ne[0] >= w.n) {
        const std::int64_t stride_elems =
            out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
        const std::int32_t stride =
            static_cast<std::int32_t>(stride_elems < w.n ? w.n : stride_elems);
        if (debug_enabled()) {
            debug_note("linear-wide-t qtype=" + std::to_string(static_cast<int>(w.qtype)) +
                       " k=" + std::to_string(w.k) + " n=" + std::to_string(w.n) +
                       " tokens=" + std::to_string(x.ne[1]) + " out_row_stride=" +
                       std::to_string(stride));
        }
        launch_t2_ptq1_wide_t(x, w, out, stride, stream);
        return;
    }
    run(x, w, a8::StoreEpilogue{reinterpret_cast<__nv_bfloat16*>(out.data), w.n, 0}, workspace,
        stream);
}

void t2_a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    run(x, w, a8::ResidualAddEpilogue{reinterpret_cast<__nv_bfloat16*>(residual.data), w.n},
        workspace, stream);
}

} // namespace ninfer::ops::detail
