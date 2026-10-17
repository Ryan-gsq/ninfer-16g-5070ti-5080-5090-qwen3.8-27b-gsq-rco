#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::unified {

// LOCAL ADAPTATION (toolchain, NOT part of the K ring port, 2026-09-29):
// Everything from here to the matching #endif is the TMA-only descriptor + kernel path. It is
// unreachable on sm_8x: launch_bf16_a16_tma_mma() routes sm_8x to the cp.async fallback (see the
// NINFER_SM8X_COMPAT branch below), and route selection cannot reach the TMA kernel there.
// It is compiled out because nvcc 13.3 + MSVC 14.51 rejects `struct alignas(128)` passed BY VALUE as
// a __grid_constant__ parameter: error C2719 "requires 128-byte alignment, will not be aligned".
// Dropping the alignas is NOT an acceptable fix: MSVC ignores the alignas(128) that CUDA 13.3 puts
// on CUtensorMap's member, so alignof(CUtensorMap) silently drops below the 128 bytes the TMA
// hardware requires (verified by static_assert). sm_120a behaviour is unchanged: the guard is false
// there, so the same code compiles exactly as before.
//
// LOCAL ADAPTATION (Windows 120a): MSVC cannot pass this alignas(128) struct by value as a
// __grid_constant__ parameter (error C2719, seen on build-120a: bf16_attn_input_gemm_mma_unified).
// The two NVFP4 TMA paths solve that by staging the descriptors through device memory; this bf16
// path has no staging path, and does not need one: launch_bf16_a16_tma_mma already falls back to
// the cp.async tile (Bf16TmaFallbackSchedule) whenever SM8X_COMPAT is set without the NVFP4 unit --
// which is exactly what the Windows sm_86/sm_89 builds compile and run. So MSVC takes the same
// fallback here, and this TMA-only block stays out of the build. The guard's stale premise is in
// CMakeLists.txt:58-64, which later made NINFER_SM120_NVFP4 unconditional for every 120a build and
// so flipped this condition to true on 120a.
#if (!defined(NINFER_SM8X_COMPAT) || defined(NINFER_SM120_NVFP4)) && !defined(_MSC_VER)

struct alignas(128) Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};

inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    CUtensorMap result{};
    const std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    const std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                              static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(pointer),
        dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("BF16 TMA requires complete row/K tiles");
    return {bf16_tma_map(p.weight, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            bf16_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector), "r"(row),
                   "r"(smem_addr(barrier))
                 : "memory");
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    const __grid_constant__ Bf16TmaDescriptors descriptors, Output output, Epilogue epilogue,
    int rows, int input_rows, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = rows / BR, tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK;
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                bf16_tma_load(a + stage * BR * BK, &descriptors.weight, kt * (BK / 64), row_begin,
                              full + stage);
                bf16_tma_load(b + stage * BT * BK, &descriptors.activation, kt * (BK / 64),
                              token_begin, full + stage);
            }
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        // Every lane must finish its shared reads before the elected lane releases this stage.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

#endif // TMA-only descriptors + kernel (see the LOCAL ADAPTATION note at the top)

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream);

// sm_8x has no tensor memory accelerator; there the same tile runs over cp.async staging.
template <class S>
using Bf16TmaFallbackSchedule = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<S::kBlockRows, S::kBlockTokens, S::kBlockK, S::kWarpRows, S::kWarpTokens,
                       S::kStages, S::kMinBlocksPerSm, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, S::kRaster, Bf16MmaSwizzle::Xor64,
                       S::kRasterGroupRows>,
    S::kStaticK>;

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
// LOCAL ADAPTATION (Windows 120a): same reason as the block above -- MSVC cannot pass the
// alignas(128) descriptors by value, so it takes the cp.async fallback that the Windows sm_86/sm_89
// builds already run, instead of the TMA path it cannot compile.
#if (defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)) || defined(_MSC_VER)
    launch_bf16_a16_mma<Bf16TmaFallbackSchedule<Schedule>>(p, output, epilogue, stream);
#else
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule>(p);
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                descriptors, output, epilogue, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
#endif
}

} // namespace ninfer::ops::detail::unified
