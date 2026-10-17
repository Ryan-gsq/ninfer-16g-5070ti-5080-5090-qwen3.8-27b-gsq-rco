#pragma once

// KVMem 按块 raw-K arena（规格里的"路 1"）—— 让"历史块"的**原始 K** 真的留下来。
//
// 为什么必须有它
//   re-RoPE 只能从 raw（pre-RoPE）K 重烘（ops/kvmem/rerope/kvmem_rerope.h:15-40；码域原地旋转
//   已被实测证否）。而 raw K 今天**只在入库那一轮**存在于 harvest 的 device staging
//   （raw_k_harvest.h:10/50-59，每层一个槽、每轮被覆盖），harvest 还是**进程级单例**
//   （raw_k_harvest.h:186-208）⇒ 换一个 chunk 就没了。mean-K 索引存的是**和**不是原始 K
//   （mean_k_index.h:9-11），指望不上。于是"把很久以前的块搬进窗口"这条链缺料。
//
// 这一层做什么：把每个**完整块**的 raw K 在入库那一刻抄出来，按 (layer, block) 定址留存。
//   * 布局与 harvest 的 staging **逐字节同构**（kv_size 最快、token 次之）：rerope 的行寻址是
//     raw_k + head_dim*(kv_head + KVHeads*(raw_token_begin + token))（kvmem_rerope_launch.cu:82-84），
//     所以一块的切片就是它要的原样 ⇒ **重烘不需要任何 reshape**，raw_token_begin = 0。
//   * 只存 K、不存 V：RoPE 只作用于 Q/K（text_context_impl.h:1002），V 与位置无关，搬页表即可
//     ⇒ V 不需要重烘、也就不需要留料。
//
// 内存账（★ 显式可配）
//   每层每块   = block_tokens(64) × kv_size(1024) × 2 B = 131 072 B = **128 KiB**
//   全部 16 层 = **2 MiB / 块 = 32 KiB / token**
//   上下文 C token = (C / block_tokens) × 2 MiB = **C × 32 KiB**
//     C = 32 768 ⇒ 1 GiB      C = 131 072 ⇒ 4 GiB
//     C = 262 144 ⇒ 8 GiB     C = 1 048 576 ⇒ **32 GiB** ← 远端 1M token 夹具的真实代价
//   ⚠ 这是**故意**摆出来的数：要覆盖"远端子集"就不能靠环形缓冲，必须让 arena 覆盖到那些块。
//   容量来源：① `NINFER_TERNARY_KVMEM_RAWK_ARENA_BLOCKS` 显式块数；否则 ② 由
//   `max_context / block_tokens` 推出；再被 ③ `NINFER_TERNARY_KVMEM_RAWK_ARENA_MAX_BYTES`
//   （默认 2 GiB）**夹住**。被夹住时**不是静默截断**：降级成环形，并打一行 `RING-DEGRADED`
//   说明"只覆盖最近 R 块"。
//
// 存哪儿：**pinned host**（复用项目既有的 RawKShadowHostStore，raw_k_shadow.h:92-115）。理由：
//   device 内存是最稀缺的资源（1M token 要 32 GiB device 不可能），而重烘一次只需要把一个块
//   （16 层 × 128 KiB = 2 MiB）搬回 device。device 侧只留 `layers × 128 KiB = 2 MiB` 的 landing。
//   ⚠ pinned host 不可换页：默认预算 2 GiB ≈ 1024 块 ≈ 65 536 token。要覆盖 1M token 的夹具，
//   得把 MAX_BYTES 提到 ~34 GiB，并确认机器上有那么多可 pin 的内存。
//
// 开关与纪律（与 kvmem_shadow.h / kvmem_window.h / kvmem_score.h 同形）
//   `NINFER_TERNARY_KVMEM_RAWK_ARENA=1`，默认关，opt-in，**env 只在本文件读**。
//   关掉时 `raw_k_block_arena_for()` 返回 nullptr ⇒ 入库不分配、不拷贝；重烘退回 staging 那一层
//   ⇒ OFF 臂逐位不变。
//
// 不许静默的三件事（都写进了行为）
//   1. **本轮某层没被收割** ⇒ 那层的 staging 是**上一轮的旧字节**（raw_k_harvest.h:152-165 的
//      教训）⇒ 整轮不存 / 整块不存，并留痕。
//   2. **块超出配置域**（block >= domain_blocks()）⇒ 不存、打一行（入库侧的越界拒答）。
//   3. **环形挤掉老块** ⇒ 构造时 RING-DEGRADED；此后 `has(block)==false` ⇒ 重烘侧**抛错拒答**，
//      绝不拿别的字节顶上。

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/kvmem/kvmem_shadow.h"
#include "ops/kvmem/raw_k_shadow.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// NINFER_TERNARY_KVMEM_RAWK_ARENA=1 才启用"按块留存 raw-K"。默认关。env 只在这里读。
[[nodiscard]] inline bool kvmem_rawk_arena_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_RAWK_ARENA");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

[[nodiscard]] inline std::int64_t kvmem_rawk_arena_env_i64(const char* name,
                                                           std::int64_t fallback) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    return end != value && parsed > 0 ? static_cast<std::int64_t>(parsed) : fallback;
}

// 默认字节预算 2 GiB = 1024 块 ≈ 65 536 token（见文件头的内存账）。
inline constexpr std::int64_t kRawKArenaDefaultMaxBytes = 2LL * 1024 * 1024 * 1024;

// 纯算术、无副作用：两个 arm 都必须能算同一个数（与 mean_k_index_bytes 同纪律）。
[[nodiscard]] inline constexpr std::size_t raw_k_block_bytes(std::int32_t layers, std::int32_t kv_size,
                                                             std::int32_t block_tokens) noexcept {
    if (layers <= 0 || kv_size <= 0 || block_tokens <= 0) { return 0; }
    return static_cast<std::size_t>(layers) * static_cast<std::size_t>(kv_size) *
           static_cast<std::size_t>(block_tokens) * static_cast<std::size_t>(kKvmemRawKBytesPerElement);
}

[[nodiscard]] inline constexpr std::size_t raw_k_block_arena_bytes(std::int32_t layers,
                                                                  std::int32_t blocks,
                                                                  std::int32_t kv_size,
                                                                  std::int32_t block_tokens) noexcept {
    if (blocks <= 0) { return 0; }
    return raw_k_block_bytes(layers, kv_size, block_tokens) * static_cast<std::size_t>(blocks);
}

// 一块 raw-K 的留存器。构造要分配（pinned host + 2 MiB device）⇒ **不许在 CUDA graph 捕获里构造**
// （与 RawKShadowHarvest 同纪律，raw_k_harvest.h:22-29）。
class RawKBlockArena {
public:
    RawKBlockArena(std::int32_t layers, std::int32_t kv_size, std::int32_t block_tokens,
                   std::int32_t max_context_tokens)
        : layers_(layers), kv_size_(kv_size), block_tokens_(block_tokens),
          per_block_bytes_(raw_k_block_bytes(layers, kv_size, block_tokens)),
          landing_arena_(raw_k_block_bytes(layers, kv_size, block_tokens)) {
        if (layers <= 0 || layers > 64 || kv_size <= 0 || block_tokens <= 0 ||
            per_block_bytes_ == 0) {
            throw std::invalid_argument("raw_k_block_arena: geometry is invalid");
        }
        // ① 配置域：由 --max-context 推出，或由显式块数覆盖。
        const std::int64_t derived =
            max_context_tokens > 0
                ? static_cast<std::int64_t>(max_context_tokens) / block_tokens +
                      (max_context_tokens % block_tokens != 0 ? 1 : 0)
                : 0;
        const std::int64_t requested_blocks =
            kvmem_rawk_arena_env_i64("NINFER_TERNARY_KVMEM_RAWK_ARENA_BLOCKS", derived);
        if (requested_blocks <= 0) {
            throw std::invalid_argument("raw_k_block_arena: block capacity is not derivable");
        }
        domain_blocks_ = static_cast<std::int32_t>(
            requested_blocks > 0x7FFFFFFFLL ? 0x7FFFFFFFLL : requested_blocks);
        // ② 字节预算夹住 ⇒ 显式降级成环形。
        const std::int64_t max_bytes = kvmem_rawk_arena_env_i64(
            "NINFER_TERNARY_KVMEM_RAWK_ARENA_MAX_BYTES", kRawKArenaDefaultMaxBytes);
        const std::int64_t budget_blocks =
            static_cast<std::int64_t>(max_bytes / static_cast<std::int64_t>(per_block_bytes_));
        ring_blocks_ = domain_blocks_;
        if (budget_blocks > 0 && budget_blocks < domain_blocks_) {
            ring_blocks_ = static_cast<std::int32_t>(budget_blocks);
        }
        if (ring_blocks_ <= 0) {
            throw std::invalid_argument("raw_k_block_arena: byte budget leaves no room for one block");
        }
        host_.emplace(static_cast<std::size_t>(ring_blocks_) * per_block_bytes_);
        resident_.assign(static_cast<std::size_t>(ring_blocks_), -1);
        covered_cols_.assign(static_cast<std::size_t>(ring_blocks_), 0);
        Tensor flat = landing_arena_.alloc(DType::BF16, {kv_size_, layers_ * block_tokens_});
        if (flat.bytes() != per_block_bytes_) {
            throw std::logic_error(
                "raw_k_block_arena: landing allocation does not match the block size");
        }
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            landing_[static_cast<std::size_t>(layer)] =
                flat.slice(1, layer * block_tokens_, block_tokens_);
        }
    }

    RawKBlockArena(const RawKBlockArena&)            = delete;
    RawKBlockArena& operator=(const RawKBlockArena&) = delete;
    RawKBlockArena(RawKBlockArena&&)                 = delete;
    RawKBlockArena& operator=(RawKBlockArena&&)      = delete;
    ~RawKBlockArena()                                = default;

    [[nodiscard]] std::int32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::int32_t kv_size() const noexcept { return kv_size_; }
    [[nodiscard]] std::int32_t block_tokens() const noexcept { return block_tokens_; }
    // 配置域：能留存的最大块号（不含）。超出即"越界"：入库侧不存、重烘侧拒答。
    [[nodiscard]] std::int32_t domain_blocks() const noexcept { return domain_blocks_; }
    // 实际槽数。`ring_blocks() < domain_blocks()` ⇒ 已降级成环形（老块会被挤掉）。
    [[nodiscard]] std::int32_t ring_blocks() const noexcept { return ring_blocks_; }
    [[nodiscard]] bool is_ring() const noexcept { return ring_blocks_ < domain_blocks_; }

    // 这个块现在在不在 arena 里（越界 / 从未入库 / 已被挤掉 ⇒ false）。
    [[nodiscard]] bool has(std::int32_t block) const noexcept {
        if (block < 0 || block >= domain_blocks_ || ring_blocks_ <= 0) { return false; }
        const std::int32_t slot = block % ring_blocks_;
        return static_cast<std::size_t>(slot) < resident_.size() &&
               resident_[static_cast<std::size_t>(slot)] == block;
    }

    // 入库：把一个块的**所有层**从 device staging 抄进 pinned host。块是原子的——任一层拿不到
    // 可信源（本轮没被收割）就整块不存（半块 = 错料，宁可没有）。`source_at(layer)` 返回该层该块的
    // 128 KiB 起点，返回 nullptr 表示"这层不可信"。
    template <class SourceAt>
    void store_block(std::int32_t block, SourceAt&& source_at, cudaStream_t stream) {
        if (block < 0 || block >= domain_blocks_) {
            ++rejected_out_of_domain_;
            return;
        }
        const std::int32_t slot = block % ring_blocks_;
        const std::int32_t occupied = resident_[static_cast<std::size_t>(slot)];
        if (occupied >= 0 && occupied != block) { ++evicted_blocks_; }
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const void* source = source_at(layer);
            if (source == nullptr) {
                ++rejected_unharvested_;
                return;  // 整块不存
            }
            const cudaError_t status = cudaMemcpyAsync(block_bytes(layer, slot), source,
                                                       per_layer_block_bytes(),
                                                       cudaMemcpyDeviceToHost, stream);
            if (status != cudaSuccess) {
                ++copy_failures_;
                return;  // 整块不标记为 resident
            }
            ++copies_;
        }
        resident_[static_cast<std::size_t>(slot)] = block;
        covered_cols_[static_cast<std::size_t>(slot)] = block_tokens_;
        ++stored_blocks_;
    }

    // 有效列水位：块自块首起连续留存了多少列。0 = 无可用列（不在 arena / 只收到过空洞段）。
    [[nodiscard]] std::int32_t cols_for(std::int32_t block) const noexcept {
        if (!has(block)) { return 0; }
        return covered_cols_[static_cast<std::size_t>(block % ring_blocks_)];
    }

    // ★ 部分块入库（2026-09-24，run8D 尾块 507 双层取料全 miss 后新增）：把本轮写入块 `block`
    // 列区间 [col_off, col_off+cols) 的 raw-K 抄进槽内对应偏移，并把"自块首起连续有效列"的水位
    // 向前推。文档**尾块**的列是逐轮长出来的，永远等不到"整块再入库补齐" ⇒ 只收整块会让尾块
    // 永久缺料。防错料两条硬规则（取代旧的"只收整块"）：
    //   ① 列区间必须接上水位或与已收区重叠（col_off <= covered）⇒ 有**空洞**即拒收并留痕；
    //      col_off == 0 视为**新住户**（跨请求同块号是不同内容，单 lane 纪律 §3.14 之内才可读），
    //      水位重置为本次列数，绝不把上一住户的尾巴当自己的。
    //   ② 块仍是原子的：任一层本轮没被收割 ⇒ 整块不收、水位不动（与整块入库同纪律）。
    // `source_at(layer)` 返回该层**列 col_off** 的起点，nullptr = "这层不可信"。
    template <class SourceAt>
    void store_block_cols(std::int32_t block, std::int32_t col_off, std::int32_t cols,
                          SourceAt&& source_at, cudaStream_t stream) {
        if (cols <= 0 || col_off < 0 || col_off > block_tokens_ ||
            cols > block_tokens_ - col_off) {
            ++rejected_out_of_domain_;
            return;
        }
        if (block < 0 || block >= domain_blocks_) {
            ++rejected_out_of_domain_;
            return;
        }
        const std::int32_t slot = block % ring_blocks_;
        std::int32_t& covered   = covered_cols_[static_cast<std::size_t>(slot)];
        const std::int32_t occupied = resident_[static_cast<std::size_t>(slot)];
        if (occupied >= 0 && occupied != block) {
            ++evicted_blocks_;
            covered = 0;
        } else if (occupied < 0) {
            covered = 0;
        }
        if (col_off == 0) {
            covered = 0;  // 新住户从块首写起：旧尾巴一律作废
        } else if (col_off > covered) {
            ++rejected_partial_gap_;
            if (!gap_reported_) {
                gap_reported_ = true;
                std::fprintf(stderr,
                             "raw_k_block_arena: GAP block=%d col_off=%d covered=%d ⇒ 这段不收"
                             "（缺列的块宁缺勿错，重烘侧会拒答）\n",
                             block, col_off, covered);
            }
            return;
        }
        const std::size_t col_bytes = static_cast<std::size_t>(kv_size_) *
                                      static_cast<std::size_t>(kKvmemRawKBytesPerElement);
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const void* source = source_at(layer);
            if (source == nullptr) {
                ++rejected_unharvested_;
                return;  // 整块不收（水位不动）
            }
            const cudaError_t status = cudaMemcpyAsync(
                static_cast<std::uint8_t*>(block_bytes(layer, slot)) +
                    static_cast<std::size_t>(col_off) * col_bytes,
                source, static_cast<std::size_t>(cols) * col_bytes, cudaMemcpyDeviceToHost, stream);
            if (status != cudaSuccess) {
                ++copy_failures_;
                return;  // 整块不标记推进
            }
            ++copies_;
        }
        resident_[static_cast<std::size_t>(slot)] = block;
        if (covered < col_off + cols) { covered = col_off + cols; }
        ++stored_blocks_;
    }

    // 重烘前：把一个块按层搬回 device landing（H2D，同一条 stream ⇒ 与后面的 kernel 有序）。
    void load_block(std::int32_t block, cudaStream_t stream) {
        if (!has(block)) { throw std::logic_error("raw_k_block_arena: block is not resident"); }
        const std::int32_t slot = block % ring_blocks_;
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const cudaError_t status =
                cudaMemcpyAsync(landing(layer), block_bytes(layer, slot), per_layer_block_bytes(),
                                cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("raw_k_block_arena: H2D failed: ") +
                                         cudaGetErrorString(status));
            }
        }
    }

    // 该层的 landing 起点（与 staging 同构 ⇒ 直接当 rerope 的 raw_k_bf16，raw_token_begin = 0）。
    [[nodiscard]] const void* device_block(std::int32_t layer) const {
        if (layer < 0 || layer >= layers_) {
            throw std::out_of_range("raw_k_block_arena: layer is out of range");
        }
        return landing_[static_cast<std::size_t>(layer)].data;
    }

    [[nodiscard]] std::size_t per_block_bytes() const noexcept { return per_block_bytes_; }
    // 单层一块的字节数。`per_block_bytes_` 是**整块（layers_ 层）**的大小，两者差 layers_ 倍 ——
    // 这一对量的混淆就是 2026-09-24 那次 0xC0000005 的来源，见 block_bytes() 的注释。
    [[nodiscard]] std::size_t per_layer_block_bytes() const noexcept {
        return per_block_bytes_ / static_cast<std::size_t>(layers_);
    }
    [[nodiscard]] std::uint64_t stored_blocks() const noexcept { return stored_blocks_; }
    [[nodiscard]] std::uint64_t evicted_blocks() const noexcept { return evicted_blocks_; }
    [[nodiscard]] std::uint64_t rejected_out_of_domain() const noexcept {
        return rejected_out_of_domain_;
    }
    [[nodiscard]] std::uint64_t rejected_unharvested() const noexcept { return rejected_unharvested_; }
    [[nodiscard]] std::uint64_t copy_failures() const noexcept { return copy_failures_; }

    // 构造时打一次：容量、字节、是否环形降级。**降级必须说出来**。
    void log_configuration() const {
        std::fprintf(stderr,
                     "raw_k_block_arena: ARMED domain_blocks=%d ring_blocks=%d layers=%d kv_size=%d "
                     "block_tokens=%d per_block=%zu B reserved=%.2f GiB\n",
                     domain_blocks_, ring_blocks_, layers_, kv_size_, block_tokens_, per_block_bytes_,
                     static_cast<double>(static_cast<std::size_t>(ring_blocks_) * per_block_bytes_) /
                         (1024.0 * 1024.0 * 1024.0));
        if (is_ring()) {
            std::fprintf(stderr,
                         "raw_k_block_arena: RING-DEGRADED ring_blocks=%d < domain_blocks=%d ⇒ 只覆盖"
                         "**最近 %d 块**（约 %lld token）；更早的块 has()==false，重烘会拒答（不是"
                         "静默用错料）。要全覆盖请提高 NINFER_TERNARY_KVMEM_RAWK_ARENA_MAX_BYTES —— "
                         "1M token 的完整史需要约 32 GiB\n",
                         ring_blocks_, domain_blocks_, ring_blocks_,
                         static_cast<long long>(ring_blocks_) * block_tokens_);
        }
    }

    // 诊断：把 resident 块的字节写成自描述 blob，给**外部 oracle** 比对（判据①）。
    // 布局（小端，int32 header）：magic 'RKBA' | layers | kv_size | block_tokens | ring_blocks
    //   | count | count × int32 block_id | 每块按层序的 BF16 载荷（block_id 升序）
    // 永不抛（诊断不许把引擎带走）：写不成就 stderr 一行。
    void dump_to_file(const char* path) const {
        if (path == nullptr || path[0] == '\0') { return; }
        std::vector<std::int32_t> blocks;
        for (std::int32_t block = 0; block < domain_blocks_; ++block) {
            if (has(block)) { blocks.push_back(block); }
        }
        if (blocks.empty()) {
            std::fprintf(stderr, "raw_k_block_arena: dump skipped (nothing resident)\n");
            return;
        }
        FILE* file = std::fopen(path, "wb");
        if (file == nullptr) {
            std::fprintf(stderr, "raw_k_block_arena: dump open failed path=%s\n", path);
            return;
        }
        const std::int32_t header[6] = {0x41424B52 /* 'RKBA' */, layers_, kv_size_, block_tokens_,
                                        ring_blocks_, static_cast<std::int32_t>(blocks.size())};
        std::size_t written = std::fwrite(header, sizeof(std::int32_t), 6, file);
        written += std::fwrite(blocks.data(), sizeof(std::int32_t), blocks.size(), file);
        for (const std::int32_t block : blocks) {
            const std::int32_t slot = block % ring_blocks_;
            for (std::int32_t layer = 0; layer < layers_; ++layer) {
                const std::uint8_t* source =
                    static_cast<const std::uint8_t*>(block_bytes(layer, slot));
                written += std::fwrite(source, 1, per_layer_block_bytes(), file);
            }
        }
        std::fclose(file);
        std::fprintf(stderr, "raw_k_block_arena: dump blocks=%zu bytes=%zu path=%s\n", blocks.size(),
                     written, path);
    }

private:
    // 一个**槽**住整块（layers_ 层），层在槽内相邻：
    //   槽步长 = per_block_bytes_（整块），层步长 = per_layer_block_bytes()（单层一块）。
    //
    // ★ 2026-09-24 的 0xC0000005（engine-debug 定位，实测 12:59）：原式把层当成"隔 ring_blocks_
    // 个块"，`(layer*ring_blocks_ + slot) * per_block_bytes_`。ring_blocks_=1024、per_block=2 MiB
    // 时 layer=1 的地址 = host + 2 GiB = **正好等于整个 host 缓冲区的末尾**（host_bytes=2 GiB），
    // 于是第一块的第 1 层那笔 D2H 写越界 ⇒ 访问违例、进程 rc=3221225477、连接被重置。
    // 证据行（当时打的临时探针）：`dst=…5260 0000`（layer0）→ `dst=…D260 0000`（layer1，差 128 MiB）
    // 且 `bytes=2097152`（应为单层的 131072）。同时 D2H 的**长度**也用了整块字节数 ⇒ 逐层写 2 MiB。
    // 只改地址式与长度**两端对齐**，不改任何策略、不改 device landing 的布局。
    [[nodiscard]] void* block_bytes(std::int32_t layer, std::int32_t slot) const {
        return static_cast<std::uint8_t*>(host_->data()) +
               (static_cast<std::size_t>(slot) * static_cast<std::size_t>(layers_) +
                static_cast<std::size_t>(layer)) *
                   per_layer_block_bytes();
    }

    [[nodiscard]] void* landing(std::int32_t layer) const {
        return landing_[static_cast<std::size_t>(layer)].data;
    }

    std::int32_t layers_          = 0;
    std::int32_t kv_size_         = 0;
    std::int32_t block_tokens_    = 0;
    std::int32_t domain_blocks_   = 0;
    std::int32_t ring_blocks_     = 0;
    std::size_t  per_block_bytes_ = 0;
    std::optional<RawKShadowHostStore> host_;    // pinned 载荷：layers × ring_blocks × per_block
    DeviceArena                landing_arena_;   // 只放"一个块的 layers 层"
    std::array<Tensor, 64>     landing_{};
    std::vector<std::int32_t>  resident_;        // [ring_blocks]：槽里住着的 block id（-1 = 空）
    std::vector<std::int32_t>  covered_cols_;    // [ring_blocks]：自块首起连续有效列（部分块水位）
    bool                       gap_reported_ = false;
    std::uint64_t rejected_partial_gap_     = 0;
    std::uint64_t stored_blocks_          = 0;
    std::uint64_t evicted_blocks_         = 0;
    std::uint64_t rejected_out_of_domain_ = 0;
    std::uint64_t rejected_unharvested_   = 0;
    std::uint64_t copy_failures_          = 0;
    std::uint64_t copies_                 = 0;
};

// 进程寿命单例，形状与 raw_k_shadow_harvest_for() 一致（raw_k_harvest.h:186-208）：开关关 ⇒
// nullptr（**不分配**）；每次调用的几何必须一致（不一致就抛，防止两个调用方各自猜同一个对象）。
[[nodiscard]] inline RawKBlockArena* raw_k_block_arena_for(std::int32_t layers, std::int32_t kv_size,
                                                           std::int32_t block_tokens,
                                                           std::int32_t max_context_tokens) {
    if (!kvmem_rawk_arena_enabled()) { return nullptr; }
    static RawKBlockArena* instance           = nullptr;
    static std::int32_t    instance_layers    = 0;
    static std::int32_t    instance_kv_size   = 0;
    static std::int32_t    instance_bt        = 0;
    if (instance == nullptr) {
        instance        = new RawKBlockArena(layers, kv_size, block_tokens, max_context_tokens);
        instance_layers = layers;
        instance_kv_size = kv_size;
        instance_bt      = block_tokens;
        instance->log_configuration();
    } else if (instance_layers != layers || instance_kv_size != kv_size || instance_bt != block_tokens) {
        throw std::logic_error("raw_k_block_arena: geometry changed between calls");
    }
    return instance;
}

// 入库驱动：把这一轮 harvest 的**完整块**抄进 arena。放在 mean-K 索引 append 的紧邻处
// （text_prefill_impl.h）——那里同时具备：staging 活着、这一轮的起点已知、且在任何 CUDA graph
// 捕获之外（捕获里不能同步、不能分配）。
//
// ★ 2026-09-24 改（坑表 §5.56）：首尾**半块也收**——文档尾块的列跨轮长成，"半块等下次整块入库
// 补齐"对尾块永远不成立（它没有下次）。错料防线从"只收整块"换成 store_block_cols 的两条硬规则
// （列区间接水位、空洞拒收留痕 + 逐层可信），重烘侧按实际列数取（program_impl 的 block_cols）。
// 旧文注（留档）："存了就是错料（重烘会把半块烘成整块，另半截是垃圾）"——该担忧已由列数显式化解除。
//
// 整轮门槛：`layers_harvested() != layers()` ⇒ **整轮不存**（有层的槽里是上一轮的旧字节）。
template <class Harvest>
inline void raw_k_block_arena_store_round(const Harvest& harvest, std::int32_t block_tokens,
                                          std::int32_t max_context_tokens, cudaStream_t stream) {
    if (!kvmem_rawk_arena_enabled()) { return; }
    if (block_tokens <= 0) { return; }
    RawKBlockArena* arena =
        raw_k_block_arena_for(harvest.layers(), harvest.kv_size(), block_tokens, max_context_tokens);
    if (arena == nullptr) { return; }
    const std::int32_t width = harvest.round_width();
    if (width <= 0) { return; }
    if (harvest.layers_harvested() != harvest.layers()) {
        std::fprintf(stderr,
                     "raw_k_block_arena: ROUND-SKIPPED harvested=%d of %d layers ⇒ 本轮不存任何块"
                     "（半轮 = 错料）\n",
                     harvest.layers_harvested(), harvest.layers());
        return;
    }
    const std::int32_t origin = harvest.round_first_token();
    if (origin < 0) { return; }
    // ★ 2026-09-24 改（run8D：尾块 507 永久缺料 ⇒ 取料双层全 miss）：首尾**半块也收**，列区间
    // 显式入库（store_block_cols 的两条防错料硬规则接替"只收整块"）；重烘侧按**实际列数**取。
    const std::int32_t first_block = origin / block_tokens;
    const std::int32_t last_block  = (origin + width - 1) / block_tokens;  // 含
    for (std::int32_t block = first_block; block <= last_block; ++block) {
        if (block < 0 || block >= arena->domain_blocks()) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                std::fprintf(stderr,
                             "raw_k_block_arena: OUT-OF-DOMAIN block=%d domain_blocks=%d ⇒ 不存"
                             "（重烘侧会拒答，不会静默用错料）\n",
                             block, arena->domain_blocks());
            }
            continue;
        }
        const std::int32_t block_begin = block * block_tokens;
        const std::int32_t limit       = block_begin + block_tokens;
        const std::int32_t end         = origin + width < limit ? origin + width : limit;
        const std::int32_t c0          = origin > block_begin ? origin - block_begin : 0;
        const std::int32_t cols        = end - (block_begin + c0);
        if (cols <= 0) { continue; }
        const std::int32_t offset_tokens = block_begin + c0 - origin;
        arena->store_block_cols(
            block, c0, cols,
            [&](std::int32_t layer) -> const void* {
                // 逐层复核：这一层本轮真的被写过吗（没写 ⇒ nullptr ⇒ 整块不存）。
                if (!harvest.layer_harvested(layer)) { return nullptr; }
                const std::uint8_t* base =
                    static_cast<const std::uint8_t*>(harvest.staging_for_round(layer).data);
                return base + static_cast<std::size_t>(offset_tokens) *
                                  static_cast<std::size_t>(harvest.kv_size()) *
                                  static_cast<std::size_t>(kKvmemRawKBytesPerElement);
            },
            stream);
    }
}

} // namespace ninfer::ops::detail
