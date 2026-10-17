#pragma once

// KVMem 选择探针（**最小切片**）· 只做三件事：按 query 打分 → 选块 → 打一行证据日志
//
// 【它是什么】把已经在库里的三件套接上主路径的**第一刀**：
//     kvmem_retrieve（官方 KVMem 论文 Eq.10 的 query-conditioned 打分器）
//   → kvmem_select  （host：sink/recent 恒留 + 中段按分降序）
//   → 一行 fprintf（对齐官方实现的三口径：selected 非连续 / skip>0 / window≈budget）
//
// 【它不是什么】**不碰页表、不碰 cache position、不碰 attention envelope、不 arm 窗口**。
//   ⇒ 可见集不变、OFF 臂逐位不变、**不需要 re-RoPE**（那属于第二刀：见 kvmem_window.h 的 arming 前置）。
//   ⇒ 因此本探针**不会让任何答案变对或变错**，它只回答"选择器选得对不对"。
//
// 【开关】NINFER_TERNARY_KVMEM_SCORE=1（默认关）。OFF 臂：不分配、不启动、不读索引、不写日志。
//   ⚠️ 开关与 env 只在这个头文件里读（与 kvmem_shadow.h 同一条纪律：开关的读取点必须唯一）。
//
// 【为什么只有打分器需要新代码】索引（mean_k_index，prefill 时按 chunk 喂）、raw-K 收割
//   （raw_k_harvest）、窗口载体（kvmem_window）**都已经接在引擎里**；缺的正是中间那句"按分选块"。
//
// 【两处必须记住的接缝】
//   1. 索引存的是每块 key 的 **和**（不是 mean）⇒ 打分器要用 block_tokens 自己除；而索引里的计数是
//      **F32**，打分器要 **int32** ⇒ 这里在 host 侧按 blocks_written/tail_fill 造 int32 数组
//      （尾部不满的块必须带**真实填充**，不能带名义块大小 —— mean_k_index.h:127-131）。
//   2. query 必须是**内容帧**（de-RoPE 之后的位置无关帧），与索引同帧 ⇒ 引擎里那个可用的点只有
//      `qn`（RMS 归一化后、`ops::rope` 覆盖之前）。这就是为什么 accumulate() 的调用点在 rope 之前。
//
// 【已知边界（写死在代码里，别当没看见）】
//   * **多 lane 不安全**：mean-K 索引是**全进程一份**（mean_k_index.h:248-251 自陈），本探针的
//     score 缓冲同样是全进程一份 ⇒ 只在单路（--max-concurrency 1）下有意义。多路时必须 per-lane。
//   * query span 的**分段**口径：≤ MAXQ（默认 256）的 span 一次内核调用打完；**更长的 span 从
//     2026-10-04 起改为"取该 span 末尾 ≤ SCORE_SPAN_MAX（默认 4096）个 token、按 MAXQ 切段求和"**
//     （B05：此前是整段跳过 ⇒ 该 chunk 一分不打 ⇒ KEPT 0 ⇒ 回落词法排序）。分段在数学上是正确的：
//     内核用 `atomicAdd` 累加、契约是 `sum_b score[b] == n_query_tokens`（kvmem_retrieve.h:117）
//     ⇒ 段和 == 整段一遍。文档 ingest 那种几万 token 的 chunk 仍受 SCORE_SPAN_MAX 约束
//     （O(块数×query 长) 会拖死 prefill）；`NINFER_TERNARY_KVMEM_SCORE_SPAN_SEGMENT=0` 回到旧的
//     跳过行为 —— 那也正是这条改动的**负控**。
//   * ★ 2026-09-26（语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把 chunk 的最后 N 个
//     token 当检索 query 打分（官方 kvmem_set_query_span 的等价物：query 是"问"，不是整段
//     ingest），MAXQ 的门限看 **span 长度**、不看 chunk 长度 —— 解掉"探针只在短 query chunk
//     上打分（skip long chunk tokens=1024 > MAXQ=256）"那块拦路石。默认 0 = 旧行为（逐位不变）。
//   * 本探针**不改可见集** ⇒ 它与"答案对不对"无关；要验证"被选中 ⇒ 真的进了 softmax"必须等第二刀。

#include "ops/kvmem/kvmem_retrieve_launch.h"
#include "ops/kvmem/kvmem_select.h"
#include "ops/kvmem/mean_k_index.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// ---- 开关（唯一读取点；env 只读一次，与 kvmem_shadow_enabled() 同形）---------------------------
//
// ★ 2026-10-02 **默认翻转**（用户裁决："我们肯定是要开着的"）。理由不是口味，是实测：
//   内容打分是低显存环这条路上"修中段丢针"的那一刀，把它默认关着 = 把修挡在门外。
//   出厂形态（启动器只设五个 ring env、不设 KVMem 那三行）实测：池 4000 多针 n=6 **hit=2/6**，
//   两轮的错答一律是题面 filler 里的 `300`（静默错），而打开打分后是 turn1 6/6 + turn2 6/6。
//
//   三态规则（**显式优先**）：
//     NINFER_TERNARY_KVMEM_SCORE=0        ⇒ 关（负控：只有显式要求才关）
//     NINFER_TERNARY_KVMEM_SCORE=1        ⇒ 开（显式）
//     NINFER_TERNARY_KVMEM=0（大开关）    ⇒ 关（"不要 KVMem"这句话仍然算数）
//     都没表态 且 NINFER_KV_WINDOW > 0    ⇒ **开**（环被请求 = 低显存配置 = 这条修真正有用的场景）
//     都没表态 且 环没开                  ⇒ 关（全池档没有可回捞的页，开着只是白花力气）
//   ⇒ **修跟着二进制走**，不再依赖交付启动器是否写那三行 env。
inline bool kvmem_score_enabled() noexcept {
    static const bool on = [] {
        const auto explicit_zero = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] != '\0' && v[0] == '0';
        };
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE");
        if (v != nullptr && v[0] != '\0') { return v[0] != '0'; }   // 子开关表态优先
        if (explicit_zero("NINFER_TERNARY_KVMEM")) { return false; } // 大开关 =0 也算关
        const char* w = std::getenv("NINFER_KV_WINDOW");
        if (w == nullptr || w[0] == '\0') { return false; }
        return std::strtol(w, nullptr, 10) > 0;                      // 环开着 ⇒ 默认开
    }();
    return on;
}

// 数值旋钮的读法：**0 与负数一律当"未设"**。这是刻意口径，不是疏漏 ——
// 例如 QUERY_TAIL 给 0 会让打分窗口变空、"默默什么都不打"，比回落默认值危险得多
// （见 accumulate 里那段 "未设/0 ⇒ 64 更安全" 的注释）。
// ⚠️ 布尔开关**不要**用它：`=0` 在这里被读成"未设"⇒ 回落成默认值 ⇒ 负控永不生效（B25）。
// 布尔开关一律用下面的 kvmem_score_env_flag()。
inline std::int32_t kvmem_score_env_i32(const char* name, std::int32_t fallback) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') { return fallback; }
    const long parsed = std::strtol(v, nullptr, 10);
    return parsed > 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

// 布尔开关的读法：**设了就用** —— `=0` 就是关、`=非0` 就是开，只有"未设/空串"才回落 fallback。
// 存在的理由（B25，2026-10-07 实测）：`NINFER_TERNARY_KVMEM_SCORE_SPAN_SEGMENT=0` 是本改动
// **文档化的负控**，但它若走 kvmem_score_env_i32，0 会被当成"未设"⇒ 回落 1 ⇒ 该负控永不生效、
// 与其配对的 "skip long query span" 分支不可达（实测：三条臂上正控/负控逐行同值）。
inline bool kvmem_score_env_flag(const char* name, bool fallback) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') { return fallback; }
    return std::strtol(v, nullptr, 10) != 0;
}

// ---- ★ ActKV (arXiv:2609.31395) retention policy -------------------------------------------------
//
// The two mechanisms ActKV v1.3 kept and measured -- LRFU retention (Sec. 5.2) and semantic
// protection (Sec. 8.1). Switch AND environment are read HERE and nowhere else, the same
// single-read-point discipline the rest of this header follows.
//
// NAMING: this fork drives KVMem through `NINFER_KV_*` environment variables rather than the
// upstream `--kvmem-*` option family (`grep -c "kvmem" serve_options.cpp` is 0 here, and the whole
// KVMem component predates the CLI). The ActKV switches therefore follow the environment
// convention already established by their neighbours (`NINFER_TERNARY_KVMEM_SCORE*`, single read
// point in this file) instead of inventing a second, parallel option family. The upstream option
// spellings are recorded in each comment so the two backends can still be compared.
//
// OFF IS BIT-IDENTICAL. With `action_lrfu` off the ranking input is the raw retrieval score and
// `begin_round`/`commit` are never called; with `protect_semantics` off no protected page is ever
// computed or consulted. Both default to off, so the engine reproduces its previous readings
// exactly -- and that reproduction IS the negative control.

// NINFER_TERNARY_KVMEM_ACTION_LRFU=1  (upstream: --kvmem-action-lrfu, default off)
[[nodiscard]] inline bool kvmem_actkv_lrfu_enabled() noexcept {
    static const bool on =
        kvmem_score_env_flag("NINFER_TERNARY_KVMEM_ACTION_LRFU", false);
    return on;
}

// NINFER_TERNARY_KVMEM_LRFU_LAMBDA (upstream: --kvmem-lrfu-lambda, default 0.50)
// NINFER_TERNARY_KVMEM_LRFU_TOP_FRAC (upstream: --kvmem-lrfu-top-frac, default 0.90)
//
// Both default to the v1.3 measured values. They come from the 262k NIAH round in v1.3's own
// `docs/actkv-retention-flags.md` (which is NOT part of the patch), so they are a STARTING POINT
// for this tree rather than a calibrated pair -- this fork's pages are different from v1.3's, so
// the values must be re-calibrated locally before they are trusted (see the report's open items).
[[nodiscard]] inline double kvmem_actkv_lrfu_lambda() noexcept {
    static const double value = [] {
        const char* text = std::getenv("NINFER_TERNARY_KVMEM_LRFU_LAMBDA");
        if (text == nullptr || text[0] == '\0') { return 0.50; }
        char* end = nullptr;
        const double parsed = std::strtod(text, &end);
        if (end == text || *end != '\0' || !(parsed > 0.0) || !(parsed < 1.0)) {
            std::fprintf(stderr,
                         "kvmem_actkv: NINFER_TERNARY_KVMEM_LRFU_LAMBDA=%s is outside (0,1); "
                         "falling back to 0.50\n",
                         text);
            return 0.50;
        }
        return parsed;
    }();
    return value;
}

[[nodiscard]] inline double kvmem_actkv_lrfu_top_frac() noexcept {
    static const double value = [] {
        const char* text = std::getenv("NINFER_TERNARY_KVMEM_LRFU_TOP_FRAC");
        if (text == nullptr || text[0] == '\0') { return 0.90; }
        char* end = nullptr;
        const double parsed = std::strtod(text, &end);
        if (end == text || *end != '\0' || !(parsed > 0.0) || !(parsed <= 1.0)) {
            std::fprintf(stderr,
                         "kvmem_actkv: NINFER_TERNARY_KVMEM_LRFU_TOP_FRAC=%s is outside (0,1]; "
                         "falling back to 0.90\n",
                         text);
            return 0.90;
        }
        return parsed;
    }();
    return value;
}

// NINFER_TERNARY_KVMEM_RETRIEVAL_TEMPERATURE (upstream: --kvmem-retrieval-temperature, default 0)
// 0 keeps the raw cosine scale, so the default path is byte-identical to before the switch.
[[nodiscard]] inline double kvmem_actkv_retrieval_temperature() noexcept {
    static const double value = [] {
        const char* text = std::getenv("NINFER_TERNARY_KVMEM_RETRIEVAL_TEMPERATURE");
        if (text == nullptr || text[0] == '\0') { return 0.0; }
        char* end = nullptr;
        const double parsed = std::strtod(text, &end);
        if (end == text || *end != '\0' || !(parsed >= 0.0)) {
            std::fprintf(stderr,
                         "kvmem_actkv: NINFER_TERNARY_KVMEM_RETRIEVAL_TEMPERATURE=%s is not "
                         "non-negative; falling back to 0 (raw cosine scale)\n",
                         text);
            return 0.0;
        }
        return parsed;
    }();
    return value;
}

// NINFER_TERNARY_KVMEM_PROTECT_SEMANTICS=1  (upstream: --kvmem-protect-semantics, default off)
[[nodiscard]] inline bool kvmem_actkv_protect_enabled() noexcept {
    static const bool on =
        kvmem_score_env_flag("NINFER_TERNARY_KVMEM_PROTECT_SEMANTICS", false);
    return on;
}

// A diagnostic-only switch, mirroring upstream's KVMEM_PROTECT_TRACE.
[[nodiscard]] inline bool kvmem_actkv_trace() noexcept {
    static const bool on = kvmem_score_env_flag("NINFER_TERNARY_KVMEM_ACTKV_TRACE", false);
    return on;
}

// The one LRFU accumulator for this lane. Process-global for the same reason every other piece of
// this probe is (mean-K index, score buffer): it is meaningful only under --max-concurrency 1.
// A per-lane form is what a multi-lane build would need, and that is stated rather than implied.
[[nodiscard]] inline ops::actkv::LrfuState& kvmem_actkv_lrfu_state() {
    static ops::actkv::LrfuState state{ops::actkv::LrfuConfig{
        kvmem_actkv_lrfu_lambda(), kvmem_actkv_lrfu_top_frac(),
        kvmem_actkv_retrieval_temperature(), /*deferred=*/true}};
    return state;
}

// ---- ActKV Sec. 8.1 protected spans --------------------------------------------------------------
//
// The two never-compress spans, published per REQUEST by the frontend (frontend.cpp already knows
// both frontiers in token space -- it uses them for the query span). Absolute token indices, the
// same coordinate system `abs_query_begin/end` uses.
//
// Published unconditionally on every request, `-1` included: a request that yields no instruction
// prefix (or no user turn) must CLEAR a stale span instead of inheriting the previous request's.
// That inheritance bug is already recorded in this file for the query span, and the same failure
// here would pin hot but unrelated pages.
struct KvMemProtectSpans {
    std::int32_t instruction_begin = -1;
    std::int32_t instruction_end   = -1;
    std::int32_t task_begin        = -1;
    std::int32_t task_end          = -1;
    std::int32_t total_tokens      = 0;
};

[[nodiscard]] inline KvMemProtectSpans& kvmem_actkv_spans() {
    static KvMemProtectSpans spans;
    return spans;
}

inline void kvmem_publish_protect_spans(std::int32_t instruction_begin,
                                        std::int32_t instruction_end,
                                        std::int32_t task_begin, std::int32_t task_end,
                                        std::int32_t total_tokens) noexcept {
    KvMemProtectSpans& spans = kvmem_actkv_spans();
    spans.instruction_begin = instruction_begin;
    spans.instruction_end   = instruction_end;
    spans.task_begin        = task_begin;
    spans.task_end          = task_end;
    spans.total_tokens      = total_tokens;
}

// The pages the current request's protection covers, or empty when the switch is off / the spans
// are unusable. `first_block` is added so the page ids come back in the same id space the
// selector uses (the selector's `scores` array starts at `first_block`, and for every caller in
// this tree `first_block` is 0 -- the parameter exists so this cannot silently mean the wrong
// thing if that ever changes).
[[nodiscard]] inline std::vector<std::uint32_t> kvmem_actkv_protected_pages(
    std::int32_t page_tokens, std::int32_t first_block) {
    std::vector<std::uint32_t> pages;
    if (!kvmem_actkv_protect_enabled()) { return pages; }
    const KvMemProtectSpans& spans = kvmem_actkv_spans();
    if (spans.total_tokens <= 0) { return pages; }
    const actkv::ProtectSpan instruction{spans.instruction_begin, spans.instruction_end};
    const actkv::ProtectSpan task{spans.task_begin, spans.task_end};
    pages = actkv::semantic_pages(instruction, task, page_tokens, spans.total_tokens);
    if (first_block > 0) {
        for (std::uint32_t& page : pages) {
            if (page < static_cast<std::uint32_t>(first_block)) { page = 0; continue; }
            page -= static_cast<std::uint32_t>(first_block);
        }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    }
    return pages;
}

// ---- 探针状态（进程级单例，与 mean-K 索引同寿命）---------------------------------------------
struct KvMemScoreProbe {
    // device 缓冲
    float*        score            = nullptr;  // [capacity_blocks]，每 chunk 前清零（打分器只 ADD）
    std::int32_t* block_tokens_dev = nullptr;  // [capacity_blocks]，尾块带真实填充（device 侧）
    // host staging
    std::vector<float>        score_host;
    std::vector<std::int32_t> tokens_host;
    // LOCAL (2026-10-02, the caller step): the block ids the last finish() selected, ASCENDING.
    // This is what the ring consumes as its `preferred` set. It exists because this file used to be a
    // probe that only LOGGED the selection ("本探针不改可见集"); a caller that reads it is the
    // difference between "we can see the ranking" and "the ranking decides what the model reads".
    // EMPTY means "no fresh selection" (feature off, not armed, or nothing scored this chunk) and the
    // caller must fall back to its previous behaviour -- never read empty as "keep nothing".
    std::vector<std::int32_t> kept;

    std::int32_t capacity_blocks = 0;
    std::int32_t block_tokens    = 0;      // == kPagedKVPageSize，由调用方传入（house rule：不许写死）
    std::int32_t chunk_tokens    = 0;      // live 平面的 token 行数（= q_layer_stride 口径）
    std::int32_t n_blocks        = 0;      // 本 chunk 打分的"历史"块数（= 打分开始时的 blocks_written）
    // ---- ★ 分数的**出处**（2026-09-26 语义档要用：静默用错查询/错时刻 = 静默错答案）------------
    std::int32_t query_span_begin  = 0;    // 本次 query span 在 live 平面里的起点（QUERY_TAIL 档）
    std::int32_t query_span_tokens = 0;    // 本次真正打分的 query token 数（= span 长度）
    // ★ D-12（2026-10-02）：**绝对** query span（= 最后一轮 role=user 的 token 区间）+ 本 chunk 的绝对
    //   起点。两者都已知时优先于尾窗规则（见 kvmem_score_accumulate）。-1 = 未知 ⇒ 回落尾窗。
    std::int32_t abs_query_begin = -1;
    std::int32_t abs_query_end   = -1;
    std::int32_t abs_chunk_begin = -1;
    bool logged_span_miss        = false;   // 空交集只喊一次（见 accumulate 的回落说明）
    std::int32_t scored_n_blocks   = 0;    // 这份分数描述块 [0, scored_n_blocks)（finish 时登记）
    bool scored_last_chunk = false;        // 最近一个 prefill chunk 真打过分（语义档的准入闸）
    std::int32_t layers_total    = 0;
    // B08 证据：本轮索引**真的**从多少层喂入（= harvest 的逐层覆盖戳计数，见 append_round 的跳过
    // 计数）。`harvest_layers == harvest_layers_total` 就是"层 0 也在索引里"的正向读数；-1 = 本轮
    // 没有报（未开环 / 未 arm / 走的是旧路径）。
    std::int32_t harvest_layers       = -1;
    std::int32_t harvest_layers_total = 0;
    std::int32_t n_heads         = 0;
    std::int32_t n_kv_heads      = 0;
    std::int32_t head_dim        = 0;
    bool armed    = false;                 // 本 chunk 正在打分
    bool scored_this_chunk = false;        // 本 chunk 真的累加过（否则 finish 打出来的是全零假日志）
    bool failed   = false;                 // 出过致命几何/分配错 ⇒ 之后一律不打分（但引擎继续跑）
    bool logged_once = false;
    bool logged_span_segment = false;   // B05: 长 span 改为切段打分，只喊一次（它是"行为变了"的证据行）
};

inline KvMemScoreProbe& kvmem_score_probe() {
    static KvMemScoreProbe probe;
    return probe;
}

// LOCAL (2026-10-02, the caller step): the block ids selected by the most recent kvmem_score_finish().
// ASCENDING (kvmem_select.h:30-33 -- never score order). Empty = no fresh selection, see the note on
// KvMemScoreProbe::kept. Read by ProgramImpl::select_retrieval_pages as the ring's `preferred` set.
inline const std::vector<std::int32_t>& kvmem_score_selected_blocks() noexcept {
    return kvmem_score_probe().kept;
}

// The per-block scores behind that selection: index i is the score of block i (first_block is 0 at our
// single call site). Published because the ring's restore budget is SMALLER than the selector's
// default budget, so the caller has to rank the kept set again and take its top `budget` -- truncating
// the ascending id list would silently keep the OLDEST blocks instead of the best ones.
inline const std::vector<float>& kvmem_score_host_scores() noexcept {
    return kvmem_score_probe().score_host;
}

// How many blocks the published scores describe ([0, n)). Zero = nothing to read.
inline std::int32_t kvmem_score_scored_blocks() noexcept {
    return kvmem_score_probe().scored_n_blocks;
}

// ---- ①b 绝对 query span（D-12, 2026-10-02）------------------------------------------------------
//
// 官方口径：检索 query 是**最后一轮 role=user**（服务层捕获，捕获与打分分两步；出处
// E:\infer-docs\02-交接与状态\KVMem-移植-交接-20260921.md:308 "官方捕获整个被标记的 query span
// （服务层口径：取最后一轮 role=user）"）。frontend 在准备 prompt 时把该轮的 token 区间（**绝对下标**）
// 交到这里，打分时再与当前 chunk 求交 —— 这就是那份"两步"的落地，不再靠"尾 N token"去猜问句在哪。
//
// 与整个探针一样，这份状态是**进程级**的 ⇒ 只在 --max-concurrency 1 下有意义（多路必须 per-lane）。
// B08：本轮索引的逐层覆盖读数 —— `MeanKIndex::append_round` 只喂"本轮真被写过的层"，这里把它的计数
// 送到 SELECT 行（`harvest_layers=K/L`）。K == L 即"层 0 也在索引里"；K < L 时 append_round 另有一行
// 说明跳过的是哪些层、以及为什么（那些槽里还是上一轮的字节）。
inline void kvmem_score_set_harvest_layers(std::int32_t harvested, std::int32_t total) noexcept {
    KvMemScoreProbe& p     = kvmem_score_probe();
    p.harvest_layers       = harvested;
    p.harvest_layers_total = total;
}

// begin < 0 = 未知 ⇒ 回落 QUERY_TAIL 尾窗规则：那既是既有行为，也是这条改动的**负控**。
inline void kvmem_score_set_query_span(std::int32_t begin, std::int32_t end) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    p.abs_query_begin = begin;
    p.abs_query_end   = end;
}

// 本 chunk 的**绝对** token 起点（= 该序列此刻已写入 KV 的 token 数，即 text_kv_base）。只有它已知，
// 绝对 span 才能落到 chunk 内的局部下标上；未知则同样回落尾窗。
inline void kvmem_score_set_chunk_origin(std::int32_t abs_begin) noexcept {
    kvmem_score_probe().abs_chunk_begin = abs_begin;
}

inline void kvmem_score_fail(const char* why) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    p.failed = true;
    p.armed  = false;
    std::fprintf(stderr, "kvmem_score: DISABLED (%s)\n", why);
}

// ---- ① chunk 开始：确保容量 + 清零 + 记下"历史块数"--------------------------------------------
//
// history_blocks 取**打分开始那一刻**的 blocks_written()：本 chunk 自己还没进索引（索引在 chunk
// 之后才 append_round），所以它天然是"历史"——这正是检索该对的东西。
inline void kvmem_score_begin(std::int32_t requested_capacity, std::int32_t history_blocks,
                              std::int32_t tail_fill, std::int32_t block_tokens,
                              std::int32_t layers_total, std::int32_t n_heads,
                              std::int32_t n_kv_heads, std::int32_t head_dim,
                              cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.failed) { return; }

    // LOCAL (D-12, 2026-10-02): a round that ends up NOT scoring -- because its chunk does not contain
    // the query span (see kvmem_score_accumulate) -- must not leave the PREVIOUS round's selection
    // behind. That selection answers another query (possibly another request's), and the ring would
    // restore the pages THAT query asked for. "Empty = no fresh selection" is already the contract the
    // caller implements (context.cpp then falls back to the lexical ranking), so clearing here is what
    // makes that fallback honest instead of accidental. Decode never calls this, so the last prefill
    // chunk's selection stays valid for the whole decode phase (the official `step` semantics).
    // ★ 2026-10-02 实测撤回（曾在此处 `p.kept.clear()`）：**不能每轮清空上一个选择**。
    //   池 4000 多针臂的日志直读：**一个请求只有 turn1 的 ~42 条 SELECT，turn2 一条都没有** ——
    //   带缓存前缀的那一轮走"后缀预填"，其 chunk 不产生打分轮（采集钩子只在 `Phase::Prefill` 武装），
    //   所以 turn2 的整段解码**靠继承 turn1 的选择**（内容打分挑出来的那批页）。
    //   一旦在这里清空，继承被抹掉 ⇒ `select_retrieval_pages` 退回词法 IDF ⇒ 词面不相交的问句必丢
    //   （实测把 turn2 由 6/6 打到 3/6）。"陈旧选择"在这里是**承重**的，不是需要清理的脏状态。
    //   契约不变：**只有 finish() 真的打了分才会覆盖 kept**；本轮没打分 ⇒ 保留上一次的选择。
    if (requested_capacity <= 0 || history_blocks <= 0 || block_tokens <= 0) { p.armed = false; return; }

    if (p.score == nullptr || p.capacity_blocks < requested_capacity) {
        // 只在**首个 chunk** 分配；prefill 不在 CUDA graph 捕获期（捕获期禁 cudaMalloc，坑表 §3.7）
        if (p.score != nullptr) { (void)cudaFree(p.score); p.score = nullptr; }
        if (p.block_tokens_dev != nullptr) { (void)cudaFree(p.block_tokens_dev); p.block_tokens_dev = nullptr; }
        const std::size_t bytes = static_cast<std::size_t>(requested_capacity) * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void**>(&p.score), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&p.block_tokens_dev),
                       static_cast<std::size_t>(requested_capacity) * sizeof(std::int32_t)) !=
                cudaSuccess) {
            kvmem_score_fail("cudaMalloc");
            return;
        }
        p.capacity_blocks = requested_capacity;
        p.score_host.assign(static_cast<std::size_t>(requested_capacity), 0.0F);
        p.tokens_host.assign(static_cast<std::size_t>(requested_capacity), 0);
        std::fprintf(stderr, "kvmem_score: ARMED capacity_blocks=%d layers_total=%d heads=%d kv_heads=%d "
                             "head_dim=%d\n", requested_capacity, layers_total, n_heads, n_kv_heads, head_dim);
    }

    p.n_blocks     = history_blocks;
    p.block_tokens = block_tokens;
    p.layers_total = layers_total;
    p.n_heads      = n_heads;
    p.n_kv_heads   = n_kv_heads;
    p.head_dim     = head_dim;
    p.armed        = true;
    p.scored_this_chunk = false;
    p.scored_last_chunk = false;

    // block_tokens：尾块（blocks_written-1）用真实填充，其余用名义块大小
    for (std::int32_t i = 0; i < history_blocks; ++i) {
        p.tokens_host[static_cast<std::size_t>(i)] =
            (i == history_blocks - 1) ? (tail_fill > 0 ? tail_fill : block_tokens) : block_tokens;
    }
    if (cudaMemcpyAsync(p.block_tokens_dev, p.tokens_host.data(),
                        static_cast<std::size_t>(history_blocks) * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        kvmem_score_fail("block_tokens H2D");
        return;
    }
    if (cudaMemsetAsync(p.score, 0, static_cast<std::size_t>(history_blocks) * sizeof(float), stream) !=
        cudaSuccess) {
        kvmem_score_fail("score memset");
        return;
    }
}

// ---- ② 每层一次：把该层的 query 行累加到 score 上 ----------------------------------------------
//
// 调用点必须在 `ops::rope(qn, ...)` **之前**：rope 会原地覆盖 qn，之后就没有内容帧的 query 了。
inline void kvmem_score_accumulate(std::int32_t fidx, const void* q, std::int32_t n_query_tokens,
                                   std::int32_t chunk_tokens, cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.armed || p.failed) { return; }
    if (n_query_tokens <= 0 || p.n_blocks <= 0) { return; }

    // ★ query span（2026-09-26 语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把本 chunk 的
    //   **最后 N 个 token** 当检索 query（官方 kvmem_set_query_span 的等价物）。MAXQ 的门限看
    //   **span 长度**、不看 chunk 长度。
    //
    //   ★ 默认值 2026-10-02 从 0（整 chunk）改成 64（缺陷 D-9）。依据是低池多针臂上的**单旋钮**
    //   剂量-响应 —— 池 4000（63 页，回捞预算 31 页）、同一支二进制、只改这一个 env、n=6：
    //       TAIL=256 -> turn1 6/6，turn2 **2/6**（第二个针静默丢；GATE RED）
    //       TAIL= 64 -> turn1 6/6，turn2 6/6
    //       TAIL= 32 -> turn1 6/6，turn2 6/6
    //   机理：末 chunk（1024 token）的"最后 256 行"里约 200 行是题面 filler + **上一轮的答案**
    //   （答案文本正好是第一个针页的强匹配），而 score 是对 query 行**求和**（契约
    //   `sum_b score[b] == n_query_tokens`，见 finish 的内置 oracle）⇒ 泛化页/旧目标页压过当前问句
    //   的目标页；池小（预算薄）时就直接表现为丢针。跨度收回问句长度即恢复。
    //   旧行为（整 chunk）现在只能显式给一个 >= chunk 的 N 拿到；但 chunk=1024 时整 chunk 会被
    //   MAXQ=256 的门限**直接 skip**（打分器一声不吭地什么都不做），所以"未设/0 ⇒ 64"更安全。
    const std::int32_t query_tail = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL", 64);
    const std::int32_t maxq = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_MAXQ", 256);
    std::int32_t span_begin  = 0;
    std::int32_t span_tokens = (chunk_tokens > 0 && chunk_tokens < n_query_tokens) ? chunk_tokens
                                                                                  : n_query_tokens;
    // ★ D-12：**绝对 query span（最后一轮 user）与本 chunk 求交**，只在它"问句大小"时才生效。
    //   · 交集为空 ⇒ 本 chunk 里没有问句（span 陈旧，或这块在 span 之外）⇒ **回落尾窗规则**
    //     （= 修前行为；绝不比"什么都不打分"差，见下面的报错行）。
    //   · 交集非空但**比 MAXQ 长** ⇒ 也**回落尾窗规则**，不截成"span 的尾 256"。
    //     依据（2026-10-02 实测）：题面把 42k 正文与问句放在**同一条** user 消息里时 span = 整条消息，
    //     截尾 256 等价于把 QUERY_TAIL 从 64 放宽到 256 —— 而那正是低池丢针的那一档（256⇒turn2 2/6）。
    //     "span 比 MAXQ 长"意味着**我们没能定位到一个问句**，此时老老实实用已验证的尾窗更安全。
    //   · 只有 span 本身 ≤ MAXQ（真正的"问句跨度"）才旁路尾窗规则 —— 那才是这条改动的用武之地。
    const bool abs_span_known = p.abs_query_begin >= 0 && p.abs_query_end > p.abs_query_begin &&
                                p.abs_chunk_begin >= 0 && chunk_tokens > 0 &&
                                !kvmem_score_env_flag("NINFER_TERNARY_KVMEM_SCORE_SPAN_OFF", false);
    bool span_applied = false;
    if (abs_span_known) {
        const std::int32_t local_begin = p.abs_query_begin - p.abs_chunk_begin;
        const std::int32_t local_end   = p.abs_query_end - p.abs_chunk_begin;
        const std::int32_t first       = local_begin > 0 ? local_begin : 0;
        const std::int32_t last        = local_end < chunk_tokens ? local_end : chunk_tokens;
        if (last > first && (last - first) <= maxq) {
            span_begin   = first;
            span_tokens  = last - first;
            span_applied = true;
        } else if (!p.logged_span_miss) {
            // LOCAL FIX (2026-10-02, found the hard way): an EMPTY intersection must NOT disable the
            // scorer for the rest of the request. The first version simply returned here -- and because
            // the span is a process-global published per request, a STALE span (turn 1's prompt-wide
            // span, left behind when the suffix-only prefill of a cached-prefix turn 2 never came
            // through the publishing path) made every turn-2 chunk "not contain the query" => nothing
            // scored => the ring fell back to the lexical ranking => the very lever this change exists
            // for was switched off silently. Measured cost: turn2 6/6 -> 3/6 at pool 4000, with ZERO
            // SELECT lines for turn 2 in the log. Falling back to the tail rule is the honest failure
            // mode: that IS the pre-D-12 behaviour, and it is never worse than scoring nothing.
            p.logged_span_miss = true;
            std::fprintf(stderr,
                         "kvmem_score: query span [%d,%d) not usable for chunk [%d,%d) (empty overlap "
                         "or longer than MAXQ=%d) -- falling back to the tail rule (QUERY_TAIL)\n",
                         p.abs_query_begin, p.abs_query_end, p.abs_chunk_begin,
                         p.abs_chunk_begin + chunk_tokens, maxq);
        }
    }
    if (!span_applied && query_tail > 0 && chunk_tokens > 0) {
        const std::int32_t tail = query_tail < chunk_tokens ? query_tail : chunk_tokens;
        span_begin  = chunk_tokens - tail;
        span_tokens = tail;
    }
    // LOCAL FIX (B05): a span longer than MAXQ used to be skipped outright -- the scorer did nothing
    // for that chunk, the ring fell back to the lexical ranking, and the symptom was a retrieval that
    // "kept nothing" (ledger B05). The span's official meaning is "the LAST user turn"
    // (frontend.cpp:1068-1097, matching kvmem-qw3's service-layer rule), so a long span means a long
    // user turn, not a broken one -- and the official service layer captures it WHOLE, having removed
    // its 512-token truncation and allocated query rows dynamically (see
    // 储备技术-可借鉴修复调研-本地.md §5.2). Our scoring kernel has a fixed query tile of MAXQ rows, so
    // the same coverage is reached by scoring the span in MAXQ-sized segments and SUMMING them: the
    // kernel accumulates with atomicAdd (kvmem_retrieve_launch.cu:243) under the contract
    // `sum_b score[b] == n_query_tokens` (kvmem_retrieve.h:117), so a sum over segments equals one
    // pass over the whole span -- and the finish-time oracle still holds, because it is told how many
    // tokens were actually scored (see query_span_tokens below).
    // Two bounds keep the cost sane and honour the D-12 measurement:
    //   * at most SCORE_SPAN_MAX tokens are scored per span (default 4096 = 16 segments), taken from
    //     the END of the span: the question sits at the end of a user turn, while the far end of a long
    //     message is where the filler D-12 blamed lives (measured: widening the query from 64 to 256
    //     tokens cost turn2 6/6 -> 2/6 at pool 4000);
    //   * SCORE_SPAN_SEGMENT=0 restores the skip, which is also this change's negative control.
    //     ⚠️ 这个开关**必须**走 kvmem_score_env_flag（不是 env_i32）：后者把 `=0` 读成"未设"⇒
    //     回落成 1 ⇒ 这条负控曾经**永不生效**、下面的 skip 分支不可达（B25，2026-10-07 实测：
    //     三条臂上正控与负控的读数逐行同值）。改回 env_i32 就会把这条负控重新废掉。
    const bool segment_span     = kvmem_score_env_flag("NINFER_TERNARY_KVMEM_SCORE_SPAN_SEGMENT", true);
    const std::int32_t span_max = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_SPAN_MAX", 4096);
    if (span_tokens > maxq) {
        if (!segment_span || maxq <= 0) {
            if (!p.logged_once) {
                p.logged_once = true;
                std::fprintf(stderr, "kvmem_score: skip long query span tokens=%d (> MAXQ=%d); chunk=%d begin=%d\n",
                             span_tokens, maxq, chunk_tokens, span_begin);
            }
            return;
        }
        const std::int32_t scored = span_max > 0 && span_tokens > span_max ? span_max : span_tokens;
        if (!p.logged_span_segment) {
            p.logged_span_segment = true;
            std::fprintf(stderr,
                         "kvmem_score: long query span %d token(s) > MAXQ=%d -> scoring its last %d "
                         "token(s) in %d segment(s); span=[%d,%d) chunk=[%d,%d)\n",
                         span_tokens, maxq, scored, (scored + maxq - 1) / maxq, span_begin,
                         span_begin + span_tokens, p.abs_chunk_begin, p.abs_chunk_begin + chunk_tokens);
        }
        span_begin += span_tokens - scored;   // keep the END of the span
        span_tokens = scored;
    }
    p.chunk_tokens      = chunk_tokens;   // live 平面的行数（= q_layer_stride）
    p.query_span_begin  = span_begin;
    p.query_span_tokens = span_tokens;    // 供 finish 的 oracle 自检用（正确口径 = 真正打分的 token 数）

    ops::MeanKIndex* index = ops::mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                                  p.capacity_blocks);
    if (index == nullptr || index->blocks_written() <= 0) { return; }
    if (fidx < 0 || fidx >= index->layers()) { return; }

    const Tensor& sums = index->layer_sums(fidx);
    if (sums.data == nullptr) { return; }

    KvMemRetrieveConfig cfg;
    cfg.n_layers          = 1;                      // 逐层调用
    cfg.n_layers_total    = p.layers_total;         // head_w = 1/(总层数×query 头数)
    cfg.n_heads           = p.n_heads;
    cfg.n_kv_heads        = p.n_kv_heads;
    cfg.head_dim          = p.head_dim;
    cfg.n_blocks          = p.n_blocks;
    cfg.q_layer_stride    = chunk_tokens;           // live plane 的 token 数（>= n_query_tokens）
    cfg.kbar_layer_stride = p.capacity_blocks;      // >= n_blocks
    cfg.budget_blocks     = 0;                      // 掩码只用在这里；探针不设带（Official 规则见下）
    cfg.sink_blocks       = 0;
    cfg.recent_blocks     = 0;
    cfg.mask_mode         = KvMemRetrieveMaskMode::Never;   // 探针要"全历史"的原始分数，不套带掩码
    cfg.dtype             = KvMemRetrieveDtype::BF16;

    // One call per MAXQ-sized segment -- a single call for the ordinary short span, which keeps the
    // common path bit-identical. `p.score` is accumulated by the kernel, so the segments sum exactly
    // as one pass over the whole span would have.
    for (std::int32_t offset = 0; offset < span_tokens; offset += maxq) {
        const std::int32_t segment = std::min(maxq, span_tokens - offset);
        cfg.n_query_tokens = segment;
        cfg.q_token_begin  = span_begin + offset;
        const cudaError_t status =
            ops::kvmem_retrieve_scores(cfg, p.score, q, static_cast<const float*>(sums.data),
                                       p.block_tokens_dev, stream);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "kvmem_score: launch failed layer=%d: %s\n", fidx,
                         cudaGetErrorString(status));
            kvmem_score_fail("launch");
            return;
        }
    }
    p.scored_this_chunk = true;
}

// ---- ③ chunk 收尾：D2H + 选块 + 一行证据日志 ---------------------------------------------------
//
// 放在与本文件同类的位置：prefill chunk 的尾巴、**任何捕获之外**（现有 kvmem_index 探针就在那儿
// 做 cudaMemcpyAsync + cudaStreamSynchronize，理由照抄它：一次同步换一个外部可核的数）。
//
// 免 oracle 自检：契约保证 `sum_b score[b] == n_query_tokens`（与带掩码与否无关）⇒ sum_score 就是
// 内置 oracle。对不上说明打分器没跑、跑了半截、或几何配错 —— 三种都得当场喊出来。
inline void kvmem_score_finish(const char* label, std::int32_t query_tokens,
                               cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.armed || p.failed || p.n_blocks <= 0) { return; }
    p.armed = false;
    if (!p.scored_this_chunk) { return; }   // 本 chunk 没打过（如长 ingest）⇒ 不打印全零假日志
    // ★ 出处登记（2026-09-26 语义档的准入闸）：这份分数描述块 [0, n_blocks)，且来自**最近一个
    //   chunk 的 query**。语义档只在 scored_last_chunk 为真时才用它装窗（宁可退压力档，也不装
    //   一个来路不明的窗）。
    p.scored_n_blocks   = p.n_blocks;
    p.scored_last_chunk = true;
    const std::int32_t q_used = p.query_span_tokens > 0 ? p.query_span_tokens : p.chunk_tokens;

    // 选块的带（只影响这一行日志；本探针不改可见集 ⇒ 不改变引擎行为）。
    // 默认取"小预算"以便**逼出选择**：budget 32768 / sink 4096 / recent 0（官方那台仪器用的就是这组）。
    const std::int32_t budget_tokens = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens   = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    const std::int32_t recent_tokens = std::getenv("NINFER_TERNARY_KVMEM_SCORE_RECENT") != nullptr
                                           ? kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_RECENT", 0)
                                           : 0;

    const std::size_t bytes = static_cast<std::size_t>(p.n_blocks) * sizeof(float);
    if (cudaMemcpyAsync(p.score_host.data(), p.score, bytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        kvmem_score_fail("score D2H");
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        kvmem_score_fail("sync");
        return;
    }

    double sum_score = 0.0;
    for (std::int32_t i = 0; i < p.n_blocks; ++i) { sum_score += static_cast<double>(p.score_host[i]); }

    KvMemSelectConfig scfg;
    scfg.total_blocks  = p.n_blocks;
    scfg.budget_blocks = budget_tokens / p.block_tokens;
    scfg.sink_blocks   = sink_tokens / p.block_tokens;
    scfg.recent_blocks = recent_tokens / p.block_tokens;
    // ---- ActKV policy (Sec. 5.2 LRFU + Sec. 8.1 protection), off by default ------------------
    //
    // The LRFU evidence for this round is handed over BEFORE the selection and committed AFTER it,
    // which is the whole point of the two-phase form: the returned `sel.kept` is exactly the set
    // allowed to grow, so a block the retrieval scored but the budget dropped does not accumulate
    // heat on a slot it never won. Both calls are no-ops when the switch is off.
    ops::actkv::LrfuState* lrfu = nullptr;
    if (kvmem_actkv_lrfu_enabled()) {
        lrfu = &kvmem_actkv_lrfu_state();
        lrfu->begin_round(p.score_host.data(), static_cast<std::size_t>(p.n_blocks));
    }
    std::vector<std::uint32_t> protected_pages;
    if (kvmem_actkv_protect_enabled()) {
        protected_pages = kvmem_actkv_protected_pages(p.block_tokens, 0);
    }
    KvMemActKvInputs actkv;
    actkv.lrfu            = lrfu;
    actkv.protected_pages = protected_pages.empty() ? nullptr : &protected_pages;
    actkv.trace           = kvmem_actkv_trace();
    const KvMemSelectResult sel = ops::kvmem_select_blocks(scfg, p.score_host.data(), 0, actkv);
    if (lrfu != nullptr) { lrfu->commit(sel.kept); }
    // LOCAL (2026-10-02, the caller step): publish the selection for the ring. Written AFTER the
    // selector returns and only when this chunk really scored, so a failed/skipped round leaves the
    // previous selection intact instead of emptying it.
    p.kept = sel.kept;

    // 选择结构的形状（这是判据的核心：非连续 ⇒ 真在选择）
    std::int32_t runs = 0;
    for (std::size_t i = 0; i < sel.kept.size(); ++i) {
        if (i == 0 || sel.kept[i] != sel.kept[i - 1] + 1) { ++runs; }
    }
    const std::int32_t skip = p.n_blocks - static_cast<std::int32_t>(sel.kept.size());
    const std::int32_t kept_first = sel.kept.empty() ? -1 : sel.kept.front();
    const std::int32_t kept_last  = sel.kept.empty() ? -1 : sel.kept.back();

    // ★ 把**选中的块号本身**打出来：没有它就无法核对"含针的块有没有被选中"（聚合量 runs/skip 做不到）。
    //   与官方实现同形（它也是逐行 `KVMEM_TRACE selected <ids...>`）。短 span 与切段后的长 span 都会
    //   出现（B05）；只有 `SCORE_SPAN_SEGMENT=0` 的旧跳过路径不再产出这一行 —— 那正是负控要看的东西。
    {
        std::fprintf(stderr, "kvmem_score: KEPT");
        for (std::size_t i = 0; i < sel.kept.size(); ++i) {
            std::fprintf(stderr, " %d", static_cast<int>(sel.kept[i]));
        }
        std::fprintf(stderr, "\n");
    }

    // ★ ActKV evidence (2026-10-16): one line per scoring round when either ActKV mechanism is on,
    //   so "the policy is off" and "the policy is on but had nothing to do" are distinguishable in
    //   the log. Reported AFTER the commit, so `rounds` counts this round. Absent from the log
    //   entirely when both switches are off -- an OFF arm must not gain a new line.
    if (kvmem_actkv_lrfu_enabled() || kvmem_actkv_protect_enabled()) {
        std::fprintf(stderr, "kvmem_score: ACTKV lrfu=%s protect=%s %s\n",
                     kvmem_actkv_lrfu_enabled() ? "on" : "off",
                     kvmem_actkv_protect_enabled() ? "on" : "off",
                     lrfu != nullptr
                         ? kvmem_actkv_lrfu_state().describe(static_cast<std::size_t>(p.n_blocks))
                               .c_str()
                         : "lrfu n/a");
    }

    std::fprintf(stderr,
                 "kvmem_score: SELECT label=%s n_blocks=%d budget_blocks=%d sink_blocks=%d "
                 "recent_blocks=%d kept=%zu runs=%d skip=%d window_tokens=%zu "
                 "sink_kept=%d recent_kept=%d scored_kept=%d candidates=%d kept_range=[%d,%d] "
                 "sum_score=%.3f query_tokens=%d span_mode=%s span_abs=[%d,%d) chunk_abs=%d "
                 "span_local=[%d,%d) oracle_ok=%d scale_floor=%.6f harvest_layers=%d/%d\n",
                 label, p.n_blocks, scfg.budget_blocks, scfg.sink_blocks, scfg.recent_blocks,
                 sel.kept.size(), runs, skip,
                 sel.kept.size() * static_cast<std::size_t>(p.block_tokens), sel.sink_kept,
                 sel.recent_kept, sel.scored_kept, sel.candidates, kept_first, kept_last, sum_score,
                 q_used, p.abs_query_begin >= 0 ? "abs" : "tail", p.abs_query_begin, p.abs_query_end,
                 p.abs_chunk_begin, p.query_span_begin, p.query_span_begin + p.query_span_tokens,
                 (q_used > 0 && sum_score > 0.5 * q_used && sum_score < 2.0 * q_used) ? 1 : 0,
                 static_cast<double>(ops::kvmem_retrieve_scale(p.head_dim)), p.harvest_layers,
                 p.harvest_layers_total);
}

// 逐块分数导出（给离线判据用：核对"含针的块"有没有被选中）。只在 switch 打开且本 chunk 打过时写。
inline void kvmem_score_dump_if_requested(const char* label) noexcept {
    if (!kvmem_score_enabled()) { return; }
    const char* path = std::getenv("NINFER_TERNARY_KVMEM_SCORE_DUMP");
    if (path == nullptr || path[0] == '\0') { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.n_blocks <= 0) { return; }
    FILE* f = std::fopen(path, "ab");
    if (f == nullptr) { return; }
    std::fprintf(f, "# %s n_blocks=%d\n", label, p.n_blocks);
    for (std::int32_t i = 0; i < p.n_blocks; ++i) {
        std::fprintf(f, "%d %.6f\n", i, static_cast<double>(p.score_host[static_cast<std::size_t>(i)]));
    }
    std::fclose(f);
}

} // namespace ninfer::ops::detail
