// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : src\kvmem_gc.cpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream. No Windows port was needed:
//              every POSIX call (pthread_*, sched_*, setpriority, <unistd.h>) is
//              already inside `#if defined(__linux__)` upstream, with an #else
//              that rejects a worker_cpu pin on other platforms. Verified by
//              compiling this file with MSVC on Windows (see exp/kvmem-port).
// ===========================================================================
#include "qw3/kvmem_gc.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace qw3 {
namespace {
double decay(uint64_t age, uint32_t half) {
    return std::exp2(-static_cast<double>(age) / std::max(1u, half));
}
uint64_t elapsed(uint64_t now, uint64_t then) { return now > then ? now - then : 0; }
}

KvMemGcDecision kvmem_gc_score(const KvMemGcFeatures &f, const KvMemGcConfig &c) {
    const double recency = decay(elapsed(f.step, std::max(f.birth, f.last_use)), c.half_life_steps);
    const double frequency = std::clamp(f.frequency / 4.0, 0.0, 1.0);
    const double semantic = f.score_valid ? std::clamp(f.semantic_heat, 0.0, 1.0) : 0.0;
    KvMemGcDecision d;
    if (f.anchor) { d.level = 4; d.reasons = GcAnchor; }
    else if (elapsed(f.step, std::max(f.birth, f.last_use)) < c.grace_steps ||
             f.frequency >= 2.0 || (f.score_valid && f.high_score)) {
        d.level = 3; d.reasons = 0;
        if (elapsed(f.step, std::max(f.birth, f.last_use)) < c.grace_steps) d.reasons |= GcRecent;
        if (f.frequency >= 2.0) d.reasons |= GcFrequent;
        if (f.score_valid && f.high_score) d.reasons |= GcHighScore;
    } else if (f.duplicate) { d.level = 0; d.reasons = GcDuplicate; }
    else if (f.superseded || (f.score_valid && f.low_streak >= c.low_observations)) {
        d.level = 1; d.reasons = f.superseded ? GcSuperseded : GcLowScore;
    }
    if (!f.score_valid) d.reasons |= GcUnknownScore;
    d.score = 20.0 * d.level + 19.0 * std::max({recency, frequency, semantic});
    return d;
}

KvMemGc::Lease::Lease(std::shared_ptr<Shared> s, const std::vector<uint32_t> &ids)
    : shared_(std::move(s)), ids_(ids) {
    // IDs are acquired while the caller holds a read view or another lease.
    for (uint32_t id : ids_) {
        if (id >= shared_->capacity) throw std::out_of_range("GC lease ID");
    }
    for (uint32_t id : ids_) {
        auto &e = shared_->entries[id];
        e.pins.fetch_add(1, std::memory_order_acq_rel);
        e.last_use.store(shared_->step.load(std::memory_order_relaxed), std::memory_order_release);
        e.touches.fetch_add(1, std::memory_order_release);
    }
}
KvMemGc::Lease::~Lease() {
    for (uint32_t id : ids_) shared_->entries[id].pins.fetch_sub(1, std::memory_order_release);
}

KvMemGc::KvMemGc(KvMemGcConfig c, uint32_t capacity, PrepareRelease prepare)
    : config_(c), shared_(std::make_shared<Shared>(capacity)), prepare_(std::move(prepare)),
      history_(capacity), dead_(capacity, 0) {
    if (!capacity || !c.max_live_blocks || !c.scan_blocks || !c.batch_blocks || !c.poll_ms ||
        !c.half_life_steps || !c.low_observations || !prepare_) {
        throw std::invalid_argument("invalid KVMem GC configuration");
    }
#if defined(__linux__)
    if (c.worker_cpu >= 0) {
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (c.worker_cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) != 0 ||
            !CPU_ISSET(c.worker_cpu, &allowed)) throw std::invalid_argument("GC CPU is outside the process affinity mask");
    }
#else
    if (c.worker_cpu >= 0) throw std::invalid_argument("GC CPU affinity requires Linux");
#endif
    view_ = std::make_shared<View>();
    worker_ = std::thread([this] { run(); });
}
KvMemGc::~KvMemGc() { stop(); }
void KvMemGc::stop() {
    stopping_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
}
uint64_t KvMemGc::generation(uint32_t id) const {
    return id < shared_->capacity ? shared_->entries[id].generation.load(std::memory_order_acquire) : 0;
}
void KvMemGc::register_block(uint32_t id, bool sealed) {
    if (id >= shared_->capacity) throw std::out_of_range("GC block capacity exceeded");
    auto &e = shared_->entries[id];
    e.flags.store(0, std::memory_order_release);
    e.generation.fetch_add(1, std::memory_order_acq_rel);
    e.birth.store(step(), std::memory_order_release);
    e.last_use.store(step(), std::memory_order_release);
    e.touches.fetch_add(1, std::memory_order_release);
    e.flags.store(Registered | Gpu | (sealed ? Sealed : 0u), std::memory_order_release);
    count_.store(id + 1, std::memory_order_release);
}
void KvMemGc::forget_tail(uint32_t first) {
    const uint32_t end = count_.load(std::memory_order_acquire);
    for (uint32_t id = first; id < end; ++id) shared_->entries[id].flags.store(0, std::memory_order_release);
    count_.store(std::min(first, end), std::memory_order_release);
}
void KvMemGc::update_flags(uint32_t id, uint32_t mask, uint32_t values) {
    if (id >= shared_->capacity) return;
    auto &flags = shared_->entries[id].flags;
    // The executor is the sole flag producer; the worker only reads flags.
    const uint32_t prior = flags.load(std::memory_order_relaxed);
    const uint32_t next = (prior & ~mask) | (values & mask);
    if (prior != next) {
        flags.store(next, std::memory_order_release);
        shared_->entries[id].touches.fetch_add(1, std::memory_order_release);
    }
}
void KvMemGc::used(uint32_t id) {
    if (id >= shared_->capacity) return;
    auto &e = shared_->entries[id];
    uint64_t prior = e.last_use.load(std::memory_order_relaxed);
    const uint64_t now = step();
    while (prior < now && !e.last_use.compare_exchange_weak(prior, now, std::memory_order_release, std::memory_order_relaxed)) {}
    e.touches.fetch_add(1, std::memory_order_release);
}
uint64_t KvMemGc::begin_step() { return shared_->step.fetch_add(1, std::memory_order_acq_rel) + 1; }
std::shared_ptr<const KvMemGc::View> KvMemGc::read_view() const { return std::atomic_load(&view_); }
bool KvMemGc::is_live(uint32_t id, const std::shared_ptr<const View> &v) const {
    if (!v || id >= v->dead_generation.size() || !v->dead_generation[id]) return true;
    if (id >= shared_->capacity) return false;
    const auto &e = shared_->entries[id];
    // An old reader may legitimately select a pending victim. Its new working
    // lease rescues it without forcing a GPU scorer retry.
    return v->dead_generation[id] != e.generation.load(std::memory_order_acquire) ||
           e.pins.load(std::memory_order_acquire) != 0;
}
std::shared_ptr<KvMemGc::Lease> KvMemGc::pin(const std::vector<uint32_t> &ids) {
    return std::make_shared<Lease>(shared_, ids);
}
void KvMemGc::submit_scores(std::shared_ptr<const ScoreEvent> event) { std::atomic_store(&scores_, std::move(event)); }
KvMemGc::Stats KvMemGc::stats() const {
    return {step(), examined_.load(), candidates_.load(), cancelled_.load(), retired_count_.load(),
            reclaimed_.load(), shadow_.load(), deferred_.load(), gaps_.load(), failures_.load(),
            shadow_reuses_.load(), ticks_.load(), worker_ns_.load(), max_tick_ns_.load()};
}
bool KvMemGc::readers_before(uint64_t epoch) {
    for (const auto &v : old_views_) if (v.first <= epoch && !v.second.expired()) return true;
    return false;
}
bool KvMemGc::protected_now(uint32_t id) const {
    const auto &e = shared_->entries[id];
    const uint32_t flags = e.flags.load(std::memory_order_acquire);
    const uint32_t count = count_.load(std::memory_order_acquire);
    const uint32_t recent = std::min(count, recent_.load(std::memory_order_acquire));
    return (flags & (Registered | Sealed | RawBacked | ValueBacked)) !=
               (Registered | Sealed | RawBacked | ValueBacked) ||
           (flags & (Gpu | Io | Mandatory | CpuCached | Anchor)) ||
           e.pins.load(std::memory_order_acquire) || id >= count - recent ||
           id >= indexed_.load(std::memory_order_acquire);
}
void KvMemGc::publish() {
    auto prior = std::atomic_load(&view_);
    old_views_.emplace_back(prior->epoch, prior);
    auto next = std::make_shared<View>();
    next->epoch = ++epoch_;
    next->dead_generation = dead_;
    std::atomic_store(&view_, std::shared_ptr<const View>(std::move(next)));
    dirty_view_ = false;
}

void KvMemGc::score_event(const ScoreEvent &s) {
    if (s.sequence <= score_sequence_) return;
    const bool gap = s.sequence != score_sequence_ + 1 || s.scorer_epoch != scorer_epoch_;
    if (gap) ++gaps_;
    scorer_epoch_ = s.scorer_epoch;
    score_sequence_ = s.sequence;
    std::vector<uint32_t> order;
    const size_t n = std::min({s.scores.size(), s.generations.size(), s.eligible.size(), s.selected.size(), history_.size()});
    for (uint32_t id = 0; id < n; ++id) {
        if (s.eligible[id] && std::isfinite(s.scores[id]) && s.generations[id] == generation(id)) order.push_back(id);
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return s.scores[a] < s.scores[b]; });
    const bool distinct = order.size() > 1 && s.scores[order.front()] != s.scores[order.back()];
    size_t selected = 0;
    for (uint32_t id : order) if (id < s.selected.size() && s.selected[id]) ++selected;
    const double cutoff = 1.0 - static_cast<double>(selected) / std::max<size_t>(1, order.size());
    for (size_t i = 0; i < order.size();) {
        size_t end = i + 1;
        while (end < order.size() && s.scores[order[end]] == s.scores[order[i]]) ++end;
        const double q = (0.5 * (i + end - 1)) / std::max<size_t>(1, order.size() - 1);
        for (size_t k = i; k < end; ++k) {
            const uint32_t id = order[k];
            refresh_history(id, step());
            auto &h = history_[id];
            if (h.generation != s.generations[id]) continue;
            const bool fresh = h.score_valid && elapsed(s.step, h.score_step) <= config_.half_life_steps;
            const bool new_step = h.score_sequence == 0 || s.step > h.score_step;
            const bool near = selected > 0 && q >= cutoff - 0.05;
            const bool hit = id < s.selected.size() && s.selected[id];
            if (gap || !fresh || h.score_sequence + 1 != s.sequence) h.low_streak = 0;
            if (distinct && q < 0.2 && !near && !hit) {
                if (new_step) ++h.low_streak;
            } else h.low_streak = 0;
            if (gap || !fresh) { h.q_count = 0; h.q_cursor = 0; }
            if (distinct) {
                if (new_step || !h.q_count) {
                    h.q_cursor = h.q_count ? (h.q_cursor + 1) % h.q.size() : 0;
                    h.q_count = std::min<uint32_t>(h.q_count + 1, h.q.size());
                    h.q[h.q_cursor] = q; h.q_step[h.q_cursor] = s.step;
                } else h.q[h.q_cursor] = std::max(h.q[h.q_cursor], q);
            }
            if (distinct && (q >= 0.8 || near)) { h.high = true; h.high_step = s.step; }
            h.score_valid = distinct;
            h.score_step = s.step;
            h.score_sequence = s.sequence;
        }
        i = end;
    }
}

void KvMemGc::refresh_history(uint32_t id, uint64_t now) {
    auto &e = shared_->entries[id]; auto &h = history_[id];
    const uint32_t flags = e.flags.load(std::memory_order_acquire);
    const uint64_t gen = generation(id);
    if (!(flags & Registered) || gen != h.generation) {
        if (h.counted && !h.dead && !h.shadow && live_count_) --live_count_;
        h = History{}; h.generation = gen;
        h.birth = e.birth.load(std::memory_order_acquire); h.last_decay = now;
        h.counted = (flags & Registered) != 0;
        if (h.counted) ++live_count_;
        if (dead_[id]) { dead_[id] = 0; dirty_view_ = true; }
    }
    const uint64_t use = e.last_use.load(std::memory_order_acquire);
    if (h.shadow && e.touches.load(std::memory_order_acquire) != h.candidate_touches) {
        h.shadow = false; h.candidate_step = 0; ++live_count_;
        if (use > h.last_counted_use) ++shadow_reuses_;
    }
    h.frequency *= decay(elapsed(now, h.last_decay), config_.half_life_steps);
    h.last_decay = now;
    if (use > h.last_counted_use) { h.frequency += 1.0; h.last_counted_use = use; }
}

KvMemGcDecision KvMemGc::decision(uint32_t id, uint64_t now) const {
    const auto &e = shared_->entries[id]; const auto &h = history_[id];
    const uint32_t flags = e.flags.load(std::memory_order_acquire);
    KvMemGcFeatures f;
    f.birth = h.birth; f.last_use = e.last_use.load(std::memory_order_acquire);
    f.step = now; f.frequency = h.frequency * decay(elapsed(now, h.last_decay), config_.half_life_steps);
    f.score_valid = h.score_valid && elapsed(now, h.score_step) <= config_.half_life_steps;
    f.high_score = h.high && elapsed(now, h.high_step) <= config_.half_life_steps;
    f.low_streak = h.low_streak;
    for (uint32_t i = 0; i < h.q_count; ++i) {
        if (elapsed(now, h.q_step[i]) <= config_.half_life_steps)
            f.semantic_heat = std::max(f.semantic_heat, h.q[i] * decay(elapsed(now, h.q_step[i]), config_.half_life_steps));
    }
    f.anchor = flags & Anchor; f.duplicate = flags & Duplicate; f.superseded = flags & Superseded;
    return kvmem_gc_score(f, config_);
}

void KvMemGc::tick() {
    const uint64_t now = step();
    if (active_step_ != now) {
        if (auto active = std::atomic_load(&active_)) for (uint32_t id : active->ids()) used(id);
        active_step_ = now;
    }
    // Observe already-delivered fresh evidence before committing any release.
    if (auto s = std::atomic_exchange(&scores_, std::shared_ptr<const ScoreEvent>{})) score_event(*s);
    old_views_.erase(std::remove_if(old_views_.begin(), old_views_.end(),
        [](const auto &v) { return v.second.expired(); }), old_views_.end());
    // Reclaim only after every earlier directory reader and resource lease is
    // gone. No polling of a device fence and no foreground participation.
    const size_t retire_scan = std::min<size_t>(retired_.size(), config_.scan_blocks);
    for (size_t n = 0; n < retire_scan; ++n) {
        Retired item = std::move(retired_.front()); retired_.pop_front();
        auto *it = &item;
        auto &e = shared_->entries[it->id];
        auto &h = history_[it->id];
        if (generation(it->id) != it->generation) {
            // A rewind owns removal of the old allocations. Never release the
            // new lineage using a stale logical block ID.
            continue;
        }
        if (protected_now(it->id) || e.touches.load(std::memory_order_acquire) != it->touches ||
            decision(it->id, now).level != it->level) {
            dead_[it->id] = 0; h.dead = false; h.candidate_step = 0;
            ++live_count_; ++cancelled_; dirty_view_ = true;
            continue;
        }
        if (readers_before(it->before_epoch)) { ++deferred_; retired_.push_back(std::move(item)); continue; }
        // An old reader can acquire a working lease (or publish fresh scores)
        // and drop its view between the first protection check and the weak
        // reader check. Recheck AFTER quiescence before touching an allocation.
        // New readers already see the tombstone and cannot acquire this ID.
        if (protected_now(it->id) || e.touches.load(std::memory_order_acquire) != it->touches ||
            std::atomic_load(&scores_)) {
            retired_.push_back(std::move(item));
            continue;
        }
        if (it->release()) {
            ++reclaimed_;
            if (config_.trace) std::fprintf(stderr, "[kvmem-gc] action=reclaim step=%llu block=%u generation=%llu\n",
                (unsigned long long)now, it->id, (unsigned long long)it->generation);
        } else retired_.push_back(std::move(item));
    }
    if (cursor_ == 0) scan_limit_ = std::max(scan_limit_, count_.load(std::memory_order_acquire));
    const uint32_t scan = std::min(config_.scan_blocks, scan_limit_ - cursor_);
    for (uint32_t n = 0; n < scan; ++n) {
        const uint32_t id = cursor_++;
        auto &e = shared_->entries[id]; auto &h = history_[id];
        const uint32_t flags = e.flags.load(std::memory_order_acquire);
        refresh_history(id, now);
        if (!(flags & Registered)) continue;
        ++examined_;
        if (h.dead || h.shadow || protected_now(id)) {
            if (!h.shadow) h.candidate_step = 0;
            continue;
        }
        auto d = decision(id, now);
        const uint64_t touch = e.touches.load(std::memory_order_acquire);
        if (d.level >= 3 || live_count_ <= config_.max_live_blocks) {
            if (h.candidate_step) ++cancelled_;
            h.candidate_step = 0; continue;
        }
        if (!h.candidate_step || h.candidate_touches != touch || h.candidate_level != d.level) {
            h.candidate_step = now; h.candidate_touches = touch; h.candidate_level = d.level;
            ++candidates_; continue;
        }
        if (elapsed(now, h.candidate_step) >= config_.candidate_steps) ready_.push_back({d.score, id, 0});
    }
    // Finish a bounded incremental sweep before ranking; a low ID must not
    // outrank a colder block simply because it was scanned first.
    if (cursor_ != scan_limit_) { if (dirty_view_) publish(); return; }
    cursor_ = 0; scan_limit_ = count_.load(std::memory_order_acquire);
    for (auto &item : ready_) {
        item.score = decision(item.id, now).score;
        item.last_use = shared_->entries[item.id].last_use.load(std::memory_order_acquire);
    }
    std::sort(ready_.begin(), ready_.end(), [](const Candidate &a, const Candidate &b) {
        if (a.score != b.score) return a.score < b.score;
        if (a.last_use != b.last_use) return a.last_use < b.last_use;
        return a.id < b.id;
    });
    uint32_t batch = 0;
    for (const auto &item : ready_) {
        if (batch >= config_.batch_blocks || live_count_ <= config_.max_live_blocks) break;
        const uint32_t id = item.id; auto &h = history_[id]; auto &e = shared_->entries[id];
        if (generation(id) != h.generation || h.dead || h.shadow || !h.candidate_step ||
            elapsed(now, h.candidate_step) < config_.candidate_steps || protected_now(id) ||
            decision(id, now).level != h.candidate_level || h.candidate_level >= 3) continue;
        if (e.pins.load(std::memory_order_acquire) || e.touches.load(std::memory_order_acquire) != h.candidate_touches) continue;
        const auto d = decision(id, now);
        auto release = config_.shadow ? std::function<bool()>{} : prepare_(id);
        if (!config_.shadow && !release) continue;
        if (config_.trace) std::fprintf(stderr,
            "[kvmem-gc] action=%s step=%llu block=%u generation=%llu level=%u score=%.6f reasons=%u\n",
            config_.shadow ? "shadow" : "retire", (unsigned long long)now, id,
            (unsigned long long)h.generation, d.level, d.score, d.reasons);
        if (config_.shadow) { ++shadow_; h.shadow = true; --live_count_; ++batch; continue; }
        retired_.push_back({id, h.generation, epoch_, h.candidate_touches, h.candidate_level, std::move(release)});
        h.dead = true; dead_[id] = h.generation; --live_count_; ++retired_count_; ++batch;
        dirty_view_ = true;
    }
    ready_.clear();
    if (dirty_view_) publish();
}
void KvMemGc::run() {
#if defined(__linux__)
    (void)setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
    if (config_.worker_cpu >= 0) {
        cpu_set_t cpus; CPU_ZERO(&cpus);
        CPU_SET(config_.worker_cpu, &cpus);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0) {
            ++failures_; std::fprintf(stderr, "[kvmem-gc] worker stopped: CPU affinity failed\n"); return;
        }
    }
#endif
    while (!stopping_.load(std::memory_order_acquire)) {
        const auto start = std::chrono::steady_clock::now();
        try { tick(); }
        catch (const std::exception &e) {
            ++failures_; std::fprintf(stderr, "[kvmem-gc] worker stopped: %s\n", e.what()); return;
        } catch (...) { ++failures_; std::fprintf(stderr, "[kvmem-gc] worker stopped: unknown error\n"); return; }
        const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        ++ticks_; worker_ns_ += ns;
        if (ns > max_tick_ns_.load()) max_tick_ns_.store(ns);
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.poll_ms));
    }
    if (config_.trace) std::fprintf(stderr,
        "[kvmem-gc] action=stop retired=%llu reclaimed=%llu shadow=%llu shadow_reuses=%llu worker_ms=%.3f max_tick_ms=%.3f\n",
        (unsigned long long)retired_count_.load(), (unsigned long long)reclaimed_.load(),
        (unsigned long long)shadow_.load(), (unsigned long long)shadow_reuses_.load(),
        worker_ns_.load() / 1e6, max_tick_ns_.load() / 1e6);
}
} // namespace qw3
