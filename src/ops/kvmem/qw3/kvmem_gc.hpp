// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : include\qw3\kvmem_gc.hpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream; only this attribution block was prepended.
// ===========================================================================
#pragma once

// CPU-only rule policy and retirement worker. No DeviceBackend dependency.
#include <atomic>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace qw3 {

struct KvMemGcConfig {
    uint32_t max_live_blocks = 0; // zero disables GC
    uint32_t half_life_steps = 8;
    uint32_t grace_steps = 4;
    uint32_t candidate_steps = 2;
    uint32_t low_observations = 3;
    uint32_t scan_blocks = 256;
    uint32_t batch_blocks = 64;
    uint32_t poll_ms = 25;
    int worker_cpu = -1;
    bool shadow = false;
    bool trace = false;
};

enum KvMemGcReason : uint32_t {
    GcOrdinary = 1, GcDuplicate = 2, GcSuperseded = 4,
    GcLowScore = 8, GcRecent = 16, GcFrequent = 32,
    GcHighScore = 64, GcAnchor = 128, GcUnknownScore = 256,
    GcPinned = 512, GcNotDurable = 1024
};

struct KvMemGcFeatures {
    uint64_t birth = 0, last_use = 0, step = 0;
    double frequency = 0.0, semantic_heat = 0.0;
    uint32_t low_streak = 0;
    bool score_valid = false, high_score = false;
    bool duplicate = false, superseded = false, anchor = false;
};

struct KvMemGcDecision {
    uint32_t level = 2, reasons = GcOrdinary;
    double score = 40.0;
};
KvMemGcDecision kvmem_gc_score(const KvMemGcFeatures &, const KvMemGcConfig &);

class KvMemGc {
public:
    enum Flags : uint32_t {
        Registered = 1, Sealed = 2, Gpu = 4, Io = 8,
        ValueBacked = 16, RawBacked = 32, Anchor = 64,
        Duplicate = 128, Superseded = 256, Mandatory = 512, CpuCached = 1024
    };
    struct View {
        uint64_t epoch = 0;
        // Stable logical IDs; a rewind/reappend gets a new generation.
        std::vector<uint64_t> dead_generation;
    };
    struct State {
        std::atomic<uint64_t> generation{0}, birth{0}, last_use{0}, touches{0};
        std::atomic<uint32_t> flags{0}, pins{0};
    };
    struct Shared {
        explicit Shared(uint32_t n) : capacity(n), entries(new State[n]) {}
        uint32_t capacity;
        std::unique_ptr<State[]> entries;
        std::atomic<uint64_t> step{0};
    };
    class Lease {
    public:
        Lease(std::shared_ptr<Shared>, const std::vector<uint32_t> &);
        ~Lease();
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        const std::vector<uint32_t> &ids() const { return ids_; }
    private:
        std::shared_ptr<Shared> shared_;
        std::vector<uint32_t> ids_;
    };
    struct ScoreEvent {
        uint64_t sequence = 0, step = 0, scorer_epoch = 0;
        std::vector<double> scores;
        std::vector<uint64_t> generations;
        std::vector<uint8_t> eligible, selected;
    };
    struct Stats {
        uint64_t step = 0, examined = 0, candidates = 0, cancelled = 0;
        uint64_t retired = 0, reclaimed = 0, shadow_candidates = 0;
        uint64_t deferred_readers = 0, score_gaps = 0, failures = 0;
        uint64_t shadow_reuses = 0, worker_ticks = 0, worker_ns = 0, max_tick_ns = 0;
    };
    // Called only by the CPU worker. Prepare captures allocation identities;
    // the returned nonblocking callback reclaims those exact allocations and
    // returns false when the allocator is busy (retry on another worker tick).
    using PrepareRelease = std::function<std::function<bool()>(uint32_t)>;

    KvMemGc(KvMemGcConfig, uint32_t capacity, PrepareRelease);
    ~KvMemGc();
    KvMemGc(const KvMemGc &) = delete;
    KvMemGc &operator=(const KvMemGc &) = delete;
    void stop();
    void register_block(uint32_t id, bool sealed);
    void forget_tail(uint32_t first);
    void set_recent_blocks(uint32_t n) { recent_.store(n, std::memory_order_release); }
    void set_indexed_blocks(uint32_t n) { indexed_.store(n, std::memory_order_release); }
    void update_flags(uint32_t id, uint32_t mask, uint32_t values);
    void used(uint32_t id);
    uint64_t begin_step();
    uint64_t step() const { return shared_->step.load(std::memory_order_relaxed); }
    uint64_t generation(uint32_t id) const;
    std::shared_ptr<const View> read_view() const;
    bool is_live(uint32_t id, const std::shared_ptr<const View> &) const;
    std::shared_ptr<Lease> pin(const std::vector<uint32_t> &ids);
    void submit_scores(std::shared_ptr<const ScoreEvent>);
    void set_active(std::shared_ptr<Lease> lease) { std::atomic_store(&active_, std::move(lease)); }
    Stats stats() const;
    const KvMemGcConfig &config() const { return config_; }

private:
    struct History {
        uint64_t generation = 0, birth = 0, last_counted_use = 0;
        uint64_t last_decay = 0, score_step = 0, score_sequence = 0, high_step = 0;
        uint64_t candidate_step = 0, candidate_touches = 0;
        uint32_t candidate_level = 0, low_streak = 0;
        double frequency = 0;
        std::array<double, 3> q{};
        std::array<uint64_t, 3> q_step{};
        uint32_t q_count = 0, q_cursor = 0;
        bool score_valid = false, high = false, counted = false, dead = false, shadow = false;
    };
    struct Retired {
        uint32_t id;
        uint64_t generation, before_epoch, touches;
        uint32_t level;
        std::function<bool()> release;
    };
    struct Candidate { double score; uint32_t id; uint64_t last_use; };
    void run();
    void tick();
    void score_event(const ScoreEvent &);
    void refresh_history(uint32_t id, uint64_t now);
    KvMemGcDecision decision(uint32_t id, uint64_t now) const;
    void publish();
    bool readers_before(uint64_t epoch);
    bool protected_now(uint32_t id) const;
    KvMemGcConfig config_;
    std::shared_ptr<Shared> shared_;
    PrepareRelease prepare_;
    std::vector<History> history_;
    std::deque<Retired> retired_;
    std::shared_ptr<const View> view_;
    std::vector<std::pair<uint64_t, std::weak_ptr<const View>>> old_views_;
    std::shared_ptr<const ScoreEvent> scores_;
    std::shared_ptr<Lease> active_;
    std::vector<uint64_t> dead_;
    std::vector<Candidate> ready_;
    std::atomic<uint32_t> count_{0};
    std::atomic<uint32_t> recent_{0};
    std::atomic<uint32_t> indexed_{UINT32_MAX};
    std::atomic<bool> stopping_{false};
    uint32_t cursor_ = 0, live_count_ = 0, scan_limit_ = 0;
    uint64_t epoch_ = 0, score_sequence_ = 0, scorer_epoch_ = 0;
    uint64_t active_step_ = 0;
    bool dirty_view_ = false;
    std::thread worker_;
    std::atomic<uint64_t> examined_{0}, candidates_{0}, cancelled_{0};
    std::atomic<uint64_t> retired_count_{0}, reclaimed_{0}, shadow_{0};
    std::atomic<uint64_t> deferred_{0}, gaps_{0}, failures_{0};
    std::atomic<uint64_t> shadow_reuses_{0}, ticks_{0}, worker_ns_{0}, max_tick_ns_{0};
};
} // namespace qw3
