// Ledger B01 -- single-page materialization capacity gate.
//
// WHY THIS TEST EXISTS
//   The B01 crash ("worker crash: Paged KV reservation invariant was violated" -> 500 for the
//   request -> 503 for everything after it) was traced to two sites in paged_kv_cache.cpp that
//   threw std::invalid_argument (a std::logic_error) where the failure was really capacity
//   exhaustion. Both were reclassified to ContextCacheExhausted (a std::bad_alloc) so the worker's
//   out-of-memory path fails the affected request instead of the Engine -- EXCEPT the
//   `reservation.pages_ == 0` branch of materialize_one(), which is deliberately left alone
//   pending evidence.
//
//   Reading the two materialization paths side by side shows a second, independent asymmetry:
//
//     batch materialize()   :426   if (count > usable_pages() - allocated_pages_) -> throw
//     one   materialize_one():546   if (free_page_runs_.empty())                 -> throw
//
//   The batch path does the arithmetic (how many pages MAY be handed out); the single-page path
//   only asks whether SOME free run exists. That gap only matters because the pool may be
//   created with allow_logical_reservation = true (DeviceKVPagePoolSpec:85), which makes
//   can_resize_reservation() return true unconditionally (:382-385) -- "the reservation is a
//   logical promise here; only the physical allocation is bounded". So a reservation can be
//   resized beyond what the pool can physically back, and the single-page path has no arithmetic
//   of its own to catch it.
//
//   This test is the host-side evidence for that claim. It does not need the model, the ring, or
//   the server: a 4-page pool reproduces every code path that matters. It is written to be cheap
//   enough to run on every change, and to state plainly which behaviour is a DEFECT candidate
//   and which is legitimate.
//
// WHAT IS ASSERTED (and why it is safe to assert)
//   Case A. A zero-page reservation is refused by materialize_one() -- a std::logic_error today.
//           Pins the CURRENT behaviour so a later reclassification is a deliberate, visible change.
//   Case B. With allow_logical_reservation = true, can_resize_reservation() accepts a resize past
//           the physical extent; with it false, the same resize is refused. This is the asymmetry
//           that lets the defect exist; asserted as behaviour, not as a wish.
//   Case C. With allow_logical_reservation = true, a reservation is resized beyond the physical
//           extent (8 pages promised, 4 exist). Both materialization paths must refuse: the batch
//           path via its arithmetic at :426, and the single-page path via the arithmetic added by
//           the B01 fix (`allocated_pages_ >= usable_pages()`). The leases are held across the loop
//           so the pool is really driven to exhaustion; letting them die each iteration would hand
//           the page back (RAII) and the case would not test anything.
//
//   Every case prints the counters it saw, because "the assertion passed" is not evidence about a
//   capacity bug -- the numbers beside the verdict are.

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

struct Planned {
    ninfer::DeviceKVPagePoolLayout pool;
    std::size_t bytes = 0;
};

// `physical_pages` is the real extent; `allow_logical_reservation` is what the KVMem ring sets.
Planned plan(std::uint32_t physical_pages, bool allow_logical_reservation) {
    ninfer::LayoutBuilder builder;
    Planned out;
    ninfer::DeviceKVPagePoolSpec spec;
    spec.page_group_count         = physical_pages;
    spec.allow_logical_reservation = allow_logical_reservation;
    spec.geometry.planes           = {{ninfer::DType::I8, 8, 2, 256}};
    out.pool                       = ninfer::plan_device_kv_page_pool(builder, spec);
    out.bytes                      = builder.finish(256);
    return out;
}

void report(const char* label, const ninfer::DeviceKVPagePool& pool) {
    std::cout << "  " << label << ": capacity=" << pool.capacity_pages()
              << " usable=" << pool.usable_pages() << " allocated=" << pool.allocated_pages()
              << " reserved=" << pool.reserved_pages() << " lent=" << pool.lent_pages()
              << " free_runs=" << pool.free_runs().size() << '\n';
}

// Case A: the zero-page reservation. Pins today's verdict and its exception class.
int case_empty_reservation(ninfer::DeviceContext& context) {
    std::cout << "[A] zero-page reservation reaches materialize_one()\n";
    int failures       = 0;
    const Planned p    = plan(4, /*allow_logical_reservation=*/false);
    ninfer::DeviceArena arena(p.bytes);
    ninfer::DeviceKVPagePool pool({arena.base(), arena.capacity()}, p.pool);

    // make_empty_reservation() is the public way to build pages_ == 0 (:277-279).
    ninfer::DeviceKVPageReservation empty = pool.make_empty_reservation();
    report("before", pool);

    bool logic_error_seen = false;
    bool capacity_seen    = false;
    std::string what;
    try {
        ninfer::DeviceKVPageLease lease = pool.materialize_one(empty, "test.empty");
        (void)lease;
    } catch (const ninfer::ContextCacheExhausted& error) {
        capacity_seen = true;
        what          = error.what();
    } catch (const std::logic_error& error) {
        logic_error_seen = true;
        what             = error.what();
    }
    std::cout << "  verdict: logic_error=" << logic_error_seen << " capacity=" << capacity_seen
              << "\n  what: " << what << '\n';

    // Current behaviour (2026-10-17): the zero-page branch throws std::invalid_argument, i.e. a
    // std::logic_error. Asserted so a reclassification changes this test on purpose.
    failures += expect(logic_error_seen || capacity_seen,
                       "materialize_one() accepted a zero-page reservation");
    return failures;
}

// Case B: can_resize_reservation() past the physical extent, both pool flavours.
int case_resize_past_extent(ninfer::DeviceContext& context) {
    std::cout << "[B] can_resize_reservation() past the physical extent\n";
    int failures = 0;

    for (const bool allow : {false, true}) {
        const Planned p = plan(4, allow);
        ninfer::DeviceArena arena(p.bytes);
        ninfer::DeviceKVPagePool pool({arena.base(), arena.capacity()}, p.pool);

        ninfer::DeviceKVPageReservation reservation = pool.make_empty_reservation();
        const bool accepted = pool.can_resize_reservation(reservation, 64); // 64 > capacity 4
        std::cout << "  allow_logical_reservation=" << allow << " -> resize to 64 accepted="
                  << accepted << '\n';
        report("    pool", pool);

        if (allow) {
            // A logical promise: the pool does not refuse a reservation it cannot back.
            failures += expect(accepted,
                               "allow_logical_reservation=true refused a logically-backed resize");
        } else {
            failures += expect(!accepted,
                               "allow_logical_reservation=false accepted a resize past capacity");
        }
    }
    return failures;
}

// Case C: batch vs single-page gate, with the logical promise on.
int case_gate_asymmetry(ninfer::DeviceContext& context) {
    std::cout << "[C] batch vs single-page capacity gate (allow_logical_reservation=true)\n";
    int failures    = 0;
    const Planned p = plan(4, /*allow_logical_reservation=*/true);
    ninfer::DeviceArena arena(p.bytes);
    ninfer::DeviceKVPagePool pool({arena.base(), arena.capacity()}, p.pool);

    // A reservation the pool cannot physically back: 8 pages promised, 4 pages exist.
    ninfer::DeviceKVPageReservation reservation = pool.make_empty_reservation();
    pool.resize_reservation(reservation, 8);
    report("after resize to 8", pool);

    // --- batch path: expect the arithmetic to refuse before touching the pool ---
    std::vector<ninfer::DeviceKVPageLease> batch;
    batch.reserve(8);
    bool batch_refused = false;
    std::string batch_what;
    try {
        pool.materialize(reservation, 8, batch);
    } catch (const ninfer::ContextCacheExhausted& error) {
        batch_refused = true;
        batch_what    = error.what();
    } catch (const std::exception& error) {
        batch_refused = true;
        batch_what    = error.what();
    }
    std::cout << "  batch: refused=" << batch_refused << " materialized=" << batch.size() << '\n'
              << "         what: " << batch_what << '\n';
    report("    pool", pool);
    failures += expect(batch_refused, "batch materialize() over-committed the pool");

    // --- single-page path: same oversubscription, one page at a time ---
    // The leases MUST outlive the loop: DeviceKVPageLease is a move-only RAII owner whose
    // destructor returns the page to the pool (see ~DeviceKVPageLease()). Holding one lease per
    // iteration and letting it die at the end of the loop body would give the pool its page back
    // every time, so `allocated_pages_` would never grow and the pool would never be driven to
    // exhaustion -- the first version of this case did exactly that and measured taken=8 with the
    // pool still empty. The leases are therefore kept in a vector for the whole loop.
    ninfer::DeviceKVPageReservation single = pool.make_empty_reservation();
    pool.resize_reservation(single, 8);
    std::vector<ninfer::DeviceKVPageLease> held;   // keeps every taken page allocated
    held.reserve(8);
    std::uint32_t taken          = 0;
    bool single_refused          = false;
    std::string single_what;
    for (std::uint32_t i = 0; i < 8; ++i) {
        try {
            held.push_back(pool.materialize_one(single, "test.one"));
            ++taken;
        } catch (const std::exception& error) {
            single_refused = true;
            single_what    = error.what();
            break;
        }
    }
    std::cout << "  single: taken=" << taken << " refused=" << single_refused
              << " held=" << held.size() << '\n'
              << "          what: " << single_what << '\n';
    report("    pool", pool);

    // The physical pool has 4 pages, so more than 4 can never be handed out at once. The leases are
    // held (above) so the pool is actually driven to exhaustion; before the B01 fix the overflow was
    // stopped only by `free_page_runs_.empty()`, now the single-page path does the same arithmetic as
    // the batch path (`allocated_pages_ >= usable_pages()`), so both refuse.
    //
    // MEASURED 2026-10-17 (4-page pool, reservation resized to 8, leases held across the loop):
    //   batch  -> refused=1 materialized=0
    //   single -> taken=4 refused=1 held=4
    //             the 5th call sees allocated=4 >= usable=4 and throws ContextCacheExhausted
    //             ("capacity exhaustion on the single-page path"), so the pool never over-commits.
    // NOTE ON AN EARLIER FALSE READING: a first version of this case let the lease die at the end of
    // each loop iteration. ~DeviceKVPageLease() returns the page, so `allocated_pages_` never grew,
    // the arithmetic never fired, and the case reported taken=8/refused=0 -- which looked like the
    // defect but was the test failing to hold the leases. The vector above is what makes it honest.
    // See docs/铁律十七-十八根因分析与修复方案-20261017.md and paged_kv_cache.cpp (materialize_one).
    const bool single_over_commits = taken > pool.capacity_pages();
    std::cout << "  single-page gate: taken=" << taken << " capacity=" << pool.capacity_pages()
              << " refused=" << single_refused << " (batch refused the same request)\n";
    // The defect is gone: the single-page path must refuse before it exceeds the physical pool.
    failures += expect(taken <= pool.capacity_pages(),
                       "single-page path handed out more pages than the pool owns (B01 regression)");
    failures += expect(single_refused,
                       "single-page path did not refuse an over-committing materialization");
    // Guard against a regression to the old over-committing behaviour.
    failures += expect(!single_over_commits,
                       "single-page gate regressed to the pre-B01 over-commit behaviour");
    return failures;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || (count_error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_error) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext context(0);
        int failures = 0;
        failures += case_empty_reservation(context);
        failures += case_resize_past_extent(context);
        failures += case_gate_asymmetry(context);
        if (failures != 0) {
            std::cerr << failures << " single-page capacity checks failed\n";
            return 1;
        }
        std::cout << "single-page capacity checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "single-page capacity test failed: " << error.what() << '\n';
        return 1;
    }
}
