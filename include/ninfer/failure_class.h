// Failure classification -- the ONE place that decides what a thrown exception MEANS.
//
// WHY THIS FILE EXISTS (audit 2026-10-04, findings F1/F2):
//   A failure's meaning used to be carried implicitly by its C++ type and then re-decided at five
//   independent places: the worker's two catch blocks (runtime/engine/engine_core.h), the
//   recover_invariant_failures_ gate, the transaction wrapper, the HTTP mapper
//   (serve/generation_service.cpp) and the metric counter. Nothing owned the answer, so the same
//   physical event -- a full KV store -- came out as HTTP 429 on one path (admission/materialization
//   wraps it as RequestError(Overloaded)) and as HTTP 500 internal_error on another (the worker OOM
//   path passes the raw exception through), and only the first of the two was counted in
//   ninfer:context_cache_exhausted_requests_total, which docs/maintainer/consolidated-line.md:24
//   promises is a 429 that IS counted.
//
//   The structural fix is not "wrap it at the second site too": it is to give the classification a
//   single owner (this header) and a single funnel (EngineCore::to_client_error), so that
//   * a new throw site cannot pick a different client-visible outcome by accident, and
//   * `--no-recover-invariant-failures` provably cannot change the outcome of a CAPACITY failure
//     (that is now a property of the table below, not a coincidence a checklist has to remember).
//
// The table is deliberately tiny and closed:
//   ContextCacheExhausted      -> Capacity   (the store refused: retry later, HTTP 429)
//   any other std::bad_alloc   -> Capacity   (host/device allocation failed: same answer to the client)
//   std::logic_error           -> Invariant  (our own accounting broke: crash unless the flag says recover)
//   anything else              -> Other      (unclassified: keep the existing behaviour, do not guess)
//
// tests/test_failure_class.cpp pins this table, so changing it is a deliberate act that fails a test
// rather than a silent drift that changes what users see.
#pragma once

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "ninfer/types.h"   // ContextCacheExhausted

namespace ninfer {

enum class FailureClass : std::uint8_t {
    Capacity,    // no room right now -> the request is refused, the engine keeps serving (HTTP 429)
    Invariant,   // the engine's own accounting broke -> governed by recover_invariant_failures
    Other,       // unclassified -> unchanged behaviour
};

// The order matters: ContextCacheExhausted derives from std::bad_alloc, so it must be tested first
// and is called out separately so the two cases can still be told apart in logs and metrics.
[[nodiscard]] inline FailureClass classify_failure(const std::exception& error) noexcept {
    if (dynamic_cast<const ContextCacheExhausted*>(&error) != nullptr) { return FailureClass::Capacity; }
    if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr) { return FailureClass::Capacity; }
    if (dynamic_cast<const std::logic_error*>(&error) != nullptr) { return FailureClass::Invariant; }
    return FailureClass::Other;
}

[[nodiscard]] inline bool is_capacity_failure(const std::exception& error) noexcept {
    return classify_failure(error) == FailureClass::Capacity;
}

[[nodiscard]] inline const char* failure_class_name(FailureClass value) noexcept {
    switch (value) {
    case FailureClass::Capacity: return "capacity";
    case FailureClass::Invariant: return "invariant";
    case FailureClass::Other: return "other";
    }
    return "other";
}

// The client-visible wording of a capacity failure. Two prefixes, one place: store exhaustion keeps
// the wording the admission path has used since 2026-10-03 (so logs stay greppable across versions)
// and a plain allocation failure says what it is instead of borrowing the store's sentence.
[[nodiscard]] inline std::string capacity_failure_message(const std::exception& error) {
    const char* prefix = dynamic_cast<const ContextCacheExhausted*>(&error) != nullptr
                             ? "context cache exhausted: "
                             : "engine out of memory during execution: ";
    return std::string(prefix) + error.what();
}

} // namespace ninfer
