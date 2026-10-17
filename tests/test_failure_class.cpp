// Pins the failure classification table (include/ninfer/failure_class.h).
//
// WHY THIS TEST EXISTS (audit 2026-10-04, finding F1): the batch's own acceptance checklist had a
// negative control that could never turn red, because the meaning of a failure was carried by its C++
// type and re-decided at five different places. Prose cannot notice when that table moves; a test can.
// From now on, "a capacity failure is never treated as an engine invariant" is checked by the
// compiler and this file, not by a sentence in a manual.
//
// What it pins, exactly:
//   1. ContextCacheExhausted      -> Capacity   (the store refused; retryable; HTTP 429)
//   2. std::bad_alloc (plain)     -> Capacity   (allocation failed; same client-visible answer)
//   3. std::logic_error           -> Invariant  (our accounting broke; recover_invariant_failures)
//   4. std::runtime_error         -> Other      (unclassified; behaviour must NOT be guessed at)
//   5. a plain std::exception     -> Other
//   6. the two capacity spellings stay distinct: store exhaustion keeps the wording the admission
//      path has used since 2026-10-03, a plain OOM says it is an OOM.
//
// Negative control inside the test: case 1 asserts Capacity *and* case 3 asserts Invariant, so
// collapsing the table to a single value fails at least one of them -- the test cannot pass by
// accident, which is exactly what the checklist could not promise.
#include "ninfer/failure_class.h"

#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

namespace {

using ninfer::ContextCacheExhausted;
using ninfer::FailureClass;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // 1. store exhaustion is a capacity event, and it is NOT an invariant
    {
        const ContextCacheExhausted exhausted("Paged KV pool cannot materialize 3 page(s): usable 63");
        failures += check(ninfer::classify_failure(exhausted) == FailureClass::Capacity,
                          "ContextCacheExhausted must classify as Capacity");
        failures += check(ninfer::is_capacity_failure(exhausted),
                          "ContextCacheExhausted must be a capacity failure");
        failures += check(std::string(ninfer::capacity_failure_message(exhausted))
                              .rfind("context cache exhausted: ", 0) == 0,
                          "store exhaustion must keep the 'context cache exhausted: ' wording");
    }

    // 2. a plain allocation failure takes the same client-visible route
    {
        const std::bad_alloc oom;
        failures += check(ninfer::classify_failure(oom) == FailureClass::Capacity,
                          "std::bad_alloc must classify as Capacity");
        failures += check(std::string(ninfer::capacity_failure_message(oom))
                              .rfind("engine out of memory during execution: ", 0) == 0,
                          "a plain bad_alloc must not borrow the store's sentence");
    }

    // 3. a broken invariant stays an invariant: this is the case `--no-recover-invariant-failures`
    //    governs, and the reason a full pool can no longer be mistaken for one.
    {
        const std::logic_error invariant("KV activation has an unavailable Device page");
        failures += check(ninfer::classify_failure(invariant) == FailureClass::Invariant,
                          "std::logic_error must classify as Invariant, not Capacity");
        failures += check(!ninfer::is_capacity_failure(invariant),
                          "an invariant must never be reported as a capacity failure");
    }

    // 4./5. anything unclassified keeps its existing behaviour
    {
        const std::runtime_error other("something else");
        failures += check(ninfer::classify_failure(other) == FailureClass::Other,
                          "std::runtime_error must classify as Other");
        const std::exception plain;
        failures += check(ninfer::classify_failure(plain) == FailureClass::Other,
                          "a plain std::exception must classify as Other");
    }

    // 6. the classifier is a table, not a guess: names are stable because logs quote them
    failures += check(std::string(ninfer::failure_class_name(FailureClass::Capacity)) == "capacity",
                      "capacity must name itself 'capacity' in logs");
    failures += check(std::string(ninfer::failure_class_name(FailureClass::Invariant)) == "invariant",
                      "invariant must name itself 'invariant' in logs");
    failures += check(std::string(ninfer::failure_class_name(FailureClass::Other)) == "other",
                      "other must name itself 'other' in logs");

    return failures == 0 ? 0 : 1;
}
