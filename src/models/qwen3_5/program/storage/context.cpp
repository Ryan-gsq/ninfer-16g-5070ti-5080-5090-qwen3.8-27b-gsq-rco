#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"
// LOCAL (KVMem content scoring, 2026-10-02): the scorer's published selection + scores. Header-only,
// host-safe (it pulls cuda_runtime, not a CUDA kernel header), so no build wiring is needed.
#include "ops/kvmem/kvmem_score.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

bool ProgramImpl::valid_continuation(const ContinuationHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < continuation_capacity &&
           ContractAccess::epoch(handle) == continuation_slots[index].generation &&
           continuation_slots[index].role == ContinuationSlotRole::Catalogued;
}

bool ProgramImpl::valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < shared_prefix_capacity &&
           ContractAccess::epoch(handle) == shared_prefix_slots[index].generation &&
           shared_prefix_slots[index].role == SharedPrefixSlotRole::Catalogued;
}

bool ProgramImpl::valid_capture_offer(const CaptureOffer& offer) const noexcept {
    if (ContractAccess::owner(offer) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    if (lane >= max_concurrency || ContractAccess::epoch(offer) != lane_epochs[lane] ||
        (requests[lane].lifecycle != Lifecycle::Prefilling &&
         requests[lane].lifecycle != Lifecycle::Active) ||
        !requests[lane].prefill) {
        return false;
    }
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    return prefill.pending_capture_offer != 0 &&
           prefill.pending_capture_offer == ContractAccess::id(offer) &&
           prefill.next_capture < prefill.capture_groups.size() &&
           prefill.cursor == prefill.capture_groups[prefill.next_capture].frontier;
}

bool ProgramImpl::materialization_pins(std::uint32_t index,
                                       std::uint64_t generation) const noexcept {
    const MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) { return false; }
    const MaterializationTransaction& transaction = *transaction_ptr;
    if (transaction.has_source && transaction.source_index == index &&
        transaction.source_generation == generation) {
        return true;
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim] && transaction.victim_indices[victim] == index &&
            transaction.victim_generations[victim] == generation) {
            return true;
        }
    }
    return false;
}

bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

std::optional<std::uint32_t> ProgramImpl::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

bool ProgramImpl::can_release_continuation_slot_strict(std::uint32_t index) const {
    if (index >= continuation_capacity || !state_store || !text_kv_addresses || !text_kv_pages ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        return false;
    }
    const SequenceState& sequence = continuation_states[index];
    if (sequence.state.fork_pending || !sequence.shared_prefix_references.empty() || !sequence.kv ||
        !text_kv_addresses->can_release(sequence.kv->text)) {
        return false;
    }
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages ||
            !backend_kv_addresses->can_release(*sequence.kv->backend)) {
            return false;
        }
    }

    const auto validate_state = [&](StateImageHandle handle, bool release_object) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
            return false;
        }
        return !release_object || total != owned ||
               state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
        if (handle == sequence.state.read || handle == sequence.state.write ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
            return true;
        }
        for (std::size_t prior = 0; prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { return true; }
        }
        return false;
    };

    if (sequence.endpoint_valid) {
        if (!validate_state(sequence.state.read, !sequence.state.read_has_external_owner() ||
                                                     sequence.state.read == sequence.state.write)) {
            return false;
        }
        if (sequence.state.write != sequence.state.read &&
            !validate_state(sequence.state.write, true)) {
            return false;
        }
    } else if (state_store->valid(sequence.state.read) ||
               state_store->valid(sequence.state.write) || sequence.state.borrows_read()) {
        return false;
    }
    if (sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read &&
        *sequence.rewrite_state != sequence.state.write &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t anchor = 0; anchor < sequence.long_anchors.size(); ++anchor) {
        const StateImageHandle handle = sequence.long_anchors[anchor].state;
        if (!repeated_before_anchor(anchor, handle) && !validate_state(handle, true)) {
            return false;
        }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated = handle == sequence.state.read || handle == sequence.state.write ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_continuation_slot_strict(std::uint32_t index,
                                                   bool written_to_disk) noexcept {
    try {
        if (!can_release_continuation_slot_strict(index)) { std::terminate(); }
    } catch (...) { std::terminate(); }
    const ActiveExclusiveBaseline baseline = active_exclusive_baseline();
    SequenceState& sequence                = continuation_states[index];
    release_sequence_kv_strict(sequence, written_to_disk);
    release_sequence_state_strict(sequence);
    retire_continuation_slot(index);
    credit_active_ownership_transfers(baseline);
}

void ProgramImpl::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    const ActiveExclusiveBaseline baseline = active_exclusive_baseline();
    SequenceState& sequence                = continuation_states[index];
    release_active_shared_references(sequence);
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
    credit_active_ownership_transfers(baseline);
}

void ProgramImpl::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_digests.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    sequence.rewrite_checkpoint      = {};
    sequence.rebuild_work            = {};
    sequence.rebuild_tail_begin      = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

detail::PhysicalResources
ProgramImpl::sequence_exclusive_state_resources(const SequenceState& sequence) const {
    if (!state_store) {
        throw std::logic_error("sequence StateImage resources have no physical store");
    }
    detail::PhysicalResources out;
    std::array<StateImageHandle, 4> states{};
    std::uint32_t state_count = 0;
    const auto add_state      = [&](StateImageHandle handle) {
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { return; }
        for (std::uint32_t index = 0; index < state_count; ++index) {
            if (states[index] == handle) { return; }
        }
        states[state_count++]                 = handle;
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    };
    const bool has_read_state  = sequence.state.read.valid();
    const bool has_write_state = sequence.state.write.valid();
    if (has_read_state != has_write_state) {
        throw std::logic_error("sequence owner has a partial primary StateImage pair");
    }
    if (sequence.state.borrows_read() &&
        (!sequence.state.fork_pending || sequence.state.read == sequence.state.write)) {
        throw std::logic_error("sequence has an invalid borrowed StateImage source");
    }
    if (has_read_state) {
        if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
            add_state(sequence.state.read);
        }
        add_state(sequence.state.write);
    }
    if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale long-anchor StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { continue; }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(state_count, states.size());
             ++index) {
            if (states[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (seen) { continue; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SequenceState& sequence) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("sequence owner resources have no physical stores");
    }
    detail::PhysicalResources out = sequence_exclusive_state_resources(sequence);

    {
        if (!sequence.kv) { throw std::logic_error("sequence owner has no KV address bundle"); }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale KV address space"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                // A shared logical page contributes to aggregate occupancy once. Releasing this
                // address cannot free either replica while another address still references it,
                // so it is not part of this owner's exact transition effect.
                if (pages.address_references(logical) > 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("resident Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
            if (addresses.active(address)) {
                const std::uint32_t mapped      = addresses.mapped_pages(address);
                const std::uint32_t entitlement = addresses.entitlement(address);
                // LOCAL FIX (ring device footprint): the Device footprint of an active
                // address is its resident pages plus its unmapped entitlement, but never
                // more than the pool. The ring keeps the excess on Host, so a demoted or
                // host-backed page is not Device footprint. Without this cap the
                // materialization check failed at the pool boundary
                // ("materialized sequence does not match its active entitlement").
                const std::uint32_t pool       = pages.physical_pool().usable_pages();
                const std::uint32_t device_cap =
                    pool == 0 ? entitlement : std::min(entitlement, pool);
                if (device_cap > mapped) {
                    const std::uint32_t delta = device_cap - mapped;
                    if (delta > std::numeric_limits<std::uint32_t>::max() - device_pages) {
                        throw std::overflow_error("owner active KV entitlement overflow");
                    }
                    device_pages += delta;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.device.main_kv_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SharedPrefixState& shared) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("shared owner resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        if (!shared.kv || !shared.identity || !state_store->valid(shared.state)) {
            throw std::logic_error("shared prefix has incomplete resident physical state");
        }
        if (state_store->checkpoint_references(shared.state) == 0) {
            throw std::logic_error("shared prefix StateImage has no checkpoint reference");
        }
        const StateReplicaResidency residency = state_store->residency(shared.state);
        if (state_store->checkpoint_references(shared.state) == 1) {
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale shared KV address"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) != 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("shared Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, out.device.main_kv_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing shared Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::active_snapshot_shared_resources(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("active snapshot source has no KV stores");
    }
    detail::PhysicalResources out;
    const auto add_kv = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, std::uint32_t frontier,
                            std::uint32_t& device_pages) {
        const std::uint32_t full_pages = frontier / static_cast<std::uint32_t>(kPagedKVPageSize);
        if (full_pages > addresses.mapped_pages(address)) {
            throw std::logic_error("active snapshot frontier exceeds its mapped KV pages");
        }
        for (std::uint32_t page = 0; page < full_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) != 1) { continue; }
            if (pages.device_resident(logical)) { ++device_pages; }
            if (pages.host_resident(logical)) {
                if (!host_kv_extents) { throw std::logic_error("missing Host KV extent store"); }
                const std::size_t stride =
                    host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
                if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                    throw std::overflow_error("active snapshot Host KV byte count overflow");
                }
                out.host.kv_bytes += stride;
            }
        }
    };
    add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, sequence.text_kv_valid,
           out.device.main_kv_pages);
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("missing Backend KV stores");
        }
        add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
               backend_kv_valid(sequence), out.device.backend_kv_pages);
    }
    return out;
}

ProgramImpl::ActiveExclusiveBaseline ProgramImpl::active_exclusive_baseline() const noexcept {
    ActiveExclusiveBaseline out{};
    // A Hybrid lane's entitlement is its admission quote, not exclusive page ownership.
    if (hybrid_prefix_cache()) { return out; }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (requests[lane].lifecycle == Lifecycle::Empty || continuation >= continuation_capacity) {
            continue;
        }
        try {
            out[lane] = ActiveExclusiveEntry{
                .continuation = continuation,
                .resources    = owner_exclusive_resources(continuation_states[continuation]),
            };
        } catch (...) {}
    }
    return out;
}

void ProgramImpl::credit_active_ownership_transfers(
    const ActiveExclusiveBaseline& baseline) noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (!baseline[lane] || requests[lane].lifecycle == Lifecycle::Empty ||
            active_continuations[lane] != baseline[lane]->continuation) {
            continue;
        }
        try {
            const detail::PhysicalResources exclusive =
                owner_exclusive_resources(continuation_states[baseline[lane]->continuation]);
            requests[lane].active_resources = checked_resource_sum(
                requests[lane].active_resources,
                positive_resource_difference(exclusive, baseline[lane]->resources));
        } catch (...) {}
    }
}

detail::PhysicalResources ProgramImpl::physical_occupancy() const noexcept {
    detail::PhysicalResources out;
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) { ++out.device.active_lanes; }
    }
    if (state_store) {
        out.device.state_slots = state_store->device_occupied();
        out.host.state_slots   = state_store->host_occupied();
    }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        // LOCAL FIX (ring physical occupancy): under a logical reservation the reserved
        // count is a promise the ring satisfies by demotion on demand, not memory already
        // held. Counting it made the Device look permanently oversubscribed
        // (allocated 1500 + reserved 938 = 2438 over a 1500-page pool) and drove the
        // capture/materialization paths into a state they could not fund.
        out.device.main_kv_pages = pool.allocated_pages() +
                                  (ring_requested() ? 0U : pool.reserved_pages());
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device.backend_kv_pages = pool.allocated_pages() +
                                     (ring_requested() ? 0U : pool.reserved_pages());
    }
    if (host_kv_arena) { out.host.kv_bytes = host_kv_arena->occupied_bytes(); }
    return out;
}

detail::PhysicalResources
ProgramImpl::materialization_deficit(const ResourceCandidateState& admission) const {
    // Pressure is relative to this candidate's real peak. Treating every dimension as scarce
    // would forbid Device-to-Host demotion even when Host capacity is available.
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), admission.demand.physical_peak_additional);
    const detail::PhysicalResources limits = admission_capacity();
    if (!ring_requested()) { return positive_resource_difference(required, limits); }
    // LOCAL PROTOTYPE (KVMem-style ring): a logical Main KV entitlement larger than the pool is
    // not a real deficit; the ring recycles resident pages. Clamp before measuring pressure.
    detail::PhysicalResources clamped = required;
    if (clamped.device.main_kv_pages > limits.device.main_kv_pages) {
        clamped.device.main_kv_pages = limits.device.main_kv_pages;
    }
    if (clamped.device.backend_kv_pages > limits.device.backend_kv_pages) {
        clamped.device.backend_kv_pages = limits.device.backend_kv_pages;
    }
    return positive_resource_difference(clamped, limits);
}

detail::PhysicalResources
ProgramImpl::guided_materialization_deficit(const ResourceCandidateState& admission,
                                            const detail::PhysicalDelta& pressure) const {
    // Pressure acts on the candidate's complete peak, not on its already-clamped deficit.  Applying
    // a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even
    // when the arena has ample slack and steer the heuristic toward unnecessary destruction.
    const detail::PhysicalResources projected_peak = positive_resource_difference(
        checked_resource_sum(admission.demand.physical_peak_additional, pressure.added),
        pressure.removed);
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), projected_peak);
    const detail::PhysicalResources limits = admission_capacity();
    if (!ring_requested()) { return positive_resource_difference(required, limits); }
    // LOCAL PROTOTYPE (KVMem-style ring): see materialization_deficit.
    detail::PhysicalResources clamped = required;
    if (clamped.device.main_kv_pages > limits.device.main_kv_pages) {
        clamped.device.main_kv_pages = limits.device.main_kv_pages;
    }
    if (clamped.device.backend_kv_pages > limits.device.backend_kv_pages) {
        clamped.device.backend_kv_pages = limits.device.backend_kv_pages;
    }
    return positive_resource_difference(clamped, limits);
}

bool ProgramImpl::physical_peak_fits(detail::PhysicalResources peak) const noexcept {
    const detail::PhysicalResources occupied = physical_occupancy();
    const detail::PhysicalResources limits   = admission_capacity();
    const auto fits_u32 = [](std::uint32_t used, std::uint32_t added, std::uint32_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    const auto fits_size = [](std::size_t used, std::size_t added, std::size_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    // LOCAL PROTOTYPE (KVMem-style ring): when the Device pool is smaller than the logical context
    // the pool is managed by the ring (pages are demoted to the Host and retrieved back), so a
    // request's logical page entitlement must NOT be compared against it. Clamping the entitlement
    // to the pool does not work either: the check is `used + added <= capacity`, so clamping
    // `added` to `capacity` forces `used == 0` and rejects every request once the pool holds
    // anything. Skip the dimension instead.
    // LOCAL FIX (ring-fits-used): zeroing only `added` is not enough: `used` itself is the
    // logical mapped count, legitimately above pool capacity by design (1660 mapped vs 1500
    // pool observed), so `used <= capacity` fails every capture past the pool even with
    // zero added. Skip the whole dimension as the comment intends, not just the added term.
    const bool ring = ring_requested() && text_kv_pages != nullptr && text_kv_addresses != nullptr &&
                      text_kv_pages->physical_pool().usable_pages() <
                          text_kv_addresses->logical_page_capacity();
    const bool ok =
        fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                 limits.device.active_lanes) &&
        fits_u32(occupied.device.state_slots, peak.device.state_slots,
                 limits.device.state_slots) &&
        (ring || fits_u32(occupied.device.main_kv_pages, peak.device.main_kv_pages,
                          limits.device.main_kv_pages)) &&
        (ring || fits_u32(occupied.device.backend_kv_pages, peak.device.backend_kv_pages,
                          limits.device.backend_kv_pages)) &&
        fits_u32(occupied.host.state_slots, peak.host.state_slots, limits.host.state_slots) &&
        fits_size(occupied.host.kv_bytes, peak.host.kv_bytes, limits.host.kv_bytes);
    return ok;
}

StateImageHandle
ProgramImpl::selected_state(const SequenceState& sequence, ReusePath reuse,
                            std::optional<runtime::CheckpointRef> checkpoint) const {
    if (reuse == ReusePath::PrivateEndpoint) {
        if (!sequence.endpoint_valid || !state_store->valid(sequence.state.read)) {
            throw std::logic_error("private endpoint StateImage is stale");
        }
        return sequence.state.read;
    }
    if (is_rewrite_checkpoint_restore(reuse) && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state)) {
        return *sequence.rewrite_state;
    }
    if (reuse == ReusePath::PrivateLongAnchor) {
        if (!checkpoint || checkpoint->kind != runtime::CheckpointKind::LongAnchor) {
            throw std::logic_error("long-anchor materialization has no selected checkpoint");
        }
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint->frontier &&
                                                    candidate.ordinal == checkpoint->ordinal;
                                         });
        if (anchor != sequence.long_anchors.end() && state_store->valid(anchor->state)) {
            return anchor->state;
        }
    }
    throw std::logic_error("materialization path has no selected StateImage");
}

std::uint32_t
ProgramImpl::selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                                RewriteCheckpointDisposition rewrite_disposition,
                                                std::optional<runtime::CheckpointRef> checkpoint,
                                                std::uint32_t reuse_base) const {
    const StateImageHandle selected   = selected_state(sequence, reuse, checkpoint);
    std::uint32_t consumed_references = 0;
    if (is_rewrite_checkpoint_restore(reuse) &&
        rewrite_disposition != RewriteCheckpointDisposition::RetainExisting) {
        if (!sequence.rewrite_state || *sequence.rewrite_state != selected) {
            throw std::logic_error("selected rewrite StateImage is unavailable");
        }
        consumed_references = 1;
    } else if (reuse == ReusePath::PrivateEndpoint &&
               rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
               sequence.rewrite_state && *sequence.rewrite_state == selected) {
        consumed_references = 1;
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier > reuse_base && anchor.state == selected) {
            if (consumed_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("consumed StateImage reference inventory overflow");
            }
            ++consumed_references;
        }
    }
    const std::uint32_t references = state_store->checkpoint_references(selected);
    if (consumed_references > references) {
        throw std::logic_error("selected StateImage reference inventory is inconsistent");
    }
    return consumed_references;
}

bool ProgramImpl::selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                               RewriteCheckpointDisposition rewrite_disposition,
                                               std::optional<runtime::CheckpointRef> checkpoint,
                                               std::uint32_t reuse_base) const {
    const StateImageHandle selected = selected_state(sequence, reuse, checkpoint);
    return state_store->checkpoint_references(selected) !=
           selected_state_consumed_references(sequence, reuse, rewrite_disposition, checkpoint,
                                              reuse_base);
}

bool ProgramImpl::can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                const RewriteCheckpointSpec& desired,
                                                const SequenceState& sequence, ReusePath reuse,
                                                std::uint32_t reuse_base) const {
    if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
        !state_store->valid(*sequence.rewrite_state) ||
        !qwen3_5::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                         sequence.rewrite_checkpoint.frontier)) {
        return false;
    }
    if (sequence.rewrite_checkpoint.frontier == desired.frontier) { return true; }
    return is_rewrite_checkpoint_restore(reuse) &&
           sequence.rewrite_checkpoint.frontier == reuse_base && desired.frontier <= reuse_base;
}

std::uint32_t ProgramImpl::device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.device_resident(addresses.logical_page(address, page))) { ++resident; }
    }
    return resident;
}

std::uint32_t ProgramImpl::shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t shared = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.address_references(addresses.logical_page(address, page)) <= 1) { continue; }
        if (page + 1U == required && frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0) {
            continue;
        }
        ++shared;
    }
    return shared;
}

std::uint32_t ProgramImpl::shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                         KVAddressSpaceHandle address,
                                                         std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && pages.device_resident(logical)) { ++resident; }
    }
    return resident;
}

bool ProgramImpl::partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address,
                                            std::uint32_t frontier) const {
    if (frontier == 0 || frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
        return false;
    }
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
    return pages.address_references(tail) > 1 || !pages.device_resident(tail);
}

std::uint32_t
ProgramImpl::missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t missing = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && !pages.device_resident(logical)) { ++missing; }
    }
    return missing;
}

std::size_t ProgramImpl::host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                              KVAddressSpaceHandle address,
                                              std::uint32_t frontier) const noexcept {
    if (!host_kv_extents) { return 0; }
    try {
        const LogicalKVPageStore& pages =
            (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
        const std::uint32_t required_pages = kv_pages_for_frontier(frontier);
        if (required_pages > addresses.mapped_pages(address)) { return 0; }
        std::size_t bytes = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) > 1) { continue; }
            if (!pages.host_resident(logical)) { continue; }
            if (page + 1U == required_pages &&
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0 &&
                partial_tail_cow_required(addresses, address, frontier)) {
                continue;
            }
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t selected_columns =
                std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (selected_columns != pages.committed_columns(logical)) {
                // A destructive private rewrite changes this tail page's content epoch, so its
                // old Host replica cannot remain part of the active entitlement.
                continue;
            }
            const std::size_t stride =
                host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
            bytes += stride;
        }
        return bytes;
    } catch (...) { return 0; }
}

qwen3_5::CheckpointSummary
ProgramImpl::checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                                StateImageHandle state, runtime::PrefillWork rebuild_work) const {
    if (!sequence.kv) { throw std::logic_error("checkpoint summary has no KV address space"); }
    if (checkpoint.frontier == 0) {
        throw std::logic_error("checkpoint summary has an empty frontier");
    }
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint summary has a stale StateImage");
    }
    const StateReplicaResidency state_location = state_store->residency(state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("checkpoint StateImage has no published replica");
    }
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? checkpoint.frontier - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? checkpoint.frontier
                                                            : 0U;
    const std::uint32_t identity_tag = capture_identity_tag();
    // LOCAL FIX (host-backed checkpoint): count the pages of this checkpoint's range whose Device
    // replica the ring gave up while a CURRENT Host replica exists. Same predicate as the publish gate
    // in capture.cpp (`scan_range`); the byte total reuses `host_kv_prefix_bytes` so the two never
    // disagree about the stride.
    const auto count_host_backed = [&](const KVAddressSpaceStore& addresses,
                                       const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                       std::uint32_t page_limit) {
        std::uint32_t count = 0;
        if (!addresses.valid(address)) { return count; }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        const std::uint32_t limit  = page_limit < mapped ? page_limit : mapped;
        for (std::uint32_t page = 0; page < limit; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (!pages.valid(logical) || pages.device_resident(logical)) { continue; }
            if (pages.host_resident(logical) && pages.host_replica_current(logical)) { ++count; }
        }
        return count;
    };
    std::uint32_t host_backed_pages = 0;
    std::uint64_t host_backed_bytes = 0;
    if (text_kv_addresses != nullptr && text_kv_pages != nullptr) {
        host_backed_pages += count_host_backed(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                               kv_pages_for_frontier(checkpoint.frontier));
        host_backed_bytes += static_cast<std::uint64_t>(
            host_kv_prefix_bytes(*text_kv_addresses, sequence.kv->text, checkpoint.frontier));
    }
    if (sequence.kv->backend && backend_kv_addresses != nullptr && backend_kv_pages != nullptr) {
        host_backed_pages +=
            count_host_backed(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                              kv_pages_for_frontier(backend_frontier));
        host_backed_bytes += static_cast<std::uint64_t>(host_kv_prefix_bytes(
            *backend_kv_addresses, *sequence.kv->backend, backend_frontier));
    }
    return qwen3_5::CheckpointSummary{
        .ref   = checkpoint,
        .scope = runtime::CheckpointScope::Private,
        .shortlist_key =
            {
                .digests      = sequence.prefix_digests.at(checkpoint.frontier),
                .frontier     = checkpoint.frontier,
                .identity_tag = identity_tag,
            },
        .state_residency = residency,
        .required_kv =
            {
                .main_frontier    = checkpoint.frontier,
                .backend_frontier = backend_frontier,
                .main_pages       = kv_pages_for_frontier(checkpoint.frontier),
                .backend_pages    = kv_pages_for_frontier(backend_frontier),
            },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
        // LOCAL FIX (host-backed checkpoint): register how much of the range is Host-backed only.
        // `host_kv_prefix_bytes` (context.cpp:824-858) already sums exactly those strides; the page
        // count repeats the publish gate's predicate (capture.cpp `scan_range`: no Device replica and a
        // CURRENT Host replica). Nothing consumes these fields yet.
        .host_backed_pages = host_backed_pages,
        .host_backed_bytes = host_backed_bytes,
    };
}

qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const SequenceState& sequence) const {
    qwen3_5::ContinuationSummary summary;
    summary.long_anchors.reserve(sequence.long_anchors.size());
    populate_continuation_summary(sequence, summary);
    return summary;
}

void ProgramImpl::populate_continuation_summary(const SequenceState& sequence,
                                                qwen3_5::ContinuationSummary& summary) const {
    validate_long_anchor_ordinals(sequence.long_anchors,
                                  context_cache.max_long_anchors_per_continuation.value_or(0));
    if (summary.long_anchors.capacity() < sequence.long_anchors.size()) {
        throw std::logic_error("continuation summary backing was not reserved");
    }
    summary.endpoint.reset();
    summary.rewrite.reset();
    summary.long_anchors.clear();
    summary.active_references = 0;
    if (sequence.endpoint_valid) {
        const runtime::CheckpointRef endpoint{
            .kind     = runtime::CheckpointKind::SessionEndpoint,
            .frontier = sequence.execution_frontier,
        };
        runtime::PrefillWork endpoint_work = sequence.rebuild_work;
        summary.endpoint =
            checkpoint_summary(sequence, endpoint, sequence.state.read, endpoint_work);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        const runtime::CheckpointRef rewrite{
            .kind     = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier = sequence.rewrite_checkpoint.frontier,
        };
        summary.rewrite = checkpoint_summary(sequence, rewrite, *sequence.rewrite_state,
                                             sequence.rewrite_checkpoint.rebuild_work);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        summary.long_anchors.push_back(
            checkpoint_summary(sequence,
                               runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                                      .frontier = anchor.frontier,
                                                      .ordinal  = anchor.ordinal},
                               anchor.state, anchor.rebuild_work));
    }
    if (!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) {
        throw std::logic_error("private continuation has no checkpoint");
    }
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        const std::size_t index = static_cast<std::size_t>(&sequence - begin);
        summary.active_references =
            continuation_slots[index].role == ContinuationSlotRole::Active ? 1U : 0U;
    }
}

qwen3_5::SharedPrefixSummary
ProgramImpl::shared_prefix_summary(const SharedPrefixState& shared) const {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        !state_store->valid(shared.state)) {
        throw std::logic_error("shared-prefix summary source is incomplete");
    }
    const StateReplicaResidency state_location = state_store->residency(shared.state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("shared-prefix StateImage has no published replica");
    }
    return qwen3_5::SharedPrefixSummary{
        .checkpoint =
            {
                .ref =
                    {
                        .kind     = runtime::CheckpointKind::SharedStablePrefix,
                        .frontier = shared.frontier,
                    },
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shared.identity->shortlist_key,
                .state_residency = residency,
                .required_kv =
                    {
                        .main_frontier    = shared.frontier,
                        .backend_frontier = shared.backend_frontier,
                        .main_pages       = kv_pages_for_frontier(shared.frontier),
                        .backend_pages    = kv_pages_for_frontier(shared.backend_frontier),
                    },
                .rebuild_work = validated_rebuild_work(shared.rebuild_work, shared.frontier),
            },
        .active_references = shared.active_references,
    };
}

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

bool ProgramImpl::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    for (std::size_t position = 0; position < sequence.shared_prefix_references.size();
         ++position) {
        const std::uint32_t index = sequence.shared_prefix_references[position];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) {
            return false;
        }
        const std::uint32_t required = static_cast<std::uint32_t>(std::count(
            sequence.shared_prefix_references.begin(),
            sequence.shared_prefix_references.begin() + static_cast<std::ptrdiff_t>(position + 1U),
            index));
        if (shared_prefix_states[index].active_references < required) { return false; }
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state.borrows_read() ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    enum class ForkEndpoint : std::uint8_t { None, Source, Destination };
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    ForkEndpoint fork_endpoint = ForkEndpoint::None) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable &&
             fork_endpoint != ForkEndpoint::Destination)) {
            return false;
        }
        if (!release_object || total != owned) { return true; }
        if (fork_endpoint == ForkEndpoint::Source) {
            return state_store->can_release_source_after_fork_abort(sequence.state.read,
                                                                    sequence.state.write, owned);
        }
        if (fork_endpoint == ForkEndpoint::Destination) {
            return state_store->can_release_destination_after_fork_abort(
                sequence.state.read, sequence.state.write, owned);
        }
        return state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto duplicates_binding = [&](StateImageHandle handle) {
        return handle == sequence.state.read || handle == sequence.state.write;
    };

    if (!validate_state(sequence.state.read,
                        !sequence.state.read_has_external_owner() ||
                            sequence.state.read == sequence.state.write,
                        sequence.state.fork_pending ? ForkEndpoint::Source : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true,
                        sequence.state.fork_pending ? ForkEndpoint::Destination
                                                    : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.rewrite_state && !duplicates_binding(*sequence.rewrite_state) &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
            repeated = sequence.long_anchors[prior].state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_active_shared_references_strict(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            std::terminate();
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

bool ProgramImpl::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                      = continuation_states.data();
    const std::uint32_t continuation       = static_cast<std::uint32_t>(&sequence - begin);
    const ActiveExclusiveBaseline baseline = active_exclusive_baseline();
    hybrid_release_lane(sequence.lane);
    release_active_shared_references_strict(sequence);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.retire();
    credit_active_ownership_transfers(baseline);
    return true;
}

void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    // A concurrent resource transaction may pin or inspect these active owners. Engine-wide
    // cleanup aborts that transaction before releasing lanes, preserving the only safe order.
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

void ProgramImpl::clear_lane_best_effort(SequenceState& sequence,
                                         RequestControl& request) noexcept {
    hybrid_release_lane(sequence.lane);
    request.retire();
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

std::uint32_t ProgramImpl::owned_checkpoint_references(const SequenceState& sequence,
                                                       StateImageHandle state) const noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

bool ProgramImpl::state_exclusive_to_sequence(const SequenceState& sequence,
                                              StateImageHandle state) const noexcept {
    if (!state_store || !state_store->valid(state)) { return false; }
    return state_store->checkpoint_references(state) ==
           owned_checkpoint_references(sequence, state);
}

void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}

void ProgramImpl::reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots) {
    const std::uint32_t owned = sequence_exclusive_state_resources(sequence).device.state_slots;
    if (slots == 0 || owned > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (owned == slots) { return; }
    if (slots - owned != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = state_store->reserve_destination();
    if (!reserved) {
        throw ninfer::ContextCacheExhausted("Device StateImage store has no free slot for the sequence reservation");
    }
    sequence.reserved_state = *reserved;
    if (sequence_exclusive_state_resources(sequence).device.state_slots != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    const bool external_source         = sequence.state.read_has_external_owner();
    state_store->commit_fork(source, destination);
    sequence.state = ActiveStateBinding{.read = destination, .write = destination};
    if (!external_source && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    refresh_state_views(sequence);
}

bool ProgramImpl::has_unsettled_state_fork() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (continuation < continuation_capacity &&
            continuation_states[continuation].state.fork_pending) {
            return true;
        }
    }
    return false;
}

void ProgramImpl::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto duplicates_binding = [&](StateImageHandle handle) {
            return handle == sequence.state.read || handle == sequence.state.write;
        };

        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state.read_has_external_owner());
        }
        if (sequence.rewrite_state) {
            release_if_unreferenced(*sequence.rewrite_state,
                                    !duplicates_binding(*sequence.rewrite_state));
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
                repeated = sequence.long_anchors[prior].state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store || sequence.state.fork_pending) { fail(); }

    try {
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
            if ((sequence.endpoint_valid &&
                 (handle == sequence.state.read || handle == sequence.state.write)) ||
                (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
                return true;
            }
            for (std::size_t prior = 0; prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { return true; }
            }
            return false;
        };

        if (sequence.endpoint_valid) {
            release_if_unreferenced(sequence.state.write, true);
            if (sequence.state.read != sequence.state.write) {
                release_if_unreferenced(sequence.state.read,
                                        !sequence.state.read_has_external_owner());
            }
        }
        if (sequence.rewrite_state) {
            const StateImageHandle handle = *sequence.rewrite_state;
            const bool duplicates_endpoint =
                sequence.endpoint_valid &&
                (handle == sequence.state.read || handle == sequence.state.write);
            release_if_unreferenced(handle, !duplicates_endpoint);
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            release_if_unreferenced(handle, !repeated_before_anchor(index, handle));
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = sequence.endpoint_valid &&
                            (handle == sequence.state.read || handle == sequence.state.write);
            repeated = repeated || (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }

    try {
        if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
            state_store->checkpoint_references(*sequence.rewrite_state) != 0) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            if (state_store->valid(anchor.state) &&
                state_store->checkpoint_references(anchor.state) != 0) {
                state_store->release_checkpoint_reference(anchor.state);
            }
        }
    } catch (...) {}

    const auto releasable = [&](StateImageHandle handle) { return state_store->valid(handle); };
    if (releasable(sequence.state.write)) { (void)state_store->release(sequence.state.write); }
    if (!sequence.state.read_has_external_owner() && sequence.state.read != sequence.state.write &&
        releasable(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    if (sequence.rewrite_state) {
        const StateImageHandle handle = *sequence.rewrite_state;
        const bool duplicates_binding =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read);
        if (!duplicates_binding && releasable(handle)) { (void)state_store->release(handle); }
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t previous = 0; !duplicate && previous < index; ++previous) {
            duplicate = sequence.long_anchors[previous].state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            duplicate = duplicate || anchor.state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_active_shared_references(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            continue;
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

const qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImpl::resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                                 std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    text_kv_addresses->resize_entitlement(sequence.kv->text, text_pages);
    if (sequence.kv->backend) {
        backend_kv_addresses->resize_entitlement(*sequence.kv->backend, backend_pages);
    }
}

void ProgramImpl::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row,
                                        compute_streams);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row,
                    compute_streams);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

void ProgramImpl::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

// LOCAL PROTOTYPE (KVMem-style ring retrieval): bring the selected Host-resident pages back onto the
// Device (H2D) and re-publish their block-table slots. Returns the retrieved pages that are Device-
// resident once this returns, which is exactly the set the retrieval mask may select (a page that was
// already resident also counts, and a failed restore simply drops out).
std::vector<std::uint32_t>
ProgramImpl::restore_kv_pages_from_host(KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                        const KVAddressSpaceHandle& address,
                                        std::span<const std::uint32_t> logical_pages) {
    std::vector<std::uint32_t> resident;
    if (!addresses.valid(address) || logical_pages.empty()) { return resident; }
    if (host_kv_extents == nullptr) {
        for (const std::uint32_t page : logical_pages) {
            if (pages.device_resident(addresses.logical_page(address, page))) {
                resident.push_back(page);
            }
        }
        return resident;
    }
    for (const std::uint32_t page : logical_pages) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (!pages.valid(logical)) { continue; }
        if (pages.device_resident(logical)) {
            resident.push_back(page);
            continue;
        }
        if (!pages.host_resident(logical)) { continue; }
        // SAFETY: never materialize into a full pool. A page that cannot be brought back is simply
        // left out of the mask, which hides it -- it can never be read from a stale Device slot.
        if (pages.physical_pool().free_runs().empty()) { break; }
        const HostKVPageReplica replica = pages.host_replica(logical);
        DeviceKVPageReservation reservation = pages.physical_pool().make_empty_reservation();
        pages.physical_pool().resize_reservation(reservation, 1);
        const HostKVAllocationConstView source =
            host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
        if (addresses.restore_page_from_host(address, page, source, reservation, device.stream)) {
            resident.push_back(page);
        }
    }
    return resident;
}

// LOCAL FIX (D-5 seam completeness, 2026-10-04): visibility is PAGE granularity, so evidence that
// straddles a page boundary is cut in half -- and the measured failure mode at a small pool is exactly
// that: the model answers with a TRUNCATED needle (`ZX-7001` -> `ZX-700`, `ARCH-4201` -> `ARCH-420`,
// 4/4 deterministic in the ledger's D-5 note; the pool-4000 multi-needle arm loses turn 2 on 4 of 6
// trials precisely because the needle's own page was brought back without its neighbour).
//
// That is what "a small window cannot hold a big question" looks like once it is measured: it is not
// that 63 pages cannot hold one needle -- it is that the resident set holds HALF of it. The cure does
// not need a bigger pool, it needs complete evidence for the pages the scorer picked.
//
// Policy: walk the ranked pages best-first and complete each one's neighbourhood while the budget
// lasts. Breadth is traded for completeness on purpose: 31 isolated pages become ~10 fully covered
// ones, and a needle needs 3 of them at most (its page plus the two the seam can be on). The budget
// stays the hard cap, so this never asks the ring to hold more than it already did.
std::vector<std::uint32_t> add_seam_neighbors(std::vector<std::uint32_t> ranked,
                                              std::uint32_t begin, std::uint32_t end,
                                              std::uint32_t budget, std::uint32_t reach) {
    std::vector<std::uint32_t> out;
    if (reach == 0 || budget == 0 || ranked.empty()) { return ranked; }
    out.reserve(ranked.size() + 2U * reach * ranked.size());
    const auto contains = [](const std::vector<std::uint32_t>& pages, std::uint32_t page) {
        return std::find(pages.begin(), pages.end(), page) != pages.end();
    };
    for (const std::uint32_t page : ranked) {
        if (page < begin || page >= end) { continue; }
        if (!contains(out, page)) { out.push_back(page); }
        for (std::uint32_t step = 1; step <= reach; ++step) {
            // FORWARD first: the measured symptom is a truncated TAIL ("...->  ZX-700"), and what is
            // missing sits in the page that follows. Backward is added next, for a needle whose
            // beginning was the part that fell on the far side of the seam.
            const std::uint32_t forward = page + step;
            if (forward < end && !contains(out, forward)) { out.push_back(forward); }
            if (out.size() >= budget) { break; }
            const std::uint32_t backward = page >= step ? page - step : 0U;
            if (page >= step && backward >= begin && !contains(out, backward)) {
                out.push_back(backward);
            }
            if (out.size() >= budget) { break; }
        }
        if (out.size() >= budget) { break; }
    }
    if (out.size() > budget) { out.resize(budget); }
    return out;
}

// How many pages on EITHER side of a selected page are brought back with it.
//   0 = the shipped behaviour, i.e. this fix's NEGATIVE CONTROL (`NINFER_KV_NEIGHBOR_PAGES=0`)
//   1 = +-1 page (the default; a 64-token page boundary can only separate two halves)
//   2 = +-2 pages, only ever if the budget allows it
std::uint32_t proto_kv_neighbor_pages() {
    static const std::uint32_t reach = [] {
        const char* text = std::getenv("NINFER_KV_NEIGHBOR_PAGES");
        if (text == nullptr) { return 1U; }
        const long value = std::strtol(text, nullptr, 10);
        return value <= 0 ? 0U : static_cast<std::uint32_t>(value);
    }();
    return reach;
}

// LOCAL PROTOTYPE (KVMem-style ring retrieval): rank the pages that are (or are about to become)
// Host-resident by lexical overlap with the recent context, and return the top `budget` page
// indices. The score is an inverse-document-frequency weighted token overlap, which is what makes
// a rare match (a variable name, an error string, a code) dominate ordinary function words. Token
// ids live on the Host in the sequence ledger, so this needs no Device work.
std::vector<std::uint32_t> ProgramImpl::select_retrieval_pages(const SequenceState& sequence,
                                                              std::uint32_t begin,
                                                              std::uint32_t end,
                                                              std::uint32_t budget) const {
    std::vector<std::uint32_t> selected;
    if (budget == 0 || end <= begin) { return selected; }
    // ---- LOCAL (KVMem content scoring, 2026-10-02) ---------------------------------------------
    // Prefer the CONTENT ranking when the scorer published one. It ranks pages by a query-conditioned
    // dot product against the mean-K index (ops/kvmem/kvmem_retrieve.h, Eq.10), so a page is found
    // because it is ABOUT the question, not because it happens to share rare WORDS with it. That
    // lexical dependence is the measured defect: with the IDF path below, a question that is
    // semantically the same but lexically disjoint lost the mid-prompt needle in 2 of 6 trials.
    //
    // The ring's restore budget is smaller than the selector's own budget, so the kept set is ranked
    // again here and cut to `budget`; ids are ascending by contract (kvmem_select.h:30-33).
    //
    // Everything below stays untouched and is the fallback -- and also this change's NEGATIVE CONTROL:
    // with NINFER_TERNARY_KVMEM_SCORE unset nothing is published, this block is skipped, and the
    // engine must reproduce the old readings exactly (the disjoint arm goes back to its pre-fix 4/6).
    {
        const std::vector<std::int32_t>& kept = ops::detail::kvmem_score_selected_blocks();
        const std::vector<float>& scores     = ops::detail::kvmem_score_host_scores();
        const std::int32_t scored            = ops::detail::kvmem_score_scored_blocks();
        if (!kept.empty() && scored > 0 && !scores.empty()) {
            std::vector<std::pair<float, std::uint32_t>> ranked;
            ranked.reserve(kept.size());
            for (const std::int32_t block : kept) {
                if (block < 0 || block >= scored) { continue; }
                const std::uint32_t page = static_cast<std::uint32_t>(block);
                if (page < begin || page >= end) { continue; }
                if (static_cast<std::size_t>(block) >= scores.size()) { continue; }
                ranked.emplace_back(scores[static_cast<std::size_t>(block)], page);
            }
            if (!ranked.empty()) {
                std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
                    return left.first != right.first ? left.first > right.first
                                                     : left.second < right.second;
                });
                if (ranked.size() > budget) { ranked.resize(budget); }
                // D-5: complete the evidence of the pages we keep, in score order, while the budget
                // lasts. The result is re-sorted below because callers binary-search it
                // (demote_kv_pages_to_host's is_preferred, context.cpp:1871).
                std::vector<std::uint32_t> by_score;
                by_score.reserve(ranked.size());
                for (const auto& entry : ranked) { by_score.push_back(entry.second); }
                selected = add_seam_neighbors(std::move(by_score), begin, end, budget,
                                              proto_kv_neighbor_pages());
                if (selected.empty()) {
                    selected.reserve(ranked.size());
                    for (const auto& entry : ranked) { selected.push_back(entry.second); }
                }
                std::sort(selected.begin(), selected.end());
                return selected;
            }
        }
    }
    // The ledger is only needed by the lexical fallback below; an empty ledger is not a reason to skip
    // the content path (which reads the index, not the ledger).
    if (sequence.ledger.empty()) { return selected; }
    constexpr std::size_t kQueryWindow = 256;
    const std::size_t query_begin =
        sequence.ledger.size() > kQueryWindow ? sequence.ledger.size() - kQueryWindow : 0;
    std::unordered_set<TokenId> query;
    for (std::size_t index = query_begin; index < sequence.ledger.size(); ++index) {
        query.insert(sequence.ledger[index]);
    }
    if (query.empty()) { return selected; }
    const std::size_t page_size = static_cast<std::size_t>(kPagedKVPageSize);
    std::vector<std::vector<TokenId>> page_tokens;
    page_tokens.reserve(end - begin);
    std::unordered_map<TokenId, std::uint32_t> document_frequency;
    for (std::uint32_t page = begin; page < end; ++page) {
        const std::size_t first = static_cast<std::size_t>(page) * page_size;
        if (first >= sequence.ledger.size()) { break; }
        const std::size_t last = std::min(first + page_size, sequence.ledger.size());
        std::vector<TokenId> distinct;
        for (std::size_t index = first; index < last; ++index) {
            const TokenId token = sequence.ledger[index];
            if (query.count(token) == 0) { continue; }
            if (std::find(distinct.begin(), distinct.end(), token) != distinct.end()) { continue; }
            distinct.push_back(token);
        }
        for (const TokenId token : distinct) { ++document_frequency[token]; }
        page_tokens.push_back(std::move(distinct));
    }
    const double documents = static_cast<double>(page_tokens.size());
    std::unordered_map<TokenId, double> weight;
    weight.reserve(document_frequency.size() * 2 + 1);
    for (const auto& entry : document_frequency) {
        weight.emplace(entry.first,
                       std::log((documents + 1.0) / (static_cast<double>(entry.second) + 1.0)) + 1.0);
    }
    std::vector<std::pair<double, std::uint32_t>> scored;
    scored.reserve(page_tokens.size());
    for (std::size_t offset = 0; offset < page_tokens.size(); ++offset) {
        double score = 0.0;
        for (const TokenId token : page_tokens[offset]) { score += weight[token]; }
        if (score > 0.0) { scored.emplace_back(score, begin + static_cast<std::uint32_t>(offset)); }
    }
    std::sort(scored.begin(), scored.end(), [](const auto& left, const auto& right) {
        return left.first != right.first ? left.first > right.first : left.second < right.second;
    });
    if (scored.size() > budget) { scored.resize(budget); }
    selected.reserve(scored.size());
    for (const auto& entry : scored) { selected.push_back(entry.second); }
    std::sort(selected.begin(), selected.end());
    return selected;
}

std::uint32_t ProgramImpl::demote_other_addresses_to_host(KVAddressSpaceStore& addresses,
                                                          LogicalKVPageStore& pages,
                                                          const KVAddressSpaceHandle& skip,
                                                          std::uint32_t target_free) {
    if (host_kv_extents == nullptr || target_free == 0) { return 0; }
    std::uint32_t freed = 0;
    std::uint32_t slots_visited = 0, skip_invalid = 0, skip_noresident = 0, skip_pins = 0,
                  skip_stalehost = 0, skip_partial = 0, skip_duplicate = 0;
    // LOCAL DIAGNOSTIC (demote-counters): the fourth way to end with freed=0 was invisible. A
    // refused host-arena reservation (HostKVExtentStore full) returns false from flush() without
    // bumping any counter -- and the tail call at the end of this function dropped the return
    // value outright. Counted here so `demote-other short` can name it. Read-only observation.
    std::uint32_t flush_failed = 0;
    // LOCAL FIX (2026-10-02): a SECOND, distinct refusal reason. `flush_failed` means the Host arena
    // was full; this one means the Host store REFUSED to publish the batch (see the `publish declined`
    // line in host_kv_store.h -- it used to be a silent `std::terminate`). They need different
    // responses from a reader: one says "raise --host-kv-mib", the other says "a page could not be
    // attached", so folding them into one counter would hide which one is happening.
    std::uint32_t flush_declined = 0;
    std::vector<LogicalKVPageHandle> batch;
    batch.reserve(64);
    const auto flush = [&]() -> bool {
        if (batch.empty()) { return true; }
        std::optional<HostKVExtentReservation> reserved = host_kv_extents->prepare(pages, batch);
        if (!reserved) { ++flush_failed; return false; }   // Host arena full: nothing more can be moved
        const std::vector<DeviceKVPageHandle> sources = host_kv_extents->device_sources(*reserved);
        // Copy on the compute stream so the D2H is ordered after the kernels that wrote them.
        pages.physical_pool().copy_to_host(sources, host_kv_extents->writable_view(*reserved),
                                           device.stream);
        // LOCAL FIX (2026-10-02): publish declines instead of terminating, so this batch simply does
        // not happen -- every Device page in it keeps its replica and its active reference (the loop
        // below never runs), i.e. the data is intact and the ring can try again later. Same shape as
        // the `prepare` refusal just above, with its own counter so the log names the real reason.
        if (!host_kv_extents->publish(std::move(*reserved))) {
            ++flush_declined;
            return false;
        }
        for (const LogicalKVPageHandle logical : batch) {
            pages.release_active_reference(logical);
            if (!pages.drop_device_replica(logical)) { return false; }
        }
        freed += static_cast<std::uint32_t>(batch.size());
        batch.clear();
        return true;
    };
    addresses.for_each_address([&](const KVAddressSpaceHandle& handle, bool active,
                                   std::uint32_t mapped) {
        if (freed >= target_free || active || handle == skip) { return; }
        ++slots_visited;
        for (std::uint32_t page = 0; page < mapped && freed + batch.size() < target_free; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(handle, page);
            if (!pages.valid(logical)) { ++skip_invalid; continue; }
            if (!pages.device_resident(logical)) { ++skip_noresident; continue; }
            if (pages.source_pins(logical) != 0) { ++skip_pins; continue; }
            // LOCAL FIX (ring decode; prescription: 交棒-ninfer解耦第四棒 §4.3): only a page that can
            // never receive another coverage commit may give up its Device replica. A partially
            // committed page keeps a live commit path (commit_coverage requires a Device replica and
            // writer_references == 1), so demoting it made the next commit throw
            // "logical KV committed coverage is not monotonic". Measured 2026-09-30 with pool 16
            // pages < prompt+output during decode.
            if (pages.committed_columns(logical) != static_cast<std::uint32_t>(kPagedKVPageSize)) {
                ++skip_partial;
                continue;
            }
            // NOTE: snapshot protection (protected_columns) is deliberately NOT consulted here.
            // Protection keeps pages resident for fork-based reuse, and reuse of a host-demoted
            // source is declined in ring mode. Demoting to the Host preserves the content and the
            // descriptor, so the checkpoint stays restorable; only the Device replica goes.
            if (pages.host_resident(logical)) {
                // Content already on the Host: just give up the Device replica.
                if (!pages.host_replica_current(logical)) { ++skip_stalehost; continue; }
                pages.set_writer(logical, false);
                pages.release_active_reference(logical);
                if (pages.drop_device_replica(logical)) { ++freed; }
                continue;
            }
            // Content not on the Host yet: the extent store pins through `can_pin_source`, which
            // needs the writer reference clear (inactive addresses normally hold none, but be safe).
            pages.set_writer(logical, false);
            // LOCAL FIX (2026-10-02, step 1 of the publish patch -- the ROOT-CAUSE half):
            // a logical page must not enter the SAME batch twice. fork/COW pages are reference-counted,
            // so one logical page can be reached through two non-live addresses; the batch then asks
            // the Host store to attach two replicas for it, the second attach throws, and with the old
            // `noexcept -> std::terminate` publish the whole batch (up to 64 pages) died with it --
            // `abort()`, no log, no dump (K's ebd7ad46). Skipping the duplicate REMOVES the cause; the
            // decline path in publish() is only the safety net behind it.
            if (std::find(batch.begin(), batch.end(), logical) != batch.end()) {
                ++skip_duplicate;
                continue;
            }
            batch.push_back(logical);
            if (batch.size() >= 64 && !flush()) { return; }
        }
    });
    // The tail flush is the last chance to bank the pending batch; its return value used to be
    // dropped here, so a failure at this exact point was indistinguishable from "no candidates".
    // flush() now counts its own refusals in flush_failed, tail call included.
    (void)flush();
    // LOCAL DIAGNOSTIC (demote-counters): when inactive addresses cannot cover the
    // deficit the live lane cannibalizes its own bankable head, so a bank-miss
    // needs to know WHY the dead residents would not drain. Name the blocker.
    if (freed < target_free) {
        std::fprintf(stderr,
                     "[ninfer] demote-other short: freed=%u want=%u slots=%u invalid=%u "
                     "noresident=%u pins=%u stalehost=%u partial=%u dup=%u flushfail=%u declined=%u pending=%zu\n",
                     freed, target_free, slots_visited, skip_invalid, skip_noresident,
                     skip_pins, skip_stalehost, skip_partial, skip_duplicate, flush_failed,
                     flush_declined, batch.size());
    }
    return freed;
}

std::uint32_t ProgramImpl::demote_kv_pages_to_host(KVAddressSpaceStore& addresses,
                                                   LogicalKVPageStore& pages,
                                                   const KVAddressSpaceHandle& address,
                                                   std::uint32_t sink_pages,
                                                   std::uint32_t target_free,
                                                   std::span<const std::uint32_t> preferred) {
    if (host_kv_extents == nullptr || !addresses.valid(address) || target_free == 0) { return 0; }
    const std::uint32_t mapped = addresses.mapped_pages(address);
    const auto is_preferred = [&](std::uint32_t page) {
        return !preferred.empty() && std::binary_search(preferred.begin(), preferred.end(), page);
    };
    // Oldest first, skipping the attention sink and (in the first pass) the pages retrieval wants to
    // keep. If that is not enough the second pass gives up the retrieved pages too, so the caller is
    // guaranteed to get the room it asked for whenever the Device can supply it at all.
    std::uint32_t demoted   = 0;
    std::uint32_t page      = sink_pages;
    bool second_pass        = false;
    while (demoted < target_free) {
        if (page >= mapped) {
            if (second_pass) { break; }
            second_pass = true;
            page        = sink_pages;
            continue;
        }
        const std::uint32_t current       = page;
        const LogicalKVPageHandle logical = addresses.logical_page(address, current);
        ++page;
        if (!second_pass && is_preferred(current)) { continue; }
        if (!pages.valid(logical) || !pages.device_resident(logical) ||
            pages.source_pins(logical) != 0) {
            continue;
        }
        // LOCAL FIX (ring decode; prescription: 交棒-ninfer解耦第四棒 §4.3): only a page that can never
        // receive another coverage commit may be demoted. The tail page the decode step is still
        // writing (or the current prefill chunk's pages) is partially committed and keeps a live
        // commit path; demoting it made the next commit throw "logical KV committed coverage is not
        // monotonic" (measured 2026-09-30: pool 16 pages, context grew past the pool during decode).
        if (pages.committed_columns(logical) != static_cast<std::uint32_t>(kPagedKVPageSize)) {
            continue;
        }
        if (!pages.host_resident(logical)) {
            // The content is not on the Host yet: copy it there first. The Host extent store pins
            // through `can_pin_source`, which requires the writer reference to be clear; an active
            // sequence keeps one on every mapped page and these pages are outside the window.
            pages.set_writer(logical, false);
            std::vector<LogicalKVPageHandle> one{logical};
            std::optional<HostKVExtentReservation> reserved = host_kv_extents->prepare(pages, one);
            if (!reserved) { break; }   // Host arena full: nothing more can be moved
            const std::vector<DeviceKVPageHandle> sources =
                host_kv_extents->device_sources(*reserved);
            // Copy on the compute stream so the D2H is ordered after the kernels that wrote it.
            pages.physical_pool().copy_to_host(sources, host_kv_extents->writable_view(*reserved),
                                               device.stream);
            // LOCAL FIX (2026-10-02): a declined publish stops the demotion exactly like the `prepare`
            // refusal above -- `break` skips `release_active_reference` below, so this page keeps its
            // Device replica and its data. Never drop a replica that has no Host copy.
            if (!host_kv_extents->publish(std::move(*reserved))) {
                // No counter here on purpose: this function has no summary line to carry it, and the
                // store already named the page and the reason in its own `publish declined` line.
                break;
            }
        } else {
            // Already preserved on the Host (demoted before and then retrieved): no transfer needed.
            if (!pages.host_replica_current(logical)) { continue; }
            pages.set_writer(logical, false);
        }
        pages.release_active_reference(logical);
        if (!pages.drop_device_replica(logical)) { continue; }
        ++demoted;
    }
    return demoted;
}

namespace {
// LOCAL PROTOTYPE: the retrieval policy's placeholder window, read once from the environment so
// the mask path can be exercised end to end without a CLI change.
//   NINFER_KV_WINDOW=<tokens>  keep only the newest N tokens on the Device (0/unset = dense)
//                             the effective window shrinks to fit the Device pool
// LOCAL PROTOTYPE (KVMem-style ring): a preserved attention "sink" is deliberately NOT supported,
// because the ring demotes the oldest pages first; a sink would read a recycled Device slot.
std::uint32_t proto_env_pages(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr) { return 0; }
    const long tokens = std::strtol(text, nullptr, 10);
    if (tokens <= 0) { return 0; }
    return static_cast<std::uint32_t>(tokens / static_cast<long>(kPagedKVPageSize));
}
std::uint32_t proto_kv_window_pages() {
    static const std::uint32_t pages = proto_env_pages("NINFER_KV_WINDOW");
    return pages;
}

std::uint32_t proto_pages_for_tokens(std::uint32_t tokens) {
    if (tokens == 0) { return 0; }
    return (tokens + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) /
           static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t proto_kv_sink_pages() {
    // StreamingLLM-style attention sink: the first pages are never demoted and stay selected.
    static const std::uint32_t pages = [] {
        const char* text = std::getenv("NINFER_KV_SINK");
        if (text == nullptr) { return 4U; }   // 4 pages = 256 tokens
        const long tokens = std::strtol(text, nullptr, 10);
        if (tokens <= 0) { return 0U; }
        return static_cast<std::uint32_t>(tokens / static_cast<long>(kPagedKVPageSize));
    }();
    return pages;
}

std::uint32_t proto_kv_retrieve_pages() {
    // Retrieval budget: how many Host-resident pages to bring back on the basis of relevance.
    static const std::uint32_t pages = proto_env_pages("NINFER_KV_RETRIEVE");
    return pages;
}
// B32 (2026-10-07): is the retrieval budget literally UNSET? proto_env_pages cannot tell "unset" from
// an explicit 0 (both return 0), and an explicit value must stay authoritative -- a switch the operator
// set and the engine silently overrode is the defect class B25 was about ("=0 must mean 0"). The
// fallback below therefore fires only when the variable is absent or empty.
bool retrieve_budget_unset() {
    const char* text = std::getenv("NINFER_KV_RETRIEVE");
    return text == nullptr || text[0] == '\0';
}

// B32 (2026-10-07): the off switch for the retrieval fallback added at its call site. Unset -- or any
// value whose first character is not '0' -- keeps the fallback ON, matching how this file reads every
// other switch. It exists only so the fallback can be turned off for a negative control.
bool retrieve_auto_enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("NINFER_KV_RETRIEVE_AUTO");
        return text == nullptr || text[0] == '\0' || text[0] != '0';
    }();
    return enabled;
}

// LOCAL FIX (ring retrieval budget, 2026-10-02): the budget used to be a hard-wired quarter of the
// pool (`pool / 4U`), and in the delivered shape that quarter is the BINDING term -- pool = 280 pages
// in the shipped tier (`--kv-capacity 17920`), so the budget was min(128 pages from
// NINFER_KV_RETRIEVE=8192, 70) = SEVENTY pages out of 636 mapped. A page whose only overlap with the
// query is ordinary body vocabulary ranks just below the cut, so "the page the question is about"
// was admitted or dropped essentially by luck -- measured: 4 hits in 6 trials for a question that is
// semantically the same but shares no rare word with the marked line.
// The share is now a percentage of the pool with the same default as the old behaviour times two,
// and -- crucially -- it is read from the ENVIRONMENT so that tuning it costs a server restart
// instead of a full rebuild of this tree (an edit to this file alone is cheap; an edit to anything
// the CUDA wrappers include is not). 0 or >100 is ignored; NINFER_KV_RETRIEVE still caps it.
std::uint32_t proto_env_percent(const char* name, std::uint32_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr) { return fallback; }
    const long value = std::strtol(text, nullptr, 10);
    if (value <= 0 || value > 100) { return fallback; }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t proto_kv_retrieve_share_percent() {
    static const std::uint32_t percent = proto_env_percent("NINFER_KV_RETRIEVE_SHARE", 50U);
    return percent;
}

// LOCAL PROTOTYPE (KVMem-style ring): the retrieval mask IS the Device-resident set. Pages past
// what has been mapped are selected too (the current chunk is writing into them). Building the mask
// page by page rather than as a range is what makes a "hole" left by a page that could not be
// demoted safe: a range would expose that hole's stale Device slot.
// LOCAL DIAGNOSTIC (mask-hidden): the mask IS the attention window, so a mapped page that is not
// Device-resident is silently unreadable -- that is the "middle of the prompt disappears" shape.
// Count exactly those pages and name the caller. `which` is the stream (text/backend), `site` is
// the installation point (mapped = the full ring mask, commit = the conservative safety net); both
// are pure labels. Nothing about the selection itself changes.
void install_resident_selection(qwen3_5::PagedKVCache* cache, std::uint32_t span_pages,
                                const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages,
                                const KVAddressSpaceHandle& address, const char* which,
                                const char* site) {
    if (cache == nullptr || span_pages == 0 || !addresses.valid(address)) { return; }
    const std::size_t words = (static_cast<std::size_t>(span_pages) + 31U) / 32U;
    std::vector<std::uint32_t> bits(words, 0U);
    const std::uint32_t mapped = addresses.mapped_pages(address);
    std::uint32_t resident = 0, hidden = 0, first_hidden = 0, last_hidden = 0;
    // Where the prompt's middle is: page index in [35%, 70%) of the span handed to this mask.
    // Printed with its bounds so the reading needs no recomputation (page P starts at token P*64).
    const std::uint32_t mid_lo = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(span_pages) * 35U) / 100U);
    const std::uint32_t mid_hi = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(span_pages) * 70U) / 100U);
    std::uint32_t mid_hidden = 0;
    for (std::uint32_t page = 0; page < span_pages; ++page) {
        const bool selected =
            page >= mapped || pages.device_resident(addresses.logical_page(address, page));
        if (selected) {
            bits[page >> 5] |= (1U << (page & 31U));
            ++resident;
        } else {
            if (hidden == 0) { first_hidden = page; }
            last_hidden = page;
            ++hidden;
            if (page >= mid_lo && page < mid_hi) { ++mid_hidden; }
        }
    }
    cache->install_selection(bits);
    // Only a non-zero reading is news: a mask that hides nothing stays silent (no per-step spam).
    if (hidden != 0) {
        std::fprintf(stderr,
                     "[ninfer] mask hidden: which=%s site=%s span=%u resident=%u hidden=%u "
                     "first_hidden=%u last_hidden=%u mid_hidden=%u mid_lo=%u mid_hi=%u\n",
                     which, site, span_pages, resident, hidden, first_hidden, last_hidden,
                     mid_hidden, mid_lo, mid_hi);
    }
}
} // namespace

void ProgramImpl::ensure_sequence_kv_lease(SequenceState& sequence, std::uint32_t main_tokens,
                                           std::uint32_t backend_tokens) {
    if (!context_cache.kv_lease_growth) { return; }
    // Only a decode step may extend the lease: materialization during admission has to leave the
    // sequence holding exactly the entitlement its plan declared.
    RequestControl& request = requests[sequence.lane];
    if (request.lifecycle != Lifecycle::Active) { return; }
    const std::uint32_t cushion    = kv_lease_cushion_pages();
    const std::uint32_t text_pages = text_kv_addresses->entitlement(sequence.kv->text);
    const std::uint32_t backend_pages =
        sequence.kv->backend ? backend_kv_addresses->entitlement(*sequence.kv->backend) : 0U;
    // The most a lease can use: every frontier up to the request's ceiling plus the drafts a
    // round may still verify past it. A lease that holds this needs no cushion beyond it, so a
    // full pool near the ceiling is not mistaken for a shortfall that ends the answer early.
    const std::uint32_t backend_ceiling =
        std::min(capacity, request.lease_ceiling + kv_lease_backend_allowance_tokens());
    const std::uint32_t reach = kv_pages_for_tokens(backend_ceiling);
    const bool main_thin =
        text_pages < reach && kv_pages_for_tokens(main_tokens) + cushion > text_pages;
    const bool backend_thin = sequence.kv->backend.has_value() && backend_tokens != 0 &&
                              backend_pages < reach &&
                              kv_pages_for_tokens(backend_tokens) + cushion > backend_pages;
    if (!main_thin && !backend_thin) { return; }

    // A full window first; when the pool cannot spare one, a step-sized extension still leaves
    // the cushion. No rung may exceed its pool or what one address can map (the cushion would
    // otherwise overshoot the context limit), so the reservation can only fail on space.
    // The ladder ends at one page group: a pool that can only spare its own allocation
    // granularity would otherwise settle with free page groups that no coarser rung can use.
    const auto target = [reach](std::uint32_t cap, std::uint32_t pages, std::uint32_t wanted) {
        return std::min(cap, std::max(pages, std::min(reach, wanted)));
    };
    const auto targets = [&](std::uint32_t extra_tokens) {
        return DeviceKVPages{
            .main = target(std::min(text_kv_pages->physical_pool().capacity_pages(),
                                    text_kv_addresses->page_capacity()),
                           text_pages,
                           kv_lease_pages_for_tokens(
                               std::min(request.lease_ceiling, main_tokens + extra_tokens))),
            .backend =
                sequence.kv->backend
                    ? target(std::min(backend_kv_pages->physical_pool().capacity_pages(),
                                      backend_kv_addresses->page_capacity()),
                             backend_pages,
                             backend_thin ? kv_lease_pages_for_tokens(std::min(
                                                backend_ceiling, backend_tokens + extra_tokens))
                                          : backend_pages)
                    : 0U,
        };
    };
    const auto grow = [&](DeviceKVPages wanted) {
        if ((main_thin && wanted.main <= text_pages) ||
            (backend_thin && wanted.backend <= backend_pages)) {
            return false;
        }
        resize_sequence_kv_entitlement(sequence, wanted.main, wanted.backend);
        // The extended lease is this owner's resource effect: keep the accounting the population
        // proofs read in step with it.
        request.active_resources.device.main_kv_pages += wanted.main - text_pages;
        request.active_resources.device.backend_kv_pages += wanted.backend - backend_pages;
        return true;
    };
    const std::uint32_t page = static_cast<std::uint32_t>(kPagedKVPageSize);
    bool space_limited       = false;
    for (const std::uint32_t extra_tokens : {kv_lease_growth_margin_tokens(), 2U * page, page}) {
        try {
            if (grow(targets(extra_tokens))) { return; }
        } catch (const std::bad_alloc&) { space_limited = true; }
    }

    // The pool cannot extend this lease. The sequence settles where the lease still covers a
    // step, so no launch can fail coverage mid-round. A space settlement records what its steps
    // asked for, so the engine can give it retained cache pages and resume growth before the
    // request's budget is bounded.
    request.lease_settled        = true;
    request.lease_space_limited  = space_limited;
    request.lease_minimum_target = targets(page);
}

std::optional<std::uint32_t>
ProgramImpl::device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                               std::uint32_t forced_span_tokens) const noexcept {
    if (!valid_sequence(sequence)) { return std::nullopt; }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity ||
        !requests[lane].lease_settled) {
        return std::nullopt;
    }
    const SequenceState& state = active_sequence(lane);
    if (!state.kv) { return std::nullopt; }
    // The settle keeps the widest single step and the caller's forced control span unused, so
    // every launch it still licenses stays inside the lease.
    const std::uint32_t step = std::max(widest_verify_window() + 1U, forced_span_tokens);
    const std::uint32_t covered =
        std::min(state.kv->backend
                     ? kv_tokens_for_pages(backend_kv_addresses->entitlement(*state.kv->backend))
                     : std::numeric_limits<std::uint32_t>::max(),
                 kv_tokens_for_pages(text_kv_addresses->entitlement(state.kv->text)));
    const std::uint32_t frontier = state.execution_frontier;
    const std::uint32_t slack    = covered > frontier + step ? covered - frontier - step : 0U;
    const std::uint32_t forced   = forced_span_tokens == 0 ? 0U : forced_span_tokens + 1U;
    return std::max(std::max(1U, slack), forced);
}

std::optional<DeviceKVLeaseShortfall>
ProgramImpl::device_kv_lease_shortfall(SequenceHandle sequence) const noexcept {
    if (!valid_sequence(sequence)) { return std::nullopt; }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
        return std::nullopt;
    }
    const RequestControl& request = requests[lane];
    if (!request.lease_settled || !request.lease_space_limited) { return std::nullopt; }
    const SequenceState& state = active_sequence(lane);
    if (!state.kv) { return std::nullopt; }
    const auto missing = [](std::uint32_t wanted, std::uint32_t held, std::uint32_t available) {
        const std::uint32_t need = wanted > held ? wanted - held : 0U;
        return need > available ? need - available : 0U;
    };
    const std::uint32_t text_held = text_kv_addresses->entitlement(state.kv->text);
    const std::uint32_t text_free = text_kv_pages->physical_pool().available_pages();
    DeviceKVLeaseShortfall out{
        .main_pages = missing(request.lease_minimum_target.main, text_held, text_free),
    };
    if (state.kv->backend && backend_kv_addresses && backend_kv_pages) {
        const std::uint32_t held = backend_kv_addresses->entitlement(*state.kv->backend);
        const std::uint32_t free = backend_kv_pages->physical_pool().available_pages();
        out.backend_pages        = missing(request.lease_minimum_target.backend, held, free);
    }
    return out;
}

bool ProgramImpl::resume_device_kv_lease(SequenceHandle sequence) noexcept {
    const std::optional<DeviceKVLeaseShortfall> shortfall = device_kv_lease_shortfall(sequence);
    if (!shortfall || shortfall->main_pages != 0 || shortfall->backend_pages != 0) {
        return false;
    }
    RequestControl& request      = requests[ContractAccess::lane(sequence).value];
    request.lease_settled        = false;
    request.lease_space_limited  = false;
    request.lease_minimum_target = {};
    return true;
}

DeviceKVPages
ProgramImpl::retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return {}; }
    try {
        const detail::PhysicalResources owned =
            owner_exclusive_resources(continuation_states[ContractAccess::index(continuation)]);
        return {.main = owned.device.main_kv_pages, .backend = owned.device.backend_kv_pages};
    } catch (...) { return {}; }
}

DeviceKVPages
ProgramImpl::retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept {
    if (!valid_shared_prefix(shared)) { return {}; }
    try {
        const detail::PhysicalResources owned =
            owner_exclusive_resources(shared_prefix_states[ContractAccess::index(shared)]);
        return {.main = owned.device.main_kv_pages, .backend = owned.device.backend_kv_pages};
    } catch (...) { return {}; }
}

// LOCAL FIX (B01 ring room): this beat used to live inside ensure_sequence_kv_mapped as a local
// lambda, so the two other places that map Device KV pages without going through it -- the DFlash
// context append (decode.cpp) and the causal-scoring lane (program_impl.cpp) -- asked the pool for
// pages without ever demoting, and hit the paged-KV reservation invariant once the pool was full
// with nothing demotable left. Same logic, one home, callable from every mapping site, with the
// early verdict inside so no caller can forget it.
void ProgramImpl::ensure_ring_room(KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                   const KVAddressSpaceHandle& address, std::uint32_t total_tokens,
                                   std::span<const std::uint32_t> preferred) {
    // No-op unless the ring is active: with a pool that covers the logical context every page is
    // resident anyway, and a demotion pass would be pure overhead on the hot path.
    if (proto_kv_window_pages() == 0 ||
        pages.physical_pool().usable_pages() >= addresses.logical_page_capacity()) {
        return;
    }
    // Keep a few pages of slack so the pool never runs at exactly 100%: the restore path always
    // needs a free Device page to work with.
    constexpr std::uint32_t kRingSlackPages = 8U;
    const auto free_pages                   = [](const LogicalKVPageStore& store) {
        const std::uint32_t usable    = store.physical_pool().usable_pages();
        const std::uint32_t allocated = store.physical_pool().allocated_pages();
        return usable > allocated ? usable - allocated : 0U;
    };
    const std::uint32_t sink_pages = proto_kv_sink_pages();
    const std::uint32_t total      = proto_pages_for_tokens(total_tokens);
    const std::uint32_t mapped     = addresses.mapped_pages(address);
    const std::uint32_t need       = total > mapped ? total - mapped : 0U;
    std::uint32_t wanted_extra     = 0;
    for (const std::uint32_t page : preferred) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.valid(logical) && !pages.device_resident(logical) &&
            pages.host_resident(logical)) {
            ++wanted_extra;   // a preferred page has to be brought back
        }
    }
    const std::uint32_t want_free   = need + kRingSlackPages + wanted_extra;
    const std::uint32_t free_now    = free_pages(pages);
    const std::uint32_t target_free = want_free > free_now ? want_free - free_now : 0U;
    if (target_free == 0) {
        (void)demote_kv_pages_to_host(addresses, pages, address, sink_pages, 0U, preferred);
        return;
    }
    // Cold pages first: inactive addresses are pure cache, so drain them before touching this
    // sequence's own resident window.
    const std::uint32_t freed_cold =
        demote_other_addresses_to_host(addresses, pages, address, target_free);
    const std::uint32_t free_after = free_pages(pages);
    const std::uint32_t target2 = want_free > free_after ? want_free - free_after : 0U;
    const std::uint32_t freed_own =
        demote_kv_pages_to_host(addresses, pages, address, sink_pages, target2, preferred);
    // LOCAL FIX (B01 ring early verdict): both demotions used to have their return value dropped, so
    // a ring that could free nothing carried on into `ensure_mapped_to_tokens` and failed later --
    // as the paged-KV reservation invariant, as an engine death, or (measured 2026-10-04, community
    // report) as a request that never returns while the prefill loop retries block after block.
    // Decide HERE, where the numbers are known.
    // `need` is the hard requirement: pages this step must map. The slack and the preferred restores
    // are best effort (retrieval quality), so falling short of `want_free` warns instead of failing.
    const std::uint32_t free_final = free_pages(pages);
    if (free_final < need) {
        // ContextCacheExhausted is a std::bad_alloc (include/ninfer/types.h:867), so the worker's
        // out-of-memory path fails this request and keeps serving -- it is not gated by
        // --recover-invariant-failures. This is the readable error the task book's F1 asks for
        // instead of a silent stall or a dead engine.
        throw ninfer::ContextCacheExhausted(
            "KVMem ring cannot free Device KV pages: this step needs " + std::to_string(need) +
            " free, " + std::to_string(free_final) + " available (target " +
            std::to_string(want_free) + " = need + slack " + std::to_string(kRingSlackPages) +
            " + restores " + std::to_string(wanted_extra) + "; freed cold " +
            std::to_string(freed_cold) + ", own " + std::to_string(freed_own) + "; pool usable " +
            std::to_string(pages.physical_pool().usable_pages()) + ", allocated " +
            std::to_string(pages.physical_pool().allocated_pages()) + ")");
    }
    if (free_final < want_free) {
        std::fprintf(stderr,
                     "[ninfer] ring short: freed=%u want=%u need=%u free=%u cold=%u own=%u -> "
                     "continuing without slack\n",
                     freed_cold + freed_own, want_free, need, free_final, freed_cold, freed_own);
    }
}

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    // ADAPTATION (master keeps kv-lease): K's tree has no kv_lease mechanism at all (zero hits for
    // kv_lease in K), so K's five-beat body never extends a lease and pasting it as-is would
    // silently disable master's own lease-growth behaviour. This port replaces only the MAPPING
    // beat; master's lease accounting is untouched. Kept FIRST because master ran it before any
    // mapping and because the beats below consume what it decides (beat 1 scores over the mapped
    // range, ensure_ring_room computes need = total - mapped, beat 3 maps into the entitlement it
    // sized).
    ensure_sequence_kv_lease(sequence, main_tokens, backend_tokens);
    // LOCAL PROTOTYPE (KVMem-style ring): when a Device pool is smaller than the logical context a
    // sequence cannot keep all of its KV on the Device. Each call then:
    //   1. ranks the pages by relevance to the recent context (retrieval preference),
    //   2. demotes the oldest pages until the pool has room for this call's pages plus slack (and for
    //      the pages retrieval wants to bring back); the preferred pages are given up last,
    //   3. maps the pages this call needs,
    //   4. brings the preferred pages back onto the Device (H2D),
    //   5. installs the mask over EXACTLY the resident set.
    // The dense configuration (pool >= logical capacity, or no window configured) never enters here.
    const std::uint32_t text_total    = proto_pages_for_tokens(main_tokens);
    const std::uint32_t backend_total = proto_pages_for_tokens(backend_tokens);
    const std::uint32_t window_pages  = proto_kv_window_pages();
    const std::uint32_t sink_pages    = proto_kv_sink_pages();
    const bool has_backend =
        backend_tokens != 0 && sequence.kv->backend && backend_kv_addresses && backend_kv_pages;
    const bool ring = window_pages != 0 && text_kv_pages != nullptr && text_kv_addresses != nullptr &&
                      text_kv_pages->physical_pool().usable_pages() <
                          text_kv_addresses->logical_page_capacity();
    std::vector<std::uint32_t> preferred;
    if (ring) {
        // Retrieval preference: a share of the pool (default half -- see the LOCAL FIX comment on
        // proto_kv_retrieve_share_percent), scored on the whole mapped range.
        // Scoring runs only when the mapped page count changed (relevance shifts slowly; within one
        // page the previous selection stays valid). Between rescores the hysteresis set is reused.
        const std::uint32_t pool   = text_kv_pages->physical_pool().usable_pages();
        const std::uint32_t share  = pool * proto_kv_retrieve_share_percent() / 100U;
        std::uint32_t budget       = std::min(proto_kv_retrieve_pages(), share);        // ---- B32 (2026-10-07): the retrieval budget had NO default. ---------------------------------
        // WHY THIS EXISTS: proto_env_pages returns 0 when NINFER_KV_RETRIEVE is unset, and NOTHING in
        // this tree sets it (no launcher, script or doc), so the shipped default is budget = 0: the
        // ring demotes pages and never brings any back. Measured on this build (2026-10-07): such a
        // request answers out of a MASKED middle with HTTP 200 and a plausible wrong answer -- the
        // exact condition generation_service.cpp warns about but does not refuse. The same
        // measurement with the budget supplied by hand: 0/6 correct at budget 0, 6/6 at 8192 tokens
        // on a content-distinct fixture. A fix that ships switched off is a fix the user does not
        // have (startup.cpp:1239), which is why the default below is ON.
        // THE CONDITION IS DELIBERATELY NARROW: only when the caller left NINFER_KV_RETRIEVE unset
        // (budget 0), only on the ring path, and only when this request's own KV (text_total pages)
        // exceeds the pool. An explicit NINFER_KV_RETRIEVE -- including 0 -- is respected untouched.
        // THE CAP RESPECTS THE RESTORE PATH'S SLACK: ensure_ring_room keeps free pages because a
        // restore needs a free slot (context.cpp:2305), so the fallback stops short of the pool
        // instead of taking every free page.
        if (budget == 0U && retrieve_budget_unset() && retrieve_auto_enabled() && text_total > pool) {
            constexpr std::uint32_t kRestoreSlackPages = 4U;
            const std::uint32_t room =
                pool > sink_pages + kRestoreSlackPages ? pool - sink_pages - kRestoreSlackPages : 0U;
            const std::uint32_t auto_budget = std::min(share, room);
            if (auto_budget != 0U) {
                budget = auto_budget;
                std::fprintf(stderr,
                             "[ring] retrieve AUTO-ENABLED: prompt %u pages > pool %u pages and "
                             "NINFER_KV_RETRIEVE is unset -> budget %u pages (skeleton %u, "
                             "restore-slack %u). Set NINFER_KV_RETRIEVE to pick a budget, or "
                             "NINFER_KV_RETRIEVE_AUTO=0 for the old (budget 0) behaviour.\n",
                             text_total, pool, budget, sink_pages, kRestoreSlackPages);
            }
        }
        // ---- P1-a BUDGET LINE (2026-10-05): the pool's split, in one auditable row. --------------
        // WHY: the plan's P1-a asks for "[ring] budgets: skeleton=S, evidence=E, recent=R, free=F
        // (pool P)" so that the NEXT person can see where the KV budget went without reading code.
        // WHAT EACH TERM IS, exactly (no invented quantities):
        //   skeleton = the attention-sink pin (proto_kv_sink_pages) -- the only region the demotion
        //              scan refuses to touch (context.cpp:1946/1952). Measured need for a real agent
        //              surface: 2,808 tokens = 44 pages (see 实验-骨架体量实测-20261005.md).
        //   evidence = the retrieval budget, min(NINFER_KV_RETRIEVE, pool * SHARE%) -- the pages the
        //              ring may bring back; it is taken FROM the pool by construction.
        //   recent   = what is left for the recency window and the sequence's own frontier.
        //   free     = pool - everything above.
        // The identity S + E + R + F == P therefore holds BY CONSTRUCTION, which is what makes this
        // row a self-check rather than decoration: if a reader ever sees the four not summing to the
        // pool, one of the terms was redefined without updating the others.
        // Logged once per request (the ring path is per-request), not per chunk, to avoid flooding.
        {
            const std::uint32_t b_skeleton = sink_pages < pool ? sink_pages : pool;
            const std::uint32_t b_evidence = budget < pool - b_skeleton ? budget : pool - b_skeleton;
            const std::uint32_t b_recent   = 0U;   // recency is whatever the request's own frontier maps
            const std::uint32_t b_free     = pool - b_skeleton - b_evidence - b_recent;
            std::fprintf(stderr,
                         "[ring] budgets: skeleton=%u, evidence=%u, recent=%u, free=%u (pool %u) "
                         "[skeleton+evidence+recent+free == pool: %s]\n",
                         b_skeleton, b_evidence, b_recent, b_free, pool,
                         (b_skeleton + b_evidence + b_recent + b_free) == pool ? "yes" : "NO");
        }
        const std::uint32_t mapped_now = text_kv_addresses->mapped_pages(sequence.kv->text);
        if (budget != 0 && mapped_now > sink_pages &&
            (mapped_now != sequence.ring_scored_mapped || sequence.ring_keep.empty())) {
            preferred = select_retrieval_pages(sequence, sink_pages, mapped_now, budget);
            sequence.ring_keep          = preferred;
            sequence.ring_scored_mapped = mapped_now;
        } else if (mapped_now == sequence.ring_scored_mapped) {
            preferred = sequence.ring_keep;
        } else {
            // Fresh or shrunken range (e.g. a recycled lane): no hysteresis applies.
            preferred.clear();
            sequence.ring_keep.clear();
            sequence.ring_scored_mapped = mapped_now;
        }
        // The hysteresis set may reference pages beyond the current mapping (stale lane reuse);
        // clamp it so logical_page() below can never throw.
        preferred.erase(std::remove_if(preferred.begin(), preferred.end(),
                                       [&](std::uint32_t page) { return page >= mapped_now; }),
                        preferred.end());
        // B01: room is made by the shared helper, which every mapping site now uses. It takes the
        // token frontier (proto_pages_for_tokens converts) and decides right here -- with these exact
        // numbers -- whether the pages this step must map can be freed at all.
        ensure_ring_room(*text_kv_addresses, *text_kv_pages, sequence.kv->text, main_tokens,
                         preferred);
        if (has_backend) {
            ensure_ring_room(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                             backend_tokens, {});
        }
    }
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, main_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, backend_tokens,
                                                      device.stream);
    }
    if (ring) {
        const std::vector<std::uint32_t> restored = restore_kv_pages_from_host(
            *text_kv_addresses, *text_kv_pages, sequence.kv->text, preferred);
        // LOCAL DIAGNOSTIC (ring retrieve): `preferred` vs `restored` separates three different
        // worlds that used to look identical from the outside --
        //   restored == preferred   -> every wanted page is Device-resident now
        //   restored  < preferred   -> the restore ran out of free slots or host arena (it walks
        //                              ascending and BREAKS, so an over-large budget loses its HIGH
        //                              tail -- that is the failure mode a budget change can create)
        //   preferred == 0          -> nothing was wanted at all (budget 0, or every page scored 0)
        // Only printed when the pair changes, so a long run does not emit one line per step.
        {
            // LOCAL FIX (defect D-10, 2026-10-02): the de-dup key used to be the PAIR OF SIZES only, so a
            // round that restored a DIFFERENT SET of the same size printed nothing -- and the printed
            // first/last then belonged to the first occurrence of that size, which invites a wrong
            // conclusion ("page X was never restored"; that exact mistake was made once already).
            // first/last are part of the key now.
            static std::size_t last_preferred = 0;
            static std::size_t last_restored  = 0;
            static std::uint32_t last_first   = 0;
            static std::uint32_t last_last    = 0;
            const std::uint32_t first_page = restored.empty() ? 0U : restored.front();
            const std::uint32_t last_page  = restored.empty() ? 0U : restored.back();
            if (preferred.size() != last_preferred || restored.size() != last_restored ||
                first_page != last_first || last_page != last_last) {
                last_preferred = preferred.size();
                last_restored  = restored.size();
                last_first     = first_page;
                last_last      = last_page;
                std::fprintf(stderr,
                             "[ninfer] ring retrieve: which=text preferred=%zu restored=%zu "
                             "first=%u last=%u\n",
                             preferred.size(), restored.size(),
                             restored.empty() ? 0U : restored.front(),
                             restored.empty() ? 0U : restored.back());
            }
        }
        // The mask IS the resident set, so a page that could not be demoted or restored can only be
        // hidden; it can never expose a stale Device slot.
        install_resident_selection(decoder ? &decoder->text_kv : nullptr, text_total,
                                   *text_kv_addresses, *text_kv_pages, sequence.kv->text, "text",
                                   "mapped");
        if (has_backend) {
            install_resident_selection(backend_kv_cache(), backend_total, *backend_kv_addresses,
                                       *backend_kv_pages, *sequence.kv->backend, "backend",
                                       "mapped");
        }
    }
}

void ProgramImpl::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                     std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_tokens);
    }
    // LOCAL PROTOTYPE: keep the retrieval masks in step with the committed frontier as well. This
    // is a CONSERVATIVE safety net (sink + the resident recent window, no retrieval); the full mask
    // is re-installed by ensure_sequence_kv_mapped before the next attention reads the cache.
    const std::uint32_t window_pages = proto_kv_window_pages();
    if (window_pages != 0 && text_kv_pages != nullptr && text_kv_addresses != nullptr &&
        text_kv_pages->physical_pool().usable_pages() <
            text_kv_addresses->logical_page_capacity()) {
        const bool has_backend =
            backend_tokens != 0 && sequence.kv->backend && backend_kv_addresses && backend_kv_pages;
        install_resident_selection(decoder ? &decoder->text_kv : nullptr,
                                   proto_pages_for_tokens(main_tokens), *text_kv_addresses,
                                   *text_kv_pages, sequence.kv->text, "text", "commit");
        if (has_backend) {
            install_resident_selection(backend_kv_cache(), proto_pages_for_tokens(backend_tokens),
                                       *backend_kv_addresses, *backend_kv_pages,
                                       *sequence.kv->backend, "backend", "commit");
        }
    }

    if (hybrid_) { hybrid_publish_blocks(sequence); }
}

void ProgramImpl::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImpl::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        text_kv_addresses->release_growth_entitlement(sequence.kv->text);
        if (sequence.kv->backend) {
            backend_kv_addresses->release_growth_entitlement(*sequence.kv->backend);
        }
    } catch (...) {}
}

void ProgramImpl::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv_strict(SequenceState& sequence,
                                             bool written_to_disk) noexcept {
    if (!sequence.kv || !text_kv_addresses || !text_kv_addresses->can_release(sequence.kv->text)) {
        std::terminate();
    }
    if (!written_to_disk) { spill_released_owner(sequence); }
    if (sequence.kv->backend &&
        (!backend_kv_addresses || !backend_kv_addresses->can_release(*sequence.kv->backend))) {
        std::terminate();
    }
    if (sequence.kv->backend && !backend_kv_addresses->release(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

qwen3_5::PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

qwen3_5::PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImpl::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}


} // namespace ninfer::models::qwen3_5::detail
