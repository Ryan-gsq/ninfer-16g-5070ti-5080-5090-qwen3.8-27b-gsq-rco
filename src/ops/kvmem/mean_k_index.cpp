#include "ops/kvmem/mean_k_index.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
// For kvmem_shadow_enabled(). Included explicitly rather than relied on transitively through
// raw_k_harvest.h -> raw_k_shadow.h -> kvmem_shadow.h: the switch is read in THIS file, and a
// transitive include that happens to hold today is not a contract (that reliance failed to compile
// the first time this function was added).
#include "ops/kvmem/kvmem_shadow.h"
#include "ops/kvmem/mean_k_index_launch.h"

#include <algorithm>
#include <stdexcept>
#include <string>

// KVMem mean-K index, host side. See mean_k_index.h for what it is, why it must be built
// incrementally, and why the stored medium is an F32 sum plus a token count rather than a mean.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "mean_k_index";

// The logical block IS one KV page, so the block size is the page size: a block has to stay
// identifiable while KVMem moves the pages that hold it. Reading the constant from the cache header
// rather than repeating "64" is the difference between a contract and a coincidence. (It lives in
// namespace ninfer, not ninfer::ops.)
constexpr std::int32_t kBlockTokens = ninfer::kPagedKVPageSize;

void require_layout(const Tensor& raw_key, std::int32_t kv_size) {
    if (raw_key.data == nullptr || raw_key.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(kSubject) + ": raw key must be a BF16 tensor");
    }
    if (raw_key.ne[0] != kv_size || raw_key.ne[1] <= 0 || raw_key.ne[2] != 1 || raw_key.ne[3] != 1) {
        throw std::invalid_argument(std::string(kSubject) + ": raw key must be {kv_size=" +
                                    std::to_string(kv_size) + ", width>0}, got {" +
                                    std::to_string(raw_key.ne[0]) + ", " +
                                    std::to_string(raw_key.ne[1]) + ", " +
                                    std::to_string(raw_key.ne[2]) + ", " +
                                    std::to_string(raw_key.ne[3]) + "}");
    }
}

std::size_t checked_bytes(std::int32_t layers, std::int32_t blocks, std::int32_t heads,
                          std::int32_t head_dim) {
    if (layers <= 0 || static_cast<std::size_t>(layers) > kMeanKMaxLayers) {
        throw std::invalid_argument(std::string(kSubject) + ": layer count out of range");
    }
    const std::size_t per_layer = mean_k_index_bytes(1, blocks, heads, head_dim);
    if (per_layer == 0) { return 0; }
    return per_layer * static_cast<std::size_t>(layers);
}

} // namespace

MeanKIndex::MeanKIndex(std::int32_t layers, std::int32_t heads, std::int32_t head_dim,
                       std::int32_t capacity_blocks)
    : layers_(layers),
      heads_(heads),
      head_dim_(head_dim),
      capacity_blocks_(capacity_blocks),
      sums_per_layer_(static_cast<std::size_t>(capacity_blocks > 0 ? capacity_blocks : 0) *
                      static_cast<std::size_t>(heads > 0 ? heads : 0) *
                      static_cast<std::size_t>(head_dim > 0 ? head_dim : 0)),
      elements_per_layer_(sums_per_layer_ +
                          static_cast<std::size_t>(capacity_blocks > 0 ? capacity_blocks : 0)),
      per_layer_bytes_(mean_k_index_bytes(1, capacity_blocks, heads, head_dim)),
      total_bytes_(checked_bytes(layers, capacity_blocks, heads, head_dim)),
      arena_(total_bytes_) {
    if (total_bytes_ == 0) {
        // A zero-capacity index is a supported state, not an error: it is what a caller that chose
        // not to reserve index memory gets, and empty() tells it so.
        return;
    }
    // ONE allocation, sliced per layer, so the arena cannot insert alignment padding BETWEEN layers
    // and make the per-layer stride disagree with per_layer_bytes() -- the same reasoning (and the
    // same machine check) as RawKShadowHarvest's constructor. The sums and the counts of a layer live
    // in that same region ([sums][counts]) rather than in a second allocation, so the count offset the
    // kernel derives cannot drift from the view built here.
    flat_ = arena_.alloc(DType::FP32, {static_cast<std::int32_t>(elements_per_layer_), layers_});
    if (flat_.bytes() != total_bytes_) {
        throw std::logic_error(std::string(kSubject) +
                               ": arena allocation does not match the declared index size");
    }
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        const Tensor region = flat_.slice(1, layer, 1);  // F32 {elements_per_layer, 1}
        views_[static_cast<std::size_t>(layer)] =
            region.slice(0, 0, static_cast<std::int32_t>(sums_per_layer_))
                .view({capacity_blocks_, heads_, head_dim_});
        counts_[static_cast<std::size_t>(layer)] =
            region.slice(0, static_cast<std::int32_t>(sums_per_layer_), capacity_blocks_);
    }
}

const Tensor& MeanKIndex::layer_sums(std::int32_t layer_index) const noexcept {
    static const Tensor kInvalid{};
    if (layer_index < 0 || layer_index >= layers_ || flat_.data == nullptr) { return kInvalid; }
    return views_[static_cast<std::size_t>(layer_index)];
}

const Tensor& MeanKIndex::layer_counts(std::int32_t layer_index) const noexcept {
    static const Tensor kInvalid{};
    if (layer_index < 0 || layer_index >= layers_ || flat_.data == nullptr) { return kInvalid; }
    return counts_[static_cast<std::size_t>(layer_index)];
}

cudaError_t MeanKIndex::zero(cudaStream_t stream) {
    if (total_bytes_ == 0 || flat_.data == nullptr) { return cudaSuccess; }
    // Not folded into append on purpose. An append that zeroed its own fresh blocks would be replayed
    // verbatim inside a captured decode graph, and a replayed memset would wipe the sums the prefill
    // accumulated -- the accumulation is state that outlives the graph. Zeroing is therefore an
    // explicit setup step on a stream the caller controls, and the append stays a single launch.
    const cudaError_t st =
        cudaMemsetAsync(flat_.data, 0, total_bytes_, stream);
    if (st != cudaSuccess) { return st; }
    blocks_written_ = 0;
    tail_fill_      = 0;
    return cudaSuccess;
}

void MeanKIndex::append_round(const RawKShadowHarvest& store, std::int32_t first_block,
                              cudaStream_t stream) {
    if (total_bytes_ == 0 || store.layers() != layers_) { return; }
    std::int32_t skipped       = 0;
    std::int32_t first_skipped = -1;
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        // LOCAL FIX (B08): feed the index only from layers whose staging slot was REALLY written this
        // round. A slot that was never written still holds the previous round's bytes -- or, on the
        // first round, whatever the allocation held -- and feeding that in is exactly the "plausible but
        // wrong" evidence raw_k_harvest.h:172-177 exists to prevent: the sums would rank blocks by keys
        // from another round with nothing looking wrong. Dropping one whole layer costs that layer's
        // contribution for every block equally, so the ranking across blocks stays comparable, and the
        // counter below is what makes "layer 0 was in this round" a reading instead of an assumption.
        if (!store.layer_harvested(layer)) {
            ++skipped;
            if (first_skipped < 0) { first_skipped = layer; }
            continue;
        }
        const Tensor view = store.staging_for_round(layer);
        // staging_for_round() returns THIS layer's slice of the packed harvest buffer, so the base
        // pointer handed to append_one already identifies the source layer, and every call is a
        // SINGLE-layer launch. The packed buffer's element stride between layers is
        // heads*head_dim*store.tokens() -- not this round's width -- but that stride is only
        // consulted by the multi-layer launch shape, which append_one deliberately does not use.
        append_one(view.data, layer, view.ne[1], first_block, store.tokens(),
                   heads_ * head_dim_ * store.tokens(), stream);
    }
    if (skipped != 0) {
        std::fprintf(stderr,
                     "kvmem index: round fed from %d of %d layer(s), skipped %d not harvested this "
                     "round (first=%d) -- those slots hold another round's bytes\n",
                     layers_ - skipped, layers_, skipped, first_skipped);
    }
}

void MeanKIndex::append_layer(const Tensor& raw_key, std::int32_t layer_index,
                              std::int32_t round_width, std::int32_t first_block,
                              cudaStream_t stream) {
    if (total_bytes_ == 0) { return; }
    if (layer_index < 0 || layer_index >= layers_) {
        throw std::invalid_argument(std::string(kSubject) + ": layer index " +
                                    std::to_string(layer_index) + " outside [0, " +
                                    std::to_string(layers_) + ")");
    }
    require_layout(raw_key, heads_ * head_dim_);
    // This entry point takes ONE layer's keys, so there is no inter-layer stride to honour; the
    // caller's tensor is self-contained.
    append_one(raw_key.data, layer_index, round_width, first_block, round_width,
               heads_ * head_dim_ * round_width, stream);
}

void MeanKIndex::append_one(const void* raw_key_bf16, std::int32_t layer_index,
                            std::int32_t round_width, std::int32_t first_block,
                            std::int32_t source_layer_stride, std::int32_t index_layer_stride,
                            cudaStream_t stream) {
    if (round_width <= 0) { return; }
    const std::int32_t full_blocks = round_width / kBlockTokens;
    const std::int32_t tail_fill   = round_width % kBlockTokens;
    const std::int32_t produced    = full_blocks + (tail_fill > 0 ? 1 : 0);
    if (produced <= 0) { return; }
    if (first_block < 0 || first_block + produced > capacity_blocks_) {
        // Refusing loudly beats writing past the reservation: an overrun here would corrupt the
        // neighbouring layer's index, which is far harder to see than an exception.
        throw std::out_of_range(std::string(kSubject) + ": appending " +
                                std::to_string(produced) + " block(s) at " +
                                std::to_string(first_block) + " exceeds the reserved capacity " +
                                std::to_string(capacity_blocks_));
    }
    (void)index_layer_stride; // the kernel derives the destination stride from blocks_per_layer

    // THE APPEND ACCUMULATES INTO THE DESTINATION, so the target blocks must already be zero. That is
    // `zero()`'s job, called once by the driver before the first append; see the header for why the
    // zeroing is NOT folded in here (a captured decode graph would replay it and wipe the prefill's
    // sums). A block that is appended to twice -- a partial tail block being completed -- is exactly
    // the case the sum+count layout exists to support, and it needs no special handling here.
    //
    // `1`, not `layers_` -- and that is not a shortcut, it is the fix for a real overrun. raw_key_bf16
    // already points at the ONE source layer this call is responsible for (append_round hands over
    // staging_for_round(layer); append_layer validates its own self-contained tensor). Passing layers_
    // made the kernel walk source layers 0..layers_-1 from that base and write destination layers
    // layer_index+0..layer_index+layers_-1: it read past the end of the harvest allocation and wrote
    // past the end of this arena for every layer above the first, silently corrupting the index.
    // The multi-layer shape still exists and is still tested -- by the standalone verifier
    // (POS-5 / POS-6), which is the only place that can supply a packed multi-layer base.
    const cudaError_t launch_status =
        detail::launch_mean_k_accumulate(raw_key_bf16, flat_.data, 1, layer_index, first_block,
                                         full_blocks, tail_fill, heads_, head_dim_, heads_ * head_dim_,
                                         kBlockTokens, source_layer_stride, capacity_blocks_,
                                         static_cast<std::int32_t>(elements_per_layer_), stream);

    // The status is CHECKED, not discarded. The launcher deliberately returns it instead of asserting
    // because its two callers want different things from a failure -- but "returns it" only helps if
    // somebody looks. Discarding it here meant a rejected launch left the index silently short of the
    // blocks the caller believes were written: the next append would continue from a cursor that
    // claims more history than the index holds, and the failure would surface much later, as a wrong
    // answer, with nothing pointing back at the launch. That is the same silent-failure shape this
    // file already refuses for capacity overruns, so it refuses here too. (Pre-existing C4834.)
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(std::string(kSubject) + ": launch failed for layer " +
                                 std::to_string(layer_index) + ", first_block " +
                                 std::to_string(first_block) + " (" +
                                 std::to_string(produced) + " block(s)): " +
                                 cudaGetErrorName(launch_status));
    }

    // max(), not assignment: append_round calls this once per layer with the SAME block range, so an
    // assignment would be idempotent too -- but a caller that interleaves layers or re-appends a tail
    // would silently move the cursor BACKWARDS, and the next append would then overwrite blocks it
    // should have continued.
    blocks_written_ = std::max(blocks_written_, first_block + produced);
    // `tail_fill`, the LOCAL computed at the top of this function -- not `tail_fill_`, which is what
    // this line used to say. `tail_fill_ = tail_fill_;` is a self-assignment: it compiled, it
    // produced no warning any of the builds surfaced, and it left the member at its previous value
    // forever, so tail_fill() could only ever report 0 or the value zero() had just written. The
    // kernel is not affected (it is handed the local directly, :182), which is exactly why the copy
    // could rot unnoticed: the one consumer of the member is the diagnostic getter.
    tail_fill_      = tail_fill;
}

MeanKIndex* mean_k_index_for(std::int32_t layers, std::int32_t heads, std::int32_t head_dim,
                             std::int32_t capacity_blocks) {
    // Read the switch in the DRIVER, like the harvest does (kvmem_shadow.h:55-61): the OFF arm must
    // allocate nothing and issue nothing. A null return is the normal "KVMem is off" answer, never an
    // error, and callers gate on it.
    if (!detail::kvmem_shadow_enabled()) { return nullptr; }
    if (layers <= 0 || heads <= 0 || head_dim <= 0 || capacity_blocks <= 0) { return nullptr; }

    static MeanKIndex* installed = nullptr;
    if (installed == nullptr) {
        installed = new MeanKIndex(layers, heads, head_dim, capacity_blocks);
        // REQUIRED before the first append: arena bytes are not guaranteed zero, and the accumulate
        // kernel ADDS into the destination, so an unwritten block would otherwise start from
        // garbage and its count would claim tokens it never saw.
        //
        // Synchronised here rather than left stream-ordered on purpose. This runs once, outside any
        // capture, on the first prefill chunk -- the same place the harvest pays its one-time
        // allocation. Leaving it async would put a memset in front of the first append on a stream
        // whose ordering the caller does not control, and a silently-lost zero would corrupt the
        // index in the one way that is hardest to attribute (a wrong ranking, much later).
        const cudaError_t zero_status = installed->zero(nullptr);
        std::fprintf(stderr,
                     "kvmem_index: installed layers=%d heads=%d head_dim=%d blocks=%d "
                     "bytes=%.1f MiB zero=%s\n",
                     static_cast<int>(layers), static_cast<int>(heads),
                     static_cast<int>(head_dim), static_cast<int>(capacity_blocks),
                     static_cast<double>(installed->bytes_total()) / (1024.0 * 1024.0),
                     cudaGetErrorName(zero_status));
        if (zero_status != cudaSuccess) {
            // Fail loudly instead of handing back an index whose counts are noise: a bad count is not
            // detectable downstream, it just moves the ranking.
            delete installed;
            installed = nullptr;
            throw std::runtime_error(std::string(kSubject) + ": zero() failed: " +
                                     cudaGetErrorName(zero_status));
        }
    }
    return installed;
}

} // namespace ninfer::ops
