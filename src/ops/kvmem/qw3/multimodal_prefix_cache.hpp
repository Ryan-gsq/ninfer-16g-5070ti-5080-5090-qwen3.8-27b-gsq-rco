// ===========================================================================
// PORTED from kvmem-qw3 -- Apache License 2.0
//   Upstream : https://github.com/kvmem/kvmem-qw3   (branch main)
//   Original : src\multimodal_prefix_cache.hpp
//   Retrieved: 2026-09-19 (archive fetched via gh-proxy); ported into the ninfer
//              ternary tree on 2026-09-21.
//   Licence  : Apache-2.0. The upstream LICENSE (11,358 B) and
//              THIRD_PARTY_NOTICES.md stay with the project; this file is a
//              derivative of the upstream file named above.
//   Changes  : NONE -- byte-identical to upstream; only this attribution block was prepended.
// ===========================================================================
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qw3::detail {

// Lightweight cache identity retained with a warm checkpoint. It deliberately
// excludes the projected embedding/storage owner so keeping a prefix warm does
// not pin a visual encoder output or device allocation.
struct MultimodalTokenIdentity {
    uint32_t token_id = 0;
    uint32_t source_token_id = 0;
    std::array<uint32_t, 3> position = {0, 0, 0};
    uint64_t cache_identity = 0;
};

constexpr uint64_t multimodal_row_cache_identity(uint64_t image_identity,
                                                  uint32_t row) {
    uint64_t h = image_identity != 0
        ? image_identity : 1469598103934665603ULL;
    for (int byte = 0; byte < 4; ++byte) {
        h ^= static_cast<uint8_t>(row & 0xffu);
        h *= 1099511628211ULL;
        row >>= 8;
    }
    return h != 0 ? h : 1;
}

inline const MultimodalTokenIdentity *multimodal_token_identity(
        const std::vector<MultimodalTokenIdentity> &identities,
        uint32_t token_id) {
    if ((token_id & 0x80000000u) == 0) return nullptr;
    const uint32_t row = token_id & 0x7fffffffu;
    if (row > 0 && row <= identities.size()) {
        const MultimodalTokenIdentity &candidate = identities[row - 1];
        if (candidate.token_id == token_id) return &candidate;
    }
    for (const MultimodalTokenIdentity &candidate : identities) {
        if (candidate.token_id == token_id) return &candidate;
    }
    return nullptr;
}

// Refine the ordinary token LCP with visual-input identity. Synthetic visual
// token IDs encode layout, not pixels: two different images with the same grid
// intentionally render the same IDs. Return the first visual position whose
// source row or M-RoPE coordinate differs. This permits [A,B,C] -> [A,B,C,D]
// to reuse a checkpoint before D while rejecting a changed/reordered A/B/C.
//
// Legacy callers may omit per-row cache identities. In that case the old
// whole-request fingerprint remains the conservative authority.
inline uint32_t multimodal_prefix_lcp(
        const std::vector<uint32_t> &tokens, uint32_t token_lcp,
        const std::vector<MultimodalTokenIdentity> &warm_identities,
        const std::vector<MultimodalTokenIdentity> &current_identities,
        uint64_t warm_fingerprint, uint64_t current_fingerprint) {
    const size_t limit =
        std::min<size_t>(token_lcp, tokens.size());
    for (size_t pos = 0; pos < limit; ++pos) {
        const uint32_t token = tokens[pos];
        if ((token & 0x80000000u) == 0) continue;
        const MultimodalTokenIdentity *warm =
            multimodal_token_identity(warm_identities, token);
        const MultimodalTokenIdentity *current =
            multimodal_token_identity(current_identities, token);
        if (!warm || !current ||
            warm->source_token_id != current->source_token_id ||
            warm->position != current->position) {
            return static_cast<uint32_t>(pos);
        }
        if (warm->cache_identity == 0 || current->cache_identity == 0) {
            if (warm_fingerprint != current_fingerprint) {
                return static_cast<uint32_t>(pos);
            }
        } else if (warm->cache_identity != current->cache_identity) {
            return static_cast<uint32_t>(pos);
        }
    }
    return static_cast<uint32_t>(limit);
}

}  // namespace qw3::detail
