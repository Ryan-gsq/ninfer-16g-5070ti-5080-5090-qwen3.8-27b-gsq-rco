#pragma once

// Row views over a TERNARY RowSplit parent: T2G128_F16S or PTQ1_0. The row-split planes are
// row-major per plane, so a contiguous row range of the parent is itself a valid RowSplit weight
// with offset code, high and scale pointers; the fused wrappers project the parent's parts through
// ops::linear with these views.
//
// Ported from the I line's ops/linear/ternary/ternary_row_view.h, which covered both packings for
// the same reason. Two things changed here relative to the T2-only version this replaces, and both
// are silent-corruption bugs if missed:
//
//   1. the plane pitches are per format (core/weight.h row_split_format), never the 32/0 default.
//      PTQ1_0 carries 24 code bytes + 2 high bytes per 128-weight group; slicing it with the 32/0
//      split reads the wrong byte ranges and yields plausible-looking garbage rather than an error.
//   2. the HIGH plane must be advanced as well. A T2 parent carries none, so nothing here used to
//      move it -- but a PTQ1_0 parent does, and leaving qhigh at row 0 while qdata moves feeds every
//      row the FIRST row's extra trits. The I line's note on the plane split is the same warning:
//      "the geometry is derived from qtype, never from a default".

#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline Weight t2_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (!is_ternary(parent.qtype) || parent.layout != QuantLayout::RowSplit || row_begin < 0 ||
        rows <= 0 || row_begin + rows > parent.n) {
        throw std::invalid_argument("t2 row view: invalid parent or row range");
    }
    const RowSplitFormat geometry     = row_split_format(parent.qtype);
    const std::int64_t groups_per_row = parent.padded_shape[1] / geometry.group;
    const std::int64_t row_offset     = static_cast<std::int64_t>(row_begin) * groups_per_row;
    Weight view                       = parent;
    view.n                            = rows;
    view.shape[0]                     = rows;
    view.padded_shape[0]              = rows;
    view.qdata                        = static_cast<const std::uint8_t*>(parent.qdata) +
                                        row_offset * geometry.code_bytes;
    view.qhigh = parent.qhigh == nullptr
                     ? nullptr
                     : static_cast<const std::uint8_t*>(parent.qhigh) +
                           row_offset * geometry.high_bytes;
    view.scales = static_cast<const std::uint8_t*>(parent.scales) +
                  row_offset * geometry.scale_bytes;
    return view;
}

} // namespace ninfer::ops::detail
