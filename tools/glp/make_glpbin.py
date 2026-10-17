#!/usr/bin/env python3
"""Convert a GLP vector into the .glpbin container the ninfer GlpState reads.

Sources accepted:
  * a GLP GGUF (spec: tensor names `direction.<N>` = layer N, F32 [n_embd]; or the
    older patch naming `steer.blk.<i>.weight`, where i is the layer index),
  * the `dirs.npz` produced by the difference-of-means pipeline (keys r_0..r_{L-1}).

The output container (little-endian) is exactly what
`src/models/qwen3_5/execution/glp_state.cpp` parses:

    magic  "GLPB"              4 bytes
    int32  layer_count         model's global layer count (rows in the padded stack)
    int32  width               n_embd (residual width)
    int32  covered_layers      rows actually present in the source vector
    int32  first_layer         global layer index of the first covered row
    uint16 directions[layer_count * width]   BF16, row-major, zero-padded

Rows outside [first_layer, first_layer + covered) are zero-filled, so an uncovered
layer's direction is all zeros and its projection is an exact no-op at runtime.

Usage:
    python make_glpbin.py --src glp49-L10-58-a1.gguf --layer-count 64 \
        --width 5120 --out glp49.bin
    python make_glpbin.py --src dirs.npz --layer-count 64 --width 5120 \
        --first-layer 10 --covered 49 --out glp49.bin
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 (round-to-nearest-even) and return uint16 bits."""
    u32 = values.astype(np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
    return ((u32 + rounding) >> 16).astype(np.uint16)


def read_gguf_directions(path: str) -> dict[int, np.ndarray]:
    """Return {layer_index: float32[n_embd]} from a GLP GGUF."""
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file (magic={magic!r})")
        struct.unpack("<I", handle.read(4))[0]  # version
        n_tensors = struct.unpack("<Q", handle.read(8))[0]
        n_kv = struct.unpack("<Q", handle.read(8))[0]

        scalar = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
                  5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
                  12: ("d", 8)}
        for _ in range(n_kv):
            key_len = struct.unpack("<Q", handle.read(8))[0]
            handle.read(key_len)
            value_type = struct.unpack("<I", handle.read(4))[0]
            if value_type == 8:  # string
                length = struct.unpack("<Q", handle.read(8))[0]
                handle.read(length)
            elif value_type == 9:  # array
                elem_type = struct.unpack("<I", handle.read(4))[0]
                count = struct.unpack("<Q", handle.read(8))[0]
                fmt, size = scalar.get(elem_type, ("B", 1))
                handle.read(size * count)
            else:
                fmt, size = scalar.get(value_type, ("I", 4))
                handle.read(size)

        tensors = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", handle.read(8))[0]
            name = handle.read(name_len).decode("utf-8")
            ndim = struct.unpack("<I", handle.read(4))[0]
            dims = [struct.unpack("<Q", handle.read(8))[0] for _ in range(ndim)]
            dtype = struct.unpack("<I", handle.read(4))[0]
            offset = struct.unpack("<Q", handle.read(8))[0]
            tensors.append((name, dims, dtype, offset))

        # Data section begins after alignment padding to 32 bytes.
        after = handle.tell()
        aligned = (after + 31) // 32 * 32
        handle.seek(aligned)
        data_start = handle.tell()

        directions: dict[int, np.ndarray] = {}
        for name, dims, dtype, offset in tensors:
            if dtype != 0:  # GGML_TYPE_F32
                raise SystemExit(f"{name}: expected F32, got ggml type {dtype}")
            if len(dims) != 1:
                raise SystemExit(f"{name}: expected 1-D, got {dims}")
            layer = _layer_from_name(name)
            if layer is None:
                continue
            handle.seek(data_start + offset)
            row = np.frombuffer(handle.read(dims[0] * 4), dtype="<f4").astype(np.float32)
            directions[layer] = row
    return directions


def _layer_from_name(name: str) -> int | None:
    # `direction.<N>` -> layer N (spec, 0-based verbatim).
    if name.startswith("direction."):
        tail = name[len("direction."):]
        head = tail.split(".")[0]
        if head.isdigit():
            return int(head)
    # `steer.blk.<i>.weight` -> layer i (older patch naming).
    if name.startswith("steer.blk."):
        parts = name.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            return int(parts[2])
    return None


def read_npz_directions(path: str) -> dict[int, np.ndarray]:
    archive = np.load(path)
    directions: dict[int, np.ndarray] = {}
    for key in archive.files:
        if key.startswith("r_"):
            directions[int(key[2:])] = archive[key].astype(np.float32)
    if not directions:
        raise SystemExit(f"{path}: no r_<i> arrays found")
    return directions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="GLP GGUF or dirs.npz")
    parser.add_argument("--out", required=True, help="output .glpbin path")
    parser.add_argument("--layer-count", type=int, default=64, help="model global layer count")
    parser.add_argument("--width", type=int, default=5120, help="residual width n_embd")
    parser.add_argument("--first-layer", type=int, default=None,
                        help="force the first covered global layer (default: min present)")
    parser.add_argument("--covered", type=int, default=None,
                        help="force the covered layer count (default: span of present layers)")
    parser.add_argument("--normalize", action="store_true",
                        help="re-normalise each row to unit length before writing")
    args = parser.parse_args()

    if args.src.lower().endswith(".gguf"):
        directions = read_gguf_directions(args.src)
    elif args.src.lower().endswith(".npz"):
        directions = read_npz_directions(args.src)
    else:
        raise SystemExit("--src must end in .gguf or .npz")

    if not directions:
        raise SystemExit("no per-layer directions found in the source")

    for layer, row in directions.items():
        if row.shape[0] != args.width:
            raise SystemExit(f"layer {layer}: width {row.shape[0]} != expected {args.width}")

    layers = sorted(directions)
    first_layer = args.first_layer if args.first_layer is not None else layers[0]
    covered = args.covered if args.covered is not None else (layers[-1] - first_layer + 1)
    if first_layer + covered > args.layer_count:
        raise SystemExit("covered range exceeds layer_count")

    stack = np.zeros((args.layer_count, args.width), dtype=np.float32)
    for layer in range(first_layer, first_layer + covered):
        row = directions.get(layer)
        if row is None:
            sys.stderr.write(f"warning: layer {layer} missing in source; zero-filling\n")
            continue
        if args.normalize:
            norm = float(np.linalg.norm(row))
            if norm > 0:
                row = row / norm
        stack[layer] = row

    bits = f32_to_bf16_bits(stack.reshape(-1))
    with open(args.out, "wb") as handle:
        handle.write(b"GLPB")
        handle.write(struct.pack("<4i", args.layer_count, args.width, covered, first_layer))
        handle.write(bits.tobytes())

    norms = np.linalg.norm(stack[first_layer:first_layer + covered], axis=1)
    print(f"wrote {args.out}: layers={args.layer_count} width={args.width} "
          f"covered={covered} first_layer={first_layer}")
    print(f"  covered row norms: min={norms.min():.4f} max={norms.max():.4f} "
          f"mean={norms.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
