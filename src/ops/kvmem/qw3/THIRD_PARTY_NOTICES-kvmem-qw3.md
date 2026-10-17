# Third-Party Notices

Except where a file or component states otherwise, this repository is
licensed under the Apache License 2.0 in [`LICENSE`](LICENSE).

## Scope

This notice covers third-party source code that is redistributed in this
repository and is directly included or compiled by the build. It intentionally
does not enumerate toolchains, system libraries, or optional dependencies that
users provide separately. A future binary release must generate a separate
dependency notice or software bill of materials from its pinned build
environment.

The source-distribution audit found the three groups below. No other vendored
source directory is part of the current build.

Two provenance questions are deliberately not classified as third-party
components here: tokenizer logic described in source comments as matching
llama.cpp, and MTP code described as ported from `qw3_ly`. Their ownership and
origin require author confirmation before release; they are recorded as
release blockers in `docs/release_baseline.md` rather than assigned a license
without evidence.

## nlohmann/json

- Upstream: <https://github.com/nlohmann/json>
- Version: 3.11.3
- Repository file: `third_party/json.hpp`
- License: MIT; see `LICENSES/nlohmann-json.txt`
- Local SHA-256:
  `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`

The hash matches the official `json.hpp` asset published for nlohmann/json
v3.11.3. The header is directly included by the model loader, model
configuration, tokenizer, KVMem archive, Anthropic adapter, CLI, and HTTP
server code.

The official amalgamated header retains attribution for code incorporated by
nlohmann/json, including:

- Hedley by Evan Nemerson, MIT;
- a C++11 utility fallback from the Abseil Authors, Apache-2.0;
- Grisu2 conversion code by Florian Loitsch, MIT; and
- UTF-8 decoder code by Björn Hoehrmann, MIT.

These are embedded upstream attributions, not separate qw3 dependencies. Their
copyright and provenance comments remain in `third_party/json.hpp`. The
Apache-2.0 terms are reproduced in the repository-level `LICENSE`, and the MIT
terms are reproduced in `LICENSES/nlohmann-json.txt`.

## cpp-httplib

- Upstream: <https://github.com/yhirose/cpp-httplib>
- Version: 0.15.3
- Repository file: `third_party/httplib.h`
- License: MIT; see `LICENSES/cpp-httplib.txt`
- Local SHA-256:
  `a3347656c71c81c3fe510fdb1ff229d7db52eb7df04d3342d340eba0b6ad50ea`

The hash matches `httplib.h` at the upstream v0.15.3 tag. The header is
directly included by `src/qw3_server.cpp`, which instantiates
`httplib::Server` for the HTTP API. The current build does not enable
cpp-httplib's optional TLS, zlib, Brotli, or other compression integrations.

## llama.cpp-derived CUDA code

- Upstream: <https://github.com/ggml-org/llama.cpp>
- Upstream commit: `57ebaf4edd99ea675f256ae2286cd99206dbfcd1`
- License: MIT; see `LICENSES/llama.cpp.txt`

The CUDA build compiles or includes the following MIT-licensed source files
that port, adapt, or duplicate code derived from the pinned llama.cpp source:

- `src/cuda_helpers.cuh`
- `src/fattn_vec_decode.cu`
- `src/gated_delta_net.cu`
- `src/mmvq_q8.cu`
- `src/mmq_q8.cu`
- `src/kernel_legacy.cu`

The files retain SPDX MIT markers and, where present, detailed descriptions of
their upstream source and qw3-specific adaptations. `src/kernel_legacy.cu`
duplicates helpers and kernels from the MIT-licensed
`src/fattn_vec_decode.cu`, so it remains in this group even though its routes
are diagnostic rather than default execution paths.

## Externally supplied dependencies

CUDA, OpenSSL, FlashInfer, CUTLASS, and CuTe DSL artifacts are not copied
into this repository. They are conditional build or runtime dependencies and
are therefore outside the source-redistribution list above. Their exact
versions and applicable notices must be captured from the pinned environment
before distributing prebuilt binaries.
