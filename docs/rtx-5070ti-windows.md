# RTX 5070 Ti: Windows, CUDA 13.4.2 and native SM120

This branch preserves the local Windows/SM120 fixes applied to upstream commit
`f118551fb401de073555807a48c50238e180e3b8`. The 18 modified source files were compared
byte-for-byte with the source used for the completed CUDA 13 native build before committing.
The existing RTX 3090 guides describe a different machine and build tree; do not substitute
their architecture or performance numbers for this configuration.

## Build configuration

Verified environment: RTX 5070 Ti 16 GB (70 SMs), Windows x64, driver 617.14,
CUDA toolkit distribution 13.4.2 (nvcc 13.4.92), MSVC 14.44.35207, CMake/Ninja,
Release `/O2 /DNDEBUG`, target `120a`. Windows enables staged TMA descriptors even
when the separate `NINFER_TMA_STAGING` cache option is OFF. Native mode activates
PDL even when the separate compatibility-route `NINFER_PDL` option is OFF.

Use a new build directory, an x64 MSVC 14.44 developer environment, and the desired
CUDA toolkit on PATH. Provide FFmpeg and curl through the existing Windows
dependency mechanism in `cmake/Dependencies.cmake` (a vcpkg CMake toolchain, or
`VCPKG_ROOT` plus `VCPKG_TARGET_TRIPLET` pointing to a prepared dependency tree).
For an environment with those prerequisites, the relevant configure options are:

```powershell
cmake -S . -B build-5070ti-native -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_SM120_NATIVE=ON -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF `
  -DNINFER_DIRECTSTORAGE=OFF -DNINFER_D3D12_RESIDENCY=OFF `
  "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
  "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe"
cmake --build build-5070ti-native --target ninfer_ops -j
```

The measured build pruned redundant PTX from `src/ops/ninfer_ops.lib` before the
final link, retaining `sm_120a` SASS to keep the Windows PE below 2 GiB. To reproduce
that step on this single-architecture build, back up that archive outside the link
inputs, then run `nvprune -arch sm_120a <archive> -o <temporary-archive>` and replace
the original archive with the successful output. Finish with:

```powershell
cmake --build build-5070ti-native --target ninfer-serve -j
```

Rebuilding the archive requires repeating the prune step. Do not prune a multi-GPU-
architecture distribution to this single architecture. The measured executable is
1,064,627,712 bytes, SHA-256
`4f876dd197845cf966cfcd72792a378094a2afc09b6276860d4b3194a43bec81`.
This is a reference artifact hash, not a promise of bit-identical builds elsewhere.
The existing `scripts/build.ps1` targets 3090-era architecture selections and is
not the entry point for this native 120a recipe.

The preserved source changes cover MSVC kernel-launch syntax, staged BF16/NVFP4
TMA descriptors, dynamic shared-memory opt-in for larger NVFP4 schedules and their
launch sites, a CUDA 12.8 SM120 convolution index workaround, portable bit counting,
consistent Windows CUDA-runtime linkage, and MSVC limits in embedded profile JSON
and CLI dispatch depth. They do not establish numerical qualification for every
NVFP4/MoE route: the completed end-to-end measurements used the mixed-GGUF model below.

## Model and serving preset

Measured artifact: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer`, a mixed-GGUF,
text/MTP v3 container. It embeds its tokenizer/template resources; the original
GGUF and conversion report are not runtime prerequisites. Model weights are not
included in this repository. Do not substitute another artifact and assume the
same memory budget or performance.

The [example launcher](../examples/rtx5070ti/serve.ps1) accepts local paths rather
than machine-specific directories:

```powershell
.\examples\rtx5070ti\serve.ps1 `
  -Executable .\build-5070ti-native\apps\ninfer-serve.exe `
  -Model .\models\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer
```

It runs in the foreground; Ctrl+C stops it. Use `-ReasoningEffort medium` or `low`
to change the default, or `-DryRun` to inspect arguments without loading the GPU.
Required runtime DLLs must be beside the executable or on PATH; FFmpeg/curl have
transitive dependencies even for text-only workloads. CUDA development tools,
headers, old builds, and raw benchmark logs do not belong in a runtime package.

| Setting | Value |
|---|---|
| Context / physical KV capacity | 147456 / 147456 (144K) |
| Concurrency | 1 |
| Actual prefill chunk | 640, `NINFER_PREFILL_ALIGN=0` |
| KV / GDN state | rk8v4 / FP16 |
| Drafting | Built-in MTP 2, ngram 31, full MTP attention window |
| CUDA Graph allowance | 72 MiB (measured graph allocation: 54 MiB) |
| LM-head draft | Disabled |
| Prefix cache | Alternative prefix caching; 4096 MiB host cache; one device snapshot |
| Sampling | temperature 1, top-p .95, top-k 20, min-p 0, presence/frequency 0, seed 42 |
| Repetition penalty | Neutral 1; no invented CLI option |
| Thinking | Enabled, xhigh default, preserve thinking enabled |

The [device profile](../examples/rtx5070ti/device-profile.json) is the earlier
calibration retained after a separate CUDA 13 native calibration did not improve
overall performance. It is specific to the 70-SM RTX 5070 Ti, not a universal GPU
profile. `auto` may update the supplied profile for an unmatched device; use a
separate file when calibrating another GPU.

`attn_prompt_fast=on` already enables the fast prompt kernel. The runtime field
`fast_prefill_kernel=false` only records the absence of the CLI force-enable flag.
The launcher clears an inherited `NINFER_PROMPT_FAST` override so the profile can
select the route, and restores the caller's environment on exit.

Thinking and preservation are usage defaults, not the mode in which the performance
tables below were measured. Requests can override defaults. The 144K window includes
reasoning and output: reserve room for generation. `xhigh` is not a fixed token budget.

## Measured evidence and limits

Measurements on 2026-09-28/29 used a resident model, single concurrency, the older
device profile, non-greedy sampling above, and **thinking off**. Cold 49K requests
had 49,539 input tokens, zero cached input, and 256 output tokens. Two seeds were
used for the comparisons below. PP/decode are server-timing rates; TTFT is client
time to first visible streamed output. These are limited workloads, not a complete
Codex project-success benchmark or a full CUDA numerical-test suite.

| Comparison | Result |
|---|---|
| chunk640/Graph72/MTP2 vs chunk256/Graph256/MTP2, head disabled | 49K PP 1762.89 vs 1622.21 tok/s; TTFT 28.168 vs 30.594 s |
| ngram31 vs ngram15, pure code copy | Decode 701.14 vs 462.33 tok/s; total 1.588 vs 2.134 s |
| ngram31 vs ngram15, two edits with remaining code copied | Decode 623.90 vs 435.42 tok/s; total 1.705 vs 2.219 s |
| Default fast profile vs explicit fast CLI flag, ngram15 | 49K PP 1761.62 vs 1761.26 tok/s; no additional gain |
| Default fast profile vs forced `NINFER_PROMPT_FAST=0`, ngram15 | 49K PP 1761.62 vs 1679.32 tok/s; TTFT 28.191 vs 29.562 s |

The chunk comparison changes the configuration combination, not just one variable.
Copy tests used an 86-line/2,807-character code fixture, generated 742 tokens, and
checked the complete expected text. The copy gain does not predict novel-code or
reasoning throughput. The forced-off fast-kernel arm generated different text and
had different draft acceptance; its higher 49K decode rate is not proof of a
general decode benefit. The fast-kernel trial comprised six fresh sessions and
30 requests, all completed; cold retrieval checks passed in all arms.

The earlier 640 continuation covered 17 sessions/70 requests, including five
near-capacity requests and 16 exact-copy checks. A near-capacity case used 146,914
input plus 256 output tokens; it does not validate that input length with a long
thinking output. Ngram31 added about 34 MiB: startup CUDA free memory was 287 MiB,
while the minimum sampled request-time NVML free memory was 982 MiB. These are
different measurement interfaces/times, not interchangeable headroom figures.

MTP attention window 8192 gave a small 49K decode gain but no near-capacity gain;
16384 showed no advantage. The example therefore retains the full window. The
new launcher was syntax-checked and dry-run checked; the underlying argument set
also completed a local packaged-server startup and a short thinking-enabled reply.
No fresh full build, full numerical suite, or end-to-end Codex task campaign was
run merely to publish this branch. The default thinking-on workload still needs
its own quality and latency comparison.
