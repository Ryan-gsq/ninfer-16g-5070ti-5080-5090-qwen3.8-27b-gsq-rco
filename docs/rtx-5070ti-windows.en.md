# RTX 5070 Ti Windows guide

[简体中文](rtx-5070ti-windows.md) · **English** · [Binary download and installation](rtx-5070ti-windows-downloads.md)

This package targets **RTX 5070 Ti 16 GB, Windows x64, one model and one concurrent
request**. It combines the CUDA 13.4.2 Native SM120a Release engine with
**NInfer Manager 1.3.1**. Double-click `NInferManager.exe` for everyday use.
Model management and monitoring run in your browser; no separate Python,
CMD or PowerShell launcher is needed.

The default profiles are **XXS 160K and S 128K**, both using the `strict`
memory policy. K means 1024 tokens. This guide covers operation, file placement,
settings and the evidence behind those defaults.

## 1. Start the manager

1. Extract the complete package to a permanent directory, such as `qwen27b`.
   Do not copy the EXE alone.
2. Put each `.ninfer` model and all its continuation volumes in `model/`.
   The software download does not include model weights.
3. Double-click `NInferManager.exe`. It has no main window or terminal;
   look for its icon in the Windows notification area.
4. Right-click the icon and open Manage Models to check the configuration.
   Choose XXS or S from the Start Model submenu.
5. Once ready, open monitoring or copy the API base and model name from the tray.

| Model ready | Stopped or not ready |
|---|---|
| ![Green N tray icon while running](assets/ninfer-tray-running.png) | ![Gray N tray icon while stopped](assets/ninfer-tray-stopped.png) |
| Mint background and dark-green N | Gray; tooltip text distinguishes loading, stopping and errors |

The management site defaults to `http://127.0.0.1:8090`, and the inference
API base to `http://127.0.0.1:18081/v1`. Opening the site from the tray establishes
the local management session. Both profiles expose
**`swift-1.5-qwen3.8-27b`** to clients, with only one loaded at a time.
Start is disabled while loading, running or stopping.

Closing the browser does not stop inference. Stopping the model leaves management
available; exiting the manager stops its model and monitoring site. A Windows Job
owns the engine and its descendants; normal shutdown first sends CTRL_BREAK.
The manager does not adopt unrelated processes merely because a port or PID matches.

### How Windows startup works

“Start with Windows” uses the fixed `NInferManager` value in the current user's
registry key `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run`.
Windows launches the manager **after that user signs in**, without administrator
permission. It is not a system service that runs before sign-in. A computer left
at the sign-in screen will not load the model.

| Independent setting | Effect when enabled |
|---|---|
| Start with Windows | Launch the tray manager after user sign-in |
| Automatically start default model | Load the selected default profile when the manager starts; initially `xxs-160k` |

**Start with Windows is off by default** and must be enabled in the site or tray.
Automatic loading of the default XXS 160K profile remains enabled in the initial
configuration. Existing personal settings keep their values; upgrading does not
turn either setting back on. Effective login startup depends on both registration
and Windows Startup Apps state.

Several copies can exist, but the current user has one startup registration:
**the last copy that explicitly enables or saves startup settings owns the entry**
and replaces its executable path. A normal or `--autostart` launch does not reclaim
registration for an older copy. Disabling startup in an old copy does not delete
an entry already pointing to a different copy. After moving the program, explicitly
enable startup from the new location to update the path.

Windows Startup Apps can independently disable this entry, and the manager respects
that state. An enabled value in a settings file does not prove Windows will launch
it; restoring login startup also requires allowing the entry in Windows.

An earlier Manager 1.3.0 check executed its registered command, automatically loaded
XXS 160K and returned a correct short reply; executing it twice did not reload the
model. That verified the older command path, not the new 1.3.1 registration rules.
**A real reboot/sign-in has not been performed to verify Windows' trigger itself.**

## 2. Where files belong

Version 1.3.1 separates the **program directory** from the **personal data directory**.
The EXE, engine, models and website remain in the program directory (PackageRoot).
Active configuration, logs and runtime state live in the data directory (DataRoot).
An installation under `Program Files` normally needs no writes to its program
directory and no administrator permission for everyday operation.

The default data directory is
`%LOCALAPPDATA%\NInferManager\<installation ID>`. The ID is derived from the
program's location, so each installation path gets a separate subdirectory. On first use at that location,
the manager copies the package's `config/` and fills missing initial configuration
from embedded defaults. Later launches use the active data-directory configuration
without repeatedly overwriting it from the package.

Only failure to create or write the preferred data directory triggers a fallback
to the program directory. Invalid configuration JSON is reported and preserved;
it does not silently cause a directory switch or reset to defaults.

```text
Program directory / PackageRoot, such as qwen27b/ or an installation under Program Files
├─ NInferManager.exe               Tray manager, Release / Windows x64
├─ engine/
│  ├─ ninfer-serve.exe              CUDA 13.4.2 Native SM120a Release engine
│  └─ *.dll                        Runtime dependencies; keep beside the EXE
├─ model/
│  ├─ *.ninfer                     Model entry files
│  └─ …                            Every continuation volume, with original names
├─ config/                         First-use configuration seeds, not the active copy
├─ wwwroot/                        Production management and monitoring website
├─ docs/                           Chinese/English guides, download page and images
├─ LICENSE                         Project license
└─ licenses/                       NVIDIA EULA, dependency licenses and sources

Personal data directory / DataRoot
%LOCALAPPDATA%\NInferManager\<installation ID>/
├─ config/
│  ├─ settings.json                Language, startup, default profile, scan paths, web port
│  ├─ profiles/
│  │  ├─ xxs-160k.json              XXS paths, API model name and launch arguments
│  │  └─ s-128k.json                S paths, API model name and launch arguments
│  ├─ chat_template.jinja           Active conversation template
│  ├─ chat_template.LICENSE         Template license
│  ├─ device-profiles.json          Active GPU calibration
│  ├─ initialized.json             First-use initialization marker
│  └─ history/                     Previous revisions saved before configuration edits
├─ runtime/                        Local process state and short-lived session data
└─ logs/                           Engine output, errors and request metrics
```

Back up `config/` in the data directory: it contains your settings and template.
`runtime/` is transient local state, not configuration to share. Saved parameters
and `config/history/` are both in the data directory. Release downloads exclude personal history,
logs, session credentials and model weights. Old engines, test launchers,
compilers, original GGUFs and raw benchmark logs do not belong in the everyday package.

Saved launch parameters apply to the **next model start**. Writes are atomic and
previous files are archived under `config/history/`. Invalid JSON is reported
and preserved instead of being silently replaced with defaults.

Relative paths follow their purpose: configuration resources such as
`config/chat_template.jinja` and `config/device-profiles.json` resolve against
DataRoot. `engine/ninfer-serve.exe`, `model/...` and model scan directories resolve
against PackageRoot. Absolute paths retain their specified location.

Moving or copying the program to a new path creates a separate data directory,
initialized from that package's `config/`. The two locations subsequently save
settings independently. Moving program files does not automatically carry later
edits from the old data directory; back up its configuration if you need those edits.
Versioned defaults live in `apps/windows-manager/config/` in the source tree and
are also embedded in the manager. Package seeds and active data are not kept in
two-way synchronization.

### Add models and change parameters

The site edits scan directories, model paths, client-facing model IDs, launch
parameters and the default profile. It scans `model/` initially and every
15 seconds, with a manual refresh available. Scanning checks v3 headers,
directories and volume completeness without loading all weights or initializing
the GPU. GGUF, safetensors and individual continuation volumes are not launch entries.
A complete file does not mean that every context setting fits in VRAM.

Question marks beside parameters provide explanations on mouse hover or keyboard
focus. The top-right Simplified Chinese / English switch persists and updates the
tray menu without changing model settings or discarding unsaved edits.

## 3. Models and default settings

The models are `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer` and its
`IQ3_S-mtp.ninfer` counterpart. These are text/MTP v3 containers converted from GGUF files that use multiple tensor quantization types
**without a usable image-understanding component**. Original GGUF files and
conversion reports are not runtime requirements. Another artifact needs its own
capacity and performance check.

| Setting | XXS default | S default |
|---|---|---|
| Profile ID | `xxs-160k` | `s-128k` |
| Context and fixed KV capacity | 163840 / 160K | 131072 / 128K |
| Prefill chunk | 1024 | 256 |
| Concurrency | 1 | 1 |
| KV / GDN state | rk8v4 / FP16 | Same |
| Drafting | MTP 2, ngram 31, full MTP attention window | Same |
| Graph allowance | 72 MiB | Same |
| `--lm-head-draft` | Off | Off |
| CPU context cache | 6144 MiB, one device snapshot | Same |
| CUDA memory policy | `strict`, equivalent to `strict-64-128` | Same |
| Default output cap | `--default-max-tokens 0` | Same |
| Sampling | temperature 1, top-p 0.95, top-k 20, min-p 0 | Same |
| Penalties and seed | presence/frequency 0, seed 42; neutral repetition penalty 1 | Same |
| Thinking | Enabled, xhigh, preserve thinking | Same |

The complete source defaults are
`apps/windows-manager/config/profiles/xxs-160k.json` and
`apps/windows-manager/config/profiles/s-128k.json`. Active profiles are read from
`config/profiles/` in the data directory.

An output cap of 0 removes the fixed default cap; clients can still supply their
own limit. It does not expand the context window. Input, reasoning and final output
share the window, so leave generation space. xhigh is an effort level, not a fixed
token budget. Editing context in the form also updates explicit KV capacity;
automatic KV can be configured in the advanced JSON editor.

The data directory's `config/chat_template.jinja` handles mid-conversation system/developer messages,
avoiding the original “System message must be at the beginning” error.
Scanning does not overwrite it with the artifact's embedded template.
`config/device-profiles.json` retains the earlier calibration for this 70-SM
RTX 5070 Ti: separate CUDA 13 Native recalibration did not improve overall performance.
It is not universal. Use a separate calibration file for another GPU; engine auto
mode can update the active file if the device does not match.

`NINFER_PREFILL_ALIGN=0` preserves the actual chunk. Inherited
`NINFER_PROMPT_FAST` and `CUDA_LAUNCH_BLOCKING` overrides are cleared.
The device profile already enables the fast prompt kernel; a reported
`fast_prefill_kernel=false` only means no additional CLI force-enable flag was supplied.

## 4. Choose a memory policy

Use one `--cuda-memory-policy` for everyday device-memory behavior.
“Shared system memory” means Windows borrowing system RAM to back CUDA device
data when VRAM is insufficient. It is separate from the engine's explicit CPU
context cache.

| Value | UI name | Meaning |
|---|---|---|
| `default` | Default policy | Plan against CUDA-reported free memory, without extra strict checks; dedicated VRAM residency is not guaranteed. |
| `mixed` | Allow borrowing system RAM | Try actual allocations beyond CUDA free and accept Shared growth; potentially more capacity, but potentially much slower. |
| `strict` | Dedicated VRAM only | Check device-data residency; verify 64 MiB spare capacity and use a 128 MiB probe step by default. |
| `strict-128-64` | Custom strict mode | Verify 128 MiB spare capacity with a 64 MiB probe step. Both numbers are MiB. |

`strict` equals `strict-64-128`. The form expands reserve and probe-step fields
only for strict mode. Reserve may be zero; the step is 1–16384 MiB.
Strict automatically selects the required Hybrid context cache.
`--host-cache-mib` independently controls CPU context-cache capacity.

### Mixed capacity is bounded

Mixed permits driver-managed borrowing; it does not force data into Shared.
The driver still decides physical placement:

- Explicit KV capacity is attempted as requested; allocation failure is reported.
- With `--kv-capacity auto`, the bound is one `--max-context` window per
  concurrent request, rounded up to KV pages. It does not keep growing idle
  prefix capacity or allocate until system RAM is exhausted.
- Mixed does not use strict reserve or probe settings. `--kv-headroom-mib`
  applies only to default auto sizing; strict auto uses the policy's first number.

Mixed and strict currently support Windows single-GPU text generation and exclude
`--wddm-evictable-budget`. Strict also conflicts with explicitly choosing the
original cache or disabling prefix reuse.

### What strict checks

Strict allows controlled real CUDA allocations instead of treating CUDA free as
the only hard ceiling. Temporary probes are released, and the final weight and
KV arenas remain contiguous. **The final KV pool is not split into many small
allocations.** Spare capacity is temporarily allocated and released at startup,
not permanently reserved.

The CPU cache is touched and transfer-warmed before a Shared baseline is established.
Checks then cover Dedicated/Shared counters, GPU accessibility, and the combined
weight/KV/Graph layout. Zero Shared growth does not mean the process has an absolute
Shared value of zero: the explicitly configured CPU cache can contribute to the baseline.

Probe OOM has fallback, and auto KV candidates have bounded retries; an explicit
context is not silently reduced. Candidates exceeding the Shared baseline are
rejected. Runtime residency failure makes health and new requests return 503.
The policy cannot permanently pin Windows/WDDM placement. One probe verified
640 MiB beyond the initial CUDA free report, then rejected the next block for
Shared growth; that does not promise an extra 640 MiB on every run.

Task Manager, NVML/`nvidia-smi` and CUDA free use different counters and sampling
times. Whole-card free memory is not a promise of an equally large contiguous CUDA
allocation. Observed Graph size is the Dedicated delta during preparation, possibly
including lazy allocations; it is not an exact Graph allocation ledger.

## 5. Built-in monitoring

No `monitor.py` is needed. The manager samples every two seconds and keeps up to
six hours of in-memory history for the current model run. Browser refresh or closure
does not clear it; manager exit or a new model run resets it.

| Metric | What it helps explain |
|---|---|
| Prefill, decode, TTFT | Prompt processing, generation speed and first-output delay |
| MTP acceptance, cache hits and reuse paths | How much drafting is accepted and whether earlier context is reused |
| KV occupancy, transfers, scheduling and pressure | Context-storage and transfer bottlenecks |
| GPU utilization, power, temperature and memory | GPU activity and capacity pressure |
| Successful, failed and rejected requests | Cumulative results, rather than the length of a recent-request list |

Live throughput uses token-counter deltas; completed-request rates use actual
request duration, with separate labels. Latency, throughput, MTP and cache-hit
aggregates have a last-hour window. Logs are read incrementally with bounded tails,
and catch-up is indicated for large existing logs. Unsupported counters remain
unknown instead of becoming zero. NVML whole-card free, CUDA free, and process
Shared baseline/growth are displayed separately.

## 6. Current package performance: 2026-09-29

This section separates the earlier user-stopped main matrix's first round from the later, independently completed three-round XXS 160K chunk 640/1024 comparison at its end. The main matrix was not resumed as a three-round campaign. Both used the current package engine/models and the AppData profiles/template effective at test time, without rebuilding the engine. After the follow-up, the XXS 160K default chunk was changed to 1024; historical tables retain the chunk used in each measurement.

**The user stopped the original main matrix at 19:26 on 29 September. Its tables below remain first-round observations, not best-of-three results.** All four strict configurations completed their first round; mixed completed the 1K, 8K, 32K and 61K requests. Its near-196K request was interrupted during prefill. Rounds two and three were not run. There are 22 complete measured requests and five warmups.

**Hardware and software:** RTX 5070 Ti 16 GB (NVML reports 16,303 MiB total), Ryzen 7 9800X3D, about 32 GB system RAM; Windows 11 build 26200, NVIDIA driver 617.14, CUDA 13.4.2 / Native SM120a Release, D3D12 residency disabled.

### Method and how to read the results

- Five configurations were tested at the time: S 64K and its default 128K used chunk 256; XXS 64K and its then-default 160K used chunk 640; the additional S mixed 196K case used chunk 640. Context and fixed KV capacity are equal. 196K means 200704 tokens.
- All other saved inference settings match section 3: one request at a time, rk8v4, MTP 2 + ngram 31, graph allowance 72 MiB, host cache 6144 MiB, xhigh thinking enabled, and no lm-head-draft. Mixed explicitly selects the same alternate prefix cache; strict selects it automatically.
- Three independent starts per configuration were planned. The completed first round used seed 42 throughout, with a small warmup (93 input / 64 output tokens) before each configuration. Warmups are excluded. Complete measured requests generated exactly 512 tokens. Planned seeds 142 and 242 were not used because the remaining queue was stopped.
- The input is a synthetic English document with repeated laboratory observations, three embedded keys and a final analysis instruction. A changing prefix prevents prompt reuse. Every accepted request has zero cached tokens, root reuse path, a complete SSE response and matching client/server token counts. This is a speed/capacity test, not a coding or answer-quality evaluation.
- Every populated row is one complete request. The count 1/3 means one observation out of three planned runs; it is neither an average nor a selected maximum. No best-of-three result or small statistically significant difference is claimed.
- Prefill is uncached input tokens divided by engine prompt time. Decode is (output tokens − 1) divided by engine predicted time. Client TTFT runs from sending the request to the first nonempty reasoning/content/tool event, including prompt processing. 22/22 valid requests (100.0%) are confirmed to contain only 512 reasoning tokens and no final content/tool calls. Reasoning counts are available for 22/22 requests; reasoning accounts for 100.0% of their output tokens. TTFT is not time to the final answer.
- GPU free is the **lowest sampled whole-card NVML free memory during that request**, in MiB. It is not a promise that cudaMalloc can allocate that amount. GPU samples are approximately every 0.5 seconds; per-process WDDM samples are approximately every second, so shorter peaks may be missed.

[All 22 complete measurements](assets/rtx5070ti-benchmark-20260929-all.csv). The CSV also includes server TTFT, full request duration, sampled power, CUDA residency snapshots where enabled, WDDM Dedicated/Shared, and combined MTP/ngram acceptance.

### Same input: 62,439 tokens, plus 512 generated tokens

| Configuration | Chunk | Valid runs | Prefill tok/s | Decode tok/s | Client TTFT s | Whole request s | GPU free MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| S 64K | 256 | 1/3 | 1,671.4 | 111.0 | 37.44 | 42.05 | 1,910 |
| XXS 64K | 640 | 1/3 | 1,692.4 | 106.9 | 36.98 | 41.77 | 3,134 |
| S 128K | 256 | 1/3 | 1,670.2 | 111.0 | 37.47 | 42.08 | 176 |
| XXS 160K | 640 | 1/3 | 1,684.7 | 106.9 | 37.15 | 41.93 | 534 |
| S mixed 196K | 640 | 1/3 | 53.0 | 4.6 | 1,177.16 | 1,288.76 | 24 |

These rows are first-round observations. The interrupted near-196K request has no complete TTFT or decode result; it is documented separately below.

### Every measured input length

Context capacity is not the number of input tokens. Read the actual input column; leave room for thinking and the answer. The near-capacity rows use different inputs and are not a same-work comparison.

| Configuration | Actual input | Valid runs | Measured round | Prefill tok/s | Decode tok/s | Client TTFT s | GPU free MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| S 64K | 993 | 1/3 | 1 | 1,940.1 | 110.4 | 0.53 | 1,910 |
| S 64K | 8,156 | 1/3 | 1 | 2,048.8 | 121.8 | 3.99 | 1,910 |
| S 64K | 32,742 | 1/3 | 1 | 1,863.5 | 123.5 | 17.64 | 1,910 |
| S 64K | 62,439 | 1/3 | 1 | 1,671.4 | 111.0 | 37.44 | 1,910 |
| XXS 64K | 993 | 1/3 | 1 | 1,873.4 | 121.3 | 0.55 | 3,134 |
| XXS 64K | 8,156 | 1/3 | 1 | 1,987.0 | 130.2 | 4.13 | 3,134 |
| XXS 64K | 32,742 | 1/3 | 1 | 1,850.9 | 121.5 | 17.73 | 3,134 |
| XXS 64K | 62,439 | 1/3 | 1 | 1,692.4 | 106.9 | 36.98 | 3,134 |
| S 128K | 993 | 1/3 | 1 | 1,941.0 | 110.2 | 0.51 | 176 |
| S 128K | 8,156 | 1/3 | 1 | 2,041.3 | 121.5 | 4.00 | 176 |
| S 128K | 32,742 | 1/3 | 1 | 1,862.1 | 123.3 | 17.62 | 176 |
| S 128K | 62,439 | 1/3 | 1 | 1,670.2 | 111.0 | 37.47 | 176 |
| S 128K (near capacity) | 127,970 | 1/3 | 1 | 1,358.8 | 92.9 | 94.34 | 176 |
| XXS 160K | 993 | 1/3 | 1 | 1,868.9 | 121.3 | 0.55 | 534 |
| XXS 160K | 8,156 | 1/3 | 1 | 1,985.3 | 130.1 | 4.12 | 534 |
| XXS 160K | 32,742 | 1/3 | 1 | 1,844.3 | 121.8 | 17.79 | 534 |
| XXS 160K | 62,439 | 1/3 | 1 | 1,684.7 | 106.9 | 37.15 | 534 |
| XXS 160K (near capacity) | 160,726 | 1/3 | 1 | 1,317.4 | 93.2 | 122.22 | 534 |
| S mixed 196K | 993 | 1/3 | 1 | 45.2 | 4.9 | 21.98 | 24 |
| S mixed 196K | 8,156 | 1/3 | 1 | 53.0 | 5.2 | 153.87 | 24 |
| S mixed 196K | 32,742 | 1/3 | 1 | 53.4 | 5.2 | 613.58 | 24 |
| S mixed 196K | 62,439 | 1/3 | 1 | 53.0 | 4.6 | 1,177.16 | 24 |
| S mixed 196K (near capacity) | — | 0/3 | — | — | — | — | — |

### Interrupted request and remaining queue

The user stopped the near-196K request and the remaining queue. This was not an engine crash or out-of-memory failure. It does not establish full-context performance at 196K.

At interruption, about 40,960 input tokens had been computed over 770.2 seconds (12.8 minutes), and the request was still in prefill. No complete TTFT or decode speed is available. This count sums complete logging intervals within this request; a boundary interval carrying 18 decode tokens from the previous request was excluded.

- s-mixed-196k / near: 0/3; no complete measurement.

### Memory and practical configuration choices

- **S 64K:** the lowest sampled GPU free across its valid runs was 1,910 MiB; peak process Shared was 6,538 MiB.
- **XXS 64K:** the lowest sampled GPU free across its valid runs was 3,134 MiB; peak process Shared was 6,538 MiB.
- **S 128K:** the lowest sampled GPU free across its valid runs was 176 MiB; peak process Shared was 6,538 MiB.
- **XXS 160K:** the lowest sampled GPU free across its valid runs was 534 MiB; peak process Shared was 6,538 MiB.
- **S mixed 196K:** the lowest sampled GPU free across its valid runs was 24 MiB; peak process Shared was 8,284 MiB.

Across valid strict requests, the recorded Shared baseline is 6,538.0 MiB; the maximum sampled increase relative to each request's own baseline is 0.0 MiB (baseline/delta available for 18/18 of 18 requests). The baseline includes the deliberately allocated CPU context cache. Shared being nonzero therefore does not by itself mean CUDA device data spilled. Mixed disables strict residency checks: its CUDA-free field is left blank, and a zero disabled-counter value is not reported as a real measurement. Changes in Mixed Shared alone cannot distinguish driver spill from explicit Host cache use.

For this 16 GB card, 64K leaves considerably more room for the desktop and other GPU applications. The shipped S 128K and XXS 160K profiles prioritize context capacity; S 128K in particular has little sampled headroom. Treat these as verified configurations for this machine, not universal guarantees for every 5070 Ti desktop. Increasing capacity does not make a short request faster, because the fixed KV pool still reserves the larger capacity.

The measured S mixed 196K / chunk 640 combination is not recommended for everyday agent use on this machine: even the 8K request took about 251 seconds, while the 61K request took about 21.5 minutes. This comparison changes capacity and chunk as well as policy; it does not imply every mixed configuration has this slowdown.

<!-- xxs-160k-chunks-20260929:start -->
### XXS 160K chunk 640/1024: three-round follow-up

This separate follow-up completed three independent starts for each chunk (six starts, twelve formal requests). Each start used a 93-input/64-output-token warmup, excluded below, followed by 8K and near-160K inputs with 512 generated tokens. Context and fixed KV capacity remain 163840; strict and all other saved inference settings stay unchanged. Paired seeds are 42/142/242; rounds 1 and 3 run 640→1024, while round 2 reverses the order. The same-round pair uses identical input messages and matching effective settings except chunk and log destination. These are new requests, not completion of the stopped main matrix.

**Practical choice:** chunk 1024 completed all 3 independent 160K strict runs on this machine. For near-160K input, paired median prefill improved by 1.54%, and TTFT was 1.845 seconds shorter. The memory cost was 374 MiB more sampled GPU use, with minimum free memory changing from 534 to 160 MiB. **After this follow-up, the XXS 160K default was changed to chunk 1024** for its measured long-input performance. You can select chunk 640 manually when you need more memory headroom for other GPU applications.

**Best whole request from three valid runs:** choose the shortest full wall time in each group. Every metric in a row belongs to that selected request; columns are not independent maxima.

| Chunk / workload | Input / output tokens | Round / seed | Prefill tok/s | Decode tok/s | Client / server TTFT s | Wall s | NVML free MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640 / 8K | 8,156 / 512 | 2 / 142 | 1,985.97 | 129.08 | 4.131 / 4.115 | 8.091 | 534 |
| 1024 / 8K | 8,156 / 512 | 2 / 142 | 1,966.74 | 131.60 | 4.170 / 4.155 | 8.055 | 160 |
| 640 / Near 160K | 160,726 / 512 | 2 / 142 | 1,318.34 | 94.83 | 122.118 / 122.111 | 127.516 | 534 |
| 1024 / Near 160K | 160,726 / 512 | 3 / 242 | 1,338.62 | 97.97 | 120.291 / 120.263 | 125.517 | 160 |

**Paired changes, 1024 relative to 640:** median [minimum, maximum] of three same-round pairs, not the ratio of the selected best rows. Positive throughput change is faster; negative TTFT change is shorter.

| Workload | Prefill change % | Decode change % | Client TTFT change s | Client TTFT change % |
|---|---:|---:|---:|---:|
| 8K | -1.01 [-1.01, -0.97] | -3.15 [-10.68, +1.95] | +0.043 [+0.040, +0.046] | +1.03 [+0.96, +1.11] |
| Near 160K | +1.54 [+1.42, +1.57] | +1.90 [-2.36, +5.51] | -1.845 [-1.867, -1.714] | -1.51 [-1.53, -1.41] |

**Memory:** chunk 640/1024 minimum sampled whole-GPU NVML free is **534 / 160 MiB**. Maximum strict Shared increase above each request's own baseline is **0.0 / 0.0 MiB** and the recorded baselines are 6,538.0 / 6,538.0 MiB. These are sampled counters, not guaranteed allocatable headroom or a permanent residency lock. Explicit Host cache contributes to the Shared baseline.

Only three paired measurements per length were collected; medians and ranges describe this sample, not statistical significance or a general speed guarantee. Complete generated output matched in 0/6 pairs; 12/12 requests are confirmed to contain only 512 reasoning tokens and no final content/tool calls. Decode can vary with generated text and draft acceptance, and TTFT is the first stream output, not the final answer. Saved profiles were unchanged during the experiment; **after it completed, the saved XXS 160K profile and default resources were updated to chunk 1024**, with all other launch settings unchanged.

[All twelve follow-up measurements](assets/rtx5070ti-xxs-chunks-20260929.csv) include server TTFT, sampling power, CUDA/WDDM counters, draft acceptance and output classification. They are separate from the main matrix's 22-request CSV.
<!-- xxs-160k-chunks-20260929:end -->

## 7. Building and supported scope

### Runtime requirements

Validated: **Windows x64, RTX 5070 Ti, driver 617.14**. This executable retains only
SM120a machine code and is not a universal CUDA build. Other compute-capability
12.0 devices are expected to be architecture-compatible but were not individually
tested. RTX 30/40-series GPUs need builds for their supported architectures.

R615 or newer is the deployment recommendation for CUDA 13.4; this package has not
validated compatibility limits on older drivers. References:
[NVIDIA GPU list](https://developer.nvidia.com/cuda/gpus),
[Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/),
[CUDA driver requirements](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver).
With dependencies present, runtime does not require the full CUDA Toolkit,
a compiler, Python, Node.js or a global .NET installation.
Recheck context, Graph allowance, reserve and calibration on another GPU;
parameter changes normally need no recompilation. 5070 Ti measurements are not
measurements for the 5090 or 3090.

### 7.1 Start with the complete source

Use the [`rtx5070ti-cuda13-native` branch of Ryan-gsq/ninfer-all](https://github.com/Ryan-gsq/ninfer-all/tree/rtx5070ti-cuda13-native).
It contains this guide, Manager 1.3.1 and the strict/mixed memory policies. The upstream default branch or an older commit may not include these changes.
For a first checkout, run this command in PowerShell; the destination directory should not already exist:

```powershell
git clone --branch rtx5070ti-cuda13-native --single-branch https://github.com/Ryan-gsq/ninfer-all.git C:\src\ninfer-all
```

If you already have the complete source from this branch, reuse that directory and adjust the paths below.
The build steps use `C:\src\ninfer-all` and first check that required files exist.

Prepare these development tools; they do not belong in the final runtime directory:

| Tool | Requirement for this route |
|---|---|
| Git and PowerShell | Fetch dependencies and run the commands below |
| Visual Studio 2022 Build Tools | Desktop development with C++, x64 MSVC v143 **14.44.35207**, Windows SDK, C++ CMake/Ninja tools |
| CUDA Toolkit | **13.4.2**, including nvcc 13.4.92, nvprune and cuBLAS development/runtime files |
| CMake / Ninja | Local CMake **4.4.3**; use the same version for a new environment. The project's 3.28 minimum does not establish older-version validation with CUDA 13.4/120a. Ninja on PATH |
| Node.js | **22.12 or newer**, matching Vite's current lockfile requirement |
| .NET SDK | **10.x**, for the self-contained Windows x64 manager |
| Python | 3.11 for conversion only; install CPU PyTorch, NumPy and download tools as described in the conversion guide |

Install CUDA from the [official 13.4.2 archive](https://developer.nvidia.com/cuda-13-4-2-download-archive).
The commands assume common VS Build Tools and CUDA installation paths; adjust them
if necessary. Adding only `cl.exe` to PATH is insufficient: headers, libraries and
the Windows SDK also need the developer environment.

Run subsequent steps in **the same PowerShell session**. `$packageRoot` is a new
output directory, not the Downloads installation currently in use. Four build jobs
were used on this 32 GB RAM machine; adjust `$jobs` to available memory.

```powershell
$ErrorActionPreference = 'Stop'
$ninferRepo = 'C:\src\ninfer-all'
$buildRoot = Join-Path $ninferRepo 'build-5070ti-native'
$vcpkgRoot = 'C:\src\vcpkg'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$publishRoot = Join-Path $ninferRepo 'dist\windows-manager'
$packageRoot = 'C:\build\ninfer-package\qwen27b'
$jobs = 4

Set-Location $ninferRepo
foreach ($required in @(
  'apps/windows-manager/NInfer.Manager.csproj',
  'apps/windows-manager/config/profiles/xxs-160k.json',
  'src/product/cuda_memory_options.h',
  'tools/convert/__main__.py'
)) {
  if (-not (Test-Path -LiteralPath (Join-Path $ninferRepo $required))) {
    throw "Incomplete source checkout: $required"
  }
}

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" -vcvars_ver=14.44 >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
$clPath = (Get-Command cl.exe -ErrorAction Stop).Source
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot\bin\x64;$cudaRoot\bin;$env:PATH"
(Get-Item -LiteralPath $clPath).VersionInfo.FileVersion
& "$cudaRoot\bin\nvcc.exe" --version
cmake --version
ninja --version
```

Confirm MSVC 14.44 (the compiler file version normally reads 19.44) and nvcc 13.4.92.
Do not reuse the older 3090 build tree. Incrementally build an already-correct
5070 Ti tree rather than reconfiguring it.

### 7.2 Acquire dependencies and build CUDA operators

This standard route uses the repository's `vcpkg.json`, with its pinned baseline
and curl, FFmpeg/zlib and pkgconf dependencies. On first configure, the vcpkg
CMake toolchain downloads and builds them; this can take substantial time.
It avoids manually assembling FFmpeg include/library paths.
[Microsoft's manifest integration documentation](https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration)
describes this automatic installation.

Use a new build directory. Do not change toolchains inside a tree configured with
the alternative prebuilt-dependency bridge. Dependencies for this route appear
under `$buildRoot\vcpkg_installed\x64-windows`.

```powershell
if (-not (Test-Path -LiteralPath $vcpkgRoot)) {
  git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
  if ($LASTEXITCODE -ne 0) { throw 'vcpkg clone failed' }
}
& "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE -ne 0) { throw 'vcpkg bootstrap failed' }

cmake -S $ninferRepo -B $buildRoot -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_MANIFEST_MODE=ON `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_SM120_NATIVE=ON -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF `
  -DNINFER_DIRECTSTORAGE=OFF -DNINFER_D3D12_RESIDENCY=OFF `
  "-DCUDAToolkit_ROOT=$cudaRoot" `
  "-DCMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe" `
  "-DCMAKE_CUDA_HOST_COMPILER=$clPath" `
  "-DCMAKE_C_COMPILER=$clPath" "-DCMAKE_CXX_COMPILER=$clPath"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $buildRoot --target ninfer_ops -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'CUDA operators build failed' }
```

The configuration is Release, `120a`, Native ON, D3D12 residency OFF and
DirectStorage OFF. Native enables PDL, and Windows uses staged TMA descriptors;
separate compatibility-route options being OFF does not disable these Native paths.
No extra unmeasured cuBLAS/A8 performance options are added.

### 7.3 Prune the archive and link the engine

The measured CUDA archive included redundant PTX and could exceed Windows' 2 GiB PE
limit at final linking. Back it up, use the same toolkit's `nvprune.exe` to retain
SM120a SASS, and replace the link input only after success. This remains a Release
build. Do not apply this single-architecture step to multi-architecture packages.

```powershell
$opsArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.lib'
$prunedArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.native.lib'
$backupRoot = Join-Path $buildRoot 'archive-backup'
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$backupName = 'ninfer_ops-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') + '.with-ptx.lib'
Copy-Item -LiteralPath $opsArchive -Destination (Join-Path $backupRoot $backupName)

& "$cudaRoot\bin\nvprune.exe" -arch sm_120a $opsArchive -o $prunedArchive
if ($LASTEXITCODE -ne 0) { throw 'nvprune failed; original archive was not replaced' }
Copy-Item -LiteralPath $prunedArchive -Destination $opsArchive -Force

cmake --build $buildRoot --target ninfer-serve -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'Engine link failed' }
```

The executable is `$buildRoot\apps\ninfer-serve.exe`. After changing CUDA operators,
build `ninfer_ops`, prune it again, then link; do not skip pruning after an archive
rebuild. Many architecture-specific kernels keep the executable large even in
Release mode. File size alone does not indicate Debug/Release, and these end-to-end
checks do not numerically qualify every NVFP4/MoE route.

### 7.4 Convert GSQ/RCO models

Follow the [download and GSQ GGUF conversion tutorial](rtx-5070ti-windows-downloads.md)
for Swift or ISTA GGUFs and the matching metadata/tokenizer, using `tools.convert`
from the same source. That tutorial includes the Python environment, downloads,
`qwen3_8_27b_gguf`, `text,mtp`, CPU conversion and `--proposal` commands.

For both defaults in this guide, convert Swift `IQ3_XXS` and `IQ3_S` separately.
Keep the documented filenames under the source tree's `converted-models/`.
The original GGUFs, metadata and conversion reports are build inputs, not runtime
package contents. The converter is not shipped with the application. Conversion
does not compile the CUDA engine or requantize the selected GGUF.

### 7.5 Build the website and tray manager

Install npm dependencies from the lockfile and build the website before publishing
C#. Vite writes to `apps/windows-manager/wwwroot/`, which the C# project copies to
the publish output. Reversing the order can publish a stale or missing website.

```powershell
node --version
dotnet --list-sdks
Push-Location (Join-Path $ninferRepo 'apps\windows-manager\web')
try {
  npm.cmd ci
  if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
  npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Website build failed' }
} finally { Pop-Location }

dotnet publish (Join-Path $ninferRepo 'apps\windows-manager\NInfer.Manager.csproj') `
  -c Release -r win-x64 --self-contained true `
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true `
  -o $publishRoot
if ($LASTEXITCODE -ne 0) { throw 'Manager publish failed' }
```

The EXE in `$publishRoot` includes the .NET runtime. Conversion Python, build-time
Node.js and the .NET SDK are not runtime-package requirements.
Manager-only changes also require no CUDA rebuild.

### 7.6 Assemble a fresh runtime directory

Copy the full vcpkg **Release bin** DLL set, including FFmpeg/curl transitive
dependencies, then add cuBLAS/Lt from the same CUDA toolkit and the x64 VC runtime.
Do not rely only on the few DLLs that happen to appear beside the CMake executable
or mix identically named files from unrelated FFmpeg/CUDA installations.
This Windows route statically links cudart, but the package below also carries
the toolkit runtime for dependent libraries. The VC files come from the installed
MSVC Redist directory; see the
[Microsoft VC Redistributable requirements](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

**Source seed profiles and on-disk runtime profiles use different outer formats.**
The embedded seeds at `apps/windows-manager/config/profiles/*.json` have `id`,
`modelPath` and other fields directly at the top level. Disk profiles require a
`schemaVersion / savedAt / profile` wrapper. The script creates that wrapper;
do not recursively copy the whole source config directory as active configuration.
Alternatively, omit package config completely and let first launch initialize
everything from embedded defaults.

```powershell
if (Test-Path -LiteralPath $packageRoot) {
  throw 'Choose a new, empty packageRoot; do not overwrite a running installation'
}
foreach ($directory in @('', 'engine', 'model', 'config\profiles', 'docs\assets', 'licenses')) {
  New-Item -ItemType Directory -Path (Join-Path $packageRoot $directory) -Force | Out-Null
}
Copy-Item -LiteralPath (Join-Path $publishRoot 'NInferManager.exe') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $publishRoot 'wwwroot') -Destination $packageRoot -Recurse
$engineDir = Join-Path $packageRoot 'engine'
Copy-Item -LiteralPath (Join-Path $buildRoot 'apps\ninfer-serve.exe') -Destination $engineDir

$dependencyRoot = Join-Path $buildRoot 'vcpkg_installed\x64-windows'
Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'bin') -File -Filter '*.dll' |
  Copy-Item -Destination $engineDir

foreach ($dll in @('cublas64_13.dll', 'cublasLt64_13.dll', 'cudart64_13.dll')) {
  $source = Join-Path $cudaRoot "bin\$dll"
  if (-not (Test-Path -LiteralPath $source)) { $source = Join-Path $cudaRoot "bin\x64\$dll" }
  if (-not (Test-Path -LiteralPath $source)) { throw "Missing CUDA DLL: $dll" }
  Copy-Item -LiteralPath $source -Destination $engineDir
}
$crtRoot = Join-Path $env:VCToolsRedistDir 'x64\Microsoft.VC143.CRT'
if (-not (Test-Path -LiteralPath $crtRoot)) {
  throw 'Locate the installed MSVC x64 redistributable directory and set crtRoot'
}
Get-ChildItem -LiteralPath $crtRoot -File -Filter '*.dll' | Copy-Item -Destination $engineDir

$seedRoot = Join-Path $ninferRepo 'apps\windows-manager\config'
Get-ChildItem -LiteralPath $seedRoot -File | Copy-Item -Destination (Join-Path $packageRoot 'config')
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $seedRoot 'profiles') -File -Filter '*.json') {
  $profile = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  $document = [ordered]@{
    schemaVersion = 1
    savedAt = [DateTimeOffset]::UtcNow.ToString('O')
    profile = $profile
  }
  $target = Join-Path $packageRoot "config\profiles\$($file.Name)"
  [IO.File]::WriteAllText($target, ($document | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
}

foreach ($name in @('rtx-5070ti-windows.md', 'rtx-5070ti-windows.en.md', 'rtx-5070ti-windows-downloads.md')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\$name") -Destination (Join-Path $packageRoot 'docs')
}
foreach ($name in @('ninfer-tray-running.png', 'ninfer-tray-stopped.png', 'rtx5070ti-benchmark-20260929-all.csv', 'rtx5070ti-xxs-chunks-20260929.csv')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\assets\$name") -Destination (Join-Path $packageRoot 'docs\assets')
}
Copy-Item -LiteralPath (Join-Path $ninferRepo 'LICENSE') -Destination $packageRoot
foreach ($port in Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'share') -Directory) {
  $copyright = Join-Path $port.FullName 'copyright'
  if (Test-Path -LiteralPath $copyright) {
    $licenseDir = Join-Path $packageRoot "licenses\$($port.Name)"
    New-Item -ItemType Directory -Path $licenseDir -Force | Out-Null
    Copy-Item -LiteralPath $copyright -Destination $licenseDir
  }
}
Get-ChildItem -LiteralPath $cudaRoot -File |
  Where-Object { $_.Name -match 'LICENSE|EULA' } |
  Copy-Item -Destination (Join-Path $packageRoot 'licenses')
```

Now copy the converted models. Remove the S entry from `$entryNames` if you only
need XXS. For different filenames or ISTA weights, select the actual model in the
website and save a separate profile.
The commands use each `.conversion.json` report's `files` list for the entry and
all volumes, without guessing volume names or overwriting existing model files.
Keep reports and outputs at their conversion location until copying is complete;
reports do not need to be included in the runtime package.

```powershell
$convertedModelRoot = Join-Path $ninferRepo 'converted-models'
$entryNames = @(
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer',
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer'
)
$modelDir = Join-Path $packageRoot 'model'
$filesToCopy = @(foreach ($entryName in $entryNames) {
  $reportPath = Join-Path $convertedModelRoot ($entryName + '.conversion.json')
  $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (-not $report.files) { throw "Conversion report has no files: $reportPath" }
  foreach ($file in $report.files) {
    [pscustomobject]@{
      source = $file.path
      target = Join-Path $modelDir ([IO.Path]::GetFileName($file.path))
    }
  }
})
foreach ($file in $filesToCopy) {
  if (-not (Test-Path -LiteralPath $file.source -PathType Leaf)) { throw "Missing model volume: $($file.source)" }
  if (Test-Path -LiteralPath $file.target) { throw "Model file already exists: $($file.target)" }
}
foreach ($file in $filesToCopy) {
  [IO.File]::Copy($file.source, $file.target, $false)
}
```

This produces the PackageRoot structure in section 2: one manager, one engine
directory, models, website, initial configuration and documentation. It does not
include the converter, build cache, test launchers or another old engine.
When redistributing software, retain the applicable project, CUDA, VC runtime and
dependency licenses.

### 7.7 Check the standalone directory and launch

Temporarily remove development tools from PATH and run engine `--help`, checking
that it does not secretly load DLLs from a development directory. This establishes
basic loading, not qualification of every inference path. Error 0xC0000135 usually
means a missing direct or transitive DLL; fix `engine/` instead of hiding the
problem behind the developer PATH.

Then launch the manager with `--no-autostart --open` to inspect configuration before
manually loading a model.

```powershell
$savedPath = $env:PATH
try {
  $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
  & (Join-Path $packageRoot 'engine\ninfer-serve.exe') --help
  if ($LASTEXITCODE -ne 0) { throw 'Packaged engine failed to start; inspect runtime dependencies' }
} finally { $env:PATH = $savedPath }

Start-Process -FilePath (Join-Path $packageRoot 'NInferManager.exe') `
  -ArgumentList @('--root', "`"$packageRoot`"", '--no-autostart', '--open') `
  -WindowStyle Hidden
```

The manager initializes AppData for this installation location. Website edits go
to DataRoot, not package seeds; changing package config does not overwrite an
existing DataRoot. Templates and device profiles resolve from DataRoot, while
engine/models resolve from PackageRoot. Moving the complete verified package to
its final location, such as Downloads/qwen27b, produces another installation ID
and first-use initialization as described in section 2.

Confirm model readiness in the site, send a short request, inspect logs and memory,
then stop the model. Management should remain available after stopping inference;
exiting the manager closes monitoring as well. Login startup requires explicit
enabling and is not enabled by this test command.

**Audit scope:** these instructions were checked against repository CMake,
the vcpkg manifest, the successful local build scripts, manager resource formats
and completed deployment checks. The measured engine used MSVC 14.44, CUDA 13.4.2
and a prebuilt FFmpeg/curl bridge. This audit did not reinstall all of vcpkg/CUDA/VS
on a blank Windows system or repeat the complete build from that environment.
Different dependency versions are not claimed to produce identical binaries.
The standard vcpkg route still needs dependency building and basic launch checks
on the reader's own development machine.

### 7.8 Completed manager validation

Manager 1.3.1 built in Release mode with zero warnings and zero errors.
All 117 backend checks and 37 platform checks passed; the latter include 19
isolated startup checks. A real Windows ACL test denied writes to the program
directory while AppData storage continued to work. The original ACL was restored
after the test.

The deployed application was also verified: 18 configuration files were imported
into AppData without content changes, and XXS 160K started successfully. Its
template and device-profile arguments pointed to AppData's `config/`, while logs
were written to AppData's `logs/`. A short API request returned `OK` (16 input and
2 output tokens), after which the model was stopped. **A real reboot/sign-in has
not been performed**; isolated startup checks do not replace that test. This is
separate from the GPU benchmark in section 6.

Manager checks require no GPU. Run from the repository root:

```powershell
dotnet build tests/windows-manager-platform/PlatformTest.csproj -c Release
dotnet tests/windows-manager-platform/bin/Release/net10.0-windows/PlatformTest.dll
dotnet run --project tests/windows-manager-backend/BackendCheck.csproj -c Release
```

Use `dotnet <assembly>` for platform checks so fake children and signal helpers
share the host. Optional `NINFER_TEST_ARTIFACT` checks a real model's headers
without loading tensors or creating a GPU context.
