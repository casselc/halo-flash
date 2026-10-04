# halo-flash: a Strix Halo–specialised Qwen3.8-Flash-Next engine on HRX

## Context

**Goal.** A narrow inference engine for one model, Qwen3.8-Flash-Next, on one machine: Ryzen AI MAX+ 395 with a gfx1151 iGPU, an XDNA2 NPU, Zen5 CPUs, 128 GB of unified memory and NVMe. It runs on the new HRX driver stack in `../hrx-system`.
- **Primary use:** an OpenAI-compatible completions server.
- **Secondary use:** in-process embedding behind a C ABI, so a harness can drive sessions directly.
- **Workload:** one user, a few concurrent sessions, mostly on similar context.

**Decisions taken:**
- This repository.
- C++20 core, kernels compiled to HSACO, and a stable C ABI.
- MIT pieces borrowed selectively from `../gufo` and `../ds4`.
- The NPU as a gated research lane, not part of the critical path.

**What exists today:**
- **gufo** (C++20/HIP, MIT) already serves Flash-Next on Halo:
  - single-user greedy decode: 25.9 tok/s AR, 32 tok/s with MTP on mixed text;
  - prefill: 1628 tok/s at pp2048 and 1292 tok/s at 128K depth;
  - 8 concurrent users: 108.7 tok/s aggregate AR;
  - a full OpenAI server and a RAM/disk prefix cache.
- gufo's profile (`docs/models/qwen3.8-flash-next/EXPERIMENTS.md`) shows where time goes:
  - routed experts stream at only 160–190 GB/s;
  - dense projections reach 30–34 of 59 TFLOPS.
- **ds4** (C, MIT) has Flash-Next on Metal and CUDA only. Its `ds4_qwen4_cuda.cuh` (2.3k lines) is a compact kernel reference. It also contributes the n-gram disk reader, expert-streaming design, disk KV format and Alibaba-continuation quality fixtures.
- **HRX** (Apache-2.0, v0.1.0, not built locally):
  - GPU via libhrx: load HSACO, `hrx_stream_dispatch`, graphs, timeline semaphores, VMM.
  - CPU via an IREE task device.
  - The NPU only via libamdf plus Loom `.xdna` images. No matmul has run on hardware yet.
  - Disk→device io_uring file reads exist in the HAL (`runtime/.../amdgpu/host_queue_file.c`) but are not exported by libhrx.

**Model facts the design rests on** (from `config.json`):
- **Size:** 125B total, about 6B active per token.
- **Layers:** 48 = 12 × (3 Gated DeltaNet + 1 sparse-attention layer).
- **Width and residual:** hidden size 2560. Four hyper-connection residual streams with a rank-320 mixer.
- **GDN:** 16 QK heads and 48 V heads, head dim 128, fp32 state.
- **Attention:** gated GQA with 24 Q heads and 2 KV heads, head dim 256, partial RoPE on 64 dims with mRoPE.
- **QSA indexer:** 4 heads × 128, 4:1 block pooling, a 2048-token budget.
- **MoE:** 512 experts, top-10 plus 1 shared, expert FFN width 640.
- **PLE n-gram table:** 51B parameters, about 95 GiB in BF16, injected at layer 2. It stays on disk and is read as 320-byte rows.
- **MTP:** one layer of about 4B parameters.
- **Vocabulary and context:** 248,320 tokens; 262K native context.
- **Vision tower:** present; out of scope until the text path is done.

**Roofline: what "as fast as possible" means here.**
- One decoded token reads about:
  - 2.6B expert parameters (11 experts × 3 × 2560 × 640 × 48 layers);
  - about 2.7B GDN and attention projection parameters;
  - 0.64B LM-head parameters;
  - about 113 MB of GDN state, read and written.
- At Q4_K experts and Q8_0 dense weights that is about 5.2 GB per token. At about 215 GB/s achievable, the AR ceiling is about 41 tok/s. gufo reaches 63% of it.
- Two levers move the ceiling itself:
  - **A cheaper dense quantization**, if it passes the quality gate. Q5/Q6-class dense weights give about 3.8 GB per token, about 55 tok/s.
  - **MTP.** The paper reports about 4.1 accepted tokens at 4-step drafting.
- Bandwidth is shared by the CPU, GPU and NPU. So the CPU and NPU help by hiding latency and adding compute during compute-bound prefill, not by adding decode bandwidth. The plan is shaped around that.

**Targets.** These are baseline-relative and measured on this machine against gufo at a pinned commit, with the same weights and protocol:
1. Single-user AR decode ≥ 85% of the bandwidth roofline at d0: ≥ 35 tok/s at gufo's quantization mix, versus 25.9.
2. MTP decode ≥ 1.5× our own AR.
3. Prefill ≥ gufo at pp2048, and no worse than gufo at 32K and 128K.
4. Restoring a shared prefix takes under 1 s for any cached prefix.
5. Every step passes the quality gates (below) before its speed counts.

## Architecture

```
halo-flash/
  include/halo_flash.h        C ABI (the server is built on it, too)
  src/core/                   gguf reader, tokenizer + Qwen chat template (from gufo, MIT), sampling, JSON/grammar constraint
  src/model/                  config, tensor binding, layout planner, CPU fp32 reference oracle
  src/rt/                     Device interface; backends: hrx (native libhrx) and hip (ROCm) for A/B
                              arena allocator (unified memory), io/ (io_uring, O_DIRECT), ngram store, expert cache
  src/kernels/                HIP-source kernels → HSACO for gfx1151 (one file per op family)
  src/engine/                 per-layer step programs, graph capture/replay, session, batch scheduler, MTP
  src/server/                 OpenAI HTTP over the C ABI
  tools/                      repack, bench, logit compare, expert profiler
  tests/                      op tests vs oracle, session/replay tests, server functional tests
```

**Runtime: HRX native first, HIP second.**
- The kernels are written in HIP C++ and compiled with clang `--offload-arch=gfx1151` (device-only). The extracted HSACO is loaded through `hrx_executable_load_data(..., "amdgpu", "gfx1151")`.
- Decode runs as one `hrx_graph` per batch shape. Positions and similar values sit in device buffers, gufo's approach, so graphs replay at any depth.
- CPU work is ordered against the GPU queue with HRX timeline semaphores and `hrx_queue_host_call`, never with blocking syncs. That CPU work is the n-gram gather, sampling, grammar masks and MTP accept/reject bookkeeping.
- A thin `rt::Device` interface also runs on ROCm HIP. Every op and the end-to-end path can then be A/B'd between HRX and ROCm, and HRX churn can't block progress.

**Memory.**
- One arena of HOST_VISIBLE|DEVICE_LOCAL unified-memory allocations, with weights page-aligned.
- Memory admission follows ds4's `ds4_rocm_memory.h` logic: MemAvailable minus CmaFree, with a 2 GiB OS reserve, so we fail cleanly before an OOM.
- Buffers the NPU may DMA from are private anonymous memory. HRX documents IO page faults on Strix NPU5 with shared or file-backed mappings.

**Weights.**
- **Phase 1** reads gufo's exact files as-is: unsloth UD-Q4_K_XL @38bb39ee, 4 shards, plus the MTP Q8_0 sidecar. That keeps baselines apples-to-apples.
- **Later**, `tools/repack` writes a native layout:
  - tensors pre-tiled for our kernels;
  - each (layer, expert) triple contiguous and 4K-aligned, for streaming;
  - the n-gram table last and page-aligned.

  It is keyed by a hash of the source GGUF, so a repack never changes semantics silently.

**Disk.**
- **n-gram table.** Rows depend only on token ids, so the CPU gathers them ahead of time: for prompt chunks before the GPU reaches layer 2, and for MTP draft tokens before verification.
  - Reads use io_uring with O_DIRECT and sorted, batched 4K-aligned reads.
  - An LRU row cache in RAM covers frequent n-grams.
  - The decode path never blocks on I/O it could have issued one step earlier.
- **Expert streaming** comes later (Phase 8). It is an option, not a requirement: Q4 resident weights (about 70 GiB) fit.
  - It frees RAM for sessions and prefix caches, and allows higher-precision experts with a hot cache.
  - Design from ds4: LRU, hot-list seeding from a profiler, prefetching the next layer while the current one computes.
  - We add io_uring and router lookahead.
- `/home` is ZFS. Phase 0 measures O_DIRECT and io_uring random-read IOPS there and on a plain ext4/xfs volume or raw partition before committing to the weight location.

**Sessions** (similar contexts, few users):
- A session holds:
  - KV for the 12 attention layers, about 24 KB per token in F16;
  - indexer keys;
  - GDN plus conv state, about 113 MB fixed;
  - MTP state and RNG.
- **Snapshots are first-class.** `fork(session)` copies the fixed recurrent state. Attention KV is shared through refcounted, copy-on-write pages, so N sessions on the same long prefix pay for it once.
- **Disk tier:** ds4's text-prefix-keyed checkpoint format (`ds4_kvstore.c`), carrying the recurrent-state payload.
- **Batched decode** groups every session's routed tokens by expert, so concurrent sessions share expert reads. This is the main multi-user win.

## Testing and robustness: Hegel property tests plus sanitizers

**Framework.** hegel-cpp, pinned with FetchContent: `GIT_TAG v0.13.1` or later, verified at P0. It is Hypothesis-based and links a prebuilt native libhegel; no Python server is needed. Tests run under GoogleTest via `hegel::test(...)`, so failures shrink and replay with `HEGEL_REPRODUCE_FAILURE`, and an example database is stored per test.

**Property tests** (`hegel::test` with generators):
- Each kernel and op is checked against the CPU oracle over generated shapes, positions, quant blocks, batch sizes and chunk splits. Tolerances are fixed before the kernel is written.
- **Quant formats:** pack then dequantize round-trips, and GEMV/GEMM results equal dequantize-then-fp32 over random blocks, including edge rows and cols.
- **Chunking:** chunked prefill equals one-shot prefill for any partition of the prompt into chunks.
- **GDN:** the chunked-parallel scan equals the sequential recurrence.
- **QSA:** the selected block set equals the oracle's top-k with the documented tie rule.
- **MoE:** expert bucketing is a permutation of the (token, slot) pairs, and padding never leaks into outputs.
- **Text:** tokenizer encode then decode is the identity on generated UTF-8. Chat-template render then parse round-trips. The tool-call parser never crashes on generated partial or malformed streams.
- **Sampling:**
  - top-k/p/min-p masks are supersets of the argmax;
  - a seeded RNG gives identical streams;
  - MTP p/q acceptance preserves the target distribution, checked statistically on small synthetic vocabularies.
- **Disk formats:** the KV checkpoint, snapshot and repack formats round-trip. Truncated or corrupt files are rejected and never misread.

**Stateful tests** (`hegel::stateful::StateMachine`). Rule selection is the engine's, so swarm testing (random subsets of rules enabled per run) comes built in.
- **Session machine.** Rules: create, sync(prefix edits), eval, decode_batch, MTP step with forced accept/reject, rewind, fork, snapshot save/restore, close.
  - A reference model replays each session's token history on the oracle, or on the GPU path in isolation.
  - Invariant: every live session's logits and next token equal its isolated replay. This is the batch-isolation and rollback-exactness property gufo and ds4 test by hand.
- **Prefix cache machine.** Rules: insert, lookup by prefix, evict, budget change, simulated restart reloading the disk tier. Invariants:
  - a hit is always a true prefix;
  - the restored state equals the recomputed state;
  - bytes stay within budget.
- **Allocator and COW KV pages machine.** Invariants: refcounts are exact, a shared page is never written, and freeing everything leaves zero bytes held.
- **Expert cache and streaming machine**, with a fake I/O layer. Rules: request, prefetch, evict, I/O completes out of order, I/O fails. Invariants:
  - an expert in the current layer's selection is never evicted;
  - data comes back byte-equal to the file;
  - failure surfaces as an error, never as stale weights.
- **n-gram store machine.** The same pattern, with short and failed reads injected.

**Concurrent stateful tests** (`hegel::stateful::run_concurrent`, rule groups):
- Server and scheduler admission: many clients submit, cancel and disconnect concurrently against the scheduler, with linearizable outcomes checked against a sequential model.
- C ABI thread-safety rules: concurrent session ops on distinct sessions, and the documented serialization on shared ones.
- The io_uring completion path.
- All of these also run under TSan.

**Sanitizers.**
- CMake presets:
  - `asan` (address,undefined, with LeakSanitizer);
  - `tsan`;
  - `ubsan-strict` (`-fno-sanitize-recover`).

  They follow hegel-cpp's own sanitizer justfile.
- GPU code can't run under host sanitizers on gfx1151. So `rt::Device` has a **CPU backend** that executes the same step programs through the oracle kernels. With it, the scheduler, sessions, caches, I/O, server and the C ABI all run fully under ASan and TSan without a GPU.
- GPU kernels get a `HALO_KERNEL_CHECKS` build: bounds and NaN asserts in the kernels, and guard pages and canaries around arena allocations, verified after each step.
- An `fuzz` preset (libFuzzer plus ASan) covers the HTTP request parser, the JSON parser and the GGUF/metadata reader, all of which consume untrusted bytes.
- CI-able targets:
  - `ctest --preset cpu-asan`, `cpu-tsan`, `cpu-ubsan` (no GPU, no weights; uses a tiny synthetic Flash-Next-shaped model, see P1);
  - `ctest --preset gpu-test`;
  - `ctest --preset gpu-checks`.

## Phases (each ends with a measured gate; no phase's speed claim counts until its quality gate passes)

**Ordering against the weight download.** The weights are still downloading over a 50 Mbps line: about 165 GB, roughly 8 hours or more. Only shard 1, the MTP sidecar and mmproj are present so far, in `../gufo/models/qwen3.8-flash-next/`.
- P0 microbenchmarks, the repo skeleton and the test infrastructure run first and need no weights.
- P1's synthetic model and P2's op tests need no weights either.
- Steps that need weights wait until the shards are complete and their sha256 checks against the HF manifest pass: real-model oracle runs, the baseline benchmarks and the quality fixtures.

**Storage.**
- The new dedicated NVMe gets a non-ZFS filesystem, xfs or ext4 (to be decided by P0 measurement), mounted for weights, the n-gram table, the expert-stream source and the disk KV tier.
- P0 measures io_uring and O_DIRECT random-read IOPS at 320 B, 4 KB and 64 KB–1 MB on that drive versus ZFS `/home`.
- The weights move there once verified. gufo's baselines are taken from the same location, so the comparison is fair.

**P0 — Bring-up and baselines.**
- Create the repo with CMake presets, a gfx1151-only build and CI-able unit tests:
  - `cpu-*` sanitizer presets;
  - a `gpu-test` preset;
  - hegel-cpp and GoogleTest via FetchContent.
- Build hrx-system with `python dev.py cmake ...` (`-DIREE_HAL_AMDGPU_TARGETS=gfx1151`, ROCm from distro 7.1 or nix 7.2.3) and run its CTS on this box.
- Microbenchmarks under native libhrx, ROCm HIP and the HRX HIP shim:
  - dispatch latency;
  - graph replay latency;
  - peak read bandwidth;
  - allocation above 64 GB;
  - host-call round trip.
- NVMe: 320 B, 4 KB and 1 MB random-read IOPS and throughput, with io_uring and with pread, on ZFS and on the alternative volume.
- **Baselines.** Use the existing gufo build at `../gufo/result/bin/{gufo,gufo-server}` (commit 8bdde80) and weights from `../gufo/models/qwen3.8-flash-next/`; build llama.cpp. When the weights are complete, record their numbers with gufo's protocol (pp2048/tg128 at depths 0, 32K and 128K; C1–C8). Run the baselines with gufo's own `tools/gufo/model_bench`.
- **Gate:** a written baseline table, and an HRX-or-HIP decision per subsystem based on the measured latency.

**P1 — Correctness spine.**
- The GGUF reader, config and tensor binding.
- The tokenizer and chat template, adapted from gufo `src/models/qwen/`.
- A CPU fp32 reference oracle for the full forward pass, including MTP and QSA. It is ported from gufo `QFN/reference.cpp`, with ds4's kernels as a second reading.
- **A synthetic tiny model.** It has the same architecture with shrunken dimensions: 8 layers in the 3:1 pattern, 16 experts, a small vocabulary and a small n-gram table. It is generated as a GGUF with seeded random weights by `tools/make_tiny`. Every test that doesn't need the real model uses it, which keeps the sanitizer and property suites fast and weight-free.
- **Quality fixtures:**
  - ds4's `data/qwen38-flash-alibaba-{100,long}` with `score_official`-style NLL;
  - greedy token traces from gufo on a fixed prompt set.
- **Gate:** the oracle reproduces gufo's greedy tokens on the fixture prompts, and the NLL matches ds4's Q4 figure (0.290 on the short set) within a pre-set tolerance.

**P2 — Correct GPU forward pass (AR, single session).**
- Straightforward kernels per op: RMSNorm, the HC mixers, GDN conv + recurrence, attention + QSA indexer/select, router top-k, MoE, the PLE inject and the LM head.
- Each kernel has an op test against the oracle, with tolerances fixed before the kernel is written (ds4's rule: never loosen a tolerance to accept a faster kernel).
- Chunked prefill, and exact replay across chunk boundaries.
- **Gate:**
  - logits within tolerance of the oracle;
  - greedy tokens identical to gufo on the fixtures;
  - long-context recall on ds4's 31K story test.

**P3 — Decode speed, the core.** This is where most of the effort goes.
- Fused per-layer kernels:
  - HC read + RMSNorm + projection;
  - GDN step with the fp32 state kept in one pass;
  - router + top-k + shared-expert gate;
  - the MoE gate/up SwiGLU + down, with the HC-combine epilogue.
- Bandwidth-oriented GEMV kernels per quant format, with per-kernel achieved-GB/s accounting against P0's measured peak.
- The whole step captured as one graph, with CPU work overlapped through semaphores.
- **Quant-mix study:** dense Q8_0 against Q6/Q5-class formats, and an LM head at lower precision. A mix is adopted only if it passes the P1 NLL gate.
- **Gate:** at least 35 tok/s at gufo's quantization mix. Per-mix numbers are reported separately.

**P4 — Prefill speed.**
- WMMA int8 and F16 GEMMs for the dense projections.
- Expert-bucketed grouped GEMMs for MoE, using gufo's `RoutedCompact` idea.
- Chunked parallel GDN, and a QSA prefill with the pooled indexer.
- The n-gram gather fully overlapped with the earlier layers.
- Avoid repeating gufo's rejected experiments, listed in its `EXPERIMENTS.md`.
- **Gate:** at least gufo's prefill at pp2048, 32K and 128K.

**P5 — MTP speculative decoding.**
- The draft layer reuses the target's QSA indices, as the paper describes.
- Rollback without replay: double-buffered recurrent-state snapshots after each verified row, using ds4's `qwen4_graph_state_swap` approach.
- Adaptive draft depth, from gufo's `mtp_policy`.
- Sampled acceptance uses p/q with residual correction.
- n-gram rows for drafted tokens are prefetched.
- **Gate:**
  - MTP outputs are identical to AR under greedy decoding, and distribution-correct under sampling (gufo's sampling-audit approach);
  - at least 1.5× AR on mixed text.

**P6 — Sessions, batching and the prefix cache.**
- Session pool, fork and snapshot, with COW KV pages.
- Batched decode with cross-session expert grouping, including MTP batches.
- Prefill chunks interleaved with active decodes.
- RAM and disk prefix tiers.
- **Gate:**
  - batched runs produce logits and tokens identical to isolated runs (gufo and ds4 both enforce this);
  - C2–C8 aggregate at least gufo's;
  - shared-prefix restore under 1 s.

**P7 — C ABI and server.**
- **`halo_flash.h`** covers:
  - engine open/close with options;
  - `session_create`, `session_fork`, `session_sync`(full token prefix; the engine reuses what it can), `eval`, `sample`, `decode_batch`, `logits`/`top_logprobs`;
  - snapshot save and load;
  - tokenize, detokenize and chat render;
  - cancellation and progress callbacks, and thread-safety rules.

  It follows ds4's engine/session idiom.
- **Server**, a thin layer over the ABI:
  - endpoints `/v1/completions` (primary), `/v1/chat/completions` and `/v1/models`, with SSE streaming;
  - Qwen tool calls (`<tool_call><function=…>`) with constrained argument decoding, adapted from gufo;
  - `/health`, `/ready` and `/metrics`;
  - keep-alive.
- **Gate:**
  - an OpenAI-SDK functional suite (ported from gufo `tests/functional`);
  - a C ABI example and test that embeds the engine in-process.

**P8 — Expert streaming from disk.**
- A hot-expert cache budget.
- io_uring O_DIRECT reads into the unified-memory cache.
- Router-lookahead prefetch.
- A profiler-generated hot list.
- **Gate:** documented decode and prefill numbers per cache size, and no regression when the cache holds every expert.

**N-lane — NPU research, gated and run in parallel from P2.**
1. **N1.** Build libamdf and Loom with XDNA enabled. Run the existing Strix Halo NPU5 CTS programs on this box.
2. **N2.** Write and measure a Loom int8 and bf16 GEMM tile on AIE2P: throughput, submission latency, and GPU↔NPU handoff cost through libamdf shared memory.
3. **N3.** Evaluate one candidate offload end-to-end. The first candidate is prefill dense projections running alongside the GPU's MoE work, since prefill is compute-bound. The second is the MTP draft layer during GPU verification.
- **Go only if** the end-to-end metric improves by at least 5% with identical outputs. Otherwise the result is recorded as a finding and the lane stays out of the hot path.

## Reuse map

| Need | Source |
|---|---|
| GGUF reader, multi-shard mmap | gufo `src/core` |
| Tokenizer, chat template | gufo `src/models/qwen/tokenizer.cpp`, `chat_template.cpp` |
| CPU oracle | gufo `src/models/qwen38_flash_next/reference.cpp`, `cpu_ops.cpp` |
| Kernel references | gufo `QFN/kernels/rocm/kernels.hip.cpp`, `mmq/` (llama.cpp, MIT); ds4 `ds4_qwen4_cuda.cuh` |
| MTP policy and costs | gufo `QFN/mtp_policy.hpp`, `mtp_costs.hpp`; ds4 state-swap rollback |
| n-gram disk reader | ds4 `ds4.c:57347-57480`, gufo `QFN/ngram.cpp` |
| Expert streaming, hot lists | ds4 `rocm/ds4_rocm_runtime.cuh:1940-2110`, `ds4_ssd.c` |
| Memory admission | ds4 `ds4_rocm_memory.h` |
| Disk KV format | ds4 `ds4_kvstore.c` |
| Quality fixtures, scorer | ds4 `gguf-tools/quality-testing/score_official.c`, `data/qwen38-flash-alibaba-*` |
| Server, functional tests | gufo `src/cli/serve/*`, `tests/functional/` |
| HRX dispatch example | hrx `libhrx/cts/tests/executable/executable_test.cpp` |

Copied MIT code keeps its notices in `NOTICE` and `licenses/`. The project is MIT; HRX is linked as an Apache-2.0 dependency.

## Verification

- **Unit, property and stateful tests:**
  - `ctest --preset cpu-asan`, `cpu-tsan` and `cpu-ubsan` on the tiny model, with no GPU. Every phase gate requires these to be clean.
  - `ctest --preset gpu-test` and `gpu-checks`: Hegel op properties against the oracle with fixed tolerances, also run under the HIP backend.
  - Any shrunk failure from Hegel is committed as a regression test, using `HEGEL_REPRODUCE_FAILURE` or an explicit case.
- **Fuzzing:** a time-boxed run of the `fuzz` preset before each server-facing change.
- **Model quality:**
  - fixture NLL with `tools/score` against ds4's Alibaba data;
  - greedy-token identity with gufo;
  - chunk-boundary and batch-isolation replay tests;
  - 31K-story recall.
- **Speed:** `tools/bench` reproduces gufo's protocol (pp2048/tg128, depths 0, 32K and 128K, C1–C8; AR and MTP, mixed and repetitive). Results go to `docs/BENCHMARKS.md`, with the commit, weights hash and the gufo/llama.cpp numbers from the same session.
  - Per-kernel achieved-GB/s and TFLOPS appear against the P0 ceilings.
- **Server:** the OpenAI SDK functional suite (completions, chat, streaming, tools, cancellation) against `halo-flash serve`. The embedding example drives two forked sessions in-process.
- **Hygiene:**
  - commits as Chuck Cassel with no AI attribution, following the standing rule;
  - no force-push;
  - no machine-local paths in committed docs.
