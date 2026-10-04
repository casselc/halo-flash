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
- **HRX** (Apache-2.0, v0.1.0; built locally at P0):
  - GPU via libhrx: load HSACO, `hrx_stream_dispatch`, graphs, timeline semaphores, VMM.
  - CPU via an IREE task device.
  - The NPU only via libamdf plus Loom `.xdna` images. (When this plan was written no matmul had run on the hardware; int8 GEMMs now run bit-exact, see `docs/N-LANE.md`.)
  - libhrx v0.1.0 does not implement stream capture (`hrx_stream_begin_capture` returns unimplemented); graphs are built explicitly with `hrx_graph_add_*_node`.
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
- Decode runs as one `hrx_graph` per batch shape. Positions and similar values sit in device buffers, gufo's approach, so graphs replay at any depth. (Implemented for single-session decode: `rt::Device` capture, stream capture on HIP and an explicit node chain on HRX; the per-step values live in `src/kernels/step_params.h`.)
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

**Mutation testing (user, 2026-10-03).** Positive controls are standing gates,
not one-off checks:
- **Planted mutants:** `tests/mutants.toml` lists exact edits to kernels,
  engine and host code, each with the test filter that must fail.
  `tools/mutants.py` applies each one, rebuilds, runs and restores, and
  refuses verdicts unless every suite passes unmutated with nothing skipped.
  Every selected mutant must be killed. A unit test checks that every
  mutant's anchor still matches exactly once, so refactors cannot silently
  retire mutants.
- **Generated mutants:** Mull (LLVM 21) on the host libraries, with a
  baseline of accepted survivors keyed by function rather than line, as
  in casselc/observability's `ci/mutants.py`. Device code is out of Mull's
  reach, so kernels rely on planted mutants.
- A mutant that survives is either a test gap, which gets a test, or is
  shown equivalent and replaced. Equivalent mutants are noted in its `why`.

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
- **Status (2026-10-03).** Done on the tiny models; the real-weight gates
  wait for the download.
  - Qwen3.5 (decider torso) and Flash-Next both match their CPU references
    on HIP and HRX (0b9874a, 5f712b8). For Flash-Next that includes LoRA
    and HF logits.
  - Session snapshots, reset and prefill (2f8b4da) are checked by a
    swarm-style Hegel program and a directed restore test.
  - Prefill runs token by token. Batched prefill is P4.

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
- **Status (2026-10-03).**
  - Routing and QSA block selection run on the device with the reference's
    tie rules (81e27fa).
  - A step is captured as a graph and replayed with device-resident step
    parameters. Each step is one upload (step parameters plus n-gram
    rows), one graph launch, and a logits download only when wanted.
  - Measured (`halo_step_stats`, tiny model, 8 layers): 373 launches per
    step, about 10 µs each when launched eagerly. Scaled to the real model
    that is about 4,100 launches per token, which caps eager decode near
    24 tok/s. Graph replay and fusion are therefore the first P3 levers.
  - HRX graph replay measured slower per node than HIP under load; this
    needs a quiet-machine remeasurement.
- **Fast-path coverage (2026-10-04).** The tiny model's widths are too small
  for the quantized kernels, so its tests only covered the float fallbacks.
  A "small" model (hidden 256, head dims 128/256, mixer rank 32;
  `make_tiny_hf_reference.py --small-quant`, `tools/make_small_quant.sh`)
  carries UD-Q4_K_XL's per-tensor types, including the Q5_K and Q8_0 expert
  layers, BF16 indexer and IQ4_NL n-gram table. On it, every fast kernel
  runs and must match the CPU reference (relative L2 3.4e-7 with float
  expert activations). int8 activations are checked against a reference
  that emulates the same quantization; values within 1e-6 of a rounding
  boundary make that comparison 1-ulp sensitive (5e-4), so the bound is
  1e-3 per position plus a tenth of the int8-vs-float distance.
- **Order of work (2026-10-03):**
  1. Device-side sampling, before fusion. Today each token downloads about
     1 MB of logits and filters them on the host.
  2. Then fusion.
  3. Then the bandwidth-oriented GEMVs and the quant-mix study.

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

- **Template facts the server must honour** (found by `src/text/chat_template`
  against HF `apply_chat_template`, 2026-10-03):
  - **Thinking defaults are opposite.** With `enable_thinking` unset,
    Flash-Next opens a think block and Qwen3.5 closes one. The parser's
    start state comes from `GenerationPrompt()`.
  - **Flash-Next always adds a reasoning-effort system block** (default
    xhigh; "high" maps to xhigh, any other value is an error). It also
    shows past thinking by default.
  - **Qwen3.5** allows a system message only first and no developer role.
    It requires a real user query, and prints scalar arguments with
    Python `str()`.
  - **Tool arguments.** OpenAI `arguments` strings must be parsed into
    objects before rendering. String arguments print raw and everything
    else as JSON, so the parser uses the tool schema to recover types.
  - **The parser does not stop at `<|im_end|>`**; the server cuts at EOS or
    a stop string. Tool calls are buffered whole (up to 1 MiB), not
    streamed as argument fragments.

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

**Results (2026-10-03; details in `docs/N-LANE.md`).**
- **N1 works.**
- **N2 shelved the strands-decider offload by its pre-set gate.** Measured
  1 TMAC/s or less; the gate needed latency within 3x of the GPU.
- **N2b, hand-written Low kernels:** 8.14 TMAC/s marginal and 2.40 TMAC/s
  end to end on all 32 tiles, bit-exact.
  - Still blocked by missing array and runtime features: memory-tile
    staging, cascade wiring, and multi-phase or continuation commands.
  - The decider stays shelved until those land in hrx-system.
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

## Addendum (2026-10-03): decision models, `/v1/systemone`, LoRA

**Requirements added by the user:**
- Run a decision model like strands-decider alongside the LLM, either by reusing it directly (perhaps on the NPU) or by adapting Flash-Next's final layers the same way.
- Serve the Jev-format `/v1/systemone` API, with matching embeddable C ABI interfaces.
- Load and train LoRA adapters.

**What strands-decider is** (Apache-2.0, github.com/strands-labs/strands-decider):
- The torso is Qwen3.5-2B-Base with a rank-16, alpha-32 LoRA on q, k, v, o, gate, up and down.
- The readout is a pointer head: a LayerNorm plus q and k projections into 256 dimensions. It scores option k from the hidden state of option k's last token against the `<answer>` position, and a masked softmax turns those scores into noul, choice or score answers.
- The prompt is `<state>…</state>`, then `<question type=…>` with numbered `<options>`, then `<answer>`.
- Several questions over one state share the state's prefix cache.
- Reported results: about 115 ms median per question on an RTX 3090, and 72.3% on JevBench.
- The schema matches TypeSafe Jev's `POST /v1/systemone`: questions are typed noul, choice or score, with criteria. Answers carry probabilities, confidence and a legend, plus usage.

**Phases:**

**P9: decision API and C ABI.** `POST /v1/systemone` (and `/v1/models`), schema-compatible with Jev and strands-decider:
- Request validation follows strands-decider's schema rules: 2 to 255 choice options, 2 to 10 score levels, and the allowed noul criteria keys.
- Confidence uses strands-decider's documented formulas: normalized maximum probability for choice, normalized standard deviation with an ordinal-smoothing floor for score.
- C ABI:
  - `halo_decide(engine, session_or_null, state, questions, answers)`, with typed question and answer structs;
  - a JSON convenience entry point;
  - `session_or_null` lets a harness ask about the live conversation without re-sending it.
- Prompt rendering is ported from `prompting.py` and checked against fixtures generated by the Python package, as for the tokenizer.
- Decisions go through a pluggable `Decider` interface:
  1. **`logprob` decider** (zero training). It renders the question with numbered options into a forked Flash-Next session and reads the next-token distribution over the option-number tokens, like the Jev eval's local baseline. It is available as soon as P2 works.
  2. **`strands` decider.** strands-decider-2B (Qwen3.5-2B, LoRA and pointer head) runs as a second resident model. It needs Qwen3.5 dense support, which reuses our GDN, attention and RMSNorm kernels, plus a dense SwiGLU MLP and no hyper-connections, QSA, PLE or MoE. It is checked against the Python reference on the published checkpoint, with JevBench-style parity.
  3. **`flash-pointer` decider.** A pointer head, with optional LoRA on Flash-Next's last k layers, reads the forked session's hidden states. The state is never re-prefilled, which makes this the cheapest decision path for a harness that already holds the conversation. It needs P11.
- **Concurrency.** Decision forwards are scheduled like prefill chunks between LLM decode steps, with a per-request priority. A strands decider pinned to the NPU (N-lane candidate 1, below) would run fully in parallel.

**P9 status (2026-10-04).** `POST /v1/systemone` is served by
`halo_serve --decider-dir` (`src/server/systemone_service`, strands-decider's
status codes: 422 with `detail`, `latency_ms` on answers), with Flash-Next and
the 2B decider resident together. Measured warm: 3.5 s for an 83-token
request, about 42 ms per token, because the Qwen3.5 GPU path still prefills a
token at a time; the reference reports 115 ms per question on an RTX 3090.
Batched prefill (P4 kernels, BF16 weights) is the fix.

**P10: variants, first-class (supersedes "LoRA inference"; user, 2026-10-03).**

Goal: train and load task-specific variants of a base model locally. Target
uses are behavior and format, domain knowledge, and tool calling (for
example, tuning tool use for the samizdat harness).

A **variant** is a named, versioned bundle that the engine applies per
session or per request, on top of an unchanged base. Any combination of
these parts is allowed:

| Part | What it changes | Runtime cost | Status |
|---|---|---|---|
| **LoRA / QLoRA deltas** on dense projections (attention, DeltaNet, shared expert, router) | behavior, format, tool use | a small low-rank matmul per adapted projection; several adapters resident, mixed in one batch | CPU reference with PEFT parity done (3088a3b); GPU on Qwen3.5 and Flash-Next (0b9874a, 5f712b8) |
| **Replacement experts (ESFT)**: retrained copies of the k experts per layer the task data routes to most | domain knowledge LoRA on attention cannot reach | none: the session's expert table points at its copies | planned |
| **Task heads**: pointer or classifier readouts, plus an optional LoRA on the last k layers | decisions, routing, scoring (`/v1/systemone` over the live session) | negligible, reuses the session's hidden states | the strands head runs today on Qwen3.5 |
| **State / prefix tuning**: a learned initial DeltaNet state plus a KV prefix for the attention layers | persona, standing instructions, house style | none: a snapshot restore through the prefix cache | planned |
| **Small-model variant**: a fully fine-tuned Qwen3.5-2B or 9B used for the task instead of, or alongside, Flash-Next | narrow, high-volume tasks | a second resident model | the Qwen3.5 torso already runs (decider) |

Engine requirements:
- **Variant manifest.** A GGUF-family file per part, plus a manifest that
  records the base model hash, part hashes, training provenance and an eval
  summary. A variant whose base hash differs from the loaded base is refused.
- **Selection per request.** A `variant` field on `/v1/completions`,
  `/v1/chat/completions` and `/v1/systemone`, a C ABI equivalent, and the
  OpenAI `model` alias `base:variant`.
- **Batching.** Sessions with different variants batch together: LoRA uses
  grouped low-rank matmuls; expert replacement is per-row expert indirection.
- **Tests per part.** Parity with the PyTorch reference on the tiny models,
  as done for LoRA. Merged and unmerged LoRA must agree. Expert-swap must be
  exact. A state variant must equal re-prefilling its training prefix.

**P11: training variants locally.**
- **Shared machinery.**
  - Frozen quantized base (QLoRA-style), FP32 master weights for trainable
    parameters, AdamW, gradient checkpointing.
  - Resumable content-bound checkpoints with durable progress, as for every
    long run.
  - Eval hooks into P12.
- **Stage 1: last-k layers.** The frozen trunk runs forward on the inference
  engine; backward covers only the last k layers plus heads or the LM head.
  This covers task heads and output-side LoRA.
- **Stage 2: full-depth backward kernels** for DeltaNet (chunked), QSA
  attention, MoE (including router gradients for ESFT), hyper-connections
  and PLE. Each is gradient-checked against PyTorch autograd on the tiny
  model, layer by layer, with tolerances fixed in advance. This unlocks full
  LoRA, ESFT and state/prefix tuning.
- **Expert selection for ESFT.** Routing statistics over the task data choose
  which experts to retrain, and are reported before training starts, so it
  is clear whether the task concentrates on a manageable set.
- **Data.**
  - Behavior and domain: user datasets.
  - Tool calling: harness traces (samizdat) converted to chat-template
    training rows with the Qwen `<tool_call>` format, plus negative and
    corrected examples.
  - Decisions: strands-decider's recipe.
- **Small-model path.** The same trainer works on Qwen3.5 (dense, so cheaper);
  it is the recommended first target for narrow tasks.

**P12: embedded LLM observability and evaluation ("halo-scope"; user, 2026-10-03).**

Goal: the engine records its own LLM telemetry directly into embedded
databases (no generic OTel pipeline). It serves log and trace search, a trace
UI, and Langfuse-style prompt, dataset, experiment and judge features, which
P11 training runs use.

**Design sources.** The user's oscope (main: Jolt; spike/oscope-core: Rust,
with a C ABI, an axum server, a vanilla-JS UI and a ratatui TUI), and the
user's casselc/observability. Both are the user's own work, and the user
permits copying from them (2026-10-03). Directly reusable pieces include the
web UI's JS/CSS, the SQL DDL and the search-language design. Rust and Go code
serves as a reference for the C++ port. NOTICE records the provenance of
copied files.

**Storage.**
- **chDB (embedded ClickHouse) for append-only, high-volume data:**
  - `llm_traces` (one row per request or session turn);
  - `llm_observations`, typed rather than attribute maps;
  - `llm_logs`;
  - `llm_scores`;
  - `engine_metrics`.
- Every table is a MergeTree ordered for time and trace lookup, with a bloom
  filter on trace id and a configurable TTL. (Sort keys and the Buffer table
  were revised by measurement; see "Single-process deltas" below.)
- **Observation fields.** Type (generation, span, event or decision), model,
  variant and adapter ids, prompt id and version, sampler parameters, exact
  input/output/cached token counts, TTFT, per-token latency percentiles, MTP
  acceptance, finish reason, tool calls, `/v1/systemone` answers, and cost
  from a price table. Payloads follow a policy (off, digest or full), with
  digest-only by default.
- **Write path, following the oscope spike.** Per-thread lock-free rings feed
  one writer thread. Batches of about 10k rows are RowBinary-encoded and sent
  with `chdb_stream_insert`, never inline RowBinary (which misparses).
  - A CRC-framed WAL makes writes crash-safe, and an insert dedup token avoids
    duplicate replays.
  - libchdb (about 566 MB) is loaded with `dlopen` as an optional component,
    with its signal handlers disabled.
  - Recording must never block or slow decode: when a ring is full the record
    is dropped and counted.
- **SQLite (WAL mode) for mutable metadata** where Langfuse would use a
  document store:
  - prompts, with versions and labels such as production and latest;
  - datasets, items, experiments and runs (run items link traces to dataset
    items);
  - evaluators (LLM-as-judge templates, rule checks, decider questions);
  - annotation queues and comments;
  - saved searches and the price table;
  - variant manifests (P10).
- Migrations are ordered and checksummed in both stores.

**API and UI** on the engine's HTTP server, plus C ABI calls for embedding
harnesses:
- **Search.** A Lucene-like query language, ported in spirit from oscope's
  `query.rs`. Field filters on model, variant, session, status, duration,
  tokens and tool. Values reach SQL only as escaped literals, and only an
  allowlist of plans is accepted: no user SQL.
  - Endpoints: histogram, facets, trace (waterfall with logs and token
    timeline), sessions, and a services/RED view.
- **Langfuse-style resources.**
  - **Prompts:** CRUD with versioning, labels and template rendering.
    Requests can reference `prompt: name@label`.
  - **Datasets** and items.
  - **Experiments.** Run a dataset against a variant and prompt version, as a
    batch on the engine itself. Results and traces are linked.
  - **Evaluators** produce scores:
    - LLM-as-judge on local Flash-Next or a variant;
    - `/v1/systemone` decisions with typed noul/choice/score outputs, which
      suit judging;
    - deterministic checks (tool-call validity, JSON schema, regex, exact
      match).
  - Scores attach to traces, observations, sessions and run items.
  - Annotation queues for human scoring.
- **Web UI.** Served by the engine. Views: search, trace waterfall, sessions,
  prompts with diff, datasets, experiment comparison across variants, and
  score dashboards.
- **TUI.** A thin HTTP client over the same API, written in C++ with FTXUI,
  following oscope's ratatui screens.

**Schema decisions adopted from casselc/observability** (user pointer,
2026-10-03). The repository is the user's own. The user permits copying its
code into halo-flash; each copied file is recorded in NOTICE with its
provenance.
Its measured shape C, typed spans plus offloaded payloads, used half the
storage and 40% less insert CPU than Langfuse's own tables, and searched
faster.
- **`llm_observations`** follows its typed `llm_spans`.
  - Columns:
    - identity and time: trace, span and parent ids, start and end times;
    - classification: type, name, user and session ids, tags, status, level;
    - model: provider and model;
    - usage: input and output tokens, plus `usage_details
      Map(LowCardinality(String), UInt64)` for input, output, cache_read and
      reasoning;
    - prompt and tools: prompt name and version, tool name;
    - content references: `input_refs`, `output_refs`, `system_ref`, with byte
      counts and a truncated flag.
  - Engine-only additions: variant and adapter ids, completion start time
    (TTFT), per-token latency percentiles, MTP acceptance, sampler
    parameters, tool-call count and ids, prefix-cache hit tokens, and decision
    answers.
  - Sort key `(project, toStartOfMinute(start_time), xxHash32(trace_id),
    span_id)`, with bloom filters on trace, session and user ids.
- **Payloads are content-addressed** in `llm_payloads`. Keyed BLAKE3-128
  hashes go in a ReplacingMergeTree with `ZSTD(3)` and a text index on
  lowercased content. Agents resend the whole conversation every turn, so
  deduplication by message shrinks storage about 6×. List pages never read
  content.
- **Scores are append-only facts** with `supersedes` and assert/retract,
  resolved at query time. Nothing is ever updated in place.
- **Cost is computed at query time** from bitemporal price facts in SQLite,
  loaded into a dictionary, and kept separate from any provided cost.
- **Write path.**
  - RowBinary through `chdb_stream_insert_n` (length-delimited).
  - A Null-engine staging table with plain types, plus a materialized view
    into the typed table. chDB's stream insert cannot do `INSERT ... SELECT
    FROM input()`, and parsing straight into LowCardinality types costs about
    2×.
  - Batches of up to 10k rows or 1 s (no Buffer table; see the deltas
    below). Every chDB statement costs about 1 to 1.5 ms of CPU, and every
    MergeTree part about 40 to 60 ms.
  - `non_replicated_deduplication_window` on every materialized-view target,
    so WAL replays stay exactly-once.
- **One process owns the database.** chDB allows one data path per process,
  and no other process can open it, so all reads go through the engine's API.
  That matches halo-scope's design. Clean shutdown works when every
  connection is closed first and then `chdb_shutdown()` is called (libchdb
  26.9.0). Exiting with a connection open segfaults.
- **`engine_metrics`** uses a series table plus points tables
  (`DoubleDelta`/`Gorilla` codecs).
- **Not adopted, because they serve multi-edge collection only:**
  arrival-day/late-part partitioning, envelopes, attribute-items text
  indexes, rollups, the semantic-convention mapping views, and S3 custody.
  We write typed columns directly, so no mapping is needed.
- **Langfuse reference:** its `research/langfuse.md` (the v4.46 data model,
  and the Postgres-side entities SQLite replaces) is the checklist for
  prompts, datasets, runs, evaluators, score configs and price matching.

**Single-process deltas from the observability design** (measured in the
halo-scope spike on libchdb 26.9.0, 2026-10-03).

**Kept from the observability design:**
- typed tables;
- content-addressed payloads;
- append-only scores;
- inserts through a Null staging table and a materialized view. This is
  still the cheapest path: RowBinary straight into LowCardinality columns
  costs about twice as much per batch.

**Dropped or changed, because they served fleet ingestion:**
- **Observation layout.**
  - Key: `(session_id, trace_id, start_time)`, with monthly partitions, a
    bloom filter on `trace_id` and a minmax index on `start_time`.
  - At 10M rows, session, trace and recent-50 lookups take 4-7 ms, against
    5-31 ms for the time-first daily layout.
- **No Buffer tables.** Their flushes bypass insert tokens.
- **Exactly-once comes from the writer's own CRC-framed WAL.** The insert
  token is the frame's position: log id plus sequence number.
- **Batches cut at 10k rows or 1 s.** chDB CPU per row falls from about
  46 µs at 250-row batches to about 6 µs at 10k.
- **Payloads.**
  - Hashes are unkeyed BLAKE3-128.
  - The writer skips payloads it has already written durably. An agent
    transcript that resends its history sent 95% fewer payload rows.
  - A text index on lowercased content.

**Process rules:**
- Close every chDB connection and call `chdb_shutdown()` before exit.
- TSan builds run without chDB: its threads crash inside TSan's allocator.

**Harness annotation ABI (user, 2026-10-03).** This is not a generic
tracing API. An embedding harness (samizdat or similar) attaches its own
steps to engine-owned traces through a few typed calls, so tool executions
appear in the same waterfall as the generations that requested them:
- **Steps:**
  - `halo_scope_step_begin(trace, parent, kind, name)`, with kind one of tool,
    retry, plan, memory or custom;
  - `halo_scope_step_end(step, status, output)`.
- **Tool results:** `halo_scope_tool_result(trace, tool_call_id, status,
  output, latency)`. It links to the generation observation that emitted the
  call, by the tool-call id the engine assigned when it parsed `<tool_call>`.
- **Scores:** `halo_scope_score(target, name, value | category | bool, source,
  comment)`, for harness-side outcomes such as task success, test results or
  user feedback.
- **Context:** `halo_scope_trace_meta(trace, key, value)` for session and user
  ids, tags and release.

The same operations are exposed as HTTP endpoints for out-of-process
harnesses. Records go through the same rings and writer, with typed columns
and payloads under the same capture policy.

**Status (2026-10-04).** Implemented as `include/halo_scope.h` over
`scope::Annotator`: steps become SPAN observations (tag `step:<kind>`),
tool results TOOL observations carrying the engine's tool-call id, scores
typed `llm_scores` facts (assert, supersede, retract), and trace context
`llm_traces` facts that queries merge per field (latest non-empty value;
tags accumulate). Output content follows the capture policy (off, digest,
full). Tested with directed cases, a Hegel property (every OK return is
exactly the rows written), concurrent threads under TSan, and a chDB round
trip. HTTP endpoints for out-of-process harnesses come with the P7 server.

Complete traces (generation, then tool call, tool result and the next
generation) are what P11 turns into tool-calling training rows. Harness
scores label them.

**Training integration (P11).** Every training run is an experiment: dataset
version, base and variant hashes, checkpoints and eval scores per step. A
variant is promoted to a "production" label only through an evaluator gate.
The same records feed the variant manifest's eval summary.

**Tests.**
- Hegel state machines for the ring-to-writer pipeline: drops counted and
  never silent, a WAL replay equal to the lost batch, no duplicate rows after
  a crash at any point.
- A crash test that kills the process during writes, then replays.
- The search compiler is property-tested: generated queries never produce
  unescaped SQL, and results equal a reference filter over generated rows.
- An hours-long chDB soak test, given jolt-chdb's earlier thread-status
  corruption report.
- A UI end-to-end smoke test.

**N-lane update.** The strands-decider torso becomes candidate 1 for the NPU, ahead of the prefill offload: a 2B dense, compute-bound forward that would otherwise compete with the LLM for the GPU. The go/no-go rule is unchanged, and it adds that LLM decode tok/s must not regress while decisions run. (Outcome: shelved after N2/N2b; see `docs/N-LANE.md`. The decider runs on the GPU.)

**Downloads.** Qwen3.5-2B-Base (about 4 GB in BF16) and the strands-decider checkpoint, which competes with the Flash-Next download on the DSL line.

## Addendum (2026-10-03): lessons from vLLM, SGLang, llama.cpp and others

Agreed with the user. Each item names its phase; the tests follow the same
rules as everything else (an oracle, Hegel properties, and planted mutants).

1. **Radix prefix cache with recurrent-state checkpoints (P6).** From
   SGLang's RadixAttention and its Mamba radix cache, and vLLM's hybrid
   prefix caching.
   - Attention KV is reusable at any token, but DeltaNet state is
     reusable only where it was saved.
   - During prefill, snapshot the recurrent state at fixed intervals (about
     every 1-2K tokens) and at turn boundaries, into a radix tree keyed by
     token prefix, with LRU eviction and the RAM and disk tiers.
   - A new session restores the nearest checkpoint and prefills only the
     gap.
   - Property: restoring checkpoint c and prefilling the gap gives exactly
     the state of a direct prefill.
2. **Sampling on the device (P3, ahead of fusion; P5).** From vLLM and
   FlashInfer's sort-free top-k/top-p.
   - The kernel downloads only the token, plus top-k logprobs when
     requested.
   - `core/sampling` (the gufo semantics) is the oracle.
   - The MTP accept/reject step and the residual draw also run on device.
3. **Prompt-lookup (n-gram) drafting alongside MTP (P5).** From llama.cpp
   lookup decoding and vLLM's n-gram speculation.
   - Agent output copies heavily from its context, so matching the recent
     suffix against the context gives long drafts almost for free.
   - Drafts are verified in the same batched pass as MTP drafts.
   - The drafter is chosen by recent acceptance.
   - Property: under greedy decoding the output is identical to AR.
4. **Grammar engine with jump-forward (P7).** From xgrammar (vLLM and
   SGLang), llguidance (llama.cpp) and SGLang's jump-forward.
   - Evaluate embedding llguidance (C API) or xgrammar rather than porting
     gufo's JSON constraint.
   - Tokens the grammar forces (the JSON skeleton of a tool call) are
     appended as one prefill chunk instead of decoded one at a time.
   - Properties:
     - constrained outputs always parse;
     - jump-forward output equals token-by-token constrained decoding.
5. **Shared-prefix ("cascade") attention and prefix-aware scheduling (P6).**
   From FlashInfer and SGLang.
   - Batched sessions that share a prefix compute attention over it once
     per batch.
   - The scheduler prefers the longest cache match.
6. **Graphs per batch-size bucket (P6).** From vLLM and SGLang graph
   capture: batched decode replays a graph captured for its padded bucket.
   Padding must never leak into outputs; this is already a stated
   invariant.
7. **Overlapped scheduling (P3/P6).** From SGLang's zero-overhead scheduler
   and vLLM's async scheduling.
   - Step N+1's host work (n-gram gather and prefetch, detokenization,
     grammar masks, sampling bookkeeping) runs while the GPU executes step
     N.
   - It is ordered with semaphores, as already planned.
8. **Reference parsers and metric names (P7).**
   - vLLM and SGLang's Qwen3 tool-call and reasoning parsers provide
     differential fixtures for our streaming parser.
   - `/metrics` uses vLLM's Prometheus metric names, so existing
     dashboards work.
9. **Quant-mix study candidates (P3).**
   - ExLlamaV3's EXL3 (QTIP trellis) gives better quality per bit at 3-4
     bits, which means fewer bytes per token for the experts. It needs its
     own dequant kernels and is gated by the P1 NLL check.
   - llama.cpp's per-tensor overrides (`-ot`) are the model for P8
     expert-placement configuration.

**Considered and not adopted:**
- Fully general paged KV. Our KV is about 24 KB per token over 12
  attention layers; contiguous storage with copy-on-write fork pages is
  enough.
- Tensor and expert parallelism, and multi-node KV transfer
  (Mooncake/LMCache). There is one device with unified memory.
- ktransformers-style CPU expert offload. CPU and GPU share the memory
  bandwidth.
- llama.cpp context shifting. It re-rotates KV, which has no DeltaNet
  equivalent.

## Addendum (2026-10-03): Responses API, stateful sessions, compaction

Agreed with the user. The model is not tied to an endpoint: every API
renders into the same chat template and parses the output back. vLLM and
SGLang already serve `/v1/responses` for open models.

**P7 extension: `/v1/responses` as the primary stateful API for harnesses.**
Completions and Chat Completions remain for compatibility.
- **Input.** A plain string or typed items: messages, `function_call_output`
  and reasoning items. Also `instructions`, function tools, `tool_choice` and
  the sampling settings.
- **Mappings onto the template and grammar.**
  - `reasoning.effort` maps onto Flash-Next's reasoning-effort system block.
  - `text.format` (JSON schema) uses the P7 grammar engine.
  - Built-in hosted tools are out of scope; function tools only.
- **Streaming.** Typed SSE events: output text, function-call argument and
  reasoning deltas. They are produced from the streaming tool-call parser's
  events.
- **Response store.** `store`, `previous_response_id` and conversations live
  in SQLite, in the halo-scope metadata store. Each stored response links to
  its trace.
- **Session binding.** A response id maps to a live engine session or a
  radix-cache checkpoint (P6). An evicted state is rebuilt by re-rendering
  the stored items and prefilling them.
- **Why it matters for this model.** Flash-Next's template keeps past
  thinking by default. Chat Completions clients usually drop
  `reasoning_content` when resending history, so the rendered history
  diverges from the cached prefix and the cache misses from the first
  assistant turn. Stored responses keep reasoning items and continue the
  live session, so only new items are prefilled.
- **Tests.**
  - The OpenAI SDK functional suite.
  - Property: a stateful continuation (`previous_response_id`) yields
    exactly the tokens of a stateless resend of the full history.
  - Stateful swarm tests over store, evict, rebuild and continue.

**Status (2026-10-04).** `src/server/openai_request` parses Chat
Completions, Responses and Completions bodies into one `GenerationRequest`
(template messages, tools, options, sampling, limits). Responses items map
onto template turns as described above; a property checks that the same
conversation sent through either API renders the identical Flash-Next
prompt. Unsupported features (n > 1, images, hosted tools, item
references, encrypted reasoning) fail with an error naming the parameter.
`src/server/openai_response` builds each API's output from the streaming
parser's events: Chat chunks and completions, Responses events (numbered,
every item added then done) and response objects, raw Completions text.
Properties: streamed deltas concatenate to the final object, streaming and
non-streaming builds agree, and a Responses output sent back as input items
is the assistant turn Chat reports. `src/server/http` is the HTTP/1.1
layer (incremental parser, keep-alive, chunked SSE, bounded connections);
`src/server/openai_service` serves `/v1/models`, `/v1/completions`,
`/v1/chat/completions`, `/v1/responses` and `GET /v1/responses/{id}` over
a `TextGenerator` interface, with stop sequences (byte-at-a-time semantics,
chunking-independent) and an in-memory response store for
`previous_response_id`. `FlashNextGenerator` runs it on the engine
(device sampling, stop tokens, whole-UTF-8 pieces) and keeps the engine's
state when a prompt extends everything already run; `halo_serve` is the
binary.

Measured on the real model (2026-10-04, HIP, greedy): a tool-calling
chat request (320 prompt tokens) returned reasoning and a parsed
`get_weather` call in 17 s, almost all of it token-at-a-time prefill; the
Responses continuation with the tool's output (`previous_response_id`)
reused 368 of 399 prompt tokens, including the first turn's reasoning,
and finished in 5.3 s. Next: P4 batched prefill (the dominant cost), the
SQLite response store and per-session engine state (P6).

**Context management and compaction (P6 with P12 evals).**
1. **Truncation** (`truncation: "auto"`): drop the oldest items over budget.
   This costs a re-prefill of the changed prefix.
2. **Summarizing compaction.** Past a threshold, the model or a cheap
   variant summarizes older turns into a compaction item.
   - The re-prefill of the compacted prefix runs in the background while
     the session is idle.
   - The radix cache keeps both versions until the compacted one is warm.
   - OpenAI's compaction wire format is to be checked against the current
     API documentation before it is matched.
3. **Model-specific compaction (research, gated by evals).** The DeltaNet
   layers carry a fixed-size state; only the 12 attention layers' KV grows,
   and QSA reads at most `indexer_top_k` tokens per step.
   - Evict old attention KV blocks that QSA rarely selects, keeping the
     recurrent state intact.
   - It is lossy, so it ships only if halo-scope experiments (task metrics
     and LLM-as-judge on long agent transcripts) show no regression at the
     chosen budget.
