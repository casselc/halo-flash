# Training variants locally: P11 stage 1 and the P10 manifest

Stage 1 trains LoRA adapters on the last k layers of a Qwen3.5 dense model
(Qwen3.5-2B first) plus the LM head's loss or a classifier head, and ships
the result as a P10 variant. Stage 2 (full-depth backward, Flash-Next) is
not covered here; see the end of this file.

## Decision: the stage-1 backward runs on the CPU

The trunk is frozen, so layers 0 .. L-k-1 run once per training example on
the inference engine (`Qwen35Gpu::PrefillToLayer`) and their output, the
residual stream entering layer L-k, is cached. Only the tail (layers L-k ..
L-1, the final norm and a head) runs every epoch, forward and backward. The
question was where the tail runs.

Measured first (2026-10-06, the machine shared with four perf streams, load
average 10-14):

| Work | CPU (FP32, 16 threads, AVX-512) | GPU |
|---|---:|---:|
| 6144 x 2048 projection, T = 256, forward (x W^T) | 7.8 ms (825 GFLOP/s) | 0.64 ms (BF16 on FP16 WMMA, 10.1 TFLOPS; BENCHMARKS.md) |
| same, backward dX (dY W) | 8.1 ms (796 GFLOP/s) | no kernel (W is not transposed in any existing GEMM) |
| LM head 248320 x 2048, 64 rows, forward / backward | 87 / 138 ms | -- |
| CPU GEMM with AVX2 only / 8 threads / 32 threads | 575 / 480 / 980 GFLOP/s | |

Cost model for one 256-token example at k = 2 (layer 22 DeltaNet, layer 23
attention, 111 M frozen parameters), LoRA rank 16 on every projection, 16
LM-loss positions: forward 57 GFLOP, recompute 57, backward dX 57, LM head
2 x 16: about 200 GFLOP, so about 0.25 s on the CPU and 20-30 ms of GEMM on
the GPU. The GPU is about 10x faster per example.

The CPU was chosen anyway, for stage 1:
1. **The cost is small enough.** The trunk is cached once, the tail is the
   only per-epoch work, and stage-1 data is thousands of short examples:
   minutes per epoch (measured below).
2. **A GPU tail needs about six new kernels** (transposed GEMM, DeltaNet
   recurrence backward, attention backward, norm / RoPE / gate backward),
   each with its property test, next to the code the perf streams churn.
   Stage 2 needs them in their chunked, full-depth form anyway; writing
   throwaway sequential versions first was not worth it.
3. **The GPU is shared.** It sits behind a 15-minute lock with four perf
   streams. A CPU tail leaves it to inference: training holds the lock only
   because the memory rule says so for any model load.
4. **Exactness.** FP32 with fixed-order reductions makes every run
   bit-reproducible at any thread count, so "resume continues bit for bit"
   is a test (below), not a hope.

Revisit when the tail per epoch costs more than ~10x the trunk cache, for
k >= 8, or in stage 2.

Measured after building (Qwen3.5-2B, 16 threads, `halo_train bench`):

| k | T | loss positions | forward | forward + backward | per token |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 16 | 187 ms | 384 ms | 1.50 ms |
| 2 | 256 | 16 | 258 ms | 622 ms | 2.43 ms |
| 4 | 256 | 16 | 384 ms | 1071 ms | 4.18 ms |
| 2 | 64 | 1 | 137 ms | 278 ms | 4.34 ms |

The model predicted 0.25 s for k = 2, T = 256; it took 0.62 s. The
difference is the non-GEMM work (the DeltaNet recurrence and its backward
run in scalar double-accumulating loops, attention per head) and, for short
examples, the LM head: a 1-row product with 2 GB of FP32 weights that the
packed GEMM copied whole on every call. Streaming B for 1-4 rows (commit
22636fb) was predicted to take the T = 64 case from 278 ms to about 130 ms;
it took 206 ms (predicted vs actual: 130 vs 206; the rest is per-layer
work that does not shrink with T). In the demonstration below a 16-example
step of ~30-token examples took 2.3 s (median), about 210 tokens/s.

The trunk cache on the GPU (prefill through 22 of 24 layers, one example at a
time, about 30 tokens each) ran at 530 tokens/s: 69 s for the 1,200 training
examples, 17 s for the 300 held-out ones.

## What the trainer computes

`src/train/last_k.{hpp,cpp}` (`LastK`): the tail of `model::Qwen35Reference`,
in FP32 on the CPU, from the cached boundary residual:
- every Qwen3.5 operator of the last k layers: RMSNorm (+1 folded),
  Gated DeltaNet (causal conv + SiLU, L2-normalized q/k, the gated delta
  recurrence with GGUF's tiled value heads, gated RMSNorm), gated attention
  (q/k norms, partial NEOX RoPE, GQA, sigmoid gate), SwiGLU MLP;
- LoRA (`y += alpha/r * B A x`) on any of attn_q/k/v/output, attn_qkv,
  attn_gate, ssm_alpha/beta/out, ffn_gate/up/down, initialized as PEFT
  does (A uniform +-1/sqrt(in), B = 0, so step 0 is the base model);
- heads: the frozen LM head (cross-entropy over completion tokens), or a
  classifier (LayerNorm + Linear on the last token's final hidden state);
- backward with activations recomputed per layer (gradient checkpointing at
  layer boundaries; a DeltaNet layer's backward keeps every position's state,
  T x heads x d x d floats, 268 MB at T = 256 on the 2B).

A packed 6x32 FP32 GEMM (`src/train/gemm.cpp`) does the projections; every
output element is produced by one task in a fixed order, so results do not
depend on the thread count.

### Gradient checks against PyTorch autograd

`tools/qwen35_grad_reference.py` runs the same tail in float64 through HF
transformers and PEFT on the tiny Qwen3.5 model (8 layers, k = 4: three
DeltaNet layers and one attention layer, LoRA rank 4 with random A and B on
all twelve projection kinds, 31 adapted tensors). The boundary input is
replaced by a leaf tensor so autograd covers exactly the tail. LoRA tensors
go through `tools/convert_lora_gguf.py` (the base converter's row/column
maps; a gradient maps like its parameter). Tolerances were written into
`tests/train/grad_check_test.cpp` before the first run:

| Quantity | Tolerance | LM head | classifier head |
|---|---:|---:|---:|
| loss (relative) | 1e-5 | 1.2e-7 | 4.4e-7 |
| d loss / d residual entering layer 4, 5, 6 (DeltaNet) | 1e-4 rel L2 | 1.8e-5, 1.8e-5, 1.7e-5 | 1.6e-6, 1.5e-6, 1.4e-6 |
| ... entering layer 7 (attention), the final norm | 1e-4 | 1.6e-6, 1.5e-6 | 4.1e-7, 3.6e-7 |
| 62 LoRA gradient tensors, worst | 1e-4 | 4.0e-5 (blk.4 ssm_alpha B) | 8.7e-6 |
| head gradients (norm w/b, out w/b) | 1e-4 | -- | <= 9.1e-7 |

The trunk capture itself (`Qwen35Reference::StepToLayer`) equals HF's
`hidden_states[L-k]` to 9.2e-7.

## Training runs: data, optimizer, checkpoints, resume

`halo_train` (tools/train.cpp) from a JSON run config:
`cache` (trunk, resumable shards) -> `train` (resumable chunks) -> `export`
(variant) -> `eval` (engine, base vs variant).

- **Data**: JSONL, `{"prompt", "completion"}` for the LM head (prompt and
  completion tokenized separately, loss on every completion token) or
  `{"text", "label"}` for a classifier. Never truncated silently.
- **Trunk cache**: shards of 64 examples, each written atomically with a
  SHA-256 trailer and keyed by base hash, data hash, tokenizer hash, layer
  and engine; building resumes, a stale cache is refused.
- **Optimizer**: AdamW with torch's semantics over FP32 master weights,
  separate learning rates for LoRA and head, global-norm clipping, linear
  warmup and decay (the strands-decider recipe's shape).
- **Checkpoints**: step, weights and both moments, bound to base, data,
  config, cache and parameter layout hashes; atomic write (temp, fsync,
  rename, fsync dir) with a SHA-256 trailer. A checkpoint of another run is
  refused.
- **Durable progress**: one fsync'd JSON line per step; on resume the lines
  past the checkpoint are dropped atomically and a resume event appended.
- **Determinism**: the batch of a step is a function of (seed, step) only.

**Crash-tested resume** (`tools/train_resume_test.sh`, tiny model): a run is
killed after a step before its checkpoint, inside a checkpoint write (temp
synced, never renamed), and by an external SIGKILL at a random moment;
the rest runs in 0.3 s budgeted chunks. Its final checkpoint is
byte-identical to an uninterrupted run's (run at a different thread count),
with identical per-step losses, and a changed config is refused. The ASan
and TSan builds produce the same final checkpoint bits.

The same on Qwen3.5-2B (the demonstration below): run A trained 150 steps
uninterrupted; run B was killed (SIGKILL) after step 37, resumed from its
step-30 checkpoint and finished. The two final checkpoints are
byte-identical (SHA-256 3817b544...), and so are all 150 per-step losses.

## Demonstration on Qwen3.5-2B

**Task** (`tools/make_sentiment_task.py`, seed 7): templated reviews, a
fifth of them negated ("was not delightful"), answered with a letter code:

    Review: Honestly, the food was not annoying at all.
    Is this review positive or negative? Answer A for positive, B for negative.
    Answer: A

1,200 training and 300 held-out examples; the held-out adjectives never
appear in training, so the gain has to come from the base model's own
sense of the words.

**Run**: Qwen3.5-2B-Base (BF16 GGUF, content hash 16dab8d2...), k = 2
(layers 22 and 23), LoRA rank 16 / alpha 32 on all twelve projection kinds
(1.36 M trainable parameters), LM-head loss on the answer token, batch 16,
150 steps (2 epochs), AdamW lr 2e-4, 10 warmup steps, clip 1.0. Trunk cache
on the GPU: 86 s; training on the CPU: 421 s.

| Step | 1 | 5 | 10 | 20 | 30 | 50 | 100 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| training loss (batch) | 6.35 | 4.58 | 0.81 | 0.52 | 0.43 | 3e-4 | 0.025 | 1e-4 |

Held-out (300 examples):

| Metric | base | variant | measured by |
|---|---:|---:|---|
| loss on the answer token | 6.54 | 8.6e-6 | trainer (CPU tail) |
| loss on the answer token | 6.54 | 8.7e-6 | engine: GPU prefill with the LoRA loaded through the manifest |
| exact next token (full vocabulary) | 0.000 | 1.000 | both |
| A vs B (the more likely of the two answer tokens) | 0.507 | 1.000 | engine |

The base model is at chance between A and B, and its most likely next token
is a newline (it starts `\n\n<think>`). The trainer's own held-out eval
(the CPU tail) and the engine's (every layer on the GPU, the variant loaded
and hash-checked through `manifest.json`) agree to 3e-8 in loss. Held-out
accuracy reached 1.000 by step 50, so this task saturates; it demonstrates
the pipeline, not a hard benchmark.

**Served** (`halo_serve --qwen35-model ... --variant .../manifest.json
--model-id qwen3.5-2b`): `/v1/models` lists `qwen3.5-2b` and
`qwen3.5-2b:sentiment-ab`. Greedy `/v1/completions` on held-out prompts:
the base continues with `\n\n<think>`, the variant (by alias or by the
`variant` field, identically) answers ` A` / ` B` correctly on all eight
tried; `qwen3.5-2b:nope` is 404. The variant then keeps generating (`A B C`)
since only the answer token was trained, with no stop.

## Variants (P10)

`src/model/variant.{hpp,cpp}`, `include/halo_variant.h`.

A variant is a directory with `manifest.json` and its part files:

```json
{"format": "halo-variant", "format_version": 1, "name": "sentiment-ab", "version": 1,
 "base": {"architecture": "qwen35", "name": "Qwen3.5-2B-Base", "content_sha256": "..."},
 "parts": [{"kind": "lora", "file": "lora.gguf", "sha256": "...", "bytes": 5441984}],
 "provenance": {"trainer": "...", "code_commit": "...", "data_sha256": "...", "config": {...},
                "config_sha256": "...", "trunk_cache_sha256": "...", "checkpoint_sha256": "...", ...},
 "eval": [{"name": "heldout (engine)", "metric": "accuracy", "value": 0.97, "baseline": 0.61,
           "n": 300, "dataset_sha256": "..."}]}
```

- Part kinds: `lora` (llama.cpp adapter GGUF), `classifier_head`,
  `pointer_head` (strands-decider head GGUF). Part files are relative names
  inside the variant directory (no `..`, no absolute paths).
- **Base content hash**: SHA-256 of the GGUF file (a split model hashes its
  shards' hashes), with SHA-NI (about 2 GB/s); a sidecar cache keyed by size,
  mtime and inode (in `$HALO_HASH_CACHE_DIR` or next to the file) makes
  repeated loads free.
- **Load-time refusal**: another base hash or architecture, a part whose
  size or SHA-256 differs, a missing part, a LoRA whose shapes do not fit the
  base, an unknown format version, two heads.
- **Several resident**: `VariantRegistry` holds any number (shared, so a
  request keeps one alive across an unload); `Qwen35Gpu::AddAdapter` /
  `SelectAdapter` keeps their LoRAs on the GPU and applies one per call.
- **Per request**: a `variant` field or the model alias `base:variant` on
  `/v1/completions`, `/v1/chat/completions`, `/v1/responses` and
  `/v1/systemone` (both given must agree); unknown variants are 404 on the
  OpenAI routes, 422 on `/v1/systemone`; `/v1/models` lists
  `<id>:<variant>`. `halo_serve --qwen35-model M --qwen35-tokenizer T
  --variant manifest.json ...` serves a Qwen3.5 model's variants;
  `--decider-dir D --variant ...` adds pointer-head variants to the decider.
  Flash-Next refuses variants until stage 2. One variant per request; mixed
  variants in one batch are later work.
- **C ABI**: `halo_variant_registry_open / _close / _base_hash`,
  `halo_variant_load / _unload / _count / _name / _manifest_json`,
  `halo_variant_parse_model_alias`.
- **Merged = unmerged**: W + (alpha/r) B A materialized in FP32 agrees with
  the unmerged adapter to 1.3e-6 (CPU) and 1.4e-6 (GPU) relative L2 on the
  tiny model (bounds 1e-5 and 1e-4, fixed in advance); interleaved selection
  of two adapters and the base matches the CPU reference (worst 2.0e-5) and
  reselecting an adapter reproduces it exactly.

## Eval reporting (for halo-scope)

`src/train/eval_report.hpp`: an `EvalReport` (run, variant and version,
base hash, dataset name and hash, metric, value, optional baseline, n,
step, engine, time) goes to an `EvalSink`. `JsonlEvalSink` appends durable
lines; `ToVariantEval` fills the manifest's eval summary. The trainer
reports held-out loss, accuracy and exact match at step 0 (the base, for an
LM-head LoRA) and every `eval_every` steps; `halo_train eval` reports the
engine's numbers for base and variant. halo-scope (stream X) can add a sink
that records each report as an experiment run item and score; nothing here
depends on it.

## What stage 2 needs

- **GPU backward kernels**, each gradient-checked like stage 1: transposed
  BF16/quantized GEMM for dX, chunked (WY) DeltaNet backward, QSA attention
  backward, MoE (routed experts and router gradients for ESFT),
  hyper-connections, PLE. The CPU tail stays as their oracle.
- **Flash-Next hooks** (files owned by the perf streams):
  `FlashNextGpu::PrefillToLayer` (the trunk capture `Qwen35Gpu` has), and
  per-call adapter selection like `Qwen35Gpu::AddAdapter/SelectAdapter`
  (Flash-Next binds one LoRA at creation), so variants can be served and
  `FlashNextGenerator` can stop refusing them.
- **Batched mixed variants**: grouped low-rank matmuls per row.
- **Pointer-head training** (the strands recipe; the decider already serves
  pointer-head variants) and a halo-scope EvalSink.
- **Faster tail**: the DeltaNet recurrence backward is scalar; a chunked
  form would also cut the CPU cost.
