# Benchmarks and quality gates

Each section records what was measured, against which weights and build, so
that numbers stay comparable. Weights are unsloth Qwen3.8-Flash-Next
UD-Q4_K_XL (HF revision 38bb39ee), four shards verified against the HF
sha256s.

## Quality

### Short set (P1/P2 gate), 2026-10-04

Teacher-forced scoring on ds4's `qwen38-flash-alibaba-100` fixtures
(`tools/score.cpp`, `halo_score`).
- **Rendering:** prompts are rendered as ds4 renders non-thinking Qwen3.8
  chat.
- **Agreement:** counted where local tokens align byte for byte with the
  API's.
- **Gate (fixed before measuring):** NLL ≤ 0.305 and API top-token
  agreement ≥ 91%.

| Engine | Kernels | Cases | Target tokens | NLL | API top-token agreement | Gate |
|---|---|---:|---:|---:|---:|---|
| halo-flash, HIP, eager | P2 (correctness) | 98 | 5,568 | **0.2859** | **92.46%** | pass |
| ds4 Q4 on DGX Spark (ds4's own GGUF) | — | 100 | 5,696 | 0.2903 | 92.08% | reference |

ds4's NLL row also includes the two cases that have damaged emoji metadata.
Its agreement column uses the same 5,568 tokens as ours. ds4's Q4 is a
different quantization from UD-Q4_K_XL, so the comparison is a sanity check,
not an identity.

### Greedy identity with gufo

`halo_run --device hip` against `gufo prompt --raw -t 0 --speculative off`
(gufo 8bdde80), with the same weights.
- "The capital of France is", 24 tokens: **identical**, including one step
  with a top-2 logit margin of 0.069.
- token-trace sha256 (gufo):
  `6e29a188870a63130d14c840b32d7a59d561f81c612ba57e62031dadb98c33e2`.

## Speed

Measured 2026-10-03/04 while other jobs were running, so the absolute
numbers are noisy.

| Measurement | Value |
|---|---|
| Model load (79 GiB resident) | 24 s with the parallel O_DIRECT loader; > 15 min with per-tensor copies of mapped memory |
| Decode, P2 kernels, eager | ~5.6 steps/s; gufo `prompt` reported 5.2 tok/s in its own verbose output |
| Graph instantiation | 8,000 nodes: ~14 ms on HIP, ~2 ms on HRX |
| Graph replay per node (trivial kernels) | ~3 µs on HIP, ~5 µs on HRX (loaded machine) |

### Bytes per decoded token at this quant mix

| Part | Size |
|---|---:|
| Dense, mostly Q8_0: `attn_qkv` 956 MiB, `ssm_out` 574, `attn_gate` 574, hyper-connection mixers 638, LM head 644, router (F32) 240 | ~4.5 GiB |
| Routed experts, 10 of 512 per layer (Q4_K gate/up, Q5_1 down) | ~1.4 GiB |
| **Total per token** | **~5.9 GiB** |

At the P0 read ceiling of 241 GB/s, that is about 26 ms per token, a decode
roofline of about 38 tok/s at this mix.
