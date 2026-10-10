# K0: deterministic classification of the archived fp16-KV failures

## Verdict

The two archived failures are reproducible on both frozen **main 61e3df0** and
**DP 0810614**. They are a pre-existing precision-path interaction, not a new
failure introduced by DP's MTP catch-up. The original strict decode-relative-L2
bound `<0.01` genuinely fails. It remains unchanged; the full GPU suite is still
not green.

Classification is bounded to two prompts, one small-q fixture, HIP/gfx1151 and
the two builds. A runtime root cause, universal error bound or current real-model
quality verdict is not established. K0's freeze/reproduce/classify prerequisite
is complete; unchanged K2 baseline measurement may use **default fp32 KV**.

## Experiment and evidence

Fresh standalone diagnostic builds link each worktree's freshly built halo_gpu.
Same RelWithDebInfo, Clang 21.1.8 and ROCm 7.2.3 as main; gtest/Hegel disabled
because the standalone probe directly uses the archived prompt vectors.

Each case resets two engines, prefills its single chunk, then decodes the first
prompt token, matching `Sessions.Fp16OnlyKvPrefillEqualsDefault`. Three repeats
per case/config/build. MTP tracking is false. Missing fixture/device fails closed.
The fixture has 8 layers, hidden 256 and vocabulary 248320.

| Mode | 21-token case relative L2 | 19-token case relative L2 | Decode route-slot differences (21 / 19) |
|---|---:|---:|---:|
| Defaults | 0.0725041417191737 | 0.0279758433827148 | 0 / 1 |
| Exact-projection prefill | 0.000173359647112 | 0.000034891220137 | 0 / 0 |
| fp32 residual prefill | 0.000037673947132 | 0.000399558006205 | 0 / 0 |
| GDN-fp32 flag | unchanged | unchanged | 0 / 1 |
| Float expert activations | 0.000031388004489 | 0.000033945686400 | 0 / 0 |

All **60 observations** have bit-identical prefill logits across KV modes and
stable decode repeats within each engine. All recorded numeric metrics match
across the two revisions. This is not a cross-revision byte comparison of entire
logit arrays. Both default failure values match the archived reports at printed
precision. The 21-token case also changes decode argmax (6910 → 98632).

The res32/float-expert controls are consistent with a non-exhaustive interaction
involving fp16 residual prefill and int8 expert activation arithmetic. They do
not isolate the originating operation. The exact-projection control changes
several precision choices, so it does not individually identify int8 projection
GEMMs. The GDN flag has **no observable effect on this fixture; liveness is
unconfirmed**, and it is not evidence excluding the GDN path. Routing differs
only in the 19-token case (first differing layer 6); reject routing changes as a
necessary explanation for the larger 21-token failure.

The second run enables routing observation. A first, uninstrumented run already
reproduced the same default numbers; the preregistered observation control held
within 1e-10. No performance timing is inferred from these instrumented probes.

Historical real-model fp16-only KV quality evidence in BENCHMARKS.md:94 is
anchored to main **6f3a1cf**, default off. It neither validates the per-case 1%
bound nor constitutes a fresh quality test of 61e3df0/0810614.

## Reproduction and artifacts

Stream directory: `/data/halo-work/streams/K0`.

- `probe/{CMakeLists.txt,kv_precision_probe.cpp}`: common deterministic probe.
- `run_probe.py`: lock-scoped sequential runner with fixture/probe/binary hashes,
  one-minute per-command deadline and take/release entries in streams/LOG.md.
- `20261009T034702Z-results.json`: 24 observations before routing instrumentation.
- `20261009T034843Z-results.json`: 60 observations with the controls above.
- `20261009T034843Z-summary.txt`: archived validation output.
- `20261009T034843Z-provenance.json`: **post-run** attestation of clean source
  commits/trees, build settings and hashes matching the measured probe/binaries.
  It is explicitly not represented as a run-time provenance capture.

Fixture SHA256: `9c9040c45ee823fa62b7a4a7336dd463e11b123f0ad265a6a71a347bc78c6807`.

```sh
flock /data/halo-work/gpu.lock env HALO_GPU_LOCKED=1 \
  python3 /data/halo-work/streams/K0/run_probe.py
python3 /data/halo-work/streams/K0/summarize.py \
  /data/halo-work/streams/K0/20261009T034843Z-results.json
```

GLM 5.3 independently checked token transcription, bound semantics and control
results: accept with conditions. Added source/build attestation and archived
validation; treated the GDN control as non-exculpatory; limited scope and causal
claims as above. No production changes or relaxed tests.
