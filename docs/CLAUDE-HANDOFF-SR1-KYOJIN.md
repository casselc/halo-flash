# Claude handoff: SR1 and Kyojin follow-up

Updated 2026-10-10 after inspecting the user's latest native SR1 run.
Audience: a fresh Claude session continuing the work. This document is the entry
point; detailed history/evidence stays in the stream STATE files and cited artifacts.

## Current status and first actions

**SR1's full native e2e gate is now PASS.** The earlier P/cleanup failures are closed
by run `20261010T164531Z`, with ten cycles, zero mismatches and final exit 0. The
corrections are still uncommitted; do not confuse the printed old HEAD `f875b70`
with a commit containing the tested fixes.

**K0 classification and K2 baseline calibration are complete.** Two archived
fp16-KV failures reproduce on both main and DP. K2 has measured the baseline tail
cost, but no merged variant or speedup verdict exists. GPU ownership was released
after the sweep; recheck current processes before any new job because Samizdat
may have resumed since that observation.

Recommended next actions:

1. Read `/data/halo-work/platform/yard-sr1/STATE.md`, review the working diff and
   package the intended SR1 source/tests/models/tooling for closeout. Preserve
   regenerated trace artifacts and unfamiliar/user work. No new SR1 functional
   correction is presently indicated by the latest run.
2. Continue K2 with actual tail expert routing and a marginal-cost projection;
   resolve cancellation semantics before writing a runtime merge variant.
3. Run the separate **guest-resume/SR3J native rerun** when the user is ready.
   Its fixed branch `d0a35d6` still has an open native gate; SR1 passing does not
   validate guest resume.
4. Keep the fp16-KV root-cause investigation open. Do not weaken its test bound,
   change precision defaults, or describe the full Halo GPU suite as green.

## Latest native SR1 run

Host: `user@100.64.0.10`, `nuc`, FreeBSD `15.1-STABLE`.
Run: **`20261010T164531Z`**, N=10, grace=5s, start_delay=0s; user invoked the normal
`SR1_N=10 SR1_KEEP=no SR1_DIAG_DRIFT=no` harness after targeted binary/script copy.

Evidence:

- NUC: `~/yard/ops/logs/sr1-10-20261010T164531Z/` — `summary.txt`, `results.csv`,
  `cycles.csv` and per-command JSON/summary files.
- The final segment of `~/yard/ops/logs/sr1-10-e2e.log` ends with
  **`GATE PASS (0 mismatches)`**, `OK`, sandbox destroyed and **`end rc=0`**.
- Local copy: `/tmp/opencode/sr1-10-20261010T164531Z/`. Original logs remain on NUC;
  the temporary local copy is convenient, not the durable source of record.

Checks verified:

- **10/10 cycles:** every suspend/resume succeeds, every drift_bytes=0, markers
  and starts/stops preserved; key unavailable when suspended, available on resume.
- **D:** genuine workspace edit refused, no jail left, fresh token, explicit forced
  resume succeeds and the service sees the edit (reported drift 86,016 bytes).
- **K:** SIGTERM-resistant service escalates to SIGKILL in 1021.978 ms; web remains
  active and the shared project key remains loaded.
- **T:** failed second start times out at 2046.520 ms; no jail, snapshot retained,
  fresh token; retry refuses failed-start writes; explicit forced retry succeeds.
- **P:** pause/unpause and port behavior pass; suspend from Paused is graceful.
- **X:** externally removed jail detected as Failed, loss counted once.
- **L:** zero service jails, snapshots, artifacts, processes and mounts; project
  unmounted/key unloaded; terminal lifecycle row Removed. `l-down.json` shows one
  unmount NotDone/retry before successful unmount and key unload.
- **S:** project crypto-shred succeeds, dataset is gone and subsequent key load fails.
- **R probe:** ZFS refuses changing a clone's key, as expected; not a functionality
  promise of independent per-clone encryption keys.

Timing (p50): suspend wall **381.422 ms**, resume wall **187.063 ms**, paired cycle
sum **573.309 ms**. The harness bound **1000 ms + ready-step 32.989 ms** passes.
The paired metric is computed from each cycle's suspend+resume sum, then its p50,
not by adding the two independently computed p50s (`sr1-10-e2e.sh:503–507`).
The quoted bound is this harness's check, not a new general platform latency SLA.
Three component predictions missed: destroy **207.907 ms** vs 2–20; lifecycle
suspend **281 ms** vs 50–250; start-to-ready **81.940 ms** vs 15–80. Leave these
visible as profiling/prediction follow-ups; the functional gate is green.

Deployment provenance was checked by matching NUC and local SHA256:

- Binary `/tmp/opencode/yard-svc-load-order` ↔ `~/yard/bin/yard-svc`:
  `687b9f9a8e37487874cacb93a89f9e6816b223dced51d89241f7e30913acd339`.
- `ops/sr1/sr1-10-e2e.sh` ↔ NUC script:
  `8c54a486a3ec640b6db98a2e1bda9e6b6c0b2aff42384b006adf94367ad5c029`.
- Worktree HEAD remains `f875b70`; tested corrections are a working-tree build.
  NUC's fish PATH did not expose `sha256`; `/sbin/sha256` works.
- `go version -m` confirms Go1.27.1, FreeBSD/amd64, CGO disabled, revision
  `f875b70a997cdcfd0b6b109134bbb53b19a84224`, `vcs.modified=true`.
  These facts establish deployed artifact identity and dirty-build metadata,
  not a fully archived source-to-build attestation of the exact working patch.
  Freeze/package the intended source snapshot during closeout.

## Completed SR1 corrections and verification

Worktree: `/data/halo-work/platform/yard-sr1`, branch `sr1-service-jails`.

### Load-order correction and strict drift policy

Native diagnostics isolated an 8192-byte mount-time write despite identical
visible content/metadata and empty successful `zfs diff`. DTrace captured
`zil_replay → zil_destroy → dsl_dataset_dirty`, with zil_destroy true and no
application replay records, on OpenZFS `2.4.4-1-FreeBSD_g71a9f9578`. Object-number
aggregates were not objset-filtered; exact block/copy layout was not established.
Evidence run: `20261008T200436Z`; see STATE and `ops/sr1/sr1-11-mount-drift.sh`.

Implemented:

- Lifecycle `Keys.LoadKey`/`KeyLoaded` is **key-only**, before any writable mount.
- Restore syncs and reads `written@snapshot`, refuses genuine positive drift,
  then mounts/validates/starts/readies under one operation lock.
- Explicit admin `Workspaces.LoadKey` / `ws-load` stays load+mount.
- No 8192-byte subtraction, tolerance bypass, implicit force or checkpoint rebase.
- Accepted failed mount/start can itself dirty the checkpoint; subsequent retry
  must refuse until explicit `-force`. Backend regressions cover both startup
  file edits and metadata-only writes.
- Harness explicitly admin-loads pre-created kill/hang clones before initial Up;
  initial Up has no checkpoint and needs an already mounted workspace.

### Policy isolation and operator removal

- `Config.SessionPolicies` scopes a manual command's policy to its target.
  Same-class peers remain Active; shared keys remain loaded while needed.
- Down order: Removed fence → own session effects settle → orphan cleanup →
  await project cleanup. External `jail -r` may leave nullfs views even when
  Instance reports absent; cleaning those first avoids the old unmount deadlock.
- `CleanupSession` retains `session.json` through project settlement, allowing
  interrupted waits and late checkpoint Delete/Exists observations to retry.
  Metadata retires only after settlement. Single-operator scope is documented.
- Unknown-session down errors; AddSession snapshots config under its mutex;
  cleanup uses the captured generation; usage text reflects the new ordering.
- Shred accepts Removed only after destroy Done; nonforced unmount/unload remain
  prerequisites to deleting wrapped keys.

### Models/tooling and local evidence

User required reuse of `casselc/observability`; pinned public test-only module:
`github.com/casselc/observability/quintgo v0.0.0-20261002204101-ca7ae8a6250f`.

- Typed ITF decoding + `connect.Driver`, explicit action map, after-step controller
  equality and SQLite reload replay. No silently skipped actions.
- `qobs`/`validate.Checker` fake-OS runtime order monitors:
  `spec/sr0/load_order.qnt` and `operator_remove.qnt` plus YAML bindings.
  Correct order conforms, old order is disabled, missing observation rejected.
- Core lifecycle model reflects key-only load and opaque physical writes left
  by a partial/unacknowledged restore. It does not pretend every kernel mount
  resource/orphan is represented by the core instance set.
- Recorded local checks passed: mandatory `tools/check-load-order.sh`, full
  `go test -race ./...`, FreeBSD crossbuild, **41/41 mutants caught**.
- Earlier bounded model run: **44 simulation rows × 3000 × 100**, seed 0x5eed,
  **27 scripted modules**, failures 0; 9 keysScripts. Current archive has
  **400 gzip ITF traces**. Model successes are bounded evidence, not a proof.
- Test-only toolkit imports; no production weaving/exporter was introduced.

Read entry points: `cmd/yard-svc/main.go`, `internal/lifecycle/{machine,types,settled}.go`,
`internal/service/{backend,workspace,removal}.go`, associated regressions and
`spec/sr0/README.md`. STATE and README now record the latest native pass.

## Completed Kyojin K0 and K2 work

### K0: frozen failure classification

Clean source worktrees `/data/halo-work/streams/K0/base` (main `61e3df0`) and
`K0/wt` (DP `0810614`). Common standalone probe in `K0/probe/`; no production
source changes. Fresh HIP builds, same compiler/ROCm configuration as main.

Two archived single-chunk prompts (21 and 19 tokens) reproduce decode relative-L2
**0.0725041417191737 / 0.0279758433827148 on both SHAs**, MTP off. Default prefill
logits are bit-identical between KV modes; decode repeats stable.

- First run `20261009T034702Z`: 24 observations, no route instrumentation.
- Second `20261009T034843Z`: **60 observations**, five controls, two cases, two
  revisions, three repeats. All recorded numeric metrics match between builds.
- Exact-projection prefill, fp32 residual prefill and float expert activations
  each bring both cases below the original 1% bound.
- GDN-fp32 flag has no observed effect; liveness unconfirmed, so not exculpatory.
- Default routed-id differences: 0 for the larger 21-token failure, 1 at layer 6
  for 19 tokens. Routing changes are not necessary to explain the larger failure.
- The 21-token case flips decode argmax. This is a genuine precision-path failure
  against the test's strict bound, not random measurement noise.

Classification is **pre-existing on main and reproduced on DP with MTP disabled**,
bounded to this fixture/backend/cases. MTP-enabled behavior is not established by
this probe. The controls suggest a precision-path interaction; they do not prove
the specific causal interaction. Root cause remains open; no
tolerance relaxation or fully-green-suite claim. Historical real-model fp16-KV
quality evidence is anchored to `6f3a1cf`, not a fresh quality test of these builds.

Artifacts: `K0/20261009T034843Z-{results,provenance}.json`, `-summary.txt`, per-mode
logs, `run_probe.py`, `summarize.py`, `attest.py`, STATE/PREDICTIONS. Provenance is
explicitly **post-run**, verifying clean commits/trees and matching probe/binary
hashes, not a falsely claimed run-time capture. GLM reviewed the evidence; its
conditions are resolved or bounded explicitly in `docs/K0-KV-CLASSIFICATION.md`.

### K2: CPU preflight and unchanged-baseline GPU sweep

Worktree `/data/halo-work/streams/K2/wt`, branch `experiment/kyojin-tail-analysis`,
starts `055ac0b` (docs-only successor to engine baseline `0810614`).

CPU discoveries and artifacts:

- Defaults already promote calls >2048 tokens to **8192-token chunks**; direct
  2049/2304/4097 are one-chunk controls. First default tail is direct 8193.
- The server holds back one prompt token for decode, shifting the cold/no-split
  request boundary to **8194 tokens**. Cache reuse/checkpoint splits change calls.
- Static hypothetical merge scratch: +814,468 device/+10,304 host bytes for a
  one-token tail; +181,444,064 device/+2,637,824 host for 256 tokens. Formula
  estimate, not measured high-water. Excludes KV, weights, MTP and other overhead.
- A naive merge removes an intermediate cancellation poll. That remains a real
  design constraint; polling before the merged chunk is not equivalent.
- `tools/prefill_tail_analysis.py`, five CPU tests (including 20,000 token
  conservation cases), `docs/K2-CPU-{DIRECT,SERVER}.json`. Two preserved benchmark
  logs projected onto default settings: 19 records, 17 ineligible, 2 unknown
  checkpoint aggregates; no production-prevalence claim.

The fresh unchanged `halo_run` binary measured run **`20261009T150959Z`**, shared
GPU lock, pinned SCLK2900/fclk7, explicit fp32KV, MTP off, context16384,
chunk2048/long8192, slice2048, dense-prefix on, n-gram cache4GiB, 3 reps per length:

| Direct tokens | Wall-prefill median | Reported tail-span median |
|---:|---:|---:|
| 8192 | 4.938 s | — |
| 8193 | 5.058 s | 112.3 ms |
| 8448 | 5.296 s | 347.7 ms |
| 8449 | 5.344 s | 363.9 ms |

Wall ranges are 66–72 ms; tail-span ranges 4.9/1.1/0.7 ms. Prespecified broad
ranges held. Warm-up file `prose_8888.txt` actually yields **3485 tokens**, logged.
The tool's "device" field is host wall time after upload through synchronized
completion, including dispatch/waiting, not pure kernel events.

No variant A/B, merge speedup, actual expert histogram or measured allocation
high-water exists. Tail-span optimistic ceilings are ~2.22%/6.57% of wall for
1/256 tokens; a merge still pays added token work and can change tiling/cache cost.

Artifacts under K2: `20261009T150959Z-{baseline,summary}.json`, length-specific
logs/logits, `run_baseline.py`, `summarize_baseline.py`, STATE/PREDICTIONS;
reports `docs/K2-TAIL-ANALYSIS.md`, `K2-BASELINE-CALIBRATION.md`, `KYOJIN-QUEUE.md`.
GLM reviewed the CPU hypothesis; one alleged broken test was a source misread
(2049 vs8193), checked and rejected rather than blindly accepted.

## Earlier closeout commits

These were finished earlier in the conversation; keep them separate from the
uncommitted SR1/K2 follow-up. Confirmed commit/worktree heads where noted:

| Work | Commit | Evidence/status |
|---|---|---|
| Spill round-six | `62ba395` | GPT ship review plus fresh GLM agreement; recorded server tests51/51; SE worktree head |
| Yard key service | `b352a5b` | Stable spill operation identity, renewal, regenerated replay and real Go/C++ integration; keysvc head |
| Yard broker | `e9b5ba0` | Outcome tests/docs; recorded74/74 mutants caught; broker head |
| Main-based serve shutdown | `ef578a1` | Generators destroyed before engines; rebuilt real request200 + SIGTERM exit0; shutdown worktree head |
| MTP DP closeout | `0810614` | State-only/exact-column batched catch-up evidence and lifetime fix; historical GPU suite88pass/16fixture-skips/1failure |
| Kyojin queue | `055ac0b` | Original reviewed queue; DP and K2 initial head |
| SR1 initial closeout record | `f875b70` | Old HEAD only; does not contain subsequent fixes tested by the latest native run |

Keysvc's recorded closeout evidence includes174 traces/8782 steps,10 mutants,
model failures0 and two actual signed real-model spill/restore cycles. Full raw
evidence and scope belong in its STATE/docs; do not imply new reruns were done
while writing this handoff. Broker has untracked `reviews-r1/` and `reviews-r2/`;
preserve them. No push/merge occurred as part of this handoff.

## Open work and evidence limits

| Item | Next action / constraint |
|---|---|
| SR1 closeout | Review and commit intended dirty source/model/tooling/400-trace changes; tested artifact has no new commit yet |
| SR1 timing predictions | Profile ~208ms destroy and lifecycle overhead; retain missed predictions instead of relabeling them as passed |
| Guest resume / SR3J | `/data/halo-work/platform/yard-resume` clean head `d0a35d6`; patched bhyve/entropy fixes still await native rerun |
| K0 precision root cause | Isolate the fp16 residual/int8 expert interaction; strict failing test remains open; default fp16-only KV stays off |
| K2 runtime design | Actual routing, marginal-cost projection, cancellation/checkpoint semantics, exactness/state checks, then prescribed A/B |
| K1/K3/K4 | History-copy CPU shadow analysis, single-wave attention resource/traffic analysis, expert-multiplicity histogram/compiler resources remain queued |
| MTP chooser calibration | Fresh fixed-depth cost sweep still needed; do not reuse old1.37ms catch-up coefficient as current state-only cost |
| GPU clock helper | User updated sudoers to card1; K2 runner uses it. Generic `tools/gpu_clock.sh` still hardcodescard0; durable PCI-path selection remains to do |
| Toolkit integration | Fake-OS conformance/test-only quintgo is done; production step capture/exporter and planned keysvc wiring remain distinct integration work |

Guest-resume background: old vmnet run `20261008T012430Z` failed13 resume checks;
fixes include bhyve SR3 patches0001+0002 for all NET modes, BUILDINFO/p4 gate,
checking each same-snapshot clone's own record, and Fortuna88-write + delayed
verification before entropy Done. Native rerun commands/design/user decisions are
in yard-resume STATE lines363–392 and `docs/guest-resume.md`. Default restarted
services, no-host-entropy policy, epoch assumption and crash-mid-run semantics
remain explicit open design items. Historical SR3J four-network pass is not the
new guest-resume-duty gate.

## Worktrees, artifacts, and operating rules

- **SR1:** `/data/halo-work/platform/yard-sr1`; source/model/tooling/trace changes
  dirty. Reuse `source ./env.sh`; mandatory checker requires Quint. Latest native
  result is recorded in STATE and spec README. `/tmp/opencode/yard-svc-load-order`
  is the tested FreeBSD binary, but the source tree is the durable rebuild input.
- **K2:** `/data/halo-work/streams/K2/wt`; analysis scripts, JSON/reports and queue
  updates are uncommitted. Stream-level runners/STATE/PREDICTIONS/raw artifacts
  live in `/data/halo-work/streams/K2` outside the git worktree—archive explicitly
  when packaging. Same applies to K0's external standalone probe/artifacts.
- **DP:** `/data/halo-work/streams/DP/wt`; prior logs and STATE in its parent.
- **Other yard:** `yard-keysvc`, `yard-broker`, `yard-resume` under platform.
  Earlier committed changes do not automatically propagate to other worktrees.

Operating rules:

1. Read `/data/halo-work/streams/RULES.md`. Use isolated worktrees; never push,
   force, amend or merge as part of this continuation. Preserve user work.
   Commit author when closing authorized work: Chuck Cassel,
   `619504+casselc@users.noreply.github.com`; no AI/coauthor trailers.
2. GPU jobs: `flock /data/halo-work/gpu.lock env HALO_GPU_LOCKED=1 <script>`;
   log take/release in streams/LOG.md, holds≤~15min, one large model instance,
   builds≤-j8. No main-build writes, no `pkill -f`, no lock bypass, no secret/env
   file reads. Inspect running model owners; don't stop unrelated user work.
3. Last K2 release verified no model/probe processes,0% GPU busy,GTT18,661,376bytes,
   auto clocks and free lock. This is historical evidence, not a promise about
   current availability. Samizdat via Lemonade auto-loads a llama backend on a
   chat request; coordinate its pause before a future large-model experiment.
   The lock serializes cooperating jobs; it does not prevent an independent
   Lemonade client from auto-loading a model.
4. Current GPU's PCI address `0000:c7:00.0`, DRM `card1`. Boot log: simpledrm got
   minor0 before amdgpu gotminor1. Updated passwordless sysfs rules covercard1;
   a durable helper/sudoers pair should use the stable PCI path consistently.
5. NUC inspections via unprivileged SSH are fine; root setup/e2e is user-run.
   Deploy SR1's binary and harness specifically. A full SR1 ops deployment can
   overwrite newer guest-resume scripts; if needed, deploy SR1 first, then
   yard-resume `tools/deploy-sr3j.sh`.
6. Before kernel/performance variants: consult ROOFLINE kept AND rejected rows,
   pre-register numeric predictions, use pinned in-engine alternating4rounds×3reps
   and the appropriate exactness/precision-quality/state gates. Baseline-only
   calibration is not a keep decision or a fully-green release claim.

No additional GPU/native jobs were launched while writing this handoff. No
commits were created during the handoff; the new document itself is uncommitted.
