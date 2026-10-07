# G3: a fair GPU scheduler for halo_serve (design, 2026-10-07)

Design and model only: nothing here is built. Model: `spec/scheduler/` (Quint 0.32, `quint run` only), checked by
`spec/scheduler/run.sh`, results in `spec/scheduler/results.txt`. Predictions (written before any G3 code):
`/data/halo-work/streams/B/PREDICTIONS-G3.md`, summarised in section 9.

## 0. Recommendation in one paragraph

One **engine thread** owns the GPU and runs one **unit** at a time: a prefill slice (one exact 2048-token chunk
while anyone else is runnable, 8192 when alone), a **batched decode unit** (StepBatch over every decoding session,
a short burst of steps), or a decider evaluation. Between units it **fences** departed requests (parking their
slice-aligned progress as a retained prefix) and **picks** the next unit: a decider with **credit** first, then
**start-time fair queuing by GPU time over sandbox principals** (the decider class is one more principal once out of
credit). HTTP threads **admit** under the same lock against a **reservation ledger** (retained prefixes + every
admitted request's reservation; eviction counts what it frees), with a per-principal in-flight limit; a refusal is
a **429 with Retry-After** that takes the request off the queue in the same critical section. Clients get **SSE
heartbeats** while queued or prefilling, and every turn carries an **idempotency key**: a retry of a finished turn
replays the stored answer, a retry of an in-flight one adopts it. The model holds all 11 properties for this design
(two seeds, 20000 traces each); today's scheduling and each of 13 mutants (one rule off each) break the property
they are named for.

## 1. Today (main ac2ec31)

- `FlashNextGenerator::GenerateIds` holds one `std::timed_mutex` (`flash_next_generator.cpp:81`) from `Prepare`
  (prefix reuse + the whole prefill) to the last decode step: **FIFO-ish, one request at a time, to its end**. A
  queued request polls its departure every 20 ms (F5). There is no queue bound, no priority, no 429.
- `/v1/systemone` (strands-decider-2B, a separate model) has **its own mutex** (`systemone_service.cpp:22`): it runs
  concurrently with Flash-Next on the GPU, so its latency under load is unmeasured contention (HARNESS D1).
- `SessionCache` (SESSIONS.md section 8) retains up to 8 sessions / 8 GB, LRU, and evicts until `KvRoomFor` says the
  prompt fits. Stream F measured that `DropSession` returns its pages to `AvailableBytes` at once (F3 does not occur
  on the eviction path; Reset/Restore and the indexer window still keep pages charged).
- Departure (SESSIONS.md section 9): stopped before each prefill chunk after the first and before each decode step;
  a stopped prefill drops its live session, keeping checkpoints parked before the stop. Opt-in SSE keepalive while a
  stream waits is on branch fix/audit-f5-f7 (1450f76), off by default.
- SESSIONS.md section 8: "Overlap needs the scheduler (G3) and batched decode". MTP.md: `SpecStep` follows one
  session (`mtp_owner_`); StepBatch stays AR.

## 2. Workload and measured costs

One user, several agent sandboxes (principals) sending long-context Responses turns (8K-200K tokens, with
`previous_response_id`, so later turns reuse the retained prefix), plus small latency-sensitive `/v1/systemone`
decider calls (egress triage, injection screening, harness decisions: SYSTEMONE.md uses 1-5).

| Work | Cost (measured) | Source |
|---|---|---|
| Cold prefill | 1.25 s at 2K, ~5.0 s at 8K, 22-25 s at 32K, 93-96 s at 128K, ~200 s at 256K | BENCHMARKS, SESSIONS 7 |
| One 2048-token chunk | ~1.25 s near position 0, ~1.7 s at 128K, ~1.95 s at 200K (fit: 0.60 + 1.74e-6 x depth ms/token) | derived |
| 2048 vs 8192 chunks | bit-identical logits; 8192 is ~7% faster at 8K, +4.3 GB of chunk buffers | ROOFLINE 11 |
| Warm later turn | prepare 0.40-0.43 s (exact cache), first step 38-43 ms | SESSIONS 8 |
| Decode step | 37 ms short, ~40 at 32K, 42.6 at 128K, 45 at 256K | BENCHMARKS |
| StepBatch, byte-identical per entry | B=2 52.9 ms, B=4 80.3, B=8 134.3 (27.0 / 37.8 / 49.8 / 59.6 tok/s aggregate) | SESSIONS 3 |
| MTP cycle (one session) | verify(4) ~72 ms + 3 drafts ~13.5 ms; 1.1-1.4x AR | MTP, SESSIONS 6 |
| Decider (2B) | 145 ms (83 tokens), 314 ms (3 questions, 248 tokens) | SYSTEMONE |
| Session memory | 118 MB recurrent + 81.4 KB/token (fp16-only KV 32.3); fork 17-29 ms + 151 MB own | SESSIONS 2, 7 |

## 3. Q1: the scheduling unit (preemption points)

**Recommendation.** The engine's existing boundaries are the units; nothing is preempted inside a kernel or a
graph:
- **Prefill slice = one engine chunk.** 2048 tokens whenever another request is runnable, 8192 when the prefill is
  alone. Both are exact (bit-identical to an unsplit prefill, BENCHMARKS "Consistency"; any split at a multiple of
  16 is exact, SESSIONS 8). A slice is the preemption bound: <= 1.3 s shallow, <= 2 s at 200K depth. Needs an
  engine entry point that runs *one chunk of a bound session* without the ~140 ms per-`Prefill`-call fixed cost
  (stream H); bind is 0 ms and the chunk buffers are engine-owned, so sessions can alternate chunks.
- **Decode unit = a burst of StepBatch steps** over every decoding session (k steps, k chosen so the unit is
  ~0.25-0.5 s: about 8 steps at B=1, 4 at B=4). One step would be the finest grain, but each unit boundary is a
  host round-trip and a scheduling decision; a burst keeps the decoder's GPU share honest while bounding the wait
  of a newcomer to well under a slice.
- **MTP cycle** (when `--mtp-model` lands) is one decode step of its session: drafted + verified + committed rows
  are atomic (the rollback lives inside SpecStep). A session in a spec cycle cannot join a StepBatch in the same
  unit (section 5).
- **Decider evaluation**: one unit (if the decider is scheduled; see decision D2).

Alternatives considered: (a) whole requests (today): unbounded wait, rejected by the model (`today`, `noPreempt`
break `boundedWait`); (b) 8192-token slices always: 7% faster prefill under contention but a 5-8 s preemption
bound, too long for deciders; (c) 512-token slices: ~4x more slice switches, and the routed experts lose weight
reuse below 2048 (BENCHMARKS P4 table: 2048-token chunks were the step to 124 tok/s); (d) preemption inside a chunk:
impossible without changing kernels.

## 4. Q2: policy, fairness and admission

**Fairness.** Start-time fair queuing (SFQ) over **principals** (one per sandbox, from the broker's
`X-Halo-*`/`X-Halo-Session-Id`; unknown callers are one shared principal), charged by **GPU time actually used**
(the unit's measured duration; a batched decode unit charges each participant its share). Within a principal,
FIFO. A principal that was idle **joins at the system virtual time** (the start tag of the last unit picked: it
banks no credit while idle). Joining at the minimum over the *other runnable* principals is not enough: a principal
briefly not runnable (departed, then adopted by its retry) is left out of that minimum and then waits for the
joiner to catch up; the model found this at 20000 samples (`boundedWait` 11 > 10 ticks). This is what makes a cold 128K prompt coexist with other sandboxes: it gets its share, the
others keep theirs.

**Decider priority with a budget.** A decider call carries the `systemone` class. While the class has **credit**
(a token bucket: `DEC_BURST` evaluations, refilled one per `DEC_REFILL`), it runs at the next unit boundary ahead of
fair queuing. Out of credit, it is the principal `D` in the fair queue: still served, never starving the sandboxes.
The model's mutants: no priority (`noDeciderPrio`: a decider waits for fair turns: `deciderLatency` breaks) and
unlimited priority (`strictDeciderPrio`: a decider flood starves the agents: `boundedWait` breaks).

**Starvation freedom** is checked in the model as the bounded-wait state property whose violation is the
mechanism: a runnable request of a connected client is served again within `WAIT_BOUND` ticks
(in service + other principals' catch-up + the decider budget). Liveness proper (eventually served) follows from
the bound; the bound is the useful statement because the client's idle timeout is a duration.

**Admission and queue bounds.** At arrival (HTTP thread, scheduler lock):
1. idempotency (section 7): replay, adopt, or drop a late duplicate;
2. per-principal limit: at most `PLIMIT` active (not known departed) agent requests per principal (1 by default:
   a sandbox's harness is sequential per conversation; the broker also keeps one generation in flight per sandbox).
   Beyond it: 429;
3. memory: the request's reservation (prompt - reused prefix + max_tokens + slack, + MTP drafts when on) against the
   ledger (section 6), evicting unpinned retained prefixes LRU; if it still does not fit: 429.
4. otherwise queued. The queue length is bounded by principals x PLIMIT + the decider credit, so no global queue cap
   is needed; a maximum queue wait (REPORT 5.6) is not needed either, since the wait is bounded by fairness.

**429 with Retry-After vs queueing.** Queue whenever the request can be served within the fairness bound (it can,
if it was admitted); answer **429 only for a request that can never run now**: over the principal's limit, or
short of memory after every evictable prefix went. `Retry-After` is an integer number of seconds: for the limit,
the principal's in-flight request's expected remaining time capped at 30 s; for memory, 5-30 s. The 429 and the
removal from the queue are **one critical section** (AUDIT N4; mutant `queued429` breaks `rejectedNeverRuns`: a
"not done" answer for work that then runs).

Alternatives: per-request fairness (one sandbox opening several conversations takes more); strict priority classes
(REPORT 5.6's interactive / background / eval: these remain the broker's job; inside halo_serve, weights per
principal give the same lever); deadline scheduling (needs deadlines nobody sends).

## 5. Q3: batched decode and a newcomer's chunked prefill

- All decoding sessions step together (StepBatch, <= 8; beyond 8, rotate groups of 8 by fair order). Each entry is
  byte-identical to the session stepped alone (SESSIONS 3), so batching never changes outputs: the scheduler can
  batch freely.
- A newcomer's prefill slices **alternate** with decode units under SFQ: decoders stall for one slice (the
  inter-token gap is one slice, <= 1.3-2 s), and the newcomer's TTFT is its own slices plus at most one decode unit
  between each. With 2 principals backlogged each gets ~50% of GPU time; with a cold 128K prompt that means the
  decoder runs at ~13.5 tok/s in bursts and the prefill takes ~10% longer for a 300-token answer beside it.
- Model: the batch is every decoding request (`wBatched`); a decode unit charges every participant.
- Graphs: StepBatch caches graphs per session list (LRU 16); a changing batch membership re-captures. M3 measures
  the capture cost at a membership change; if large, keep membership stable within a burst.
- **B x MTP follow-up.** SpecStep's draft block follows one session (`mtp_owner_`). Until B-session x (k+1)-row
  verify batches exist, the scheduler runs MTP only when exactly one session decodes (B=1); with B >= 2 it uses
  StepBatch (AR). Measured costs say this is also the faster choice: MTP is ~1.1-1.4x AR at B=1, StepBatch B=2 is
  1.40x aggregate. The follow-up (P5/P6): verify batches of B x (k+1) columns; gate it on a measured win over B-AR.
  MTP tracking must be re-established after Fork/Bind (AUDIT E6), and `KvRoomFor` must reserve `kMaxDrafts` rows.

## 6. Q4: memory, eviction and departure

**Reservation ledger, not the memory reading.** Admission compares the reservation against
`CAP - (retained prefixes + every admitted request's reservation)`, not against `AvailableBytes` at the moment.
The reading only shows pages already mapped; two admitted requests then grow into the same memory over their
prefills (AUDIT D1, E3: the decision at `Prepare` must hold for 200 s). Mutant `memoryReadingOnly` breaks
`noOvercommit`. The ledger is engine-level bytes (`SessionMemory`), recalibrated against `AvailableBytes` at idle
(the reading is the ground truth for what other processes took; the ledger is the truth for what we promised).

**The F3 lesson.** Eviction must see what it freed: the next check after an eviction must count the released bytes,
else the loop empties the cache and still refuses (AUDIT F3). Stream F measured that `DropSession` does give its
pages back (`hipMemAddressFree`), so the ledger term is right today; the model keeps the rule as `evictSeesFreed`
with the mutant `releasedStaysCharged` (breaks `refuseOnlyIfShort`), the analogue of tests/mutants.toml's
`guard-ignores-released`. Reset/Restore and the indexer window still keep charged pages: they stay in the ledger.

**Pins.** A retained prefix that an admitted request plans to reuse is pinned until that request's first unit
binds it (a fork then shares its pages). Evicting it would leave the request's reservation too small for the
prefill it now has to redo (mutant `evictPlanned` breaks `withinReservation`). A started fork keeps its shared
pages when its parent prefix is evicted: they become its own (the model's `inherited`).

**Bind at start, not at admission.** The reused prefix is chosen when the request's first unit starts: by then a
departed predecessor may have parked a longer prefix (mutant `bindAtAdmission` breaks `noRecompute`). With
idempotency keys a retry adopts its in-flight attempt instead, so the mutant is equivalent for keyed clients; the
rule protects keyless clients, and the model checks it on the keyless instance. With
concurrency this is also AUDIT D9: a checkpoint fork must be taken from rows no concurrent step writes; forks below
the immutable bound (`ForkSession`'s `immut[]`) satisfy it, and the engine thread is the only writer.

**Departure at every point.** A departure (FIN, a failed write, the send deadline, or a vanished peer found by
heartbeat failure / TCP keepalive within `ACK_TO`) is acted on at the next unit boundary:
| Where the request is | Effect of the departure |
|---|---|
| waiting (in flight to the server) | late arrival: admitted or refused as usual, then fenced at the first boundary (idempotency may drop it as a late duplicate) |
| queued | removed at the next boundary; never touches the cache |
| mid-prefill | the slice in progress finishes; its slice-aligned progress is **parked** as a retained prefix (exact: slices are 2048-aligned), not dropped (today drops the live session; mutant `dropOnCancel` breaks `noRecompute` when the client retries) |
| mid-decode | the burst in progress finishes; the session keeps exactly what ran (SESSIONS 9) |
| finished, answer lost | the stored answer is replayed to a retry (section 7) |
Work done for a departed client is bounded by `ACK_TO + MAXUNIT` (`workAfterGoneBounded`; mutants `noFence`,
`noDeadPeerProbe`).

## 7. Q5: client timeouts and retries (the D3 loop)

The loop (AUDIT D3): nothing is sent while a request queues or prefills, the client's idle timeout fires, the SDK
retries, and the retry queues behind the dead attempt. Three rules break it, each with its mutant:
1. **Heartbeats** (`: keepalive` SSE comments every few seconds while queued or prefilling): the client's idle timer
   never fires while the server holds its request (`noSpuriousTimeout`; mutant `noHeartbeat`). A failed heartbeat
   write is also the departure signal for a peer gone without FIN. Non-streaming requests cannot be kept alive:
   their callers need a timeout above the fairness bound + their own work, or should stream.
2. **Idempotency keys.** Every turn carries a key (`Idempotency-Key` header, defaulting to a digest of
   (principal, previous_response_id, input)). On arrival: a finished key **replays** its stored answer (Responses
   store) instead of generating again (`generatedOnce`); an in-flight key whose client left is **adopted**: the
   retry's connection takes over its stream (resuming from the next event) (`noSelfInflicted429`: without it the
   retry is refused by its own dead attempt's limit slot); a late duplicate of a key already in flight is dropped.
   Mutant `noIdempotency` breaks both.
3. **Park on cancel + bind at start**: a retry after a departure reuses the departed attempt's progress.

**Outcomes and the caller rule** (AMBIGUITY.md):
| Call | done | not done | unknown: settle bound / resolving observation |
|---|---|---|---|
| Responses turn (stream) | `response.completed` received | 429 (never ran, never will: removed atomically), 4xx/5xx, or `response.cancelled` | connection lost before the terminal event: settles when the server fences (<= `ACK_TO` + one unit) or finishes; observe by retrying **with the same key** (replay / adopt answers it) or `GET /v1/responses/{id}` |
| Decider call (non-stream) | 200 | 429 / 4xx | lost answer: settles at the evaluation's end (<= `DEC_BOUND`); retry with the same key (cheap, idempotent) |
| 429 | | not done, by construction | Retry-After |
Caller rule: **retry only with the same idempotency key, after Retry-After or on a lost connection; never a fresh
key for the same turn**. The unsafe rule, a fresh key per retry, is the mutant `noIdempotency`.

## 8. Q6: observability

Per request (observer GENERATION attributes and `/metrics`, vLLM names where they exist): `queue_wait_ms` (arrival
to first unit), `ttft_ms`, number of **preemptions** (times another unit ran between two of its units) and the
total **preempted time**, slices run, decode units and batch sizes, `cached_tokens` and whether the prefix was bound
from a parked departure, outcome (completed / cancelled-where / 429-why / replayed / adopted). Global:
`vllm:num_requests_waiting` / `running` (queued, not locked: fixes H8), `vllm:request_queue_time_seconds`,
`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` (shows the slice gaps),
`vllm:num_preemptions_total`, per-principal GPU seconds and virtual-time lag, decider credit and out-of-credit
count, 429s by reason, ledger reserved / mapped / retained bytes, evictions, heartbeats sent, departures by point.

## 9. Q7: milestones, tests and predictions

| # | Milestone | Tests |
|---|---|---|
| M1 | Engine: one-chunk prefill entry point on a bound session (no per-call fixed cost), chunk size per call | gpu_tests: interleaving chunks of 2-3 sessions equals each prefilled alone, bitwise (Hegel swarm extension of `Sessions.SwarmMatchesFreshReplay`); predicted switch cost <= 15 ms |
| M2 | `Scheduler` class (host-only): units, SFQ, decider credit, ledger admission, fence, park, 429 atomic, idempotency | server_tests: a **Hegel state machine against the model**: the same actions (send, arrive, lose, abort, vanish, timeout, unitEnd, tick) on a fake engine with tick costs, with the model's 11 properties as invariants; each model mutant as a `tests/mutants.toml` entry that the machine kills |
| M3 | Wire `FlashNextGenerator` through it; StepBatch decode units; heartbeats on by default | the existing `OpenAiService.ServingModelUnderDeparturesStallsAndRetries` extended to 3 concurrent clients; small-model GPU: concurrent requests' tokens equal each served alone (exact cache) |
| M4 | Decider class (D2's choice) and metrics | real-model soak: 4 sandboxes (yard battery) + decider load; predictions below |
| M5 | ITF trace replay (as yard `internal/lifecycle/replay_test.go`): `quint run --out-itf` traces of the design replayed against the M2 class, comparing state after each step | worthwhile once M2 exists: it pins the code to the model's transitions cheaply; the Hegel machine covers what traces miss |

Predictions (PREDICTIONS-G3.md; warm later turns of ~500 new tokens and ~300 output tokens):
| Sandboxes | Today TTFT | G3 TTFT median / worst | Per-session tok/s | Aggregate tok/s |
|---:|---|---|---:|---:|
| 1 | 0.45 s | 0.45 / 0.45 s | 27.0 | 27.0 |
| 2 | mean ~5.5 s, worst ~11 s | 0.50 / 0.9 s | 18.9 | 37.8 |
| 4 | mean ~17 s, worst ~33 s | 0.6 / 1.9 s | 12.5 | 49.8 |
A cold 128K prompt beside one decoder: prefill ~105 s (94 alone), the decoder ~13.5 tok/s in bursts; a credited
decider waits <= ~2.3 s if scheduled as a unit (vs up to ~94 s today if it shared the mutex; today it does not:
it contends concurrently, unmeasured).

## 10. The model (`spec/scheduler/`)

**Shape: shared state, not Choreo.** The scheduler is one process: HTTP threads and the engine thread act on one
queue under one lock, and the engine thread runs units one at a time. Nothing inside it passes messages, so a
shared-state machine with atomic actions per critical section is the faithful shape. The only process boundary is
the client's connection, and it is modelled as an environment with every outcome of an HTTP call: lost request,
late effect (a request that arrives after its client left; work that runs after it left), lost answer, duplicate
delivery (the client's retry), a 429, and departure with or without FIN. Choreo would fit if the broker and
halo_serve were modelled together (the broker's token bucket and one-in-flight rule); not needed for G3.

**Durations.** Time is ticks (one tick ~0.6 s, half a 2048-token slice). A slice costs 2, a decode unit 1, a decider
1; units run to their end; the client idle timeout is 5 ticks, a vanished peer is known gone after 2, Retry-After is
3, the decider credit refills 1 per 8 ticks. Two conversations (sandboxes A, B), two turns each (3 and 5 slices,
the second extends the first), up to 12 decider calls, 2 client faults a run, memory for 9 KV units. A unit maps
the page it writes when it starts (EnsureKv before the chunk or step), so the memory properties see it.

**Bounds the properties check.** `WAIT_BOUND` = the unit in service (2) + the other sandbox's catch-up after its
join (3) + the decider principal's catch-up (3) + two credited deciders (2) = 10 ticks (~6 s); `DEC_BOUND` = the
unit in service + one credited decider ahead = 4 ticks (~2.4 s); `GONE_BOUND` = `ACK_TO` + one unit = 4 ticks.
The client idle timeout (5) is below `WAIT_BOUND` on purpose: without heartbeats a fairly served request can time
out, which is the D3 trigger.

**Model bugs found while running it** (fixed before the table below): admission counted the arriving request in
its own ledger (false memory 429s); the unit kind was chosen before the reused prefix was bound (an extra slice);
a 429 sent to a late attempt released the client's newer request. The fourth was a design bug, not a model bug: the
fair-queuing join rule above.

**Rules** (`Rules` record; the design sets all; each mutant in `schedTest.qnt` turns one off): preempt,
deciderPrio, deciderBudget, fenceDeparture, deadPeerProbe, heartbeat, parkOnCancel, bindAtStart, reserveLedger,
evictSeesFreed, pinPlanned, atomic429, idempotency.

**Properties**: `rejectedNeverRuns`, `boundedWait`, `deciderLatency`, `workAfterGoneBounded`,
`noSpuriousTimeout`, `noOvercommit`, `withinReservation`, `refuseOnlyIfShort`, `noRecompute`,
`noSelfInflicted429`, `generatedOnce` (and `safety`, their conjunction).

**Results** (`spec/scheduler/results.txt`, seed 0x5eed, 120 steps; 20000 traces for the design, 5000 per mutant):

| Instance | Property | Expect | Result | What breaks it |
|---|---|---|---|---|
| design (seeds 0x5eed, 0xbeef) | `safety` (all 11) | ok | ok, 2 x 20000 | |
| design, keyless clients (`noIdempotency`) | `noRecompute`, `boundedWait` | ok | ok | park on cancel + bind at start protect keyless retries |
| today (one mutex, FIFO, to the end) | `boundedWait` / `deciderLatency` | VIOLATED | VIOLATED / VIOLATED | a turn or a decider waits for another sandbox's whole request |
| `noPreempt` | `boundedWait` | VIOLATED | VIOLATED | whole requests as units |
| `noDeciderPrio` | `deciderLatency` | VIOLATED | VIOLATED | a decider waits for fair turns behind slices |
| `strictDeciderPrio` | `boundedWait` | VIOLATED | VIOLATED | a decider flood starves the sandboxes |
| `noFence` | `workAfterGoneBounded` | VIOLATED | VIOLATED | a departed request runs on |
| `noDeadPeerProbe` | `workAfterGoneBounded` | VIOLATED | VIOLATED | a peer gone without FIN is never found |
| `noHeartbeat` | `noSpuriousTimeout` | VIOLATED | VIOLATED | AUDIT D3: an idle timeout while queued / prefilling |
| `dropOnCancel` | `noRecompute` | VIOLATED | VIOLATED | the retry re-prefills what the departed attempt computed |
| `bindAtAdmission` (keyless) | `noRecompute` | VIOLATED | VIOLATED | the prefix chosen before the predecessor parked |
| `memoryReadingOnly` | `noOvercommit` | VIOLATED | VIOLATED | AUDIT D1/E3: two admissions grow into the same memory |
| `releasedStaysCharged` | `refuseOnlyIfShort` | VIOLATED | VIOLATED | AUDIT F3: eviction cannot see what it freed |
| `evictPlanned` | `withinReservation` | VIOLATED | VIOLATED | a planned prefix evicted under an admitted request |
| `queued429` | `rejectedNeverRuns` | VIOLATED | VIOLATED | AUDIT N4: a 429'd request still runs |
| `noIdempotency` | `generatedOnce` / `noSelfInflicted429` | VIOLATED | VIOLATED / VIOLATED | a lost answer regenerated; a retry refused by its own attempt |

Witnesses (design, 20000 traces): preempted 99.97%, batched 70.2%, decider between slices 99.8%, decider out of
credit served fairly 100%, parked on cancel 13.0%, evicted 34.1%, memory 429 94.7%, retry after 429 75.3%, client
timeout 40.9%, fenced after FIN 21.7% / after a vanished peer 6.1%, late arrival 44.1%, replayed 7 traces,
adopted 36.6%, second turn done 0.9%; keyless instance (5000): limit 429 30.1%, prefix bound later 4.9%.

**Limits of the model.** Two principals and one conversation each; the eviction candidate is the other
conversation's prefix only; decode units charge each participant a whole unit (the doc's rule charges shares);
memory is in abstract units with no page rounding or fork constant; simulation, not exhaustive checking.

## 11. Decisions for the user

**Decided 2026-10-07: the user accepted every recommendation below.** D1 2048 contended / 8192 alone; D2 measure
the decider's GPU contention first (M4), flash-pointer deciders as units; D3 one in-flight turn per sandbox; D4
burst 4, refill 2/s; D5 default key = digest of (principal, previous_response_id, input); D6 heartbeats on (main
24cab4b: `halo_serve --sse-keepalive` defaults to 15 s).

- **D1 slice size**: 2048-token slices under contention, 8192 alone (recommended), or always 8192 (7% faster
  contended prefill, 5-8 s preemption bound).
- **D2 the 2B decider**: keep it concurrent on the GPU (today: no wait for a slice, unmeasured contention) or move
  it into the scheduler as units (bounded wait <= one slice + one decider, ~1.3-2.3 s). Recommended: measure
  contention first (HARNESS D1) in M4; flash-pointer deciders (P9, forks of Flash-Next sessions) must be units.
- **D3 PLIMIT**: one in-flight turn per sandbox principal (recommended) or more (sub-agents in one sandbox would
  then share its fair share, not multiply it).
- **D4 decider budget**: burst and refill rate (recommended: burst 4, refill 2/s, per the SYSTEMONE hot-path uses).
- **D5 idempotency default key** when the client sends none: digest of (principal, previous_response_id, input)
  (recommended) or none (retries of keyless clients regenerate).
- **D6 heartbeats default on** in halo_serve (recommended; the A2 branch has them opt-in).
