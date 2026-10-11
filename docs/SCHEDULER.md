# G3: a fair GPU scheduler for halo_serve (design, 2026-10-07)

Design, model and build. Model: `spec/scheduler/` (Quint 0.32, `quint run` only), checked by
`spec/scheduler/run.sh`, results in `spec/scheduler/results.txt`. Predictions (written before any G3 code):
`/data/halo-work/streams/B/PREDICTIONS-G3.md`, summarised in section 9. Fairness is two-level since A2A-MCP D9 (chain
root, then sandbox: section 4). What is built and measured: section 12.

## 0. Recommendation in one paragraph

One **engine thread** owns the GPU and runs one **unit** at a time: a prefill slice (one exact 2048-token chunk
while anyone else is runnable, 8192 when alone), a **batched decode unit** (StepBatch over every decoding session,
a short burst of steps), or a decider evaluation. Between units it **fences** departed requests (parking their
slice-aligned progress as a retained prefix) and **picks** the next unit: a decider with **credit** first, then
**start-time fair queuing by GPU time over sandbox principals** (the decider class is one more principal once out of
credit). The fair queue has **two levels** (A2A-MCP D9): the **chain root** a request serves first, then the sandbox
principal within that root, so an agent chain that fans out to several sandboxes holds one share, not one per
sandbox. HTTP threads **admit** under the same lock against a **reservation ledger** (retained prefixes + every
admitted request's reservation; eviction counts what it frees), with a per-principal in-flight limit; a refusal is
a **429 with Retry-After** that takes the request off the queue in the same critical section. Clients get **SSE
heartbeats** while queued or prefilling, and every turn carries an **idempotency key**: a retry of a finished turn
replays the stored answer, a retry of an in-flight one adopts it. The model holds all 12 properties for this design
(two seeds, 20000 traces each); today's scheduling and each of 14 mutants (one rule off each) break the property
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

**Fairness: two levels (A2A-MCP D9).** Start-time fair queuing (SFQ) by **GPU time actually used** (the unit's
measured duration; a batched decode unit charges each participant a whole unit), on two levels:
1. over **chain roots**: a request carries the chain it serves (`X-Halo-Chain-Id`, stamped by yard-relay on
   deliveries and forwarded by the callee's broker); a sandbox outside any chain is its own root, so lone agents
   see no change;
2. within a root, over its **sandbox principals** (`X-Halo-Principal`; unknown callers are one shared principal);
3. within a principal, FIFO.
Each root and principal has a virtual time; the pick takes the root with the least, then its principal with the
least. A root that was idle **joins at the system virtual time** (the least tag among the backlogged roots at the
last pick: it banks no credit while idle), and a principal joins at its root's own system virtual time. Joining at
the minimum over the *other runnable* principals is not enough: a principal briefly not runnable (departed, then
adopted by its retry) is left out of that minimum and then waits for the joiner to catch up; the model found this
at 20000 samples (`boundedWait` 11 > 10 ticks). This is what makes a cold 128K prompt coexist with other
sandboxes: it gets its share, the others keep theirs.

Why two levels: with one level, a chain whose root calls B, C and D at once holds three shares (and a loop of
agents calling each other crowds out an unrelated interactive sandbox). With two, the chain holds one share and
its sandboxes split it: a request that shares its root with another sandbox waits about twice as long between
units (`WAIT_CHAIN` = 2 x `WAIT_BOUND` - one unit, section 10), and an unrelated root's service stays within
`LAG_BOUND` of the chain's (`chainFair`; the one-level mutant `oneLevelSfq` breaks it). Batching adds **rides**: a
decoding request is in every decode unit, also one another root starts, and is charged for it; the lag bound
counts one root's catch-up of rides.

**Request headers** (set by the broker, never by a sandbox):
| Header | Meaning | Default when absent |
|---|---|---|
| `X-Halo-Principal` | the sandbox principal (fair-queuing principal, D3 limit) | `anonymous` (one shared principal) |
| `X-Halo-Chain-Id` | the chain root the request serves (A2A-MCP 5.7, 9) | the principal itself |
| `Idempotency-Key` | the turn's key (section 7) | with a principal: `d5:` sha256(principal, previous_response_id, input, API); without one: none |
yard-relay stamps `yard_chain` on deliveries and the callee's broker forwards it as `X-Halo-Chain-Id`; the broker
sets `X-Halo-Principal` from the sandbox's tap identity (A2A-MCP 3: "headers from the VM never carry identity").
halo_serve trusts these headers as it trusts the broker's other `X-Halo-*` headers: **the broker must strip any
`X-Halo-*` and `Idempotency-Key` header a sandbox sends and set its own** (a sandbox that could pick its principal
or chain could take another's share or its key's replay). The default key applies only to principal-stamped
requests: an anonymous caller repeating a prompt (an eval sampling N answers) means it, and must not get a replay.

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

**How a request was answered: `Halo-Answer`.** Every admitted response (OpenAI routes and `/v1/messages`,
streamed or not, success or error) carries the response header `Halo-Answer` with one of:
- `generated`: a new generation ran for it (with a key whose earlier attempt was fenced, this is a regeneration:
  the fenced attempt's tokens were spent and are in no answer's usage);
- `adopted`: it took over the generation already running for its idempotency key (within the keyed grace);
- `replayed`: its key's generation had finished; the stored answer was sent, with no engine work.
The usage an adopted or replayed answer reports is that one generation's. A server that keeps no keys (`--scheduler
off`, or a build before G3) sends no `Halo-Answer` at all: a client can tell from its absence that keys are not
honoured. A 429 carries none (nothing was admitted).

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
3, the decider credit refills 1 per 8 ticks. Three conversations (sandboxes A, B, C), two turns each (3 and 5
slices, the second extends the first); each sandbox serves its own chain or the chain X (drawn at the start, so a
run has a fan-out of two or three sandboxes under X, or none); up to 12 decider calls, 2 client faults a run, memory
for 14 KV units, 48 ticks a run (160 steps: long enough for a second turn with three sandboxes). A unit maps the
page it writes when it starts (EnsureKv before the chunk or step), so the memory properties see it.

**Bounds the properties check.** `WAIT_BOUND` (a request alone in its chain root) = the unit in service (2) + the
catch-up of the two other agent roots after their join (3 each) + the decider root's catch-up (3) + two credited
deciders (2) = 13 ticks (~8 s); `WAIT_CHAIN` (a request whose root holds another sandbox's request too) = 2 x
`WAIT_BOUND` - one unit = 24 ticks (~14 s): it gets half its root's turns, and its root-mate's decode rides in the
other roots' batches are charged to the root; the simulation's worst case is this bound (22 breaks at 48 ticks with
seed 0xbeef, 24 holds on both seeds). `LAG_BOUND` (two agent roots backlogged together: their GPU service since
then stays within) = one unit of each (2 x 2) + one root's catch-up of rides (2 + 1) = 7 ticks. `DEC_BOUND` = the
unit in service + one credited decider ahead = 4 ticks (~2.4 s); `GONE_BOUND` = `ACK_TO` + one unit = 4 ticks.
The client idle timeout (5) is below `WAIT_BOUND` on purpose: without heartbeats a fairly served request can time
out, which is the D3 trigger.

**Model bugs found while running it** (fixed before the table below): admission counted the arriving request in
its own ledger (false memory 429s); the unit kind was chosen before the reused prefix was bound (an extra slice);
a 429 sent to a late attempt released the client's newer request. The fourth was a design bug, not a model bug: the
fair-queuing join rule above. With two levels: the first `chainFair` bound (one unit of each, 4) broke in the design:
a decoding root rides batches another root starts while catching up; and the first two `WAIT_CHAIN` derivations (18,
then 20) undercounted the root-mate's tag ties and rides (found at 36 and 48 ticks).

**Found by the M5 trace replay** (the code against the model's own traces, below), each a place where the model
knew more or less than the server can, fixed in the model (the code was right): FIFO within a principal was by the
order clients *sent* requests (model ids), but the server sees only arrival order (now `seq`, the admission order);
admission decided "late duplicate" vs "adopt" on the client's true connection state, but the server knows only what
it saw (FIN, or a failed probe): a peer that vanished a moment ago looks connected, so its retry adopts the
in-flight attempt, which then lives or dies with that connection; fences were one request a step, but the code
fences every departed request at a boundary in one critical section (an arrival cannot land between two fences);
ties in virtual time were a nondeterministic choice, now the root (then principal) holding the request admitted
first, as the code breaks them.

**Rules** (`Rules` record; the design sets all; each mutant in `schedTest.qnt` turns one off): preempt,
deciderPrio, deciderBudget, fenceDeparture, deadPeerProbe, heartbeat, parkOnCancel, bindAtStart, reserveLedger,
evictSeesFreed, pinPlanned, atomic429, idempotency, twoLevel.

**Properties**: `rejectedNeverRuns`, `boundedWait` (with groups: `WAIT_BOUND` / `WAIT_CHAIN`), `deciderLatency`,
`workAfterGoneBounded`, `noSpuriousTimeout`, `noOvercommit`, `withinReservation`, `refuseOnlyIfShort`,
`noRecompute`, `noSelfInflicted429`, `generatedOnce`, `chainFair` (and `safety`, their conjunction).

**Results** (`spec/scheduler/results.txt`, seed 0x5eed, 160 steps; 20000 traces for the design, 5000 per mutant but
`generatedOnce`'s, which needs 20000: a replay is rare with three sandboxes):

| Instance | Property | Expect | Result | What breaks it |
|---|---|---|---|---|
| design (seeds 0x5eed, 0xbeef) | `safety` (all 12) | ok | ok, 2 x 20000 | |
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
| `oneLevelSfq` | `chainFair` | VIOLATED | VIOLATED | a chain fanned out to two sandboxes takes two shares (A2A-MCP D9) |

Witnesses (design, 20000 traces, after M5's corrections): preempted 100%, batched 98.4%, decider between slices
99.96%, decider out of credit served fairly 100%, parked on cancel 17.6%, evicted 88.5%, memory 429 94.4%, retry
after 429 87.0%, client timeout 55.7%, fenced after FIN 35.3% / after a vanished peer 12.9%, late arrival 54.5%,
replayed 8 traces, adopted 30.5%, second turn done 33.3%, chain fan-out (a root with two backlogged sandboxes beside
another root) 35.2%; keyless instance (5000): limit 429 40.3%, prefix bound later 3.8%. A run takes ~10 min per 20000 traces.

**Limits of the model.** Three principals and one conversation each, one chain (X) besides the lone roots, a
conversation's chain fixed for the run; an admission that needs room evicts every evictable prefix at once (the
code evicts least recently used first until it fits); decode units charge each participant a whole unit (as the
code does); memory is in abstract units with no page rounding or fork constant; simulation, not exhaustive
checking; the bounds are the simulation's worst case, derived after the fact for `WAIT_CHAIN` and `LAG_BOUND`.

## 11. Decisions for the user

**Decided 2026-10-07: the user accepted every recommendation below.** D1 2048 contended / 8192 alone; D2 measure
the decider's GPU contention first (M4), flash-pointer deciders as units; D3 one in-flight turn per sandbox; D4
burst 4, refill 2/s; D5 default key = digest of (principal, previous_response_id, input); D6 heartbeats on (main
24cab4b: `halo_serve --sse-keepalive` defaults to 15 s). Also accepted (A2A-MCP.md D9): two-level fair queuing,
chain root then sandbox (section 4; model re-checked, section 10).

- **D1 slice size**: 2048-token slices under contention, 8192 alone (recommended), or always 8192 (7% faster
  contended prefill, 5-8 s preemption bound).
- **D2 the 2B decider**: keep it concurrent on the GPU (today: no wait for a slice, unmeasured contention) or move
  it into the scheduler as units (bounded wait <= one slice + one decider, ~1.3-2.3 s). Recommended: measure
  contention first (HARNESS D1) in M4; flash-pointer deciders (P9, forks of Flash-Next sessions) must be units.
  **Measured (section 12): keep it concurrent** (`--decider-mode concurrent`, the default): p90 421 ms beside a
  prefill against 1.9 s as a unit with 2048-token slices (8.7 s with 8192 alone); its price at 2 deciders/s is
  Flash-Next decode -20% and prefill ~1.5x. `--decider-mode scheduled` exists for flash-pointer deciders.
- **D3 PLIMIT**: one in-flight turn per sandbox principal (recommended) or more (sub-agents in one sandbox would
  then share its fair share, not multiply it).
- **D4 decider budget**: burst and refill rate (recommended: burst 4, refill 2/s, per the SYSTEMONE hot-path uses).
- **D5 idempotency default key** when the client sends none: digest of (principal, previous_response_id, input)
  (recommended) or none (retries of keyless clients regenerate).
- **D6 heartbeats default on** in halo_serve (recommended; the A2 branch has them opt-in).

## 12. Implementation and measurements (stream G3, 2026-10-07)

| # | Commit | What | Tests |
|---|---|---|---|
| model | a63f49e, M5 commit | two-level SFQ (chain root, then sandbox), `chainFair`, `oneLevelSfq`; M5's corrections (section 10) | `run.sh`: failures 0 |
| M1 | 35406cd | `Prefill(..., chunk_tokens)`: a slice is one engine call on the bound session | `Sessions.InterleavedSlicesMatchAlone` (bitwise); mutant |
| M2 | 123a3bc | `Scheduler` (host-only policy) | Hegel machine `SchedulerModel.PropertiesHoldUnderTheModelsActions` (12 properties) + 8 directed; 15 `sched-*` mutants |
| M3 | ade4858, 18d3405 | `ScheduledGenerator` (engine thread, leases, StepBatch decode units, MTP only at B = 1), headers, 429 + Retry-After, D5 key; a completing slice runs the first decode step, a checkpoint inside a slice is parked on the way | `ScheduledGenerator.ConcurrentRequestsEqualEachServedAlone` (small model, tokens equal each request served alone), `OpenAiService.ServingModelUnderDeparturesStallsAndRetries` on the scheduler with 3 clients; 5 mutants |
| M4 | 83f5899, 0fdb6a4 | `DeciderGate` (`--decider-mode concurrent|scheduled`), `RunDecider` units, `halo:sched_*` / `halo:decider_*` metrics; the idle engine wakes on an admission | `ScheduledDecider.*`, `SystemOneService.EvaluationsGoThroughTheDeciderGate`; 4 mutants |
| M5 | M5 commit | `spec/scheduler/replay.sh` + `itf_events.py`: `quint run --mbt --out-itf` traces as server events; `SchedulerReplay.ModelTracesMatchTheCodeStepByStep` replays them against `Scheduler` (admission outcomes, fences, picks, every virtual time), `Scheduler::Boundary` | 24 checked-in traces (1143 compared steps), 300 more on 6 seeds all match; kills 12 of the 15 `sched-*` mutants alone, and 2 of its own |

**Serving, real model** (unpinned clocks: `tools/gpu_clock.sh` cannot pin on this host; turns of 600-1100 new
tokens on 12-15K contexts, 300 output tokens; `tools/serve_ab.py conc`):

| Sandboxes | Today (`--scheduler off`) TTFT median / worst | G3 TTFT median / worst | G3 per-session tok/s | Aggregate tok/s today / G3 |
|---:|---|---|---:|---|
| 1 | 1.17 / 1.28 s | 1.21 / 1.30 s | 25.5 | 23.4 / 23.2 |
| 2 | 7.2 / 14.9 s | 1.42 / 2.72 s | 17.0 | 22.3 / 30.7 |
| 4 | 20.9 / 51.4 s | 2.89 / 5.22 s | 10.4 | 22.5 / 37.1 |

B's P2 assumed 0.45 s turns; these turns take 1.0-1.3 s alone, and TTFT scales with them. Two fixes came out of the
first runs: under GPU-time fairness the request whose prefill had just finished waited a round of everyone's slices
for its first token (the first step now belongs to the completing slice), and a turn's prefill stopped at its
exact-mode checkpoint, costing it a second round (the checkpoint is now parked inside the slice).

**A cold 130K prompt beside a decoding sandbox**: the decoder ran 23.3 tok/s before, 11.9 tok/s while both were
backlogged, and finished 1500 tokens in 120 s; the cold prompt's first token came at 170 s (alone ~97 s; +7% for
2048-token slices; + the decoder's ~60 s of GPU). No allocation failure at this size on this branch.

**D2** (2B decider at 2 calls/s, fresh 23-24K prefills):

| | concurrent | scheduled, 8192 alone | scheduled, 2048 alone |
|---|---|---|---|
| decider beside a decode: median / p90 | 218 / 224 ms | 419 / 475 ms | 419 / 476 ms |
| Flash-Next decode beside deciders | -20% | -30% | -33% |
| decider beside a prefill: median / p90 | 380 / 421 ms | 5.6 / 8.7 s | 1.8 / 1.9 s |
| prefill time per token beside deciders | +53% | +25% | +9% (and +10% alone, for 2048 slices) |

A scheduled decider waits for the slice in progress, whole (the next call arrives just after a slice starts), so
its latency is the slice length; concurrent, the two models share the GPU at kernel granularity with little real
overlap. Decision rule fixed before measuring: concurrent, since its p90 beside a prefill (421 ms) is below a
unit's wait.

## 13. Keyed grace, and a departure that is not a stop (stream GM, 2026-10-11)

G3 forward onto main e4cabdc (merge 667d4d4), then two changes found by a cordis-shaped client: a non-streaming
chat completion with an `Idempotency-Key`, a 10 s client timeout, retried with the same key. Its answer (1200 tokens,
~45 s of decode) outlasts every attempt.

**The gap.** Section 7's adoption only works while the departed attempt is still in flight, and a departure is
fenced at the next unit boundary (~0.3 s). A client that retries at once was adopted (4 times; answered at attempt
5, one generation). A client that backs off 2 s (the OpenAI SDKs back off 0.5-8 s) found its attempt fenced every
time: 6 admissions, the decode redone 5 times, and no answer unless a restart happened to fit in 10 s.

**Keyed grace** (`Options::keyed_grace_s`, `--sched-keyed-grace`, default 30 s): a departed request with a key is
fenced only at the first boundary after the grace, measured from when the server learned of the departure. Until
then it runs on: a retry with the key adopts it, or replays it once it finished (`halo:sched_grace_finished_total`).
Keyless departures are fenced at once, as before. The price is the late-effect bound: work for a client that
never comes back grows from `GONE_BOUND` to `GONE_BOUND + GRACE` (model: `goneBound`).

**A departure is not a stop.** The service's text callback returns false both for a stop string and when a write
fails (the client left); `ScheduledGenerator::Wait` took every false for a consumer stop, so the engine finished the
request *completed* at its next token, stored the truncated text for replay and marked the key finished. A retry with
the key then got the truncated answer as a finished one (`finish_reason: stop`): one of the probe's runs got 266 of
1200 tokens this way. Which path saw the departure first (the callback, or `Wait`'s 20 ms poll) was a race. Now the
service reports its own departure through `Params::cancelled`, and `Wait` checks it before taking a false for a stop.
Main (no replay store) is not affected.

**Counting.** With adoption, every reader that left was recorded as a cancelled request with the tokens it had seen
(one 1200-token generation counted 3825). A reader that leaves a keyed request, or is taken over by a retry, is now
`detached` (finish reason) and counts no tokens; the first reader to receive the whole answer (directly, by adoption
or by replay) counts them once.

**Model** (`spec/scheduler/`): rule `keyedGrace`; caller kind `ns` (sandbox C may be a non-streaming caller whose
answer, `NS_TOK = CT + 1` decode units, outlasts its timeout); property `noRedoWithinGrace` (a retry that reaches the
server within `GRACE` of its attempt's departure never finds it fenced with decoding done); mutant `noKeyedGrace`
breaks it; witnesses `wNsTimeout`, `wNsDone`, `wFinishedDetached`. `workAfterGoneBounded` uses `goneBound`. The
replay traces were regenerated (24 traces, 1154 compared steps; 42 of their 43 adoptions come after the server knew
the client had left).

**Tests.** `Scheduler.KeyedDepartureRunsOnThroughItsGrace`, `Scheduler.KeyedDepartureFinishedWithinItsGraceIsReplayed`;
`ScheduledGenerator.ARetryWithTheKeyGetsTheWholeAnswerAfterTheCallbackFoundTheClientGone` (fails without the fix with
the truncated answer, 3 of 3 runs); the Hegel machine runs with the model's grace; mutants `sched-no-keyed-grace`,
`sg-detached-reader-counts-tokens`, `sg-departure-taken-for-stop`.

**Measured** (`retry_probe.py`, 8 attempts, 10 s timeout, greedy, reference answer sha 41d06cc6df42):

| Client | G3 as merged | + keyed grace and the fix |
|---|---|---|
| key, retry at once | answered at attempt 5, 47.6 s; 1 generation; counted 3825 tokens, 4 "cancelled" | attempt 5, 47.4 s; 1 generation; 1200 tokens, 4 "detached"; = reference |
| key, retry after 2 s | 6 generations (5 fenced), answered at 74 s with a truncated replayed answer (266 tokens, `stop`) | attempt 4, 47.2 s; 1 generation (3 adoptions); = reference |
| no key, retry after 2 s | never answered (8 generations) | unchanged |

Restarting after a fenced mid-decode departure is exact: three cuts at 5 s (300 tokens) and six keyed cuts at 10 s
(1200 tokens) both answered byte-identically to the uninterrupted reference, on this branch and on main.
