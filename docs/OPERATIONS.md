# The operation API (proposal, revision 3, 2026-10-12)

Status: **proposed**. This is halo-flash's half of cordis RFC 0007 ("Boundaries, and the operation contract").
No code changes until the contract is agreed. The model is `spec/operation/`.

Revision 3 aligns with cordis RFC 0007 revision 4, the normative text, which adopts this proposal:
- every verb carries the attempt's `not_after` and digest;
- records (held attempts, tombstones, unacknowledged outcomes) have a ceiling, and halo refuses rather than erases
  (model mutant `eraseForRoom`).

Revision 2 followed the third review of revision 1 (58e60d3). Changes:
- the durable capture is split into fetch, commit and acknowledge;
- every verb is bound to an incarnation and an owner;
- deadlines are in halo's clock, and expiry is modelled;
- a refusal closes only its copy;
- cancellation is a message, and `on_detach` is gone;
- retention is reserved when a submit is accepted;
- usage is disjoint, and the repeat guarantee is corrected.

## 1. Why

Cordis has been reconstructing an inference operation's history from a connection-oriented interface whose
server remembers for a bounded time. Today:

- **An unknown key starts a new generation.** That happens after a restart, or once the scheduler has forgotten
  the key (`replay_ttl_s` 600 s, or past `max_keys` 4096).
- **Retention is a cache.** It evicts under load.
- **The server cannot tell "never accepted" from "I no longer know".**

`Halo-Answer`, `Halo-Discarded-Tokens`, `Halo-Admitted-Sends` and keyed grace each patch one consequence of this.

## 2. Identity, clock, binding

- **An attempt is `(owner, operation id, attempt)`.**
  - The owner is the authenticated principal.
  - The operation id and attempt are issued by the platform, unique per owner.
  - The attempt is the unit of execution and accounting. A new execution is a new attempt that the platform
    authorized; halo never starts one on its own.
- **Binding.** A submit binds the attempt to a digest of its immutable parts: `api`, `request`, `not_after`, and the
  execution mode (section 4). Any later message for the attempt naming another digest gets 409 `conflict` and
  changes nothing.
- **Incarnation.** 128 random bits chosen at process start. It is per process, not a durable epoch (RFC 0007 open
  question 2): halo persists nothing an operation depends on.
  - **Every verb names the incarnation it is meant for.** Another incarnation answers `unknown` (status) or 409
    (submit, cancel, ack), and acts on nothing.
  - So an old query after a restart gets `unknown`, never the new process's `never_accepted`.
  - A delayed cancel or ack can only reach the attempt it names, in the incarnation it names. Attempt ids are never
    reused, so there is no other execution for it to touch.
- **Clock.** Every answer carries `Halo-Time`: milliseconds on halo's monotonic clock since the incarnation started.
  - Deadlines are expressed in that clock, and only halo compares them, so no two hosts' clocks are ever compared
    and halo's clock never runs backward.
  - The platform computes `not_after` from its latest `Halo-Time` reading plus its own monotonic time elapsed since
    that reading, plus the window. An underestimate only makes the window shorter.
  - Halo rejects a `not_after` more than `max_submit_age_s` (proposed 600 s) ahead.
  - **A restart restarts the clock with a new incarnation, so a deadline never outlives its clock.** This holds
    whether the platform and halo share a host or not.

## 3. Verbs

All requests carry `Halo-Incarnation`, `Halo-Not-After` and `Halo-Digest`, and are authenticated. All answers carry
`Halo-Incarnation` and `Halo-Time`.

**Every verb carries the deadline and digest.** That lets halo answer for an attempt whose submit never arrived.
With no record, it compares its clock with the carried `not_after`: `never_accepted` before it, `closed` after it.
Without a carried deadline, halo would have to guess one, and an early guess answers `closed` while a copy can still
be accepted (cordis `expiry.qnt`, `guessDeadline`). A cancel's tombstone records the carried `not_after` and
digest, so a later submit naming another digest is a conflict.

| Verb | Request | Answers |
|---|---|---|
| contract | `GET /v1/halo/contract` | incarnation, time and the guarantees per operation type (section 5) |
| submit | `POST /v1/operations` `{operation_id, attempt, not_after, api, request, mode}` | 202 `accepted`; a duplicate gets the current status; 409 stale incarnation or `conflict`; 410 past `not_after`; 429 principal limit or 503 retention/memory full, with Retry-After |
| status | `GET /v1/operations/{op}/{attempt}` | see below |
| fetch | `GET /v1/operations/{op}/{attempt}/result` | the result body and usage, while held |
| observe | `GET /v1/operations/{op}/{attempt}/stream?from=<event>` | SSE from an event offset while the body is held; 410 with the status afterwards. Leaving is an observer departing, never a cancel. Partial tokens are observations, not receipts. |
| cancel | `POST /v1/operations/{op}/{attempt}/cancel` | `cancelled_before_accept` (the id is tombstoned: a later submit is a no-op); `running` with `cancel_requested: true` (it finishes at the next unit boundary as `done`, finish `cancelled`, with its actual usage); or the current terminal status |
| ack | `POST /v1/operations/{op}/{attempt}/ack` | the platform has durably committed the result: halo releases the body and keeps the outcome until the tombstone is pruned |

**Status answers.**

| Status | Meaning | The platform |
|---|---|---|
| `running` | accepted, executing | waits |
| `done` | finished (finish reason, usage, `expires_at` in halo time); the body can be fetched | fetch, durably commit receipt and settlement, then ack |
| `released` | acknowledged and released | nothing to do (it committed before acking) |
| `expired` | finished, but the capture window passed without an ack; the body is gone, the usage is not | settle at that usage, no receipt; a new attempt if the operation still needs one; ack the outcome |
| `cancelled_before_accept` | cancelled before any copy was accepted | settle at 0 (terminal) |
| `closed` | never accepted, and `Halo-Time` is past `not_after`: no copy can be accepted any more | settle at 0 (terminal) |
| `never_accepted` | not accepted *yet*: a delayed copy may still arrive | keep the reservation; resend, cancel, or wait for `closed` |
| `unknown` | another incarnation | settle at the worst case; a new attempt to run again |

`forgotten` (evicted before capture) does not exist under the promise (section 5). A server without the promise
would answer it, settled at the worst case.

**Rules the platform keeps** (each is checked in the model; section 7):

1. **Commit before ack.** Fetch the result, durably commit the receipt and settlement, then ack, then deliver the
   receipt to the workflow, which applies a receipt once per attempt. A lost ack only keeps the body longer: until
   it is acknowledged, or until its window ends and it becomes `expired`.
2. **A refusal closes the copy, not the attempt.** A 429 or 503 says this copy was not accepted. Another copy may
   still be. The reservation stays until a terminal answer: `done`, `expired`, `cancelled_before_accept`, `closed`
   or `unknown`.
3. **Settle `never_accepted` only through a terminal answer.** That means `cancelled_before_accept` (from the cancel,
   or from a later status) or `closed`.
4. **Resend a submit exactly as first sent.** Same identity, incarnation, deadline and digest. It is never re-aimed
   at a new incarnation.
5. **Repeat a cancel until it is answered.** The cancel or its answer can be lost, and it can race acceptance or
   completion. A cancelled attempt that ran has a terminal result with its actual usage.

## 4. Usage and execution modes

- **Usage is `{prompt, cached, completion}`, all disjoint.**
  - `completion` counts every token this attempt decoded, delivered or not; a cancelled attempt included.
  - There is no `discarded`: an attempt executes at most once, so its usage is all of its work.
  - Token counts are the accounting unit. `engine_ms`, the engine time attributed to the attempt, is reported as a
    measurement, not a cost.
- **`mode`** is `relaxed` (the default) or `exact`, and is part of the binding digest.
  - Under `relaxed`, kernels for batches of 2 or more may change a greedy session's output depending on which other
    sessions share the batch. About 0.8% of token positions measured on the fixtures (docs/SESSIONS.md, "Relaxing
    exactness").
  - Under `exact` (the server's `--batch-relaxed 0`, or an attempt that runs alone), greedy output is identical on
    the same build and weights.
  - Sampled decoding with a seed is reproducible under the same conditions.

## 5. Retention: a bounded capture window, reserved at acceptance

- **The body is held until acknowledged, or until the capture window ends.** The window is `capture_s`, advertised;
  each `done` carries `expires_at`.
  - After the window, the outcome (status and usage) is kept until it is acknowledged, then pruned with its
    tombstone. The model's `dropExpiredOutcome` mutant shows why: pruning an unacknowledged outcome turns "it ran"
    into "closed".
  - **The capture window is the promise.** Halo never releases a body before its ack or its `expires_at`, and never
    evicts under load. This replaces RFC 0007's "until captured", which needs unbounded memory, with a bounded,
    stated one.
- **Space is reserved at acceptance.** Each submit reserves the most its result can need:
  - `max_tokens` × the bytes-per-token bound;
  - tool-call arguments and reasoning text, which are tokens of the same output;
  - the stream's event history, which is the body.

  If the reservations would exceed the retention budget, halo refuses with 503 and Retry-After. Many attempts
  finishing together therefore cannot overflow it (mutant `noReserve`).
- **Records are bounded, never erased to make room.** A record is a held attempt, a tombstone or an
  unacknowledged outcome. Each accepted attempt and each cancel-before-accept holds one until it is pruned.
  - At the `max_records` ceiling, halo refuses the submit or the cancel with 503, reason `records`, and
    Retry-After. A refused cancel leaves the outbox waiting for `closed`.
  - Erasing a tombstone or an unacknowledged outcome to make room would turn "it ran" into `closed` (mutant
    `eraseForRoom`).
  - **What a platform that stops acknowledging sees** (RFC 0007 open question 3): the contract and `/metrics` report
    records in use, unacknowledged outcomes and the oldest one's age. A 503 names `records` or `retention`, never
    a generic failure.
- **Tombstones.** An attempt id stays known until its `not_after` has passed and, if it ran, its outcome was
  acknowledged. Until then a duplicate or late submit is a no-op. Memory is about 50 bytes per id seen in one
  `max_submit_age_s` window, plus unacknowledged outcomes.
- RFC 0007 open question 3, the cost of refusals, is to be measured on the 8-agent pool. A submit reserves
  `max_tokens`, so callers that set it far above what they use will be refused earlier. The measurement should
  report that.

`GET /v1/halo/contract`:

```json
{"incarnation": "9f…", "time_ms": 123456,
 "inference": {"mode": "relaxed", "cost": "tokens: prompt, cached, completion",
               "repeat": {"external_effects": "none", "spend": "a new generation spends again",
                          "exact": "identical greedy output on the same build and weights",
                          "relaxed": "greedy output may differ with batch participation"}},
 "operations": {"apis": ["completions", "chat"], "owner": "X-Halo-Principal",
                "digest": "sha256 (hex) of the submit body",
                "retention": {"capture_s": 3600, "until": "ack or capture window", "budget_bytes": 268435456,
                              "reserve": "at acceptance: 16384 + max_tokens x 256 bytes", "reserved_bytes": 0},
                "records": {"max": 100000, "in_use": 12, "unacked_outcomes": 1, "oldest_unacked_ms": 5021},
                "dedup": {"scope": "owner, incarnation", "max_submit_age_s": 600},
                "cancel": "next unit boundary (<= one decode unit or one prefill slice); usage reported",
                "disconnect": "never cancels"}}
```

`operations` is `null` on a server run with `--operations 0`: the verbs are not served, and every field above it
still is.

A server started with `--operations off` advertises no operation contract and keeps today's header contract.

## 6. What stays, and how it maps

- The OpenAI, Anthropic and systemone endpoints are unchanged, along with `Idempotency-Key`, the `Halo-*` headers
  and keyed grace. They remain the contract for ordinary clients, and for cordis until its adapter moves (RFC 0007,
  order of work step 3).
- **An operation is a scheduler key** (`owner/op/attempt`), so admission, fairness, fencing and the memory ledger
  are reused.
- New in the scheduler's key table:
  - release on acknowledgement, plus the capture window;
  - tombstones;
  - the incarnation and deadline checks;
  - retention reservation at acceptance;
  - binding digests;
  - no detach policy: operations are cancelled only explicitly.

## 7. The models, and how they correspond

**The two models have different scopes.** Cordis's `operation.qnt` is the accounting abstraction: an attempt is
held or not, captured or unknown. Ours covers what that abstraction assumes atomic:
- executor identity and incarnations;
- delivery of every verb over a network that loses, delays and duplicates;
- halo's clock, deadlines, the capture window and tombstone pruning;
- the platform's durable fetch → commit → ack, with platform crashes;
- receipt delivery to the workflow.

| Cordis `operation.qnt` | This model |
|---|---|
| `send` | `pSubmit` (+ `pResend`, `deliver`: accepted, duplicate, stale, late or refused) |
| `worldRuns` | `xRun`; a cancel while running finishes it at its actual usage |
| `capture` (one step) | `pStatus` → `done` (fetched, volatile) → `pCommit` (durable) → `pAck` → `wApply` (deduplicated) |
| `restart` | `restart` (new incarnation and clock) and `pCrash` (the platform loses what it fetched) |
| `evict` with `PROMISE` | `xExpire` (the body goes, the outcome stays until acked); `xEvict` only without the promise |
| `unknown` → settled at R | `unknown`, `forgotten` → R; `expired` → its actual usage; `closed`, `cancelled_before_accept` → 0 |
| `resubmitSame` | `pResend` is always the same identity *within* an incarnation; never across one (fence) |

**Properties.**
- `runsOnce`: no attempt executes twice.
- `accounted`: real spend ≤ settled + reserved.
- `appliedOnce`: the workflow applies each operation's receipt once.
- `reservationsHeld`.
- `neverAcceptedTrue`: no `never_accepted`, `closed` or `cancelled_before_accept` answer is ever given for an
  attempt that ran.
- `noReceiptLost`: no result is released before it was committed.
- `promiseKept`: no body is dropped before its ack or its window.
- Status and cancel are answered against the deadline they carry (`answer(…, nb, …)`).
- `noOvercharge`: settled ≤ real; it holds where nothing becomes unknown.

`run.sh` (`quint run` only), seed 0x5eed, 10 000 samples × 60 steps; `results.txt`:

| Instance | Rule off / world | Property | Result |
|---|---|---|---|
| design | executor restarts, platform crashes | safety | ok (also 100 000 × 80 steps) |
| promised | no executor restart | safety, noOvercharge | ok, ok |
| cacheLike | eviction under load | core / noOvercharge | ok / **violated** |
| noFence | incarnation check | runsOnce | **violated**: a stale submit reruns an attempt settled as unknown |
| noDeadline | not-after | runsOnce | **violated**: a late submit is accepted after its tombstone was pruned |
| pruneEarly | tombstone lifetime | runsOnce | **violated**: a duplicate submit is accepted again |
| dropExpiredOutcome | outcome kept until acked | neverAcceptedTrue | **violated**: an expired attempt that ran is answered `closed` |
| noReserve | reservation at acceptance | promiseKept | **violated**: results finishing together overflow the budget |
| ackBeforeCommit | commit before ack | noReceiptLost | **violated**: a platform crash between ack and commit loses the result |
| cancelUntombed | cancel tombstones | accounted | **violated**: a late copy runs after `cancelled_before_accept` settled it at 0 |
| refusalCloses | a refusal closes only its copy | accounted | **violated**: a refused copy settled at 0, then another copy runs |
| noReceiptDedup | receipt dedup | appliedOnce | **violated** |
| eraseForRoom | records refused at the ceiling | neverAcceptedTrue | **violated**: a tombstone erased for room makes a run attempt read `closed` |

**Witnesses.** Every one is reached:
- done, with and without restarts (`promised`: 216 of 10 000 traces);
- committed after a platform crash;
- done after a lost ack;
- a duplicate receipt;
- done after `never_accepted`;
- closed by deadline;
- cancelled before acceptance;
- a cancelled running attempt committed;
- expired and settled at its usage;
- refused, stale, late, deduplicated, pruned;
- unknown;
- forgotten (without the promise).

**For cordis.**
- A submit lost before it arrived costs 0, not `R`: it ends `closed` or `cancelled_before_accept`.
- An expired result costs its actual usage, not `R`.
- `resubmitSame` remains the right prohibition across incarnations.

## 8. Implementation order (after agreement)

1. **Incarnation, time and contract.** `Halo-Incarnation` and `Halo-Time` on every answer, and
   `GET /v1/halo/contract`. Small; cordis can start recording them at once. **Done.**
   - Every answer the HTTP server writes is stamped: handled, refused at the API key, routing errors, parse errors
     and the connection limit's 503 (`HttpServerOptions::stamp`). The admin socket is stamped too.
   - Until stage 3, the contract's `operations` is `null`: the header contract is the one in force. `inference.mode`
     is advertised already, from `--batch-relaxed`.
   - The clock starts when the server is assembled, after the weights load.
   - Tests: `Http.IncarnationsAreRandomAndTheirClocksNeverRunBackward`,
     `Http.EveryAnswerNamesTheIncarnationAndItsClock`,
     `Http.TheContractNamesTheIncarnationItsClockAndTheInferenceMode`.
   - Mutants `opapi-*`, 5 of 5 killed.
   - The server suite passes under ASan (99 tests).
   - A live `halo_serve` on the small model answered the contract, a 404 and a completion with the same
     incarnation and an increasing `Halo-Time`.
2. **Operation records**: binding digests, the incarnation and deadline checks, tombstones, retention
   reservation, the capture window, and ack release. **Done** (`src/server/operations.{hpp,cpp}`).
   - **A class of its own, `Operations`, not the scheduler's key table.** The record rules (tombstones, digests,
     the window, release on ack) share nothing with key replay, and are kept in one place. An accepted attempt still
     executes as an ordinary request through the scheduler's admission and fairness (stage 3).
   - The caller passes the time (the incarnation's clock). One mutex.
   - Status, cancel and ack are answered against the deadline the verb carries.
   - A record is pruned at the first verb after its not-after, once its outcome (if it ran) was acknowledged.
   - **Tests.** Eight directed tests, one per rule. `OperationsModel.PropertiesHoldUnderTheModelsActions` is a Hegel
     state machine with the model's actors:
     - a platform that keeps the contract's rules, crashing and losing what it fetched;
     - a network that loses, delays and duplicates every verb;
     - the executor, which finishes attempts (cancelled ones with their usage) and restarts under a new incarnation.

     After each action it checks runsOnce, accounted, neverAcceptedTrue, noReceiptLost, promiseKept, the record
     ceiling and the retention budget. 2,000 cases; 14 positive-control situations, each reached in every run.
   - Mutants `opapi-ops-*`, one per executor rule (fence, deadline, prune, expired outcome, reserve, erase for room,
     cancel tombstone, capture window, digest): 9 of 9 killed.
   - The server suite passes under ASan and TSan (108 tests).
   - Tested with a Hegel state machine against `operation.qnt`'s executor actions, as the G3 scheduler is.
   - Each model mutant becomes a `tests/mutants.toml` entry.
   - Measure refusals and reserved versus used bytes on the 8-agent pool.
3. **The endpoints and the stream.** **Done** (`src/server/operation_service.{hpp,cpp}`; `halo_serve --operations 1`,
   the default).
   - **Execution.** An accepted attempt runs on a worker thread through `OpenAiService::RunOperation`: the same
     rendering, scheduler admission, fairness (the owner is the principal) and turn runner as `/v1/completions` and
     `/v1/chat/completions`.
     - It has no idempotency key: an attempt is its own identity and never a replay.
     - It has no connection: it is cancelled only by the cancel verb or the server stopping.
     - A scheduler refusal (principal limit, memory) withdraws the record and answers the submit 429/503 with
       Retry-After, as a refusal of that copy.
     - A cancelled attempt keeps its final object and its actual usage.
   - **Digest.** The SHA-256 (hex) of the submit's body bytes, which carry api, request, not_after and mode. A
     submit may send it as `Halo-Digest` and must then match. Every other verb carries it with `Halo-Incarnation`
     and `Halo-Not-After`.
   - **Answers.** 202 accepted; 200 a duplicate's status; 409 stale incarnation (with the current one) or conflict;
     410 late; 400 too far ahead; 422 a mode the server does not run; 503 `records`/`retention`. Status for another
     incarnation is 200 `unknown`. Fetch of an attempt not done is 409 with its status. Observe is 410 with the
     status once the body is gone.
   - **Observation.** The stream's events (the chunks a streaming request would get) are kept with the held body.
     `?from=N` replays from event N, then follows to the end, then sends an `event: status` frame. An observer
     leaving cancels nothing.
   - **APIs** (`"api"` in the submit; the contract lists them). Each runs through an adapter
     (`server/operation_api.hpp`) that the API's service registers:
     - `completions`, `chat` and `responses` (OpenAiService);
     - `messages` (AnthropicService);
     - `systemone` (the decider, when one is loaded).

     The result is the API's own final object, and the observation stream is in that API's frame format (Responses
     and Messages events carry their `event:` names).
     - A `responses` operation may continue from a stored response with `previous_response_id`. It is not stored
       itself: its result is the operation's.
     - A `systemone` operation is one evaluation, a single unit of GPU work. A cancel that arrives while it runs
       takes effect after it, and the attempt finishes `done`. Its usage is the answer's `input_tokens` as
       `prompt_tokens`, and the evaluation's length as `engine_ms`.
     - `halo_serve` now stops the operation service (cancelling and joining its workers) before it releases the
       generator and the decider. Stage 3 released the generator first, which could pull it out from under a
       running operation at shutdown.
   - **`engine_ms`** is the request's share of every unit of GPU work it took part in: a unit's measured length
     divided by its participants. A prefill slice is one request's; a batched decode unit is split. The unit a
     request finishes in counts up to its finish.
     - It is published after every unit, so a reader that leaves early (a cancel) carries the time spent so far.
     - A replay reports 0.
     - Tests: `ScheduledGenerator.EngineTimeIsEachRequestsShareOfItsUnits`, which checks that two batched requests
       are charged exactly the units' time, split between them, and
       `ScheduledGenerator.ACancelledReaderCarriesTheEngineTimeSpent`.
     - Mutants `opapi-engine-*`: 3 of 3 killed.
     - Live (small model): 32 tokens, 86 ms; an attempt cancelled after 0.5 s, 284 tokens and 502 ms.
   - **Tests.** `OperationService.*` (7) over HTTP with a scripted generator:
     - a run, fetch and ack, with a duplicate that runs nothing;
     - incarnation, digest, deadline and mode binding;
     - a scheduler refusal that leaves no record;
     - the record ceiling;
     - a cancel before acceptance (the late copy is a no-op) and while running (finish `cancelled`, usage kept);
     - an observer that leaves (the attempt finishes), and replay from an offset;
     - the contract section.
   - Mutants `opapi-svc-*`: 6 of 6 killed. `svc-no-liveness-check` was retargeted to the shared departure check.
     The server suite passes under ASan and TSan (115 tests).
   - **Live** (`halo_serve`, small model, under the GPU lock): a submit answered 202; its duplicate 200; done with
     its usage; fetch, the stream replay and ack worked; a stale incarnation answered `unknown`. A 3,500-token
     attempt cancelled after 0.5 s finished `cancelled` at 285 tokens.
   - **Live fault probe** (`tools/op_fault_probe.py`): a minimal outbox that keeps the contract's rules against a live
     `halo_serve`. It injects lost, duplicated and late submits, cancels before acceptance and while running, lost
     acks, a platform that forgets a fetch, expiry past a 3 s capture window and server restarts, and checks every
     answer against the contract.
     - Main c4a39ff, small model, seeds 1-3, 16 operations each: 48 of 48 committed, 81 attempts, 8 incarnations,
       0 failures.
     - Every fault above was reached, with `closed` by deadline and `unknown` after each restart.
     - With `--records 3` (seeds 4-5): 11,679 refused copies (503 `records`) and 26 late copies answered 410 after
       `closed`, with 0 failures.
       - 28 of 32 operations committed. The other 4 used up their 6 attempts while tombstones held the records until
         their not-after passed.
       - A full ceiling becomes refusals that clear as deadlines pass, never lost state. A record is held for up to
         `max_submit_age_s`, so the ceiling needed is about the submit rate times that window.
       - The probe resends refused copies at once; an outbox should wait for Retry-After.
     - Positive control: the same probe against a server built with `opapi-ops-cancel-untombed` failed ("a copy
       was accepted after cancelled_before_accept").
     - This is halo's side of the RFC 0007 milestone; the milestone itself runs cordis's adapter against it.
   - **RFC 0007 milestone on the real model** (2026-10-12). Cordis ea32902 (engine, outbox adapter, journal) against
     `halo_serve` at main df4b6e8, Qwen3.8-Flash-Next UD-Q4_K_XL, `milestone.sh` with `HALO_URL`. Evidence is in the
     stream directory MS.
     - **As shipped it fails, for two reasons on cordis's side.** Halo's answers were correct in both.
       - Its counts are the stand-in's. The stand-in fails the critic twice; the real model answered `LGTM` at
         once (86 tokens), so the run had 3 engine crashes and 1 critic computation, not at least 4 and exactly 3.
       - Its operation ids repeat across fresh journals. A second run reused (owner, op, attempt 0) with another
         not-after, so another digest, and halo answered 409 `conflict` while the first run's record was held.
         RFC 0007 says ids are never reused.
     - **With only those two counts relaxed, and a fresh halo per run, 3 of 3 runs passed.** Each shipped after 3
       engine crashes with one done attempt for its one computation (settled 87), nothing unacknowledged at halo, no
       acks owed, no deliveries pending and the audit clean.
     - **The fault probe on the real model** (seed 3): 16 of 16 operations committed over 3 incarnations, 0
       failures. It reached cancels before acceptance and while running, expiry, lost acks, duplicates, `closed`
       and `unknown` after restarts.
4. **Cordis's adapter** moves to the verbs (commit before ack, receipt dedup), and the header rules
   (`observe_report.qnt`, `observe_forget.qnt`) retire for it.
