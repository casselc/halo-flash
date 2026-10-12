# The operation API (proposal, 2026-10-12)

Status: **proposed**. This is halo-flash's half of cordis RFC 0007 ("Boundaries, and the operation contract"). No
code changes until the contract is agreed. The model is `spec/operation/`.

## 1. Why

Cordis has been reconstructing an inference operation's history from a connection-oriented interface whose
server remembers for a bounded time: `Halo-Answer`, `Halo-Discarded-Tokens`, `Halo-Admitted-Sends`, keyed grace.
Each was correct, and together they are the symptom RFC 0007 names. Three facts about today's server:

- **An unknown key starts a new generation.** After a restart, or once the scheduler has forgotten a key
  (`replay_ttl_s` 600 s, or past `max_keys` 4096), a request with that key generates again. `Admitted-Sends`
  made this visible to the caller; it did not prevent it.
- **Retention is a cache, not a promise.** "600 s unless 4096 newer keys finished" evicts under load.
- **"Never accepted" and "I no longer know" are not distinguished.** Nothing about a forgotten key is kept.

## 2. Identity

- **Operation id and attempt** are issued by the platform. An attempt is the unit of execution and accounting.
  A new execution is a new attempt that the platform authorized; halo never starts one on its own.
- **Incarnation**: 128 random bits chosen at process start, returned on every answer (`Halo-Incarnation`).
  - It is per process, not a durable epoch (RFC 0007 open question 2). Halo persists nothing an operation depends
    on, so a durable epoch would distinguish nothing a fresh token does not.
  - A submit names the incarnation it is sent to, and **any other incarnation rejects it** (409, with the current
    incarnation). A late or duplicated submit therefore cannot start an attempt in a process the platform did not
    send it to.
- **Tombstones.** Within an incarnation, every attempt id that was accepted, released or abandoned is remembered
  until its submit's deadline has passed. A duplicate or late copy of the submit is then a no-op that returns the
  current status.
  - Every submit carries `not_after`, at most `max_submit_age_s` ahead (proposed 600 s). Halo rejects a submit
    that arrives after its `not_after` (410). That bounds tombstone memory to the ids seen in one window: about
    50 bytes each, so 100 k ids is 5 MB.
  - It assumes the platform and halo clocks agree to within a stated skew. Today they share a host.

## 3. Verbs

| Verb | Request | Answers |
|---|---|---|
| contract | `GET /v1/halo/contract` | incarnation, retention, deduplication, cancellation, cost reporting (section 4) |
| submit | `POST /v1/operations` `{operation_id, attempt, incarnation, not_after, api, request, on_detach}` | 202 `accepted`; or the current status for a duplicate; 409 stale incarnation; 410 past `not_after`; 429 principal limit / 503 retention or memory full, both with Retry-After and both **definitely not accepted** |
| status | `GET /v1/operations/{op}/{attempt}` | `running` (tokens so far) / `done` (finish reason, usage, result) / `unknown` (another incarnation) / `forgotten` (accepted here, released before capture; only possible if retention is not promised) / `never_accepted` |
| observe | `GET /v1/operations/{op}/{attempt}/stream?from=<event>` | SSE from an event offset. Leaving is an observer departing. Partial tokens are observations, never receipts. |
| cancel | `POST /v1/operations/{op}/{attempt}/cancel` | never accepted: tombstoned, `cancelled_before_accept` (safe to settle at 0); running: stops at the next unit boundary and finishes `done` with finish `cancelled` and its usage; done: unchanged |
| ack | `POST /v1/operations/{op}/{attempt}/ack` | the platform has captured the result: halo releases it and the tombstone stays |

- `api` selects completions, chat, responses, messages or systemone, and `request` is that API's body.
- `on_detach` is `continue` (the default) or `cancel`. It replaces keyed grace for operations: a disconnect never
  decides execution unless the submit asked it to.
- Usage is `{prompt, cached, completion, discarded}`. `discarded` is decoded work that no result carries (a
  cancel, for example). There is no per-key history to reconstruct: an attempt is executed at most once, so its
  usage is its own.
- Status and result are pulled by id (RFC 0007, "Streams and the ledger"). Completion reaches the ledger through a
  path the platform can repeat, never a one-shot notification.

**Rules the platform needs** (the model checks them; section 6):

- `never_accepted` lets the platform resend the same identity, or abandon it with a cancel and settle it at 0. It
  must not abandon it without the cancel: a late copy could still arrive and run.
- `unknown` and `forgotten` are settled at the worst case. Executing again needs a new attempt.
- Resending a submit is always under the incarnation the attempt was first sent to. A submit is never re-aimed
  at a new incarnation.

## 4. Retention, and what halo advertises

**Promise.** A finished result is held until it is acknowledged. A backstop releases it after `max_hold_s`
(proposed 24 h) and is part of the advertised promise, not an eviction. When holding another result would need
more than `max_held` results, or more memory than the retention budget, halo refuses new submits (503,
Retry-After) instead of releasing an unacknowledged result.

RFC 0007 open question 3 asks what such refusals cost. They are a measurement for the implementation stream:

- a held result is its text plus its usage, a few KB, so a 64 MB budget holds about 10 k;
- refusals should then only happen when the platform stops acknowledging, which is a fault it can see.

`GET /v1/halo/contract` returns, per operation type:

```json
{"incarnation": "9f…", "retention": {"until": "ack", "max_hold_s": 86400, "max_held": 10000},
 "dedup": {"scope": "incarnation", "max_submit_age_s": 600},
 "cancel": "next unit boundary (<= one decode unit or one prefill slice)",
 "cost": "tokens: prompt, cached, completion, discarded",
 "repeat": "no external effects; a new attempt spends again; identical text under greedy decoding on the same build, otherwise may differ"}
```

A server started with `--operations off` advertises no operation contract and keeps today's header contract.

## 5. What stays, and how it maps

- The OpenAI, Anthropic and systemone endpoints are unchanged, along with `Idempotency-Key`, the `Halo-*` headers
  and keyed grace. They remain the contract for ordinary clients, and for cordis until its adapter moves (RFC 0007,
  order of work step 3).
- **Internally, an operation is a scheduler key** (`op/attempt`), so admission, fairness, fencing, replay and the
  reservation ledger are reused unchanged. What is new is in the scheduler's key table:
  - acknowledgement-based release, replacing the TTL and LRU for operation keys;
  - tombstones and the incarnation check;
  - submit-time refusal when retention is full;
  - `on_detach` in place of grace.

## 6. The model (`spec/operation/`)

Cordis's `operation.qnt` is the platform's ledger over an executor that either holds an attempt or does not; a
submit is accepted the moment it is sent. This model adds the executor and the network between them, which is
what makes `never_accepted` safe to act on:

- submits are lost, delayed and duplicated;
- the executor restarts (a new incarnation that holds nothing);
- the executor holds at most `HOLD` attempts.

**The two actors.** The platform submits, resends, learns the incarnation, asks status and abandons. The executor
accepts, runs, evicts (only without a promise) and restarts.

**Properties.**

- `runsOnce`: no attempt executes twice.
- `accounted`: real spend ≤ settled + reserved.
- `appliedOnce`.
- `reservationsHeld`.
- `neverAcceptedTrue`: `never_accepted` is never answered for an attempt that ran.
- `noOvercharge`: settled ≤ real. It holds where nothing becomes unknown.

`run.sh` (`quint run` only), with seed 0x5eed, 10 000 samples and 40 steps:

| Instance | What it is | Property | Result |
|---|---|---|---|
| design | restarts, promise, every rule | safety | ok (also 100 000 × 80 steps) |
| promised | no restart, promise | safety, noOvercharge | ok, ok |
| cacheLike | eviction under load | safety / noOvercharge | ok / **violated** (a forgotten result costs the worst case) |
| noFence | a stale-incarnation submit is accepted | runsOnce, accounted | **violated**: a late copy reruns an attempt already settled as unknown |
| noTomb | ids forgotten once released | runsOnce | **violated**: a duplicate submit reruns a captured attempt |
| abandonUntombed | abandon without the cancel | accounted | **violated**: the late copy runs after it was settled at 0 |
| cacheNoTomb | eviction forgets the id outright | neverAcceptedTrue | **violated**: "I no longer know" answered as "never accepted" |

Every witness is reached:

- done, with and without restarts;
- done after a `never_accepted` answer;
- a duplicate deduplicated, and a stale submit rejected;
- a refusal under the promise;
- an abandoned attempt;
- an unknown, and a forgotten one (without a promise).

**For cordis.** Their model settles every unheld attempt at the worst case, `R`. With these rules, a submit lost
before it arrived costs 0 instead of `R`: answered `never_accepted`, then resent or abandoned. Their
`resubmitSame` mutant remains the right prohibition for `unknown` and `forgotten`.

## 7. Implementation order (after agreement)

1. **Incarnation and contract.** `Halo-Incarnation` on every answer and `GET /v1/halo/contract`. Small; cordis can
   start recording the incarnation at once.
2. **Operation keys in the scheduler.** Acknowledgement-based release, tombstones with `not_after`, the
   incarnation check, refusal when retention is full, and `on_detach`.
   - Tested with a Hegel state machine against `operation.qnt`'s actions, as the G3 scheduler is.
   - Each model mutant becomes a `tests/mutants.toml` entry.
   - Measure: refusals and held bytes under the 8-agent pool.
3. **The endpoints and the stream**, with HTTP tests for every status answer and fault: a lost submit, a
   duplicate, a restart between submit and status, a cancel before acceptance.
4. **Cordis's adapter** moves to the verbs, and the header rules (`observe_report.qnt`, `observe_forget.qnt`)
   retire for it.
