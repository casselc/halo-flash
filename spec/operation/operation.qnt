// -*- mode: Bluespec; -*-
// operation.qnt: halo-flash's half of the operation contract (docs/OPERATIONS.md; cordis RFC 0007,
// whose operation.qnt is the platform's ledger over an executor that either holds an attempt or does not).
//
// This model adds what the executor side needs for its answers to be safe to act on: submits travel over a
// network that loses, delays and duplicates them, and the executor is ephemeral (a restart drops everything it
// holds). The platform names each attempt (operation, attempt number) and the executor incarnation it believes
// is current; the executor answers a status query with one of
//   running | done (cost) | unknown (another incarnation) | forgotten (evicted before capture) | never accepted.
// The rules that make "never accepted" true and the accounting hold:
//  - fence: a submit names the incarnation it was sent to and is rejected by any other one, so a late copy
//    can never start an attempt in an incarnation the platform did not send it to;
//  - tomb: within an incarnation, an attempt id once accepted, captured, evicted or abandoned is remembered, so
//    a duplicate or late copy of its submit is a no-op;
//  - promise: a finished result is held until the platform captures it; when that would need more than HOLD
//    attempts held, the executor refuses the submit (a 503 with Retry-After) instead of evicting;
//  - abandon: the platform may give up an attempt the executor says it never accepted only by a cancel that
//    tombstones the id, and then settles it at 0.
//
// Properties: runsOnce (no attempt executes twice), accounted (real spend <= settled + reserved),
// appliedOnce, reservationsHeld, neverAcceptedTrue (a "never accepted" answer is never given for an attempt
// that ran), and noOvercharge (settled <= real; only where nothing becomes unknown: a promise, no restart).
module operation {

  type Rules = {
    fence: bool,     // a submit carrying another incarnation is rejected (false: accepted by whoever gets it)
    tomb: bool,      // ids are remembered for the incarnation's life (false: forgotten when captured or evicted)
    promise: bool,   // finished results are held until captured; refuse at HOLD instead of evicting
    abandonTomb: bool, // abandoning a never-accepted attempt tombstones it (false: the platform just settles it at 0)
  }
  const RULES: Rules
  const RESTART: bool  // the executor may restart

  pure val N = 2      // operations, in order
  pure val MAXA = 3   // attempts per operation
  pure val R = 2      // an attempt's worst-case cost: its reservation
  pure val HOLD = 1   // attempts the executor can hold at once
  pure val IDS = 0.to(N * MAXA - 1)

  type PSt = Unused | Pending | Settled | Lost | Abandoned
  type PRec = { op: int, inc: int, st: PSt }                 // the platform's ledger row for an attempt
  type XRec = { held: bool, ran: bool, cost: int, tomb: bool } // the executor's record (this incarnation)
  type Msg = { id: int, inc: int }                           // a submit in flight

  type S = {
    done: int, tries: List[int], pinc: int,        // platform: operations applied, attempts per operation, believed incarnation
    recs: int -> PRec,
    reserved: int, settled: int, applied: List[int],
    inc: int, ex: int -> XRec,                     // executor
    net: Set[Msg],
    // ghosts
    real: int, runs: int -> int, lied: bool,
    restarts: int, neverAnswers: int, deduped: int, stale: int, refused: int, abandoned: int, unknowns: int,
    forgotten: int, evicted: int,
  }
  var s: S

  pure val X0: XRec = { held: false, ran: false, cost: 0, tomb: false }

  action init = s' = {
    done: 0, tries: [0, 0], pinc: 0,
    recs: IDS.mapBy(_ => { op: 0, inc: 0, st: Unused }),
    reserved: 0, settled: 0, applied: [0, 0],
    inc: 0, ex: IDS.mapBy(_ => X0),
    net: Set(),
    real: 0, runs: IDS.mapBy(_ => 0), lied: false,
    restarts: 0, neverAnswers: 0, deduped: 0, stale: 0, refused: 0, abandoned: 0, unknowns: 0,
    forgotten: 0, evicted: 0,
  }

  pure def pendingFor(recs: int -> PRec, op: int): bool = IDS.exists(i => recs.get(i).op == op and recs.get(i).st == Pending)
  // release an attempt's result: with tombstones its id stays known; without, the executor forgets it entirely
  pure def forget(x: XRec, tomb: bool): XRec = if (tomb) { ...x, held: false, tomb: true } else X0
  pure def heldCount(ex: int -> XRec): int = IDS.filter(i => ex.get(i).held).size()

  // ---- platform ---------------------------------------------------------------------------------------------
  // a new attempt of an operation, under a reservation of R, sent to the incarnation it believes in.
  // Operations are independent: both may be in flight at once
  action pSubmit: bool = {
    nondet op = 0.to(N - 1).oneOf()
    val id = op * MAXA + s.tries[op]
    all {
      s.tries[op] < MAXA, s.applied[op] == 0, not(pendingFor(s.recs, op)),
      s' = { ...s,
        recs: s.recs.set(id, { op: op, inc: s.pinc, st: Pending }),
        tries: s.tries.replaceAt(op, s.tries[op] + 1), reserved: s.reserved + R,
        net: s.net.union(Set({ id: id, inc: s.pinc })) },
    }
  }

  // the same attempt sent again (a timeout, a lost answer): always under the incarnation it was first sent to
  action pResend: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { r.st == Pending, s' = { ...s, net: s.net.union(Set({ id: id, inc: r.inc })) } }
  }

  // the platform learns the current incarnation (every answer carries it)
  action pLearn: bool = all { s.pinc != s.inc, s' = { ...s, pinc: s.inc } }

  // a status query for a pending attempt, answered from the executor's current state
  action pStatus: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    val x = s.ex.get(id)
    all {
      r.st == Pending,
      r.inc != s.inc or not(x.held and not(x.ran)),  // a running attempt: nothing to do
      if (r.inc != s.inc) {
        // unknown: the attempt may have run in the incarnation it was sent to; settle the worst case
        s' = { ...s, recs: s.recs.set(id, { ...r, st: Lost }), reserved: s.reserved - R, settled: s.settled + R,
               pinc: s.inc, unknowns: s.unknowns + 1 }
      } else if (x.held and x.ran) {
        // done: captured at its cost, receipt applied, the result released (acknowledged)
        s' = { ...s, recs: s.recs.set(id, { ...r, st: Settled }), reserved: s.reserved - R,
               settled: s.settled + x.cost, applied: s.applied.replaceAt(r.op, s.applied[r.op] + 1),
               done: s.done + 1,
               ex: s.ex.set(id, forget(x, RULES.tomb)) }
      } else if (x.tomb) {
        // forgotten: it was accepted here and evicted before capture; it may have run
        s' = { ...s, recs: s.recs.set(id, { ...r, st: Lost }), reserved: s.reserved - R, settled: s.settled + R,
               forgotten: s.forgotten + 1 }
      } else {
        // never accepted (so far): keep it pending; the platform may resend it or abandon it
        s' = { ...s, neverAnswers: s.neverAnswers + 1, lied: s.lied or s.runs.get(id) > 0 }
      }
    }
  }

  // give up an attempt the executor has never accepted: cancel it (a tombstone), then settle it at 0
  action pAbandon: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    val x = s.ex.get(id)
    all {
      r.st == Pending, r.inc == s.inc, not(x.held), not(x.tomb),
      s' = { ...s, recs: s.recs.set(id, { ...r, st: Abandoned }), reserved: s.reserved - R,
             ex: s.ex.set(id, { ...x, tomb: RULES.abandonTomb }), abandoned: s.abandoned + 1 },
    }
  }

  // ---- network and executor --------------------------------------------------------------------------------
  // a submit copy arrives; it may stay in flight as a duplicate
  action deliver: bool = {
    nondet m = s.net.oneOf()
    nondet keep = Set(true, false).oneOf()
    val x = s.ex.get(m.id)
    val net2 = if (keep) s.net else s.net.exclude(Set(m))
    all {
      s.net.size() > 0,
      if (RULES.fence and m.inc != s.inc) s' = { ...s, net: net2, stale: s.stale + 1 }
      else if (x.held or x.tomb) s' = { ...s, net: net2, deduped: s.deduped + 1 }
      else if (RULES.promise and heldCount(s.ex) >= HOLD) s' = { ...s, net: net2, refused: s.refused + 1 }
      else s' = { ...s, net: net2, ex: s.ex.set(m.id, { ...x, held: true }) },
    }
  }

  // a held attempt runs to its end, spending 1..R
  action xRun: bool = {
    nondet id = IDS.oneOf()
    nondet c = 1.to(R).oneOf()
    val x = s.ex.get(id)
    all {
      x.held, not(x.ran),
      s' = { ...s, ex: s.ex.set(id, { ...x, ran: true, cost: c }), real: s.real + c,
             runs: s.runs.set(id, s.runs.get(id) + 1) },
    }
  }

  // without a promise, a finished result not yet captured may be evicted under load
  action evict: bool = {
    nondet id = IDS.oneOf()
    val x = s.ex.get(id)
    all {
      not(RULES.promise), x.held, x.ran,
      s' = { ...s, ex: s.ex.set(id, forget(x, RULES.tomb)), evicted: s.evicted + 1 },
    }
  }

  // a restart: a new incarnation holding nothing, remembering nothing
  action restart: bool = all {
    RESTART, s.restarts < 2,
    s' = { ...s, inc: s.inc + 1, ex: IDS.mapBy(_ => X0), restarts: s.restarts + 1 },
  }

  action step = any { pSubmit, pResend, pLearn, pStatus, pAbandon, deliver, xRun, evict, restart }

  // ---- properties ------------------------------------------------------------------------------------------
  val runsOnce: bool = IDS.forall(i => s.runs.get(i) <= 1)
  val accounted: bool = s.real <= s.settled + s.reserved
  val appliedOnce: bool = s.applied.foldl(true, (ok, n) => ok and n <= 1)
  val reservationsHeld: bool = s.reserved == R * IDS.filter(i => s.recs.get(i).st == Pending).size()
  val neverAcceptedTrue: bool = not(s.lied)
  val noOvercharge: bool = s.settled <= s.real
  val safety: bool = runsOnce and accounted and appliedOnce and reservationsHeld and neverAcceptedTrue

  // witnesses (passed to --witnesses; each must be reached)
  val wDone: bool = s.done == N
  val wRestartedThenDone: bool = s.restarts > 0 and s.done == N
  val wNeverThenDone: bool = s.neverAnswers > 0 and s.done == N
  val wDeduped: bool = s.deduped > 0
  val wStaleRejected: bool = s.stale > 0
  val wRefused: bool = s.refused > 0
  val wAbandonedThenDone: bool = s.abandoned > 0 and s.done == N
  val wUnknown: bool = s.unknowns > 0
  val wForgotten: bool = s.forgotten > 0
}
