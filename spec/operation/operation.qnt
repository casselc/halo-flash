// -*- mode: Bluespec; -*-
// operation.qnt: halo-flash's half of the operation contract (docs/OPERATIONS.md; cordis RFC 0007, whose
// operation.qnt is the platform's accounting over an executor that either holds an attempt or does not).
//
// Scope: executor identity, delivery of every verb over a network that loses, delays and duplicates messages, the
// executor's clock and deadlines, and the platform's durable capture of a result. The platform is the ledger: its
// rows (`recs`), settlement and receipt outbox are durable; what it has fetched and not yet committed is not
// (`pCrash`). The workflow (cordis) applies receipts, which the outbox may deliver more than once.
//
// The executor (halo) is ephemeral: a restart is a new incarnation that holds and remembers nothing. Every verb
// carries the attempt's identity (operation, attempt: unique per platform) and the incarnation it is meant for;
// another incarnation answers `unknown` and acts on nothing. Time is the executor's monotonic clock; every answer
// carries it, and a deadline (`nb`, a submit's not-after) is in that clock, so no two clocks are ever compared.
//
// Rules (RULES; each has a mutant in operation_modes.qnt):
//  fence         a verb meant for another incarnation is rejected
//  deadline      a submit past its not-after is rejected; an id never accepted and past its not-after is `closed`
//  tomb          an attempt id the executor has seen stays known until its not-after has passed and, if it ran,
//                its outcome was acknowledged: a duplicate or late submit is a no-op
//  ackOutcome    an expired result keeps its outcome (usage) until acknowledged, though its body is gone
//  reserve       a submit is accepted only with a retention slot for its eventual result
//  promise       a finished result is never dropped before its ack or the end of its capture window
//  commitThenAck the platform acknowledges only what it has durably committed
//  cancelTomb    a cancel of an attempt the executor never accepted tombstones it
//  refusalOpen   a refused copy (429/503) does not end the attempt: the reservation stays until a terminal answer
//  dedupReceipt  the workflow applies a receipt once per attempt
//  metaBound     records (held attempts, tombstones, unacknowledged outcomes) are capped at META: a submit or a
//                cancel that would need one more is refused; nothing is erased to make room
//
// Every verb carries the attempt's not-after (and, in the API, its digest): a status for an attempt whose submit
// never arrived is answered against the carried deadline (cordis expiry.qnt's guessDeadline).
module operation {
  type Rules = {
    fence: bool, deadline: bool, tomb: bool, ackOutcome: bool, reserve: bool, promise: bool,
    commitThenAck: bool, cancelTomb: bool, refusalOpen: bool, dedupReceipt: bool, metaBound: bool,
  }
  const RULES: Rules
  const RESTART: bool     // the executor may restart
  const PCRASH: bool      // the platform may crash (losing what it fetched and has not committed)

  pure val N = 2          // operations
  pure val MAXA = 3       // attempts per operation
  pure val R = 2          // an attempt's worst-case cost: its reservation
  pure val HOLD = 1       // retention slots
  pure val META = 2       // records: held attempts and tombstones
  pure val W = 3          // a submit's not-after: W ticks past the platform's latest reading of the executor's clock
  pure val HOLD_T = 4     // a finished result's body is held at most this long (the advertised capture window)
  pure val MAXT = 14      // ticks per run
  pure val IDS = 0.to(N * MAXA - 1)

  type PSt = Unused | Pending | Committed | Lost | Closed
  type PRec = { op: int, inc: int, nb: int, st: PSt, refusedSeen: bool, cancelSeen: bool }
  type Tomb = TNone | TCancelled | TReleased | TExpired | TEvicted
  // held: accepted, holding a retention slot (running, or finished with its body); tomb: what remains after
  type XRec = { held: bool, ran: bool, cost: int, doneAt: int, tomb: Tomb, acked: bool, nb: int }
  type Kind = Submit | Cancel | Ack
  type Msg = { kind: Kind, id: int, inc: int, nb: int }
  type Answer = AUnknown | ARunning | ADone | AReleased | AExpired | AForgotten | ACancelled | AClosed | ANever

  type S = {
    time: int,
    // platform (durable: recs, reserved, settled, outbox; volatile: fetched, pinc)
    recs: int -> PRec, tries: List[int], pinc: int, fetched: Set[(int, int)],
    reserved: int, settled: int, outbox: Set[int],
    // workflow
    appliedIds: Set[int], applied: List[int],
    // executor
    inc: int, ex: int -> XRec,
    net: Set[Msg],
    // ghosts
    real: int, runs: int -> int, lied: bool, receiptLost: bool, promiseBroken: bool,
    restarts: int, pcrashes: int, nevers: int, deduped: int, stale: int, late: int, refused: int,
    closedByDeadline: int, cancelledBefore: int, cancelledRunning: int, expiredSettled: int, unknowns: int,
    forgotten: int, pruned: int, ackDropped: int, dupReceipts: int, recoveredAfterCrash: int,
  }
  var s: S

  pure val X0: XRec = { held: false, ran: false, cost: 0, doneAt: 0, tomb: TNone, acked: false, nb: 0 }

  action init = s' = {
    time: 0,
    recs: IDS.mapBy(_ => { op: 0, inc: 0, nb: 0, st: Unused, refusedSeen: false, cancelSeen: false }), tries: [0, 0], pinc: 0,
    fetched: Set(), reserved: 0, settled: 0, outbox: Set(),
    appliedIds: Set(), applied: [0, 0],
    inc: 0, ex: IDS.mapBy(_ => X0),
    net: Set(),
    real: 0, runs: IDS.mapBy(_ => 0), lied: false, receiptLost: false, promiseBroken: false,
    restarts: 0, pcrashes: 0, nevers: 0, deduped: 0, stale: 0, late: 0, refused: 0,
    closedByDeadline: 0, cancelledBefore: 0, cancelledRunning: 0, expiredSettled: 0, unknowns: 0,
    forgotten: 0, pruned: 0, ackDropped: 0, dupReceipts: 0, recoveredAfterCrash: 0,
  }

  pure def liveFor(recs: int -> PRec, op: int): bool =
    IDS.exists(i => recs.get(i).op == op and (recs.get(i).st == Pending or recs.get(i).st == Committed))
  // retention slots in use: with `reserve`, every accepted attempt holds one from acceptance; without, only results
  pure def slots(ex: int -> XRec, reserve: bool): int =
    IDS.filter(i => ex.get(i).held and (reserve or ex.get(i).ran)).size()

  pure def records(ex: int -> XRec): int = IDS.filter(i => ex.get(i).held or ex.get(i).tomb != TNone).size()

  // the metaBound mutant makes room by erasing a record (any tombstone) instead of refusing
  pure def eraseOne(ex: int -> XRec, victim: int): int -> XRec =
    if (not(ex.get(victim).held) and ex.get(victim).tomb != TNone) ex.set(victim, X0) else ex

  // the executor's answer to a status query for `id` meant for incarnation `qinc`
  pure def answer(st: S, id: int, qinc: int, nb: int, deadline: bool): Answer = {
    val x = st.ex.get(id)
    if (qinc != st.inc) AUnknown
    else if (x.held and not(x.ran)) ARunning
    else if (x.held) ADone
    else if (x.tomb == TReleased) AReleased
    else if (x.tomb == TExpired) AExpired
    else if (x.tomb == TEvicted) AForgotten
    else if (x.tomb == TCancelled) ACancelled
    else if (deadline and st.time > nb) AClosed
    else ANever
  }

  // settle a pending attempt at `cost` with no receipt (the operation may be attempted again, as a new attempt)
  pure def lose(st: S, id: int, end: PSt, cost: int): S =
    { ...st, recs: st.recs.set(id, { ...st.recs.get(id), st: end }), reserved: st.reserved - R, settled: st.settled + cost }

  // ---- platform -------------------------------------------------------------------------------------------
  // a new attempt of an operation with no live attempt and no committed result, under a reservation of R
  action pSubmit: bool = {
    nondet op = 0.to(N - 1).oneOf()
    val id = op * MAXA + s.tries[op]
    val nb = s.time + W
    all {
      s.tries[op] < MAXA, not(liveFor(s.recs, op)),
      s' = { ...s,
        recs: s.recs.set(id, { op: op, inc: s.pinc, nb: nb, st: Pending, refusedSeen: false, cancelSeen: false }),
        tries: s.tries.replaceAt(op, s.tries[op] + 1), reserved: s.reserved + R,
        net: s.net.union(Set({ kind: Submit, id: id, inc: s.pinc, nb: nb })) },
    }
  }

  // resend a pending attempt's submit exactly as first sent (a timeout, a lost answer, a refusal)
  action pResend: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { r.st == Pending, s' = { ...s, net: s.net.union(Set({ kind: Submit, id: id, inc: r.inc, nb: r.nb })) } }
  }

  action pCancel: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { r.st == Pending, s' = { ...s, net: s.net.union(Set({ kind: Cancel, id: id, inc: r.inc, nb: r.nb })) } }
  }

  // every answer carries the incarnation
  action pLearn: bool = all { s.pinc != s.inc, s' = { ...s, pinc: s.inc } }

  // a status query for a pending attempt. A `done` answer is only fetched (volatile) here; pCommit makes it durable
  action pStatus: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    val a = answer(s, id, r.inc, r.nb, RULES.deadline)
    val x = s.ex.get(id)
    val ack = Set({ kind: Ack, id: id, inc: r.inc, nb: r.nb })
    val lie = s.lied or s.runs.get(id) > 0
    all {
      r.st == Pending, a != ARunning,
      match a {
        | AUnknown => s' = { ...lose(s, id, Lost, R), pinc: s.inc, unknowns: s.unknowns + 1 }
        // the commitThenAck mutant acknowledges what it has only fetched
        | ADone => s' = { ...s, fetched: s.fetched.union(Set((id, x.cost))),
                          net: if (RULES.commitThenAck) s.net else s.net.union(ack) }
        // released although the platform never committed it: the result is gone (only without commitThenAck)
        | AReleased => s' = { ...lose(s, id, Lost, x.cost), receiptLost: true }
        // the capture window passed: settled at its actual usage, acknowledged so the outcome can be pruned
        | AExpired => s' = { ...lose(s, id, Lost, x.cost), expiredSettled: s.expiredSettled + 1, net: s.net.union(ack) }
        | AForgotten => s' = { ...lose(s, id, Lost, R), forgotten: s.forgotten + 1 }
        | ACancelled => s' = { ...lose(s, id, Closed, 0), cancelledBefore: s.cancelledBefore + 1, lied: lie }
        | AClosed => s' = { ...lose(s, id, Closed, 0), closedByDeadline: s.closedByDeadline + 1, lied: lie }
        | ANever => s' = { ...s, nevers: s.nevers + 1, lied: lie }
        | ARunning => s' = s
      },
    }
  }

  // the refusalOpen mutant: a refused copy is taken as the attempt's end and settled at 0
  action pTakeRefusal: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { not(RULES.refusalOpen), r.st == Pending, r.refusedSeen, s' = lose(s, id, Closed, 0) }
  }

  // the answer to a cancel, `cancelled_before_accept`, closes the attempt at 0 without waiting for its not-after
  action pTakeCancel: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { r.st == Pending, r.cancelSeen, s' = { ...lose(s, id, Closed, 0), cancelledBefore: s.cancelledBefore + 1 } }
  }

  // durably commit a fetched result: the row settles at its cost and its receipt enters the outbox
  action pCommit: bool = {
    nondet f = s.fetched.oneOf()
    val id = f._1
    val r = s.recs.get(id)
    all {
      s.fetched.size() > 0,
      if (r.st == Pending)
        s' = { ...s, recs: s.recs.set(id, { ...r, st: Committed }), reserved: s.reserved - R,
               settled: s.settled + f._2, outbox: s.outbox.union(Set(id)), fetched: s.fetched.exclude(Set(f)),
               recoveredAfterCrash: if (s.pcrashes > 0) s.recoveredAfterCrash + 1 else s.recoveredAfterCrash }
      else s' = { ...s, fetched: s.fetched.exclude(Set(f)) },
    }
  }

  // acknowledge a committed result, as often as the platform likes (acks are lost too)
  action pAck: bool = {
    nondet id = IDS.oneOf()
    val r = s.recs.get(id)
    all { r.st == Committed, s' = { ...s, net: s.net.union(Set({ kind: Ack, id: id, inc: r.inc, nb: r.nb })) } }
  }

  action pCrash: bool = all { PCRASH, s.pcrashes < 2, s' = { ...s, fetched: Set(), pcrashes: s.pcrashes + 1 } }

  // the outbox delivers a receipt to the workflow, any number of times
  action wApply: bool = {
    nondet id = s.outbox.oneOf()
    val op = s.recs.get(id).op
    all {
      s.outbox.size() > 0,
      if (RULES.dedupReceipt and s.appliedIds.contains(id)) s' = { ...s, dupReceipts: s.dupReceipts + 1 }
      else s' = { ...s, appliedIds: s.appliedIds.union(Set(id)), applied: s.applied.replaceAt(op, s.applied[op] + 1) },
    }
  }

  // ---- network and executor -------------------------------------------------------------------------------
  action deliver: bool = {
    nondet m = s.net.oneOf()
    nondet keep = Set(true, false).oneOf()
    nondet c = 0.to(R).oneOf()
    nondet victim = IDS.oneOf()
    val net2 = if (keep) s.net else s.net.exclude(Set(m))
    // at the record ceiling: the design refuses; the mutant erases a tombstone and goes on
    val full = records(s.ex) >= META
    val ex1 = if (full and not(RULES.metaBound)) eraseOne(s.ex, victim) else s.ex
    val metaRefuse = full and RULES.metaBound
    val x = ex1.get(m.id)
    all {
      s.net.size() > 0,
      if (RULES.fence and m.inc != s.inc) s' = { ...s, net: net2, stale: s.stale + 1 }
      else match m.kind {
        | Submit =>
          if (x.held or x.tomb != TNone) s' = { ...s, net: net2, deduped: s.deduped + 1 }
          else if (RULES.deadline and s.time > m.nb) s' = { ...s, net: net2, late: s.late + 1 }
          else if (metaRefuse or slots(ex1, RULES.reserve) >= HOLD)
            s' = { ...s, net: net2, refused: s.refused + 1,
                   recs: s.recs.set(m.id, { ...s.recs.get(m.id), refusedSeen: true }) }
          else s' = { ...s, net: net2, ex: ex1.set(m.id, { ...x, held: true, nb: m.nb }) }
        | Cancel =>
          if (not(x.held) and x.tomb == TNone and metaRefuse) s' = { ...s, net: net2, refused: s.refused + 1 }
          else if (not(x.held) and x.tomb == TNone)
            // answered `cancelled_before_accept` (the platform sees it: cancelSeen); the cancelTomb mutant answers
            // the same without remembering the id
            s' = { ...s, net: net2, ex: ex1.set(m.id, if (RULES.cancelTomb) { ...x, tomb: TCancelled, nb: m.nb } else x),
                   recs: s.recs.set(m.id, { ...s.recs.get(m.id), cancelSeen: true }) }
          else if (x.held and not(x.ran))
            // stops at the next unit boundary: a finished, cancelled result with its actual usage (0..R)
            s' = { ...s, net: net2, ex: s.ex.set(m.id, { ...x, ran: true, cost: c, doneAt: s.time }), real: s.real + c,
                   runs: s.runs.set(m.id, s.runs.get(m.id) + 1), cancelledRunning: s.cancelledRunning + 1 }
          else s' = { ...s, net: net2 }
        | Ack =>
          if (x.held and x.ran) s' = { ...s, net: net2, ex: s.ex.set(m.id, { ...x, held: false, tomb: TReleased, acked: true }) }
          else if (x.tomb == TExpired) s' = { ...s, net: net2, ex: s.ex.set(m.id, { ...x, acked: true }) }
          else s' = { ...s, net: net2 }
      },
    }
  }

  // a message is lost
  action drop: bool = {
    nondet m = s.net.oneOf()
    all { s.net.size() > 0, s' = { ...s, net: s.net.exclude(Set(m)),
                                   ackDropped: if (m.kind == Ack) s.ackDropped + 1 else s.ackDropped } }
  }

  // a held attempt runs to its end, spending 1..R. Without a reserved slot (the reserve mutant) a result that
  // finds the retention budget full is dropped at once: the promise is broken
  action xRun: bool = {
    nondet id = IDS.oneOf()
    nondet c = 1.to(R).oneOf()
    val x = s.ex.get(id)
    val full = not(RULES.reserve) and slots(s.ex, false) >= HOLD
    all {
      x.held, not(x.ran),
      s' = { ...s,
        ex: s.ex.set(id, if (full) { ...x, held: false, ran: true, cost: c, doneAt: s.time, tomb: TExpired }
                         else { ...x, ran: true, cost: c, doneAt: s.time }),
        real: s.real + c, runs: s.runs.set(id, s.runs.get(id) + 1),
        promiseBroken: s.promiseBroken or full },
    }
  }

  // the capture window ends: the body goes, the outcome stays until acknowledged (the ackOutcome mutant drops it)
  action xExpire: bool = {
    nondet id = IDS.oneOf()
    val x = s.ex.get(id)
    all {
      x.held, x.ran, s.time >= x.doneAt + HOLD_T,
      s' = { ...s, ex: s.ex.set(id, { ...x, held: false, tomb: TExpired, acked: not(RULES.ackOutcome) }) },
    }
  }

  // without a promise, a finished result may be evicted under load before its window ends
  action xEvict: bool = {
    nondet id = IDS.oneOf()
    val x = s.ex.get(id)
    all {
      not(RULES.promise), x.held, x.ran,
      s' = { ...s, ex: s.ex.set(id, { ...x, held: false, tomb: TEvicted }), promiseBroken: true },
    }
  }

  // a tombstone is pruned once its not-after has passed and, if it ran, its outcome was acknowledged.
  // The tomb mutant prunes as soon as nothing is held
  action xPrune: bool = {
    nondet id = IDS.oneOf()
    val x = s.ex.get(id)
    all {
      not(x.held), x.tomb != TNone,
      RULES.tomb implies (s.time > x.nb and (x.acked or not(x.ran))),
      s' = { ...s, ex: s.ex.set(id, X0), pruned: s.pruned + 1 },
    }
  }

  action restart: bool = all {
    RESTART, s.restarts < 2,
    s' = { ...s, inc: s.inc + 1, ex: IDS.mapBy(_ => X0), restarts: s.restarts + 1 },
  }

  action tick: bool = all { s.time < MAXT, s' = { ...s, time: s.time + 1 } }

  action step = any {
    pSubmit, pResend, pCancel, pLearn, pStatus, pTakeRefusal, pTakeCancel, pCommit, pAck, pCrash, wApply,
    deliver, drop, xRun, xExpire, xEvict, xPrune, restart, tick,
  }

  // ---- properties ------------------------------------------------------------------------------------------
  val runsOnce: bool = IDS.forall(i => s.runs.get(i) <= 1)
  val accounted: bool = s.real <= s.settled + s.reserved
  val appliedOnce: bool = s.applied.foldl(true, (ok, n) => ok and n <= 1)
  val reservationsHeld: bool = s.reserved == R * IDS.filter(i => s.recs.get(i).st == Pending).size()
  val neverAcceptedTrue: bool = not(s.lied)
  val noReceiptLost: bool = not(s.receiptLost)
  val promiseKept: bool = not(s.promiseBroken)
  val noOvercharge: bool = s.settled <= s.real
  val core: bool = and { runsOnce, accounted, appliedOnce, reservationsHeld, neverAcceptedTrue, noReceiptLost }
  val safety: bool = core and promiseKept

  // witnesses (each must be reached)
  val wDone: bool = s.applied == [1, 1]
  val wRestartedThenDone: bool = s.restarts > 0 and wDone
  val wCrashedThenCommitted: bool = s.recoveredAfterCrash > 0
  val wAckLostThenDone: bool = s.ackDropped > 0 and wDone
  val wDupReceipt: bool = s.dupReceipts > 0
  val wNeverThenDone: bool = s.nevers > 0 and wDone
  val wClosedByDeadline: bool = s.closedByDeadline > 0
  val wCancelledBefore: bool = s.cancelledBefore > 0
  val wCancelledRunningCommitted: bool = s.cancelledRunning > 0 and s.outbox.size() > 0
  val wExpiredSettled: bool = s.expiredSettled > 0
  val wRefused: bool = s.refused > 0
  val wStale: bool = s.stale > 0
  val wLate: bool = s.late > 0
  val wDeduped: bool = s.deduped > 0
  val wPruned: bool = s.pruned > 0
  val wUnknown: bool = s.unknowns > 0
  val wForgotten: bool = s.forgotten > 0
}
