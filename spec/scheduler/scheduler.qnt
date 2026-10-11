// -*- mode: Bluespec; -*-
// scheduler.qnt: the G3 fair scheduler of halo_serve (docs/SCHEDULER.md), as a
// shared-state machine. One engine thread runs one unit of GPU work at a time
// (a prefill slice, a batched decode unit or a decider evaluation); between
// units it fences departed requests and picks the next unit. HTTP threads admit
// or refuse requests under the same lock. The process boundary is the client's
// connection, so the clients are modelled with every outcome of an HTTP call:
// a lost request, a late effect (a request that arrives after its client left;
// work that runs after it left), a lost answer, and duplicate delivery (the
// client's retry). Durations are ticks; one tick is half a 2048-token prefill
// slice (~0.6 s at the measured costs, SCHEDULER.md "Model").
//
// The caller and scheduler rules are the fields of `RULES`; the design sets them
// all true and each mutant (schedTest.qnt) turns one off.
module scheduler {
  import basicSpells.* from "./spells/basicSpells"

  type Rules = {
    preempt: bool,         // units are slices and decode units, picked by fair queuing (false: today's FIFO, run to completion)
    deciderPrio: bool,     // a decider with credit runs at the next unit boundary (false: a decider is just principal D)
    deciderBudget: bool,   // decider priority is credit-limited (false: strict priority, unlimited)
    fenceDeparture: bool,  // a departed request is dropped at the next unit boundary (false: it runs to the end)
    deadPeerProbe: bool,   // a peer that vanished without FIN is found within ACK_TO (heartbeat unacked, keepalive)
    heartbeat: bool,       // SSE comments while queued or prefilling keep the client's idle timer alive
    parkOnCancel: bool,    // a departed prefill parks its slice-aligned progress as a retained prefix (false: dropped)
    bindAtStart: bool,     // the reused prefix is chosen when the first slice starts (false: at admission)
    reserveLedger: bool,   // admission against the reservation ledger (false: against the memory reading only)
    evictSeesFreed: bool,  // the next check sees what an eviction freed (false: F3, released pages stay charged)
    pinPlanned: bool,      // a retained prefix an admitted request plans to reuse is not evictable
    atomic429: bool,       // a 429 and the removal from the queue are one critical section (AUDIT N4)
    idempotency: bool,     // the idempotency key (conversation, turn): a retry of a completed turn replays the stored
                           // answer; a retry of an in-flight one adopts it (false: every retry is a new request)
    twoLevel: bool,        // fair queuing over chain roots first, then over the sandboxes of a root (A2A-MCP D9)
                           // (false: one level, over sandbox principals; a chain that fans out takes several shares)
    keyedGrace: bool,      // a departed request with a key keeps running for GRACE ticks before it is fenced, so a
                           // retry with the same key adopts it, or replays it if it finished meanwhile (false: a
                           // keyed departure is fenced at the next boundary like any other)
  }
  const RULES: Rules

  // ---------------------------------------------------------------- numbers
  pure val SLICE = 2        // ticks per prefill slice (2048 tokens, ~1.25 s)
  pure val DECODE = 1       // ticks per decode unit (a burst of StepBatch steps for every decoding session)
  pure val DEC_COST = 1     // ticks per decider evaluation
  pure val MAXUNIT = 2      // the longest unit: a request waits at most this for the unit in progress
  pure val CT = 5           // client idle timeout (no byte received for CT ticks)
  pure val ACK_TO = 2       // a vanished peer is known gone this long after it vanished (heartbeat + keepalive)
  pure val RA = 3           // Retry-After of a 429
  pure val DEC_BURST = 1    // decider credit: at most this many prioritised evaluations banked
  pure val DEC_REFILL = 8   // one credit every DEC_REFILL ticks
  pure val CAP = 14         // KV units the sessions may hold
  pure val PLIMIT = 1       // active (not known departed) requests per principal
  pure val NDEC = 12        // decider requests per run
  pure val MAXATT = 4       // attempts per conversation (turns and retries)
  pure val MAXT = 48        // ticks per run (three sandboxes: long enough for a second turn)
  pure val MAXFAULTS = 2    // client faults per run (timeouts are not counted: durations cause them)
  // bounds the properties check (SCHEDULER.md "Model" derives them)
  // in service + catch-up of the two other agent roots and of D + two credited deciders
  pure val WAIT_BOUND = MAXUNIT + 3 + 3 + 3 + 2
  // a request that shares its chain root with another sandbox gets half its root's turns (its root-mate takes the
  // others, and the mate's decode rides in the other roots' batches are charged to the root too), so it waits for
  // about two unchained rounds, the unit in service counted once. The simulation's worst case: 22 breaks (seed
  // 0xbeef, 20000 traces of 48 ticks), 24 holds on both seeds.
  pure val WAIT_CHAIN = 2 * WAIT_BOUND - MAXUNIT
  // chain fairness: two agent roots backlogged together get GPU time within LAG_BOUND ticks of each other.
  // Start-time fair queuing alone gives one unit of each (2 * MAXUNIT); batching adds rides: a decoding root is in
  // every batch, also one a third root starts while catching up to the waiting root's tag (at most MAXUNIT +
  // DECODE ticks of batches, each charged in full to every participant)
  pure val LAG_BOUND = 2 * MAXUNIT + MAXUNIT + DECODE
  pure val DEC_BOUND = MAXUNIT + 2 * DEC_COST  // in service + one credited decider ahead
  pure val GONE_BOUND = ACK_TO + MAXUNIT      // work started for a departed client
  pure val GRACE = 4        // a departed keyed request runs this long before it is fenced (keyedGrace)
  pure val NS_TOK = CT + 1  // a non-streaming turn's answer: its decoding alone outlasts the caller's timeout

  pure val CONVS = Set(1, 3, 5)                 // conversations of sandboxes A, B and C
  pure def principalOf(c: int): str = if (c == 1) "A" else if (c == 3) "B" else "C"
  // chain roots (A2A-MCP 5.7): a request carries its chain root, defaulting to its own principal; X is a chain
  // that may hold several sandboxes (a fan-out); D is the decider class
  pure val AGENT_ROOTS = Set("A", "B", "C", "X")
  pure val ROOTS = AGENT_ROOTS.union(Set("D"))
  pure val PRINCIPALS = Set("A", "B", "C", "D")
  pure def promptOf(turn: int): int = if (turn == 1) 3 else 5  // in slices; turn 2 extends turn 1

  // ---------------------------------------------------------------- types
  type Phase = Waiting | Lost | Queued | Running | Finished | Cancelled | Rejected | Merged
  type Ans = NoAns | Pending | Got | AnsLost | Got429
  type Kind = Idle | Slice | Batch | Decide
  type Req = {
    id: int, p: str, conv: int, turn: int, dec: bool, prompt: int, maxTok: int,
    reused: int, done: int, decoded: int, inherited: int, need: int,
    ph: Phase, conn: bool, fin: bool, leftAt: int, heard: int, admittedAt: int, lastServed: int,
    started: bool, goneWork: int, ans: Ans, prio: bool, served: bool, spurious: bool,
    root: str, chained: bool, seq: int, ns: bool,
  }
  // ns: a non-streaming caller (cordis's outbox, an SDK without stream=true): it hears nothing until the answer, so
  // its timeout fires while the server still holds its request, and it retries the turn with the same key
  type Client = { turn: int, busy: bool, attempts: int, retryAt: int, lastAns: Ans, root: str, ns: bool }
  type Engine = { kind: Kind, until: int, rs: Set[int] }
  // vtime/vnow: virtual time per chain root and the root-level system virtual time; pvt/pvnow: virtual time per
  // principal and each root's own system virtual time (the second level)
  // seq: admissions so far (the server orders requests by arrival: it cannot know the order they were sent in)
  type Srv = { held: int -> int, charged: int, vtime: str -> int, vnow: int, pvt: str -> int, pvnow: str -> int,
               credit: int, seq: int }
  // svc: GPU ticks started for each agent chain root; co: the ordered pairs of agent roots both backlogged at the
  // last pick; base: their service difference when they became backlogged together
  type Ghost = { computed: int -> int, maxPrompt: int -> int, evicted: int -> int, falseRefusals: int, faults: int, selfLimit: int,
                 events: Set[str], svc: str -> int, co: Set[(str, str)], base: (str, str) -> int, redo: int }

  // ---------------------------------------------------------------- state
  var time: int
  var reqs: int -> Req
  var nextId: int
  var clients: int -> Client
  var decSent: int
  var eng: Engine
  var srv: Srv
  var gh: Ghost

  // ---------------------------------------------------------------- views
  // the root the scheduler queues a request under: its chain root (design) or its own principal (one level)
  pure def rootOf(r: Req): str = if (RULES.twoLevel) r.root else r.p
  pure def isActive(r: Req): bool = r.ph == Queued or r.ph == Running
  val allReqs: Set[Req] = reqs.keys().map(i => reqs.get(i))
  val actives: Set[Req] = allReqs.filter(r => isActive(r))
  // the server's knowledge of a departure: FIN at once; a vanished peer after ACK_TO (if probed)
  def knownGone(r: Req): bool =
    not(r.conn) and (r.fin or (RULES.deadPeerProbe and time >= r.leftAt + ACK_TO))
  def hasWork(r: Req): bool = r.dec or r.done < r.prompt or r.decoded < r.maxTok
  // a departed keyed request within its grace keeps running (keyedGrace); fencing waits for the grace to run out
  // the grace starts when the server learns of the departure (FIN at once; a vanished peer ACK_TO later)
  // (a request that reaches the server after its client left: at its admission)
  pure def goneAt(r: Req): int = max(r.admittedAt, if (r.fin) r.leftAt else r.leftAt + ACK_TO)
  def inGrace(r: Req): bool = RULES.keyedGrace and RULES.idempotency and not(r.dec) and time < goneAt(r) + GRACE
  def fenceDue(r: Req): bool = knownGone(r) and not(inGrace(r))
  def runnable(r: Req): bool = isActive(r) and hasWork(r) and not(RULES.fenceDeparture and fenceDue(r))
  val runSet: Set[Req] = actives.filter(r => runnable(r))
  val fencePending: bool = RULES.fenceDeparture and actives.exists(r => fenceDue(r))
  pure def minId(s: Set[Req]): int = s.fold(1000000, (a, r) => if (r.id < a) r.id else a)
  // the id of the request admitted first (FIFO is by arrival at the server, not by when the client sent it)
  pure def firstIn(s: Set[Req]): int =
    s.fold((1000000, 0), (a, r) => if (r.seq < a._1) (r.seq, r.id) else a)._2
  pure def sumHeld(h: int -> int): int = CONVS.fold(0, (a, c) => a + h.get(c))
  // pages actually mapped: retained prefixes plus each active request's own pages
  // the page a unit in progress writes is mapped when the unit starts (EnsureKv before the chunk or step)
  def inUnit(r: Req): int = if (eng.rs.contains(r.id) and (eng.kind == Slice or eng.kind == Batch)) 1 else 0
  val physical: int = sumHeld(srv.held) +
    actives.filter(r => not(r.dec)).fold(0, (a, r) => a + r.done - r.reused + r.decoded + r.inherited + inUnit(r))
  // the reservation ledger: retained prefixes plus each active request's reservation
  pure def ledgerUsed(h: int -> int, rs: Set[Req], charged: int): int =
    sumHeld(h) + rs.filter(r => isActive(r) and not(r.dec)).fold(0, (a, r) => a + r.need + r.inherited) +
    (if (RULES.evictSeesFreed) 0 else charged)
  pure def reuseOf(h: int -> int, c: int, prompt: int): int = if (h.get(c) <= prompt) h.get(c) else 0
  // free memory for an admission: the reservation ledger (design) or the memory reading (mutant)
  pure def freeOf(h: int -> int, rs: Set[Req], charged: int, phys: int): int =
    if (RULES.reserveLedger) CAP - ledgerUsed(h, rs, charged)
    else CAP - phys - (if (RULES.evictSeesFreed) 0 else charged)
  def lastByte(r: Req): int = if (RULES.heartbeat and isActive(r) and not(r.ns)) time else r.heard

  // ---------------------------------------------------------------- init
  // each conversation's sandbox serves either its own chain (the default root: itself) or the chain X
  action init = {
    nondet ra = Set("A", "X").oneOf()
    nondet rb = Set("B", "X").oneOf()
    nondet rc = Set("C", "X").oneOf()
    // sandbox C's caller may be non-streaming (the cordis outbox shape)
    nondet nsC = Set(false, true).oneOf()
    val rootOfConv = Map(1 -> ra, 3 -> rb, 5 -> rc)
    all {
      time' = 0,
      reqs' = Map(),
      nextId' = 1,
      clients' = CONVS.mapBy(c => { turn: 1, busy: false, attempts: 0, retryAt: 0, lastAns: NoAns, root: rootOfConv.get(c),
                                    ns: c == 5 and nsC }),
      decSent' = 0,
      eng' = { kind: Idle, until: 0, rs: Set() },
      srv' = { held: CONVS.mapBy(c => 0), charged: 0, vtime: ROOTS.mapBy(q => 0), vnow: 0,
               pvt: PRINCIPALS.mapBy(q => 0), pvnow: ROOTS.mapBy(q => 0), credit: DEC_BURST, seq: 0 },
      gh' = { computed: CONVS.mapBy(c => 0), maxPrompt: CONVS.mapBy(c => 0), evicted: CONVS.mapBy(c => 0),
              falseRefusals: 0, faults: 0, selfLimit: 0, events: Set(),
              svc: AGENT_ROOTS.mapBy(q => 0), co: Set(), base: tuples(AGENT_ROOTS, AGENT_ROOTS).mapBy(q => 0), redo: 0 },
    }
  }

  pure def newReq(id: int, p: str, c: int, turn: int, dec: bool, prompt: int, maxTok: int, t: int, root: str, ns: bool): Req = {
    id: id, p: p, conv: c, turn: turn, dec: dec, prompt: prompt, maxTok: maxTok,
    reused: 0, done: 0, decoded: 0, inherited: 0, need: 0,
    ph: Waiting, conn: true, fin: true, leftAt: 0, heard: t, admittedAt: 0, lastServed: 0,
    started: false, goneWork: 0, ans: NoAns, prio: false, served: false, spurious: false,
    root: root, chained: false, seq: 0, ns: ns,
  }
  def ev(e: str): Ghost = { ...gh, events: gh.events.union(Set(e)) }
  // a client-side fault (lost request, abort, vanished peer, lost answer): at most MAXFAULTS a run, so that runs
  // also reach the long happy paths (both turns of both conversations) between faults
  def fault(e: str): Ghost = { ...gh, faults: gh.faults + 1, events: gh.events.union(Set(e)) }

  // ---------------------------------------------------------------- clients
  // a sandbox sends its conversation's next turn, or retries the same turn (duplicate delivery)
  action clientSend(c: int): bool = {
    val cl = clients.get(c)
    nondet mt = Set(1, 2).oneOf()
    all {
      not(cl.busy), cl.turn <= 2, cl.attempts < MAXATT, time >= cl.retryAt,
      reqs' = reqs.put(nextId, newReq(nextId, principalOf(c), c, cl.turn, false, promptOf(cl.turn),
                                      if (cl.ns) NS_TOK else mt, time, cl.root, cl.ns)),
      nextId' = nextId + 1,
      clients' = clients.set(c, { ...cl, busy: true, attempts: cl.attempts + 1 }),
      decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = if (cl.lastAns == AnsLost) ev("retryAfterLostAnswer")
            else if (cl.lastAns == Got429) ev("retryAfter429") else gh,
    }
  }
  // a decider call (/v1/systemone; non-streaming, short)
  action deciderSend: bool = all {
    decSent < NDEC,
    reqs' = reqs.put(nextId, newReq(nextId, "D", 0, 0, true, 0, 0, time, "D", false)),
    nextId' = nextId + 1, decSent' = decSent + 1,
    clients' = clients, eng' = eng, srv' = srv, time' = time, gh' = gh,
  }
  // lost request: it never reaches the server; the client hears nothing
  action loseRequest(id: int): bool = {
    val r = reqs.get(id)
    all {
      r.ph == Waiting, not(r.dec), gh.faults < MAXFAULTS,
      reqs' = reqs.set(id, { ...r, ph: Lost }),
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = fault("lostRequest"),
    }
  }
  // the client stops waiting and closes its connection (FIN)
  action leave(id: int, fin: bool, spurious: bool): bool = {
    val r = reqs.get(id)
    val cl = clients.get(r.conv)
    all {
      reqs' = reqs.set(id, { ...r, conn: false, fin: fin, leftAt: time, spurious: r.spurious or spurious }),
      clients' = clients.set(r.conv, { ...cl, busy: false, retryAt: time, lastAns: r.ans }),
      nextId' = nextId, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
    }
  }
  def waitingOn(r: Req): bool = not(r.dec) and r.conn and (r.ans == NoAns or r.ans == AnsLost)
  // idle timeout: no byte for CT ticks (lost request, lost answer, or a silent queue or prefill)
  action clientTimeout(id: int): bool = {
    val r = reqs.get(id)
    all {
      waitingOn(r), time - lastByte(r) >= CT,
      leave(id, true, isActive(r) and not(r.ns)),
      gh' = if (r.ns) ev("nsTimeout") else if (isActive(r)) ev("spuriousTimeout") else ev("timeout"),
    }
  }
  // the user aborts (Ctrl-C), at any point
  action clientAbort(id: int): bool = {
    val r = reqs.get(id)
    all { waitingOn(r), gh.faults < MAXFAULTS, leave(id, true, false), gh' = fault("abort") }
  }
  // the peer vanishes without FIN (laptop asleep, VM suspended, link down); it may come back and retry
  action peerVanishes(id: int): bool = {
    val r = reqs.get(id)
    all { waitingOn(r), gh.faults < MAXFAULTS, leave(id, false, false), gh' = fault("vanish") }
  }
  // the final answer reaches the client, or is lost (lost answer)
  action deliver(id: int): bool = {
    val r = reqs.get(id)
    all {
      r.ans == Pending, r.conn,
      reqs' = reqs.set(id, { ...r, ans: Got, heard: time }),
      clients' = if (r.dec) clients else {
        val cl = clients.get(r.conv)
        clients.set(r.conv, { ...cl, busy: false, turn: cl.turn + 1, lastAns: Got })
      },
      nextId' = nextId, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = { ...gh, events: gh.events.union(if (r.turn == 2) Set("turn2Done") else Set())
                                      .union(if (r.ns) Set("nsDone") else Set()) },
    }
  }
  action loseAnswer(id: int): bool = {
    val r = reqs.get(id)
    all {
      r.ans == Pending, not(r.dec), gh.faults < MAXFAULTS,
      reqs' = reqs.set(id, { ...r, ans: AnsLost }),
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = fault("lostAnswer"),
    }
  }

  // ---------------------------------------------------------------- admission (HTTP thread, under the scheduler lock)
  // A request reaches the server, possibly after its client left (late effect). The decision (queue, or 429
  // with Retry-After) and any eviction it needs are one critical section.
  // start-time fair queuing: a principal that becomes backlogged joins at the system virtual time `vnow` (the
  // start tag of the last unit picked), so it banks no credit while idle. (An earlier version joined at the minimum
  // over the other runnable principals; a principal briefly not runnable -- departed, then adopted -- was left out
  // of that minimum and then waited for the joiner to catch up: boundedWait broke at 20000 samples.)
  // Two levels (A2A-MCP D9): the chain root joins the root-level queue at `vnow`, and the principal joins its
  // root's queue at that root's `pvnow`.
  def joinVtime(r: Req): Srv = {
    val R = rootOf(r)
    { ...srv,
      vtime: if (runSet.exists(x => rootOf(x) == R)) srv.vtime else srv.vtime.set(R, max(srv.vtime.get(R), srv.vnow)),
      pvt: if (runSet.exists(x => x.p == r.p)) srv.pvt else srv.pvt.set(r.p, max(srv.pvt.get(r.p), srv.pvnow.get(R))) }
  }
  action arrive(id: int): bool = {
    val r = reqs.get(id)
    all { r.ph == Waiting, if (r.dec) admitDecider(r) else admitAgent(r) }
  }
  action admitDecider(r: Req): bool = {
    val prio = not(RULES.deciderBudget) or srv.credit > 0
    all {
      reqs' = reqs.set(r.id, { ...r, ph: Queued, admittedAt: time, lastServed: time, prio: prio, seq: srv.seq }),
      srv' = { ...joinVtime(r), credit: if (RULES.deciderBudget and prio) srv.credit - 1 else srv.credit,
               seq: srv.seq + 1 },
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, time' = time,
      gh' = if (prio) gh else ev("deciderOutOfCredit"),
    }
  }
  action refuse(r: Req, why: str, g: Ghost): bool = {
    val cl = clients.get(r.conv)
    all {
      // the design takes it off the queue with the answer; the mutant answers 429 and leaves it queued
      reqs' = reqs.set(r.id, { ...r, ph: if (RULES.atomic429) Rejected else Queued, ans: Got429, admittedAt: time,
                               lastServed: time, need: 0 }),
      // the 429 reaches only a client still waiting on this attempt (a late attempt's client has moved on)
      clients' = if (r.conn) clients.set(r.conv, { ...cl, busy: false, retryAt: time + RA, lastAns: Got429 })
                 else clients,
      nextId' = nextId, decSent' = decSent, eng' = eng, time' = time,
      gh' = { ...g, events: g.events.union(Set(why)) },
    }
  }
  // the idempotency key is (conversation, turn): every attempt of a turn carries the same key
  pure def sameKey(x: Req, r: Req): bool = not(x.dec) and x.id != r.id and x.conv == r.conv and x.turn == r.turn
  action admitAgent(r: Req): bool = {
    val prior = allReqs.filter(x => sameKey(x, r))
    val stored = prior.filter(x => x.ph == Finished and x.served)
    val inflight = prior.filter(x => isActive(x))
    // What the server knows of the arriving connection: gone only once it saw FIN or the probe failed (a peer that
    // vanished a moment ago still looks connected). The code decides duplicate vs adopt on this, never on the
    // client's true state (M5's trace replay found the model had used the latter).
    val arrivesGone = knownGone(r)
    if (RULES.idempotency and stored.size() > 0) all {
      // replay: the stored answer (the Responses store holds it), no engine work
      reqs' = reqs.set(r.id, { ...r, ph: Finished, ans: Pending, admittedAt: time }),
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = ev("replayed"),
    } else if (RULES.idempotency and inflight.size() > 0 and arrivesGone) all {
      // a late duplicate of a key already in flight: dropped (no answer is owed: its client left)
      reqs' = reqs.set(r.id, { ...r, ph: Merged }),
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
      gh' = ev("lateDuplicateDropped"),
    } else if (RULES.idempotency and inflight.size() > 0) {
      // adopt: the retry's connection takes over the in-flight attempt (its stream resumes); no new work. The
      // attempt now lives or dies with the adopting connection (which may itself be a peer that vanished).
      val x = reqs.get(minId(inflight))
      all {
        reqs' = reqs.set(x.id, { ...x, conn: r.conn, fin: r.fin, leftAt: r.leftAt, heard: time, goneWork: 0 })
                    .set(r.id, { ...r, ph: Merged, conn: false }),
        nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, srv' = srv, time' = time,
        gh' = ev("adopted"),
      }
    } else admitNew(r)
  }
  action admitNew(r: Req): bool = {
    // ghost: a retry that reaches the server within GRACE of its earlier attempt's departure, and finds that attempt
    // fenced with decoding done, will decode again what was already decoded (the redo keyedGrace prevents)
    val redone = RULES.idempotency and allReqs.exists(x => sameKey(x, r) and x.ph == Cancelled and x.decoded > 0 and
                                                       time < x.leftAt + GRACE)
    val gr = if (redone) { ...gh, redo: gh.redo + 1 } else gh
    val mine = actives.filter(x => x.p == r.p and not(x.dec) and not(knownGone(x)))
    val reuse0 = reuseOf(srv.held, r.conv, r.prompt)
    val need = r.prompt - reuse0 + r.maxTok
    val rq = { ...r, ph: Queued, admittedAt: time, lastServed: time, reused: reuse0, done: reuse0, need: need,
               seq: srv.seq }
    // r is still Waiting, so neither the ledger nor the reading counts it yet
    val free0 = freeOf(srv.held, allReqs, srv.charged, physical)
    // the other conversations' unpinned prefixes are the eviction candidates; when one is needed, all go (the
    // model does not order them: SessionCache evicts LRU until the prompt fits)
    val ev0 = CONVS.filter(o => o != r.conv and srv.held.get(o) > 0 and
                                not(RULES.pinPlanned and actives.exists(x => x.conv == o)))
    val canEvict = ev0.size() > 0
    // what an eviction does: the prefix is dropped; a started fork keeps its shared pages (they become its own)
    val forks = actives.filter(x => ev0.contains(x.conv) and x.started and x.reused > 0 and x.inherited == 0)
    val h2 = CONVS.mapBy(o => if (ev0.contains(o)) 0 else srv.held.get(o))
    val freed = ev0.fold(0, (a, o) => a + (if (forks.exists(x => x.conv == o)) 0 else srv.held.get(o)))
    val charged2 = srv.charged + freed
    val reqs2 = reqs.keys().mapBy(i => {
      val x = reqs.get(i)
      if (forks.contains(x)) { ...x, inherited: x.reused } else x
    })
    val all2 = reqs2.keys().map(i => reqs2.get(i))
    val phys2 = physical - freed
    val free2 = freeOf(h2, all2, charged2, phys2)
    // truth: what fits if every evictable prefix went (pins and other reservations are real)
    val trueFree = CAP - sumHeld(if (canEvict) h2 else srv.held) -
      all2.filter(x => isActive(x) and not(x.dec) and x.id != r.id).fold(0, (a, x) => a + x.need + x.inherited)
    val g1 = { ...gr, maxPrompt: gh.maxPrompt.set(r.conv, max(gh.maxPrompt.get(r.conv), r.prompt)) }
    val gEvict = { ...g1, evicted: CONVS.mapBy(o => gh.evicted.get(o) + (if (ev0.contains(o)) srv.held.get(o) else 0)),
                   events: gh.events.union(Set("evicted")) }
    // a request that shares its chain root with another sandbox's request is chained (its wait bound is the
    // chained one, both ways); the flag stays for the request's life
    val mates = actives.filter(x => not(x.dec) and rootOf(x) == rootOf(r) and x.p != r.p).map(x => x.id)
    def markChained(m: int -> Req): int -> Req =
      if (mates.size() == 0) m
      else m.keys().mapBy(i => { val x = m.get(i)  if (mates.contains(i) or i == r.id) { ...x, chained: true } else x })
    if (mine.size() >= PLIMIT) all {
      // a refusal caused only by the client's own earlier attempt of this turn is self-inflicted
      refuse(r, "limit429", { ...gr, selfLimit: gh.selfLimit + (if (r.conn and mine.forall(x => sameKey(x, r))) 1 else 0) }),
      srv' = srv,
    }
    else if (need <= free0) all {
      reqs' = markChained(reqs.set(r.id, rq)),
      srv' = { ...joinVtime(r), seq: srv.seq + 1 },
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, time' = time,
      gh' = { ...g1, events: gh.events.union(if (r.ans == NoAns and not(r.conn)) Set("lateArrival") else Set()) },
    }
    else if (canEvict and need <= free2) all {
      reqs' = markChained(reqs2.set(r.id, rq)),
      srv' = { ...joinVtime(r), held: h2, charged: charged2, seq: srv.seq + 1 },
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, time' = time,
      gh' = gEvict,
    }
    // refused: evictions already made stay made (as SessionCache::Prepare's loop does)
    else if (canEvict) all {
      refuse(r, "mem429", { ...gEvict, falseRefusals: gh.falseRefusals + (if (need <= trueFree) 1 else 0) }),
      srv' = { ...srv, held: h2, charged: charged2 },
    }
    else all {
      refuse(r, "mem429", { ...g1, falseRefusals: gh.falseRefusals + (if (need <= trueFree) 1 else 0) }),
      srv' = srv,
    }
  }

  // ---------------------------------------------------------------- the engine thread
  // fence: every departed request the server knows of leaves at a unit boundary, all at once (the code fences in
  // one critical section: an arrival cannot land between two fences); a started prefill's slice-aligned progress
  // is parked as its conversation's retained prefix (design)
  def parks(r: Req): bool = RULES.parkOnCancel and not(r.dec) and r.started and r.done > srv.held.get(r.conv)
  action fence: bool = {
    val gone = actives.filter(r => fenceDue(r))
    val parked = gone.filter(r => parks(r))
    all {
      eng.kind == Idle, RULES.fenceDeparture, gone.size() > 0,
      reqs' = reqs.keys().mapBy(i => if (gone.exists(r => r.id == i)) { ...reqs.get(i), ph: Cancelled } else reqs.get(i)),
      srv' = { ...srv, held: CONVS.mapBy(c => parked.filter(r => r.conv == c).fold(srv.held.get(c), (a, r) => max(a, r.done))) },
      nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, time' = time,
      gh' = { ...gh, events: gh.events
        .union(if (parked.size() > 0) Set("parkedOnCancel") else Set())
        .union(if (gone.exists(r => not(parks(r)) and r.fin)) Set("fencedFin") else Set())
        .union(if (gone.exists(r => not(parks(r)) and not(r.fin))) Set("fencedVanished") else Set()) },
    }
  }

  // the reused prefix a request starts from: bound when its first unit starts (design), or the one chosen at
  // admission if it is still retained (mutant; an evicted prefix leaves nothing to reuse)
  def boundOf(x: Req): int =
    if (x.started or x.dec) x.reused
    else if (RULES.bindAtStart) reuseOf(srv.held, x.conv, x.prompt)
    else if (srv.held.get(x.conv) >= x.reused) x.reused else 0
  // prefill still to do once bound: the unit kind is chosen from it
  def prefillLeft(x: Req): bool = boundOf(x) + (if (x.started) x.done - x.reused else 0) < x.prompt

  // start a unit for the requests `rs`: bind (first unit), charge, note waste for departed clients
  // Each participant's root and principal are charged the unit's cost once (a decode unit charges every
  // participant a whole unit, as before).
  action startUnit(k: Kind, rs: Set[int], cost: int, g: Ghost): bool = {
    val midPrefill = actives.exists(x => not(rs.contains(x.id)) and x.started and x.done < x.prompt and runnable(x))
    val inU = rs.map(i => reqs.get(i))
    val chRoots = inU.map(x => rootOf(x))
    val chPs = inU.map(x => x.p)
    // ghost: the agent chain roots (true roots, whatever the scheduler queues by) backlogged now, and their service
    val bl = runSet.filter(x => not(x.dec)).map(x => x.root)
    val pairs = tuples(bl, bl).filter(q => q._1 != q._2)
    val svcRoots = inU.filter(x => not(x.dec)).map(x => x.root)
    val svc2 = gh.svc.keys().mapBy(q => gh.svc.get(q) + (if (svcRoots.contains(q)) cost else 0))
    val base2 = gh.base.keys().mapBy(q =>
      if (pairs.contains(q) and not(gh.co.contains(q))) gh.svc.get(q._1) - gh.svc.get(q._2) else gh.base.get(q))
    val fanout = AGENT_ROOTS.exists(q => runSet.filter(x => not(x.dec) and x.root == q).map(x => x.p).size() >= 2) and
                 bl.size() >= 2
    all {
      eng' = { kind: k, until: time + cost, rs: rs },
      reqs' = reqs.keys().mapBy(i => {
        val x = reqs.get(i)
        if (not(rs.contains(i))) x
        else {
          val bound = boundOf(x)
          val y = if (x.started or x.dec) x
                  else { ...x, reused: bound, done: bound, need: min(x.need, x.prompt - bound + x.maxTok) }
          { ...y, started: true, ph: Running, served: true, lastServed: time,
                  goneWork: x.goneWork + (if (not(x.conn) and x.ans != Got429) cost else 0) }
        }
      }),
      srv' = {
        val rts = runSet.map(x => rootOf(x))
        val mv = rts.fold(1000000, (a, q) => if (srv.vtime.get(q) < a) srv.vtime.get(q) else a)
        // each backlogged root's own system virtual time: the least tag among its backlogged principals
        val pvnow2 = srv.pvnow.keys().mapBy(q => {
          val ps = runSet.filter(x => rootOf(x) == q).map(x => x.p)
          if (ps.size() == 0) srv.pvnow.get(q)
          else max(srv.pvnow.get(q), ps.fold(1000000, (a, p) => if (srv.pvt.get(p) < a) srv.pvt.get(p) else a))
        })
        { ...srv,
          vtime: srv.vtime.keys().mapBy(q => srv.vtime.get(q) + (if (chRoots.contains(q)) cost else 0)),
          pvt: srv.pvt.keys().mapBy(q => srv.pvt.get(q) + (if (chPs.contains(q)) cost else 0)),
          vnow: if (rts.size() > 0) max(srv.vnow, mv) else srv.vnow,
          pvnow: pvnow2 }
      },
      nextId' = nextId, clients' = clients, decSent' = decSent, time' = time,
      gh' = {
        val e1 = if (midPrefill) Set("preempted") else Set()
        val e2 = if (rs.size() >= 2) Set("batched") else Set()
        val e3 = if (rs.exists(i => not(reqs.get(i).started) and not(reqs.get(i).dec) and
                                   reuseOf(srv.held, reqs.get(i).conv, reqs.get(i).prompt) > reqs.get(i).reused))
                   Set("boundLater") else Set()
        val e4 = if (k == Decide and midPrefill) Set("deciderBetweenSlices") else Set()
        val e5 = if (fanout) Set("chainFanout") else Set()
        { ...g, events: g.events.union(e1).union(e2).union(e3).union(e4).union(e5), svc: svc2, co: pairs, base: base2 }
      },
    }
  }
  action pick: bool = {
    val rset = runSet
    val prioDec = rset.filter(r => r.dec and r.prio)
    val v = srv.vtime
    val pv = srv.pvt
    all {
      eng.kind == Idle, not(fencePending), rset.size() > 0,
      if (RULES.deciderPrio and prioDec.size() > 0) {
        val i = firstIn(prioDec)
        startUnit(Decide, Set(i), DEC_COST, ev("deciderPrio"))
      } else if (not(RULES.preempt)) {
        // today: one request at a time, to the end (the generator mutex); no batching
        val cur = rset.filter(r => r.started)
        val i = if (cur.size() > 0) firstIn(cur) else firstIn(rset)
        val r = reqs.get(i)
        if (r.dec) startUnit(Decide, Set(i), DEC_COST, gh)
        else if (prefillLeft(r)) startUnit(Slice, Set(i), SLICE, gh)
        else startUnit(Batch, Set(i), DECODE, gh)
      } else {
        // two-level start-time fair queuing by GPU time: the chain root with the least tag, then the principal
        // with the least tag within that root, then FIFO within the principal
        // Ties go to the root (then the principal) holding the request admitted first, as the code breaks them
        // (the M5 trace replay compares the code's picks with these).
        val rts = rset.map(r => rootOf(r))
        val mv = rts.fold(1000000, (a, q) => if (v.get(q) < a) v.get(q) else a)
        val tiedR = rts.filter(q => v.get(q) == mv)
        val R = rootOf(reqs.get(firstIn(rset.filter(r => tiedR.contains(rootOf(r))))))
        val inR = rset.filter(r => rootOf(r) == R)
        val ps = inR.map(r => r.p)
        val mp = ps.fold(1000000, (a, q) => if (pv.get(q) < a) pv.get(q) else a)
        val tiedP = ps.filter(q => pv.get(q) == mp)
        val p = reqs.get(firstIn(inR.filter(r => tiedP.contains(r.p)))).p
        val i = firstIn(inR.filter(r => r.p == p))
        val r = reqs.get(i)
        if (r.dec) startUnit(Decide, Set(i), DEC_COST, if (r.prio) gh else ev("deciderFair"))
        else if (prefillLeft(r)) startUnit(Slice, Set(i), SLICE, gh)
        else {
          // every decoding session steps in the same batch (StepBatch, up to 8); each root and principal pays a unit
          val b = rset.filter(x => not(x.dec) and not(prefillLeft(x)))
          startUnit(Batch, b.map(x => x.id), DECODE, gh)
        }
      }
    }
  }
  // the unit ends: its effects apply (a slice advances the prefill; a decode unit emits a token to each)
  action unitEnd: bool = {
    val rs = eng.rs
    val fin = rs.filter(i => {
      val x = reqs.get(i)
      x.dec or (eng.kind == Batch and x.decoded + 1 >= x.maxTok)
    })
    all {
      eng.kind != Idle, time >= eng.until,
      reqs' = reqs.keys().mapBy(i => {
        val x = reqs.get(i)
        if (not(rs.contains(i))) x
        else if (eng.kind == Slice) { ...x, done: x.done + 1, lastServed: time }
        else if (eng.kind == Batch) {
          val y = { ...x, decoded: x.decoded + 1, lastServed: time, heard: if (x.conn and not(x.ns)) time else x.heard }
          if (fin.contains(i)) { ...y, ph: Finished, ans: Pending } else y
        }
        else { ...x, ph: Finished, ans: Pending, lastServed: time }
      }),
      // a finished turn keeps its prompt as a retained prefix (the exact-mode checkpoint)
      srv' = { ...srv, held: CONVS.mapBy(c => {
        val f = fin.filter(i => reqs.get(i).conv == c)
        if (f.size() == 0) srv.held.get(c)
        else max(srv.held.get(c), f.fold(0, (a, i) => max(a, reqs.get(i).prompt)))
      }) },
      gh' = if (eng.kind == Slice) {
        val c = reqs.get(rs.fold(0, (a, i) => i)).conv
        { ...gh, computed: gh.computed.set(c, gh.computed.get(c) + 1) }
      } else if (fin.exists(i => not(reqs.get(i).dec) and not(reqs.get(i).conn))) ev("finishedDetached")
      else gh,
      eng' = { kind: Idle, until: time, rs: Set() },
      nextId' = nextId, clients' = clients, decSent' = decSent, time' = time,
    }
  }
  // time passes while a unit runs, or while there is nothing to run (the scheduler is work-conserving)
  action tick: bool = all {
    time < MAXT,
    if (eng.kind != Idle) time < eng.until else not(fencePending or runSet.size() > 0),
    time' = time + 1,
    srv' = { ...srv, credit: if ((time + 1) % DEC_REFILL == 0) min(DEC_BURST, srv.credit + 1) else srv.credit },
    reqs' = reqs, nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng, gh' = gh,
  }
  action stutter: bool = all {
    time >= MAXT,
    time' = time, srv' = srv, reqs' = reqs, nextId' = nextId, clients' = clients, decSent' = decSent, eng' = eng,
    gh' = gh,
  }

  action step = any {
    nondet c = CONVS.oneOf()
    clientSend(c),
    deciderSend,
    all {
      reqs.keys().size() > 0,
      nondet i = reqs.keys().oneOf()
      any { loseRequest(i), arrive(i), clientTimeout(i), clientAbort(i), peerVanishes(i), deliver(i), loseAnswer(i) },
    },
    fence,
    pick,
    unitEnd,
    tick,
    stutter,
  }

  // ---------------------------------------------------------------- properties
  // a 429 means not done: the refused request never gets engine work (AUDIT N4)
  val rejectedNeverRuns: bool = allReqs.forall(r => not(r.ans == Got429 and r.served))
  // starvation freedom, as the state property whose violation is its mechanism: a runnable request of a
  // connected client is served again within WAIT_BOUND ticks
  // (a request sharing its chain root with another sandbox gets half its root's share: the chained bound)
  val boundedWait: bool = actives.forall(r =>
    r.dec or not(r.conn) or r.ans == Got429 or not(hasWork(r)) or
    time - r.lastServed <= (if (r.chained) WAIT_CHAIN else WAIT_BOUND))
  // chain fairness (A2A-MCP D9): two agent chain roots backlogged together get GPU time within LAG_BOUND ticks of
  // each other, however many sandboxes each one holds (a fan-out does not take several shares)
  val chainFair: bool = gh.co.forall(q => {
    val d = gh.svc.get(q._1) - gh.svc.get(q._2) - gh.base.get(q)
    d <= LAG_BOUND and -d <= LAG_BOUND
  })
  // a credited decider starts within DEC_BOUND of its admission
  val deciderLatency: bool = actives.forall(r => not(r.dec and r.prio and r.ph == Queued) or time - r.admittedAt <= DEC_BOUND)
  // late effect bounded: work started for a client that left is at most GONE_BOUND ticks
  // (a keyed request's departure is fenced only after its grace: its bound grows by GRACE, the price of keyedGrace;
  // the ghost restarts when a retry adopts the request, since that work is collected)
  val goneBound: int = if (RULES.keyedGrace and RULES.idempotency) GONE_BOUND + GRACE else GONE_BOUND
  val workAfterGoneBounded: bool = allReqs.forall(r => r.goneWork <= goneBound)
  // a streaming client never times out while the server holds its request (the D3 retry trigger)
  val noSpuriousTimeout: bool = allReqs.forall(r => not(r.spurious))
  // memory: mapped pages never exceed what the sessions may hold
  val noOvercommit: bool = physical <= CAP
  // each request's own mapped pages stay within its reservation (what the ledger counts is what is mapped)
  val withinReservation: bool = actives.forall(r =>
    r.dec or r.done - r.reused + r.decoded + inUnit(r) <= r.need)
  // a retry is never refused because of its own earlier attempt (that 429 is the retry-amplification loop)
  val noSelfInflicted429: bool = gh.selfLimit == 0
  // a turn is generated at most once: a retry after a lost answer gets the stored answer, not a new generation
  val generatedOnce: bool = allReqs.forall(a => allReqs.forall(b =>
    a.id == b.id or a.dec or b.dec or a.conv != b.conv or a.turn != b.turn or
    not(a.ph == Finished and a.served and b.ph == Finished and b.served)))
  // a memory 429 only when evicting every evictable prefix would not make room (F3)
  val refuseOnlyIfShort: bool = gh.falseRefusals == 0
  // a caller that retries with the same key within GRACE of leaving never makes the server decode its turn again
  // (cordis's outbox and SDK retries: a non-streaming answer longer than the caller's timeout)
  val noRedoWithinGrace: bool = gh.redo == 0
  // retries do not recompute: prefill slices per conversation <= its longest prompt + prefixes evicted
  val noRecompute: bool = CONVS.forall(c => gh.computed.get(c) <= gh.maxPrompt.get(c) + gh.evicted.get(c))
  val safety: bool = and {
    rejectedNeverRuns, boundedWait, deciderLatency, workAfterGoneBounded, noSpuriousTimeout, noOvercommit,
    withinReservation, refuseOnlyIfShort, noRecompute, noSelfInflicted429, generatedOnce, chainFair,
    noRedoWithinGrace,
  }

  // ---------------------------------------------------------------- witnesses
  val wPreempted: bool = gh.events.contains("preempted")
  val wBatched: bool = gh.events.contains("batched")
  val wDeciderBetweenSlices: bool = gh.events.contains("deciderBetweenSlices")
  val wDeciderFair: bool = gh.events.contains("deciderFair")
  val wParkedOnCancel: bool = gh.events.contains("parkedOnCancel")
  val wBoundLater: bool = gh.events.contains("boundLater")
  val wEvicted: bool = gh.events.contains("evicted")
  val wMem429: bool = gh.events.contains("mem429")
  val wLimit429: bool = gh.events.contains("limit429")
  val wRetryAfter429: bool = gh.events.contains("retryAfter429")
  val wRetryAfterLostAnswer: bool = gh.events.contains("retryAfterLostAnswer")
  val wTimeout: bool = gh.events.contains("timeout")
  val wFencedFin: bool = gh.events.contains("fencedFin")
  val wFencedVanished: bool = gh.events.contains("fencedVanished")
  val wLateArrival: bool = gh.events.contains("lateArrival")
  val wReplayed: bool = gh.events.contains("replayed")
  val wAdopted: bool = gh.events.contains("adopted")
  val wTurn2Done: bool = gh.events.contains("turn2Done")
  val wChainFanout: bool = gh.events.contains("chainFanout")
  val wNsTimeout: bool = gh.events.contains("nsTimeout")
  val wNsDone: bool = gh.events.contains("nsDone")
  val wFinishedDetached: bool = gh.events.contains("finishedDetached")
}
