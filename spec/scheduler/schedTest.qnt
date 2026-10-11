// -*- mode: Bluespec; -*-
// schedTest.qnt: the instances of scheduler.qnt: the design, today's server as a mutant, and one mutant per
// unsafe rule (each turns one field of RULES off). run.sh checks them all.
module schedRules {
  import scheduler.* from "./scheduler"
  pure val DESIGN: Rules = {
    preempt: true, deciderPrio: true, deciderBudget: true, fenceDeparture: true, deadPeerProbe: true,
    heartbeat: true, parkOnCancel: true, bindAtStart: true, reserveLedger: true, evictSeesFreed: true,
    pinPlanned: true, atomic429: true, idempotency: true, twoLevel: true, keyedGrace: true,
  }
}
module design { import schedRules.*  import scheduler(RULES = DESIGN).* from "./scheduler" }
// today's halo_serve: one generator mutex, FIFO, each request to its end; no decider slot
module today { import schedRules.*  import scheduler(RULES = { ...DESIGN, preempt: false, deciderPrio: false }).* from "./scheduler" }
module noPreempt { import schedRules.*  import scheduler(RULES = { ...DESIGN, preempt: false }).* from "./scheduler" }
module noDeciderPrio { import schedRules.*  import scheduler(RULES = { ...DESIGN, deciderPrio: false }).* from "./scheduler" }
module strictDeciderPrio { import schedRules.*  import scheduler(RULES = { ...DESIGN, deciderBudget: false }).* from "./scheduler" }
module noFence { import schedRules.*  import scheduler(RULES = { ...DESIGN, fenceDeparture: false }).* from "./scheduler" }
module noDeadPeerProbe { import schedRules.*  import scheduler(RULES = { ...DESIGN, deadPeerProbe: false }).* from "./scheduler" }
module noHeartbeat { import schedRules.*  import scheduler(RULES = { ...DESIGN, heartbeat: false }).* from "./scheduler" }
module dropOnCancel { import schedRules.*  import scheduler(RULES = { ...DESIGN, parkOnCancel: false }).* from "./scheduler" }
// with idempotency a retry adopts its in-flight attempt, so the prefix it reuses cannot change between admission and
// start: bindAtStart protects keyless clients (decision D5), so its mutant runs without keys
module bindAtAdmission { import schedRules.*  import scheduler(RULES = { ...DESIGN, bindAtStart: false, idempotency: false }).* from "./scheduler" }
module memoryReadingOnly { import schedRules.*  import scheduler(RULES = { ...DESIGN, reserveLedger: false }).* from "./scheduler" }
module releasedStaysCharged { import schedRules.*  import scheduler(RULES = { ...DESIGN, evictSeesFreed: false }).* from "./scheduler" }
module evictPlanned { import schedRules.*  import scheduler(RULES = { ...DESIGN, pinPlanned: false }).* from "./scheduler" }
module queued429 { import schedRules.*  import scheduler(RULES = { ...DESIGN, atomic429: false }).* from "./scheduler" }
// keyless clients: also the instance on which the keyless paths (limit 429, a prefix bound later) are witnessed
module noIdempotency { import schedRules.*  import scheduler(RULES = { ...DESIGN, idempotency: false }).* from "./scheduler" }
// one-level fair queuing over sandbox principals (G3 as first designed): a chain that fans out to several sandboxes
// takes several shares
module oneLevelSfq { import schedRules.*  import scheduler(RULES = { ...DESIGN, twoLevel: false }).* from "./scheduler" }
// a keyed departure fenced at the next boundary (G3 as first built): a non-streaming caller whose answer outlasts its
// timeout retries with its key, finds its attempt fenced, and the server decodes the turn again
module noKeyedGrace { import schedRules.*  import scheduler(RULES = { ...DESIGN, keyedGrace: false }).* from "./scheduler" }
