// -*- mode: Bluespec; -*-
// operation_modes: operation.qnt's design, its worlds, and one mutant per rule (that rule off).
// design: executor restarts, platform crashes, every rule
module design {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// promised: no executor restart: nothing becomes unknown, so noOvercharge holds
module promised {
  import operation(RESTART = false, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// cacheLike: eviction under load: noOvercharge and promiseKept violated, core holds
module cacheLike {
  import operation(RESTART = false, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: false, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// noFence: fence off -> runsOnce: a stale-incarnation submit reruns an attempt settled as unknown
module noFence {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: false, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// noDeadline: deadline off -> runsOnce: a late submit is accepted after its tombstone was pruned
module noDeadline {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: false, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// pruneEarly: tomb off -> runsOnce: a duplicate submit is accepted after its tombstone was pruned early
module pruneEarly {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: false, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// dropExpiredOutcome: ackOutcome off -> neverAcceptedTrue: an expired outcome pruned unacknowledged is answered as closed
module dropExpiredOutcome {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: false, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// noReserve: reserve off -> promiseKept: results finishing together overflow the retention budget
module noReserve {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: false, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// ackBeforeCommit: commitThenAck off -> noReceiptLost: a platform crash between ack and commit loses the result
module ackBeforeCommit {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: false, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// cancelUntombed: cancelTomb off -> accounted: a late submit runs after a cancelled attempt was settled at 0
module cancelUntombed {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: false, refusalOpen: true, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// refusalCloses: refusalOpen off -> accounted: a refused copy settled at 0, then another copy runs
module refusalCloses {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: false, dedupReceipt: true, metaBound: true }).* from "./operation"
}
// noReceiptDedup: dedupReceipt off -> appliedOnce: a duplicate receipt is applied twice
module noReceiptDedup {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: false, metaBound: true }).* from "./operation"
}
// eraseForRoom: metaBound off -> neverAcceptedTrue: a tombstone erased to make room turns "it ran" into "closed"
module eraseForRoom {
  import operation(RESTART = true, PCRASH = true,
    RULES = { fence: true, deadline: true, tomb: true, ackOutcome: true, reserve: true, promise: true, commitThenAck: true, cancelTomb: true, refusalOpen: true, dedupReceipt: true, metaBound: false }).* from "./operation"
}
