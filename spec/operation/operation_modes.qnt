// -*- mode: Bluespec; -*-
// operation_modes: operation.qnt's design, its worlds and its mutants.
//  - design: restarts, a retention promise, every rule;
//  - promised: no restart, a promise: nothing becomes unknown, so noOvercharge holds;
//  - cacheLike: no promise (eviction under load): noOvercharge is violated (a forgotten result is settled at the
//    worst case), safety still holds;
//  - noFence: a late submit from an earlier incarnation starts the attempt again after it was settled as
//    unknown -> runsOnce, accounted;
//  - noTomb: a duplicate submit of a captured attempt runs it again -> runsOnce;
//  - abandonUntombed: an abandoned attempt's late submit runs after it was settled at 0 -> accounted.
//  - cacheNoTomb: an evicted result is forgotten outright, so the executor answers "never accepted" for an attempt
//    that ran: "I no longer know" told as "never accepted" -> neverAcceptedTrue.
module design {
  import operation(RULES = { fence: true, tomb: true, promise: true, abandonTomb: true }, RESTART = true).* from "./operation"
}
module promised {
  import operation(RULES = { fence: true, tomb: true, promise: true, abandonTomb: true }, RESTART = false).* from "./operation"
}
module cacheLike {
  import operation(RULES = { fence: true, tomb: true, promise: false, abandonTomb: true }, RESTART = false).* from "./operation"
}
module noFence {
  import operation(RULES = { fence: false, tomb: true, promise: true, abandonTomb: true }, RESTART = true).* from "./operation"
}
module noTomb {
  import operation(RULES = { fence: true, tomb: false, promise: true, abandonTomb: true }, RESTART = false).* from "./operation"
}
module abandonUntombed {
  import operation(RULES = { fence: true, tomb: true, promise: true, abandonTomb: false }, RESTART = false).* from "./operation"
}
module cacheNoTomb {
  import operation(RULES = { fence: true, tomb: false, promise: false, abandonTomb: true }, RESTART = false).* from "./operation"
}
