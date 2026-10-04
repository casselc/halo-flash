#ifndef HALO_SCOPE_GENERATION_OBSERVER_HPP_
#define HALO_SCOPE_GENERATION_OBSERVER_HPP_

// The OpenAI service's generations as halo-scope GENERATION observations:
// trace id from the harness (X-Halo-Trace-Id) or the request id, span id the
// request id, usage with cache reads, time to first text, per-piece latency
// percentiles, sampler settings, finish reason, and the tool-call ids a
// harness's halo_scope_tool_result links to. Prompt and output follow the
// annotator's capture policy.

#include "scope/annotate.hpp"
#include "server/observer.hpp"

namespace halo::scope {

class GenerationObserver : public server::GenerationObserver {
 public:
  explicit GenerationObserver(Annotator& annotator) : a_(annotator) {}
  void OnGeneration(const server::GenerationRecord& r) noexcept override;
  /// The observation a record becomes (exposed for tests).
  [[nodiscard]] static Observation ToObservation(const server::GenerationRecord& r);

 private:
  Annotator& a_;
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_GENERATION_OBSERVER_HPP_
