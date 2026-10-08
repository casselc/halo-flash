#ifndef HALO_SERVER_DECIDER_GATE_HPP_
#define HALO_SERVER_DECIDER_GATE_HPP_

// G3 M4 (docs/SCHEDULER.md 4 and 11, D2): how /v1/systemone evaluations (strands-decider-2B, a separate model)
// share the GPU with Flash-Next.
//   kConcurrent  one evaluation at a time under the gate's own mutex, concurrently with the engine thread's
//                units (the two models' kernels interleave on the GPU; what that costs each is D2's measurement).
//   kScheduled   each evaluation is a scheduler unit on the engine thread (ScheduledGenerator::RunDecider): a
//                decider with credit runs at the next unit boundary (D4: burst 4, refill 2/s), one out of credit
//                takes the decider class's fair share; nothing else runs on the GPU meanwhile.
// Both modes measure the wait (arrival to the evaluation's start) and the evaluation itself for /metrics:
//   halo:decider_requests_total{mode}, halo:decider_wait_seconds, halo:decider_eval_seconds (histograms).

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "server/metrics.hpp"

namespace halo::server {

class DeciderGate {
 public:
  enum class Mode : std::uint8_t { kConcurrent, kScheduled };
  /// Runs `work` as a scheduled unit and returns once it ran (false with `error`: it did not run).
  using Runner = std::function<bool(const std::function<void()>& work, std::string* error)>;

  /// kScheduled needs `scheduled`; without one the gate is kConcurrent.
  explicit DeciderGate(Mode mode = Mode::kConcurrent, Runner scheduled = {});
  /// Runs one evaluation; false with `error` when it could not run (the scheduler stopped).
  bool Run(const std::function<void()>& work, std::string* error);
  [[nodiscard]] Mode mode() const { return mode_; }
  [[nodiscard]] static const char* Name(Mode m) { return m == Mode::kScheduled ? "scheduled" : "concurrent"; }
  /// Appends the gate's series (Metrics::AddSection).
  void Render(std::string& out, const std::string& labels) const;

  struct Stats {
    std::uint64_t runs = 0, failed = 0;
    Metrics::Histogram wait{{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0}, {}};
    Metrics::Histogram eval{{0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 5.0}, {}};
  };
  [[nodiscard]] Stats stats() const;

 private:
  const Mode mode_;
  Runner scheduled_;
  std::mutex run_mu_;  // kConcurrent: one evaluation at a time
  mutable std::mutex mu_;
  Stats stats_;  // guarded by mu_
};

}  // namespace halo::server

#endif  // HALO_SERVER_DECIDER_GATE_HPP_
