#include "server/decider_gate.hpp"

#include <chrono>

namespace halo::server {

namespace {
using Clock = std::chrono::steady_clock;
double S(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
}  // namespace

DeciderGate::DeciderGate(Mode mode, Runner scheduled)
    : mode_(scheduled ? mode : Mode::kConcurrent), scheduled_(std::move(scheduled)) {}

bool DeciderGate::Run(const std::function<void()>& work, std::string* error) {
  const auto arrived = Clock::now();
  Clock::time_point started{}, ended{};
  const auto timed = [&] {
    started = Clock::now();
    work();
    ended = Clock::now();
  };
  bool ok = true;
  if (mode_ == Mode::kScheduled) {
    ok = scheduled_(timed, error);
  } else {
    std::lock_guard<std::mutex> lk(run_mu_);
    timed();
  }
  ok = ok && ended != Clock::time_point{};
  std::lock_guard<std::mutex> lk(mu_);
  if (!ok) {
    ++stats_.failed;
    return false;
  }
  ++stats_.runs;
  stats_.wait.Observe(S(arrived, started));
  stats_.eval.Observe(S(started, ended));
  return true;
}

DeciderGate::Stats DeciderGate::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

void DeciderGate::Render(std::string& out, const std::string& labels) const {
  const Stats s = stats();
  const std::string l = labels + ",mode=\"" + Name(mode_) + "\"";
  out += "# HELP halo:decider_requests_total Decider evaluations (/v1/systemone) by outcome.\n"
         "# TYPE halo:decider_requests_total counter\n";
  out += "halo:decider_requests_total{" + l + ",outcome=\"ran\"} " + std::to_string(s.runs) + "\n";
  out += "halo:decider_requests_total{" + l + ",outcome=\"failed\"} " + std::to_string(s.failed) + "\n";
  RenderHistogram(out, "halo:decider_wait_seconds", "Decider arrival to its evaluation's start, seconds.", l, s.wait);
  RenderHistogram(out, "halo:decider_eval_seconds", "Decider evaluation, seconds.", l, s.eval);
}

}  // namespace halo::server
