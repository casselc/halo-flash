#ifndef HALO_SERVER_METRICS_HPP_
#define HALO_SERVER_METRICS_HPP_

// GET /metrics in the Prometheus text format with vLLM's metric names
// (PLAN.md P7), so dashboards built for vLLM work unchanged:
//   vllm:num_requests_running                    gauge
//   vllm:prompt_tokens_total, vllm:generation_tokens_total   counters
//   vllm:request_success_total{finished_reason}  counter
//   vllm:time_to_first_token_seconds             histogram
//   vllm:time_per_output_token_seconds           histogram
//   vllm:e2e_request_latency_seconds             histogram
// plus halo:prefix_cache_hit_tokens_total (cached prompt tokens). MTP speculative decoding (docs/MTP.md) uses
// vLLM's spec-decode names:
//   vllm:spec_decode_num_drafts_total, vllm:spec_decode_num_draft_tokens_total,
//   vllm:spec_decode_num_accepted_tokens_total, vllm:spec_decode_num_accepted_tokens_per_pos_total{position}
// plus halo:mtp_decode_steps_total{depth} (decode steps by drafts run; 0: an ordinary step of a speculative
// request) and halo:mtp_catchup_tokens_total (prompt positions the draft block caught up on). Every series
// carries model_name.

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "server/http.hpp"
#include "server/observer.hpp"

namespace halo::server {

class Metrics : public GenerationObserver {
 public:
  explicit Metrics(std::string model_name) : model_(std::move(model_name)) {}
  void OnGeneration(const GenerationRecord& r) noexcept override;
  /// Requests in flight (the service brackets each generation).
  void Started() noexcept { running_.fetch_add(1); }
  void Finished() noexcept { running_.fetch_sub(1); }
  [[nodiscard]] std::string Render() const;
  void Register(HttpServer& server);

  struct Histogram {
    std::vector<double> bounds;  // upper bounds, seconds
    std::vector<std::uint64_t> counts;  // per bucket (not cumulative)
    double sum = 0;
    std::uint64_t count = 0;
    void Observe(double v);
  };

 private:
  const std::string model_;
  std::atomic<std::int64_t> running_{0};
  mutable std::mutex mu_;
  std::uint64_t prompt_tokens_ = 0, generation_tokens_ = 0, cached_tokens_ = 0;  // guarded by mu_
  std::map<std::string, std::uint64_t> finished_;                                // guarded by mu_
  SpecDecodeStats spec_;                                                         // guarded by mu_ (summed)
  Histogram ttft_{{0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
                   40.0, 80.0, 160.0, 640.0, 2560.0},
                  {}};
  Histogram tpot_{{0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
                   80.0},
                  {}};
  Histogram e2e_{{0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
                  960.0, 1920.0, 7680.0},
                 {}};
};

/// Forwards each record to several observers.
class FanOut : public GenerationObserver {
 public:
  void Add(GenerationObserver* o) { observers_.push_back(o); }
  void OnGeneration(const GenerationRecord& r) noexcept override {
    for (auto* o : observers_) o->OnGeneration(r);
  }

 private:
  std::vector<GenerationObserver*> observers_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_METRICS_HPP_
