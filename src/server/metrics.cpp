#include "server/metrics.hpp"

#include <cstdio>

#include "core/json.hpp"

namespace halo::server {

void Metrics::Histogram::Observe(double v) {
  if (counts.size() != bounds.size() + 1) counts.assign(bounds.size() + 1, 0);
  std::size_t i = 0;
  while (i < bounds.size() && v > bounds[i]) ++i;
  ++counts[i];
  sum += v;
  ++count;
}

void Metrics::OnGeneration(const GenerationRecord& r) noexcept {
  std::lock_guard<std::mutex> lk(mu_);
  prompt_tokens_ += r.prompt_tokens;
  generation_tokens_ += r.completion_tokens;
  cached_tokens_ += r.cached_tokens;
  ++finished_[r.finish_reason];
  if (r.first_text_ns > 0) ttft_.Observe(static_cast<double>(r.first_text_ns - r.start_ns) * 1e-9);
  for (auto gap : r.gaps_us) tpot_.Observe(gap * 1e-6);
  e2e_.Observe(static_cast<double>(r.end_ns - r.start_ns) * 1e-9);
}

namespace {

// Shortest round-trip form ("0.04", not "0.040000000000000001"), without
// Python's trailing ".0" on integers ("1", as Prometheus clients print).
std::string Num(double v) {
  std::string s = json::PythonFloatRepr(v);
  if (s.size() > 2 && s.ends_with(".0")) s.resize(s.size() - 2);
  return s;
}

void Hist(std::string& out, const char* name, const char* help, const std::string& labels,
          const Metrics::Histogram& h) {
  out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " histogram\n";
  std::uint64_t cum = 0;
  for (std::size_t i = 0; i < h.bounds.size(); ++i) {
    cum += i < h.counts.size() ? h.counts[i] : 0;
    out += std::string(name) + "_bucket{" + labels + ",le=\"" + Num(h.bounds[i]) + "\"} " + std::to_string(cum) + "\n";
  }
  out += std::string(name) + "_bucket{" + labels + ",le=\"+Inf\"} " + std::to_string(h.count) + "\n";
  out += std::string(name) + "_sum{" + labels + "} " + Num(h.sum) + "\n";
  out += std::string(name) + "_count{" + labels + "} " + std::to_string(h.count) + "\n";
}

}  // namespace

std::string Metrics::Render() const {
  const std::string labels = "model_name=\"" + model_ + "\"";
  std::string out;
  out += "# HELP vllm:num_requests_running Number of requests currently generating.\n"
         "# TYPE vllm:num_requests_running gauge\n";
  out += "vllm:num_requests_running{" + labels + "} " + std::to_string(running_.load()) + "\n";
  std::lock_guard<std::mutex> lk(mu_);
  const auto counter = [&](const char* name, const char* help, std::uint64_t v) {
    out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " counter\n";
    out += std::string(name) + "{" + labels + "} " + std::to_string(v) + "\n";
  };
  counter("vllm:prompt_tokens_total", "Prefill tokens processed.", prompt_tokens_);
  counter("vllm:generation_tokens_total", "Generation tokens processed.", generation_tokens_);
  counter("halo:prefix_cache_hit_tokens_total", "Prompt tokens served from engine state already computed.",
          cached_tokens_);
  out += "# HELP vllm:request_success_total Finished requests by finish reason.\n"
         "# TYPE vllm:request_success_total counter\n";
  for (const auto& [reason, n] : finished_)
    out += "vllm:request_success_total{" + labels + ",finished_reason=\"" + reason + "\"} " + std::to_string(n) + "\n";
  Hist(out, "vllm:time_to_first_token_seconds", "Time to first output text, seconds.", labels, ttft_);
  Hist(out, "vllm:time_per_output_token_seconds", "Time between output pieces, seconds.", labels, tpot_);
  Hist(out, "vllm:e2e_request_latency_seconds", "Request latency, seconds.", labels, e2e_);
  return out;
}

void Metrics::Register(HttpServer& server) {
  server.Route("GET", "/metrics", [this](const HttpRequest&) {
    HttpResponse r;
    r.headers.emplace_back("Content-Type", "text/plain; version=0.0.4");
    r.body = Render();
    return r;
  });
}

}  // namespace halo::server
