// /metrics: vLLM-named counters and histograms from generation records, in
// the Prometheus text format (cumulative buckets, +Inf equal to the count).

#include <gtest/gtest.h>

#include <string>

#include "server/metrics.hpp"

using namespace halo::server;

TEST(Metrics, CountersAndHistogramsInPrometheusText) {
  Metrics m("flash");
  GenerationRecord r;
  r.start_ns = 0;
  r.first_text_ns = 30'000'000;  // 30 ms
  r.end_ns = 900'000'000;        // 0.9 s
  r.prompt_tokens = 100;
  r.cached_tokens = 60;
  r.completion_tokens = 4;
  r.gaps_us = {20000, 30000, 60000};
  r.finish_reason = "stop";
  m.OnGeneration(r);
  r.finish_reason = "length";
  r.first_text_ns = 0;  // no text: no time-to-first-token sample
  m.OnGeneration(r);
  m.Started();
  const std::string t = m.Render();
  const auto has = [&](const std::string& line) { return t.find(line + "\n") != std::string::npos; };
  EXPECT_TRUE(has("vllm:num_requests_running{model_name=\"flash\"} 1"));
  EXPECT_TRUE(has("vllm:prompt_tokens_total{model_name=\"flash\"} 200"));
  EXPECT_TRUE(has("vllm:generation_tokens_total{model_name=\"flash\"} 8"));
  EXPECT_TRUE(has("halo:prefix_cache_hit_tokens_total{model_name=\"flash\"} 120"));
  EXPECT_TRUE(has("vllm:request_success_total{model_name=\"flash\",finished_reason=\"stop\"} 1"));
  EXPECT_TRUE(has("vllm:request_success_total{model_name=\"flash\",finished_reason=\"length\"} 1"));
  // One TTFT sample of 30 ms: below 0.02 none, at 0.04 one (cumulative).
  EXPECT_TRUE(has("vllm:time_to_first_token_seconds_bucket{model_name=\"flash\",le=\"0.02\"} 0"));
  EXPECT_TRUE(has("vllm:time_to_first_token_seconds_bucket{model_name=\"flash\",le=\"0.04\"} 1"));
  EXPECT_TRUE(has("vllm:time_to_first_token_seconds_count{model_name=\"flash\"} 1"));
  // Six gap samples (3 per request): 20, 30, 60 ms twice.
  EXPECT_TRUE(has("vllm:time_per_output_token_seconds_bucket{model_name=\"flash\",le=\"0.025\"} 2"));
  EXPECT_TRUE(has("vllm:time_per_output_token_seconds_bucket{model_name=\"flash\",le=\"0.05\"} 4"));
  EXPECT_TRUE(has("vllm:time_per_output_token_seconds_bucket{model_name=\"flash\",le=\"+Inf\"} 6"));
  EXPECT_TRUE(has("vllm:e2e_request_latency_seconds_count{model_name=\"flash\"} 2"));
  EXPECT_TRUE(has("# TYPE vllm:e2e_request_latency_seconds histogram"));
  m.Finished();
  EXPECT_NE(m.Render().find("vllm:num_requests_running{model_name=\"flash\"} 0\n"), std::string::npos);
}
