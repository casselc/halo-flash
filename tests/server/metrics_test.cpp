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

// MTP speculative decoding: vLLM's spec-decode counters summed over records, per-position acceptance and the
// halo depth histogram; requests without MTP add nothing.
TEST(Metrics, SpecDecodeCountersSumOverRequests) {
  Metrics m("flash");
  GenerationRecord r;
  r.finish_reason = "stop";
  m.OnGeneration(r);  // MTP off
  r.spec.cycles = 3;
  r.spec.drafted = 7;
  r.spec.accepted = 5;
  r.spec.catchup_tokens = 512;
  r.spec.depth = {2, 0, 2, 1, 0, 0, 0, 0};  // two ordinary steps, cycles of 2, 2 and 3 drafts
  r.spec.accepted_at = {3, 2, 0, 0, 0, 0, 0};
  m.OnGeneration(r);
  m.OnGeneration(r);
  const std::string t = m.Render();
  const auto has = [&](const std::string& line) { return t.find(line + "\n") != std::string::npos; };
  EXPECT_TRUE(has("vllm:spec_decode_num_drafts_total{model_name=\"flash\"} 6"));
  EXPECT_TRUE(has("vllm:spec_decode_num_draft_tokens_total{model_name=\"flash\"} 14"));
  EXPECT_TRUE(has("vllm:spec_decode_num_accepted_tokens_total{model_name=\"flash\"} 10"));
  EXPECT_TRUE(has("vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"flash\",position=\"0\"} 6"));
  EXPECT_TRUE(has("vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"flash\",position=\"1\"} 4"));
  EXPECT_TRUE(has("halo:mtp_decode_steps_total{model_name=\"flash\",depth=\"0\"} 4"));
  EXPECT_TRUE(has("halo:mtp_decode_steps_total{model_name=\"flash\",depth=\"3\"} 2"));
  EXPECT_TRUE(has("halo:mtp_catchup_tokens_total{model_name=\"flash\"} 1024"));
  EXPECT_TRUE(has("# TYPE vllm:spec_decode_num_draft_tokens_total counter"));
}
