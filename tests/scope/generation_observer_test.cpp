// The OpenAI service's generation records as halo-scope observations: the
// field mapping, latency percentiles, capture policy for prompt and output,
// and the tool-call ids a harness's tool result links to (a full trace: the
// generation, then the harness's tool step, in one trace).

#include <gtest/gtest.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "halo_scope.h"
#include "scope/annotate.hpp"
#include "scope/c_api.hpp"
#include "scope/generation_observer.hpp"
#include "scope/recorder.hpp"

using namespace halo;
using namespace halo::scope;

namespace {

class Rows : public Sink {
 public:
  bool Write(Table table, std::string_view rows, std::size_t, std::string_view, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    rb::Reader r(rows);
    while (!r.done()) {
      if (table == Table::kObservations) observations.push_back(DecodeObservation(r));
      else if (table == Table::kPayloads) payloads.push_back(DecodePayload(r));
      else throw std::runtime_error("unexpected table");
    }
    return true;
  }
  std::mutex mu;
  std::vector<Observation> observations;
  std::vector<Payload> payloads;
};

server::GenerationRecord Record() {
  server::GenerationRecord r;
  r.api = "chat";
  r.request_id = "chatcmpl-1";
  r.trace_id = "tr-1";
  r.session_id = "sess";
  r.user = "u";
  r.model = "flash";
  r.start_ns = 1000;
  r.first_text_ns = 5000;
  r.end_ns = 9000;
  r.prompt_tokens = 300;
  r.cached_tokens = 200;
  r.completion_tokens = 11;
  r.gaps_us = {50, 10, 40, 20, 30, 60, 70, 80, 90, 100};
  r.finish_reason = "tool_calls";
  r.sampling.temperature = 0.6f;
  r.sampling.top_k = 20;
  r.sampling.seed = 7;
  r.max_tokens = 64;
  r.tool_call_ids = {"call_a"};
  r.tool_names = {"weather"};
  r.input = "<|im_start|>user\nWeather?<|im_end|>";
  r.output = "<tool_call>...</tool_call>";
  return r;
}

}  // namespace

TEST(ScopeGenerations, RecordBecomesATypedGenerationObservation) {
  const auto o = GenerationObserver::ToObservation(Record());
  EXPECT_EQ(o.trace_id, "tr-1");
  EXPECT_EQ(o.span_id, "chatcmpl-1");
  EXPECT_EQ(o.type, "GENERATION");
  EXPECT_EQ(o.name, "chat");
  EXPECT_EQ(o.session_id, "sess");
  EXPECT_EQ(o.completion_start_ns, 5000);
  EXPECT_EQ(o.input_tokens, 300u);
  EXPECT_EQ(o.cached_tokens, 200u);
  EXPECT_EQ(o.output_tokens, 11u);
  // Nearest rank over {10, 20, ..., 100}.
  EXPECT_EQ(o.tpot_p50_us, 50u);
  EXPECT_EQ(o.tpot_p90_us, 90u);
  EXPECT_EQ(o.tpot_p99_us, 100u);
  EXPECT_EQ(o.top_k, 20u);
  EXPECT_EQ(o.seed, 7u);
  EXPECT_EQ(o.finish_reason, "tool_calls");
  EXPECT_EQ(o.tool_call_ids, std::vector<std::string>{"call_a"});
  EXPECT_EQ(o.status_code, "OK");
  // Without a harness trace the request is its own trace.
  auto r = Record();
  r.trace_id.clear();
  EXPECT_EQ(GenerationObserver::ToObservation(r).trace_id, "chatcmpl-1");
  // Without MTP the usage is input / output / cache_read; with it the draft counts ride along.
  EXPECT_EQ(o.usage_details.size(), 3u);
  r.spec.depth[3] = 2;
  r.spec.cycles = 2;
  r.spec.drafted = 6;
  r.spec.accepted = 4;
  r.spec.catchup_tokens = 128;
  const auto u = GenerationObserver::ToObservation(r).usage_details;
  const std::vector<std::pair<std::string, std::uint64_t>> mtp{
      {"mtp_cycles", 2}, {"mtp_drafted", 6}, {"mtp_accepted", 4}, {"mtp_catchup", 128}};
  ASSERT_EQ(u.size(), 7u);
  EXPECT_TRUE(std::equal(mtp.begin(), mtp.end(), u.begin() + 3));
}

TEST(ScopeGenerations, CapturePolicyAndAHarnessToolStepJoinOneTrace) {
  for (const Capture c : {Capture::kOff, Capture::kDigest, Capture::kFull}) {
    Rows sink;
    RecorderOptions opt;
    opt.writer_thread = false;
    Recorder rec(sink, opt);
    Annotator a(rec, "proj", c, [] { return std::int64_t{20000}; });
    GenerationObserver obs(a);
    const auto r = Record();
    obs.OnGeneration(r);
    // The harness runs the tool and reports it against the generation's call id.
    ASSERT_EQ(a.ToolResult("tr-1", "call_a", "weather", Status::kOk, "18C", 1000), Outcome::kOk);
    rec.Flush();
    ASSERT_EQ(sink.observations.size(), 2u);
    const auto& g = sink.observations[0];
    const auto& tool = sink.observations[1];
    EXPECT_EQ(g.project, "proj");
    EXPECT_EQ(g.trace_id, tool.trace_id);
    EXPECT_EQ(g.tool_call_ids, tool.tool_call_ids);
    EXPECT_EQ(g.input_refs.empty(), c == Capture::kOff);
    EXPECT_EQ(g.output_refs.empty(), c == Capture::kOff);
    if (c != Capture::kOff) {
      EXPECT_EQ(g.input_refs[0], PayloadHash(r.input));
      EXPECT_EQ(g.input_bytes, r.input.size());
    }
    EXPECT_EQ(sink.payloads.size(), c == Capture::kFull ? 3u : 0u);  // prompt, output, tool output
  }
}
