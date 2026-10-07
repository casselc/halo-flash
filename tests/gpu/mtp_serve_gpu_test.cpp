// MTP speculative decoding through the OpenAI service on the real model (docs/MTP.md "Server integration" item 8).
// The small fixture has no draft block, so this is gated like MtpSpec.GreedyEqualsArOnRealModel:
//   HALO_FLASH_NEXT_GGUF  the trunk (first shard), HALO_FLASH_NEXT_MTP  the MTP sidecar,
//   HALO_MTP_VOCAB        optional trimmed draft head ids (int32).
// One engine (the model holds most of the memory): the same request sequence runs through a service whose generator
// decodes without MTP, then through one with it (window 0 and the full prompt), each with the exact prefix cache and
// deterministic ids. Every greedy reply (chat with and without thinking, /v1/completions, Responses chained by
// previous_response_id, stop strings, max_tokens 1..13) must be byte-identical apart from cached-token counts;
// requests with penalties must not draft; sampled requests without penalties must draft and succeed.
// Positive controls: drafts were accepted, a generation ended inside an accepted run (the engine ran drafts past
// the stop) and the max_tokens cap lowered a cycle's drafts.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_generator.hpp"
#include "server/http.hpp"
#include "server/observer.hpp"
#include "server/openai_service.hpp"
#include "text/tokenizer.hpp"

namespace halo {
namespace {

struct Req {
  std::string name, path, body;
  enum Check { kIdentical, kSampled } check = kIdentical;
  int chain = -1;  // index of the earlier request whose response id becomes previous_response_id
};

std::vector<Req> Battery() {
  const std::string gsm =
      "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she "
      "earn? Please reason step by step, and put your final answer within \\\\boxed{}.";
  const auto chat = [](const std::string& user, int max_tokens, const std::string& extra = "") {
    return R"({"messages": [{"role": "user", "content": ")" + user + R"("}], "max_tokens": )" +
           std::to_string(max_tokens) + R"(, "temperature": 0)" + extra + "}";
  };
  const std::string nothink = R"(, "reasoning_effort": "none")";
  std::vector<Req> r;
  r.push_back({"chat_gsm_think", "/v1/chat/completions", chat(gsm, 160)});
  r.push_back({"chat_capital_nothink", "/v1/chat/completions", chat("What is the capital of France? One word.", 32, nothink)});
  r.push_back({"chat_count_nothink", "/v1/chat/completions",
               chat("Count from 1 to 30, separated by commas, nothing else.", 120, nothink)});
  for (int n : {1, 2, 3, 5, 8, 13})
    r.push_back({"chat_cut_" + std::to_string(n), "/v1/chat/completions",
                 chat("Count from 1 to 30, separated by commas, nothing else.", n, nothink)});
  r.push_back({"chat_stop_string", "/v1/chat/completions",
               chat("Count from 1 to 30, separated by commas, nothing else.", 120, nothink + R"(, "stop": ["17"])")});
  r.push_back({"completion_raw", "/v1/completions",
               R"({"prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "max_tokens": 64, "temperature": 0, "stop": ["\n\n\n"]})"});
  const auto resp = [](const std::string& input, int max_tokens) {
    return R"({"input": ")" + input + R"(", "max_output_tokens": )" + std::to_string(max_tokens) +
           R"(, "temperature": 0, "store": true, "reasoning": {"effort": "none"}})";
  };
  const int base = static_cast<int>(r.size());
  r.push_back({"resp_1", "/v1/responses", resp("Name three prime numbers below 20, as a list.", 48)});
  r.push_back({"resp_2", "/v1/responses", resp("Now three more above 20, same format.", 48), Req::kIdentical, base});
  r.push_back({"resp_3", "/v1/responses", resp("What is the sum of all six?", 7), Req::kIdentical, base + 1});
  r.push_back({"chat_penalties", "/v1/chat/completions",
               chat("Count from 1 to 30, separated by commas, nothing else.", 40,
                    nothink + R"(, "temperature": 0.8, "seed": 7, "frequency_penalty": 0.5)")});
  r.push_back({"chat_sampled", "/v1/chat/completions",
               chat("Count from 1 to 30, separated by commas, nothing else.", 40,
                    nothink + R"(, "temperature": 0.8, "top_p": 0.95, "seed": 7)"),
               Req::kSampled});
  return r;
}

class Records : public server::GenerationObserver {
 public:
  void OnGeneration(const server::GenerationRecord& r) noexcept override { all.push_back(r); }
  std::vector<server::GenerationRecord> all;
};

struct Reply {
  int status = 0;
  std::string body;
  server::SpecDecodeStats spec;
};

std::string ResponseId(const std::string& body) {
  std::smatch m;
  return std::regex_search(body, m, std::regex(R"re("id"\s*:\s*"([^"]+)")re")) ? m[1].str() : std::string();
}

std::string WithoutCachedCounts(const std::string& body) {
  return std::regex_replace(body, std::regex(R"re("cached_tokens"\s*:\s*\d+)re"), "\"cached_tokens\":_");
}

std::vector<Reply> RunBattery(engine::FlashNextGpu& gpu, const text::Tokenizer& tok, const server::SpecOptions& spec,
                              const std::vector<Req>& reqs) {
  server::FlashNextGenerator gen(gpu, tok, 4096, {.mode = server::SessionCacheOptions::Mode::kExact}, spec);
  Records rec;
  server::ServiceOptions opt;
  opt.observer = &rec;
  int next_id = 0;
  opt.new_id = [&] { return "id" + std::to_string(next_id++); };
  opt.now_s = [] { return std::int64_t{1700000000}; };
  server::OpenAiService svc(gen, opt);
  std::vector<Reply> out;
  for (const auto& q : reqs) {
    server::HttpRequest h;
    h.method = "POST";
    h.target = h.path = q.path;
    h.body = q.body;
    if (q.chain >= 0)
      h.body.insert(1, R"("previous_response_id": ")" + ResponseId(out[static_cast<std::size_t>(q.chain)].body) + "\", ");
    const std::size_t before = rec.all.size();
    const server::HttpResponse r = q.path == "/v1/chat/completions" ? svc.Chat(h)
                                   : q.path == "/v1/responses"       ? svc.Responses(h)
                                                                     : svc.Completions(h);
    Reply y{r.status, r.body, {}};
    if (rec.all.size() == before + 1) y.spec = rec.all.back().spec;
    out.push_back(std::move(y));
  }
  return out;
}

TEST(MtpServe, GreedyRepliesIdenticalThroughTheServiceOnRealModel) {
  const char* path = std::getenv("HALO_FLASH_NEXT_GGUF");
  const char* mpath = std::getenv("HALO_FLASH_NEXT_MTP");
  if (!path || !mpath) GTEST_SKIP() << "set HALO_FLASH_NEXT_GGUF and HALO_FLASH_NEXT_MTP";
  std::string error;
  auto reader = core::GgufReader::OpenFile(path, &error);
  ASSERT_TRUE(reader) << error;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  ASSERT_TRUE(weights) << error;
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  ASSERT_TRUE(tok) << error;
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       c.ple_head_dim, t.type, &error, std::size_t{1} << 30);
  ASSERT_TRUE(ngram) << error;
  auto mreader = core::GgufReader::OpenFile(mpath, &error);
  ASSERT_TRUE(mreader) << error;
  auto mtp = model::MtpWeights::Bind(*mreader, c, &error);
  ASSERT_TRUE(mtp) << error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no GPU backend";
  std::vector<rt::FileSource> sources;
  for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  auto gpu = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, 4096, &error, sources);
  ASSERT_TRUE(gpu) << error;
  ASSERT_TRUE(gpu->LoadMtp(*mtp, &error)) << error;
  if (const char* v = std::getenv("HALO_MTP_VOCAB")) {
    std::ifstream f(v, std::ios::binary | std::ios::ate);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
    ASSERT_TRUE(gpu->SetDraftVocab(ids, &error)) << error;
  }

  const std::vector<Req> reqs = Battery();
  const std::vector<Reply> off = RunBattery(*gpu, *tok, {}, reqs);
  for (std::size_t i = 0; i < reqs.size(); ++i) {
    ASSERT_EQ(off[i].status, 200) << reqs[i].name << ": " << off[i].body;
    ASSERT_FALSE(off[i].spec.active()) << reqs[i].name;
  }
  for (const std::uint32_t window : {0u, 4096u}) {
    server::SpecOptions spec;
    spec.enabled = true;
    spec.window = window;
    const std::vector<Reply> on = RunBattery(*gpu, *tok, spec, reqs);
    server::SpecDecodeStats sum;
    bool length_capped = false;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
      const auto& q = reqs[i];
      ASSERT_EQ(on[i].status, 200) << q.name << ": " << on[i].body;
      const auto& s = on[i].spec;
      if (q.check == Req::kIdentical)
        EXPECT_EQ(WithoutCachedCounts(on[i].body), WithoutCachedCounts(off[i].body)) << q.name << ", window " << window;
      if (q.name == "chat_penalties") EXPECT_EQ(s.cycles, 0u) << "a request with penalties drafted";
      if (q.check == Req::kSampled) EXPECT_GT(s.cycles, 0u) << "a sampled request without penalties did not draft";
      std::printf("window %u %-22s cycles %3llu drafted %3llu accepted %3llu capped %llu ran-past-end %llu catch-up %llu\n",
                  window, q.name.c_str(), static_cast<unsigned long long>(s.cycles),
                  static_cast<unsigned long long>(s.drafted), static_cast<unsigned long long>(s.accepted),
                  static_cast<unsigned long long>(s.capped), static_cast<unsigned long long>(s.ran_past_end),
                  static_cast<unsigned long long>(s.catchup_tokens));
      sum.cycles += s.cycles;
      sum.accepted += s.accepted;
      sum.ran_past_end += q.check == Req::kIdentical ? s.ran_past_end : 0;
      length_capped = length_capped || (s.capped > 0 && q.check == Req::kIdentical);
      if (window == 0) EXPECT_EQ(s.catchup_tokens, 0u) << q.name;
    }
    EXPECT_GT(sum.accepted, sum.cycles / 2) << "window " << window << ": drafts hardly accepted";
    EXPECT_GT(sum.ran_past_end, 0u) << "window " << window << ": no greedy reply ended inside an accepted run";
    EXPECT_TRUE(length_capped) << "window " << window << ": the max_tokens cap never lowered a cycle";
  }
}

}  // namespace
}  // namespace halo
