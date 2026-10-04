// End-to-end System One answers from the real strands-decider-2B-hobson-v19
// against the Python engine (tests/decide/data/engine.jsonl, CPU, FP32).
// Needs HALO_DECIDER_MODELS_DIR with Qwen3.5-2B-Base-BF16.gguf,
// strands-decider-lora-f32.gguf, strands-decider-head.gguf and
// strands-decider-tokenizer.gguf (see README).
//
// Tolerances fixed before the first run: every reported probability, noul,
// score and confidence within 0.01 of the reference; the same choice unless
// the reference's top-two margin is below 0.02; input token usage exact.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "decide/decider.hpp"

using halo::json::Value;
using namespace halo::decide;

TEST(StrandsDecider, AnswersMatchPythonEngine) {
  const char* dir = std::getenv("HALO_DECIDER_MODELS_DIR");
  if (!dir) GTEST_SKIP() << "set HALO_DECIDER_MODELS_DIR";
  const std::string d = dir;
  std::string error;
  auto decider = StrandsDecider::Open({d + "/Qwen3.5-2B-Base-BF16.gguf", d + "/strands-decider-lora-f32.gguf",
                                       d + "/strands-decider-head.gguf", d + "/strands-decider-tokenizer.gguf",
                                       std::getenv("HALO_DECIDER_DEVICE") ? std::getenv("HALO_DECIDER_DEVICE") : "cpu"},
                                      &error);
  ASSERT_TRUE(decider) << error;
  std::ifstream f(std::string(HALO_DECIDE_TEST_DATA_DIR) + "/engine.jsonl");
  std::string line;
  int n = 0, failures = 0;
  const int limit = std::getenv("HALO_DECIDER_CASES") ? std::atoi(std::getenv("HALO_DECIDER_CASES")) : 1000;
  double worst = 0;
  while (std::getline(f, line) && n < limit) {
    ++n;
    auto row = halo::json::Parse(line, &error);
    ASSERT_TRUE(row) << error;
    auto body = halo::json::Parse(row->find("request")->str(), &error);
    auto req = ParseRequest(*body, &error);
    ASSERT_TRUE(req) << error;
    auto got = decider->Evaluate(*req, &error);
    ASSERT_TRUE(got) << error;
    const auto& want = *row->find("response");
    bool ok = got->find("usage")->find("input_tokens")->number() == want.find("usage")->find("input_tokens")->number();
    for (const auto& [name, wa] : want.find("answers")->members()) {
      const Value* ga = got->find("answers")->find(name);
      if (!ga) { ok = false; continue; }
      const auto check = [&](double g, double w) {
        worst = std::max(worst, std::fabs(g - w));
        ok = ok && std::fabs(g - w) <= 0.01;
      };
      if (const Value* v = wa.find("noul")) check(ga->find("noul")->number(), v->number());
      if (const Value* v = wa.find("score")) check(ga->find("score")->number(), v->number());
      if (const Value* v = wa.find("confidence")) check(ga->find("confidence")->number(), v->number());
      if (const Value* probs = wa.find("probabilities")) {
        std::vector<double> ps;
        for (const auto& [k, p] : probs->members()) {
          check(ga->find("probabilities")->find(k)->number(), p.number());
          ps.push_back(p.number());
        }
        if (const Value* c = wa.find("choice")) {
          std::sort(ps.rbegin(), ps.rend());
          if (ps.size() < 2 || ps[0] - ps[1] >= 0.02) ok = ok && ga->find("choice")->str() == c->str();
        }
      }
    }
    if (!ok && ++failures <= 5)
      ADD_FAILURE() << "case " << n << "\n got " << halo::json::DumpPython(*got) << "\nwant " << halo::json::DumpPython(want);
  }
  EXPECT_EQ(failures, 0) << "of " << n;
  std::printf("%d requests, worst absolute difference %.4g\n", n, worst);
}
