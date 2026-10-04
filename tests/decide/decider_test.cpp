// End-to-end System One answers from the real strands-decider-2B-hobson-v19
// against the Python engine (tests/decide/data/engine.jsonl, CPU, FP32).
// Needs HALO_DECIDER_MODELS_DIR with Qwen3.5-2B-Base-BF16.gguf,
// strands-decider-lora-f32.gguf, strands-decider-head.gguf and
// strands-decider-tokenizer.gguf (see README).
//
// Tolerances fixed before the first run: every reported probability, noul,
// score and confidence within 0.01 of the reference; the same choice unless
// the reference's top-two margin is below 0.02; input token usage exact.

#include <hegel/gtest.h>

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

// The pointer head alone, against its definition evaluated in double:
// logit_k = <K(LN(o_k)), Q(LN(d))> / sqrt(dim), LN being torch's LayerNorm
// (biased variance, eps 1e-5). Without the 2B model this is the only test
// of the head (Mull found it untested).
TEST(StrandsDecider, PointerHeadMatchesItsDefinition) {
  namespace gs = hegel::generators;
  hegel::test([](hegel::TestCase& tc) {
    PointerHead h;
    h.hidden = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 6}));
    h.dim = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
    auto vec = [&](std::size_t n) {
      std::vector<float> v(n);
      for (auto& x : v) x = tc.draw(gs::floats<float>({.min_value = -2.0F, .max_value = 2.0F}));
      return v;
    };
    h.norm_w = vec(h.hidden);
    h.norm_b = vec(h.hidden);
    h.q_w = vec(h.dim * h.hidden);
    h.q_b = vec(h.dim);
    h.k_w = vec(h.dim * h.hidden);
    h.k_b = vec(h.dim);
    const auto decide = vec(h.hidden);
    std::vector<std::vector<float>> options(tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3})));
    for (auto& o : options) o = vec(h.hidden);

    auto ln = [&](const std::vector<float>& x) {
      double mean = 0, var = 0;
      for (float v : x) mean += v;
      mean /= x.size();
      for (float v : x) var += (v - mean) * (v - mean);
      var /= x.size();
      std::vector<double> y(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) y[i] = (x[i] - mean) / std::sqrt(var + 1e-5) * h.norm_w[i] + h.norm_b[i];
      return y;
    };
    auto affine = [&](const std::vector<float>& w, const std::vector<float>& b, const std::vector<double>& x) {
      std::vector<double> y(b.size());
      for (std::size_t r = 0; r < b.size(); ++r) {
        y[r] = b[r];
        for (std::size_t c = 0; c < x.size(); ++c) y[r] += static_cast<double>(w[r * x.size() + c]) * x[c];
      }
      return y;
    };
    const auto q = affine(h.q_w, h.q_b, ln(decide));
    const auto got = h.Logits(decide, options);
    ASSERT_EQ(got.size(), options.size());
    for (std::size_t k = 0; k < options.size(); ++k) {
      const auto key = affine(h.k_w, h.k_b, ln(options[k]));
      double dot = 0, mag = 0;
      for (std::uint32_t i = 0; i < h.dim; ++i) {
        dot += key[i] * q[i];
        mag += std::fabs(key[i] * q[i]);
      }
      const double want = dot / std::sqrt(static_cast<double>(h.dim));
      // FP32 LayerNorm and affine: relative to the terms' magnitude.
      ASSERT_NEAR(got[k], want, 1e-4 * (mag + 1.0)) << "option " << k;
    }
  });
}
