// The System One decision layer against fixtures produced by the
// strands_decider package (tools/gen_decider_fixtures.py), plus Hegel
// properties for its invariants.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/gguf.hpp"
#include "decide/systemone.hpp"

namespace gs = hegel::generators;
using halo::json::Value;
using namespace halo::decide;

namespace {

std::vector<Value> LoadJsonl(const char* name) {
  std::ifstream f(std::string(HALO_DECIDE_TEST_DATA_DIR) + "/" + name);
  EXPECT_TRUE(f.good()) << name;
  std::vector<Value> rows;
  std::string line;
  while (std::getline(f, line)) {
    std::string error;
    auto v = halo::json::Parse(line, &error);
    EXPECT_TRUE(v) << error;
    if (v) rows.push_back(std::move(*v));
  }
  return rows;
}

Request MustParse(const std::string& raw) {
  std::string error;
  auto body = halo::json::Parse(raw, &error);
  EXPECT_TRUE(body) << error;
  auto r = ParseRequest(*body, &error);
  EXPECT_TRUE(r) << error << "\n" << raw;
  return r ? *r : Request{};
}

}  // namespace

TEST(SystemOne, ValidationMatchesPydantic) {
  int failures = 0;
  const auto rows = LoadJsonl("validate.jsonl");
  ASSERT_GT(rows.size(), 100U);
  for (const auto& row : rows) {
    const auto& raw = row.find("request")->str();
    std::string error;
    auto body = halo::json::Parse(raw, &error);
    const bool got = body && ParseRequest(*body, &error).has_value();
    if (got != row.find("ok")->boolean() && ++failures <= 10)
      ADD_FAILURE() << "pydantic " << (got ? "rejects" : "accepts") << ": " << raw << " (" << error << ")";
  }
  EXPECT_EQ(failures, 0);
}

TEST(SystemOne, RenderingMatchesPythonByteForByte) {
  int failures = 0;
  const auto rows = LoadJsonl("render.jsonl");
  ASSERT_GT(rows.size(), 400U);
  for (const auto& row : rows) {
    const Request r = MustParse(row.find("request")->str());
    if (RenderState(r.state) != row.find("state_text")->str() && ++failures <= 5)
      ADD_FAILURE() << "state\n got: " << RenderState(r.state) << "\nwant: " << row.find("state_text")->str();
    const auto& want_qs = row.find("questions")->items();
    ASSERT_EQ(want_qs.size(), r.questions.size());
    for (std::size_t i = 0; i < want_qs.size(); ++i) {
      const auto& w = want_qs[i];
      const auto rq = RenderQuestion(r.questions[i]);
      bool ok = r.questions[i].name == w.find("name")->str() && KindName(rq.kind) == w.find("kind")->str() &&
                rq.text == w.find("text")->str();
      const auto& labels = w.find("slot_labels")->items();
      const auto& descs = w.find("slot_descriptions")->items();
      const auto& spans = w.find("option_spans")->items();
      ok = ok && labels.size() == rq.slot_labels.size() && spans.size() == rq.option_spans.size();
      for (std::size_t k = 0; ok && k < labels.size(); ++k) {
        ok = labels[k].str() == rq.slot_labels[k] && descs[k].str() == rq.slot_descriptions[k] &&
             spans[k].items()[0].number() == double(rq.option_spans[k].first) &&
             spans[k].items()[1].number() == double(rq.option_spans[k].second);
      }
      if (!ok && ++failures <= 5)
        ADD_FAILURE() << "question " << w.find("name")->str() << "\n got: " << rq.text << "\nwant: " << w.find("text")->str();
    }
  }
  EXPECT_EQ(failures, 0);
}

TEST(SystemOne, ReadoutMatchesPython) {
  int failures = 0;
  const auto rows = LoadJsonl("readout.jsonl");
  ASSERT_GT(rows.size(), 500U);
  for (const auto& row : rows) {
    RenderedQuestion rq;
    const auto& kind = row.find("kind")->str();
    rq.kind = kind == "noul" ? Kind::kNoul : kind == "choice" ? Kind::kChoice : Kind::kScore;
    for (const auto& v : row.find("slot_labels")->items()) rq.slot_labels.push_back(v.str());
    for (const auto& v : row.find("slot_descriptions")->items()) rq.slot_descriptions.push_back(v.str());
    std::vector<float> logits;
    for (const auto& v : row.find("logits")->items()) logits.push_back(static_cast<float>(v.number()));
    const auto probs = SlotProbabilities(logits, static_cast<float>(row.find("temperature")->number()));
    const auto got = AnswerToJson(ReadAnswer(rq, probs, row.find("ordinal_smoothing")->number()));
    const auto& want = *row.find("answer");
    // Values are rounded to 4 decimals on both sides; FP32 summation order
    // can move a value across a rounding boundary, so allow one unit.
    bool ok = got.members().size() == want.members().size();
    for (const auto& [k, wv] : want.members()) {
      const Value* gv = got.find(k);
      if (!gv) { ok = false; break; }
      if (wv.is(Value::Kind::kObject)) {
        for (const auto& [kk, wvv] : wv.members()) {
          const Value* g2 = gv->find(kk);
          ok = ok && g2 && (wvv.is(Value::Kind::kString) ? g2->str() == wvv.str()
                                                         : std::fabs(g2->number() - wvv.number()) <= 1.0001e-4);
        }
      } else if (wv.is(Value::Kind::kString)) {
        ok = ok && gv->str() == wv.str();
      } else {
        ok = ok && std::fabs(gv->number() - wv.number()) <= 1.0001e-4;
      }
    }
    if (!ok && ++failures <= 5)
      ADD_FAILURE() << "got " << halo::json::DumpPython(got) << "\nwant " << halo::json::DumpPython(want);
  }
  EXPECT_EQ(failures, 0);
}

// Properties.

TEST(SystemOne, SlotProbabilitiesAreADistributionAndShiftInvariant) {
  hegel::test([](hegel::TestCase& tc) {
    const auto logits = tc.draw(gs::vectors(gs::floats<float>({.min_value = -50, .max_value = 50}),
                                            {.min_size = 2, .max_size = 255}));
    const auto t = tc.draw(gs::floats<float>({.min_value = 0.05F, .max_value = 5.0F}));
    const auto p = SlotProbabilities(logits, t);
    double sum = 0;
    for (float v : p) {
      ASSERT_GE(v, 0.0F);
      ASSERT_LE(v, 1.0F);
      sum += v;
    }
    ASSERT_NEAR(sum, 1.0, 1e-4);
    auto shifted = logits;
    for (auto& v : shifted) v += 7.0F;
    const auto q = SlotProbabilities(shifted, t);
    for (std::size_t i = 0; i < p.size(); ++i) ASSERT_NEAR(p[i], q[i], 1e-4);
  });
}

TEST(SystemOne, ScoreReadsTheSameUnderAnyRenderingOrder) {
  hegel::test([](hegel::TestCase& tc) {
    Question q;
    q.kind = Kind::kScore;
    q.instructions = Value::String("rate");
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 2, .max_value = 10}));
    for (std::size_t i = 0; i < n; ++i) q.criteria.emplace_back("", Value::String("level " + std::to_string(i)));
    std::vector<float> by_level(n);
    for (auto& v : by_level) v = tc.draw(gs::floats<float>({.min_value = 0.0F, .max_value = 1.0F}));
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    for (std::size_t i = n - 1; i > 0; --i)
      std::swap(order[i], order[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = i}))]);
    const auto canonical = RenderQuestion(q);
    const auto permuted = RenderQuestion(q, &order);
    std::vector<float> permuted_probs(n);
    for (std::size_t i = 0; i < n; ++i) permuted_probs[i] = by_level[order[i]];
    const auto a = ReadAnswer(canonical, by_level, 0.1);
    const auto b = ReadAnswer(permuted, permuted_probs, 0.1);
    ASSERT_NEAR(a.score, b.score, 2e-4);
    ASSERT_EQ(a.legend, b.legend);
    ASSERT_NEAR(a.confidence, b.confidence, 2e-4);
  });
}

TEST(SystemOne, ConfidenceBounds) {
  hegel::test([](hegel::TestCase& tc) {
    auto p = tc.draw(gs::vectors(gs::floats<double>({.min_value = 0.0, .max_value = 1.0}), {.min_size = 1, .max_size = 12}));
    const double c = DeriveConfidence(p);
    ASSERT_GE(c, 0.0);
    ASSERT_LE(c, 1.0);
    const double s = DeriveScoreConfidence(p, tc.draw(gs::floats<double>({.min_value = 0.0, .max_value = 0.3})));
    ASSERT_GE(s, 0.0);
    ASSERT_LE(s, 1.0);
  });
}

TEST(SystemOne, ParsedRequestsAlwaysRender) {
  hegel::test([](hegel::TestCase& tc) {
    // Arbitrary JSON values in each field: either rejected, or renderable.
    const auto pick = [&](int depth) {
      switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}))) {
        case 0: return Value::String(tc.draw(gs::text({.max_size = 20})));
        case 1: return Value::Int(tc.draw(gs::integers<int>()));
        case 2: return Value::Null();
        case 3: return Value::MakeArray({Value::String("a"), Value::String(tc.draw(gs::text({.max_size = 5})))});
        default: return depth ? Value::String("x") : Value::MakeObject({{"true", Value::String("y")}, {"b", Value::Null()}});
      }
    };
    Value q = Value::MakeObject();
    if (tc.draw(gs::booleans()))
      q.set("type", Value::String(tc.draw(gs::sampled_from<std::string>({"noul", "choice", "score", "other"}))));
    q.set("instructions", pick(0));
    if (tc.draw(gs::booleans())) q.set("criteria", pick(0));
    Value body = Value::MakeObject();
    body.set("state", pick(0));
    Value qs = Value::MakeObject();
    qs.set("q", q);
    body.set("questions", qs);
    std::string error;
    auto r = ParseRequest(body, &error);
    if (!r) return;
    (void)RenderState(r->state);
    for (const auto& question : r->questions) {
      const auto rq = RenderQuestion(question);
      ASSERT_EQ(rq.slot_labels.size(), rq.option_spans.size());
      ASSERT_GE(rq.slot_labels.size(), 2U);
      for (const auto& [s, e] : rq.option_spans) ASSERT_LE(e, CodepointLength(rq.text));
    }
  }, {.test_cases = 500});
}

TEST(SystemOne, WindowFittingMatchesEngine) {
  const char* path = std::getenv("HALO_DECIDER_TOKENIZER_GGUF");
  if (!path) GTEST_SKIP() << "set HALO_DECIDER_TOKENIZER_GGUF";
  std::string error;
  auto reader = halo::core::GgufReader::OpenFile(path, &error);
  ASSERT_TRUE(reader) << error;
  auto tok = halo::text::Tokenizer::FromGguf(*reader, &error);
  ASSERT_TRUE(tok) << error;
  int failures = 0;
  const auto rows = LoadJsonl("tokens.jsonl");
  ASSERT_GT(rows.size(), 400U);
  for (const auto& row : rows) {
    const Request r = MustParse(row.find("request")->str());
    std::vector<RenderedQuestion> rqs;
    for (const auto& q : r.questions) rqs.push_back(RenderQuestion(q));
    auto fit = FitTokens(*tok, RenderState(r.state), rqs, {}, &error);
    ASSERT_TRUE(fit) << error;
    const auto ids_of = [](const Value& v) {
      std::vector<std::size_t> out;
      for (const auto& x : v.items()) out.push_back(static_cast<std::size_t>(x.number()));
      return out;
    };
    std::vector<std::size_t> state(fit->state_ids.begin(), fit->state_ids.end());
    bool ok = state == ids_of(*row.find("state_ids"));
    const auto& qids = row.find("question_ids")->items();
    const auto& opt = row.find("option_index")->items();
    for (std::size_t q = 0; ok && q < qids.size(); ++q) {
      std::vector<std::size_t> got(fit->question_ids[q].begin(), fit->question_ids[q].end());
      ok = got == ids_of(qids[q]) && fit->option_index[q] == ids_of(opt[q]);
    }
    if (!ok && ++failures <= 5) ADD_FAILURE() << "request " << row.find("request")->str().substr(0, 200);
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

// --- Gaps Mull found ----------------------------------------------------------

// A window so small that the options are cut away is an error, not an answer
// read from arbitrary tokens.
TEST(SystemOne, WindowTooSmallForTheOptionsIsAnError) {
  const char* path = std::getenv("HALO_DECIDER_TOKENIZER_GGUF");
  if (!path) GTEST_SKIP() << "set HALO_DECIDER_TOKENIZER_GGUF";
  std::string error;
  auto reader = halo::core::GgufReader::OpenFile(path, &error);
  ASSERT_TRUE(reader) << error;
  auto tok = halo::text::Tokenizer::FromGguf(*reader, &error);
  ASSERT_TRUE(tok) << error;
  const Request r = MustParse(
      R"({"state":"s","questions":{"q":{"instructions":"pick","criteria":{"alpha":"first","beta":"second"}}}})");
  const std::vector<RenderedQuestion> rqs = {RenderQuestion(r.questions[0])};
  EXPECT_TRUE(FitTokens(*tok, RenderState(r.state), rqs, {}, &error)) << error;
  error.clear();
  EXPECT_FALSE(FitTokens(*tok, RenderState(r.state), rqs, {.max_length = 8, .max_question_fraction = 0.5}, &error));
  EXPECT_NE(error.find("option"), std::string::npos) << error;
}

// The response body, key by key, as strands-decider serializes it.
TEST(SystemOne, AnswerToJsonShapes) {
  Answer noul{.kind = Kind::kNoul, .noul = 0.25};
  EXPECT_EQ(halo::json::DumpPython(AnswerToJson(noul)), R"({"type": "noul", "noul": 0.25})");
  Answer choice{.kind = Kind::kChoice, .choice = "b", .probabilities = {{"a", 0.25}, {"b", 0.75}}, .confidence = 0.5};
  EXPECT_EQ(halo::json::DumpPython(AnswerToJson(choice)),
            R"({"type": "choice", "choice": "b", "probabilities": {"a": 0.25, "b": 0.75}, "confidence": 0.5})");
  Answer score{.kind = Kind::kScore, .score = 1.5, .legend = {{"0", "low"}, {"1", "mid"}},
               .probabilities = {{"0", 0.5}, {"1", 0.5}}, .confidence = 0.0};
  EXPECT_EQ(halo::json::DumpPython(AnswerToJson(score)),
            R"({"type": "score", "score": 1.5, "legend": {"0": "low", "1": "mid"}, )"
            R"("probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.0})");
}

// Python's max() keeps the first of equal probabilities.
TEST(SystemOne, ChoiceTiesGoToTheFirstOption) {
  RenderedQuestion rq;
  rq.kind = Kind::kChoice;
  rq.slot_labels = {"x", "y", "z"};
  rq.slot_descriptions = {"", "", ""};
  const std::vector<float> probs = {0.25F, 0.375F, 0.375F};
  EXPECT_EQ(ReadAnswer(rq, probs, 0.0).choice, "y");
}

// str.isspace(), from CPython: [c for c in range(0x110000) if chr(c).isspace()].
TEST(SystemOne, PyIsSpaceMatchesCPython) {
  const std::vector<char32_t> spaces = {0x9,    0xa,    0xb,    0xc,    0xd,    0x1c,   0x1d,   0x1e,
                                        0x1f,   0x20,   0x85,   0xa0,   0x1680, 0x2000, 0x2001, 0x2002,
                                        0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
                                        0x2028, 0x2029, 0x202f, 0x205f, 0x3000};
  for (char32_t c = 0; c <= 0x3001; ++c) {
    const bool want = std::find(spaces.begin(), spaces.end(), c) != spaces.end();
    ASSERT_EQ(PyIsSpace(c), want) << std::hex << static_cast<std::uint32_t>(c);
  }
}

// Option and level counts are validated at exactly their limits.
TEST(SystemOne, CriteriaCountLimits) {
  auto choice = [](int n) {
    std::string crit;
    for (int i = 0; i < n; ++i) crit += (i ? "," : "") + std::string("\"o") + std::to_string(i) + "\":null";
    return R"({"state":"s","questions":{"q":{"type":"choice","instructions":"i","criteria":{)" + crit + "}}}}";
  };
  auto score = [](int n) {
    std::string crit;
    for (int i = 0; i < n; ++i) crit += (i ? "," : "") + std::string("\"l") + std::to_string(i) + "\"";
    return R"({"state":"s","questions":{"q":{"type":"score","instructions":"i","criteria":[)" + crit + "]}}}";
  };
  auto accepts = [](const std::string& raw) {
    std::string error;
    auto body = halo::json::Parse(raw, &error);
    return body && ParseRequest(*body, &error).has_value();
  };
  EXPECT_FALSE(accepts(choice(1)));
  EXPECT_TRUE(accepts(choice(2)));
  EXPECT_TRUE(accepts(choice(255)));
  EXPECT_FALSE(accepts(choice(256)));
  EXPECT_FALSE(accepts(score(1)));
  EXPECT_TRUE(accepts(score(2)));
  EXPECT_TRUE(accepts(score(10)));
  EXPECT_FALSE(accepts(score(11)));
  // Null criteria render as label-only options.
  const Request r = MustParse(choice(2));
  ASSERT_EQ(r.questions[0].criteria.size(), 2U);
  EXPECT_FALSE(r.questions[0].criteria[0].second.has_value());
  const Request n = MustParse(R"({"state":"s","questions":{"q":{"instructions":"i","criteria":{"true":null}}}})");
  ASSERT_EQ(n.questions[0].criteria.size(), 1U);
  EXPECT_FALSE(n.questions[0].criteria[0].second.has_value());
}

TEST(SystemOne, ScoreConfidenceEdges) {
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{0.0, 0.0, 0.0}, 0.1), 0.0);  // no mass
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{1.0}, 0.1), 1.0);
  EXPECT_EQ(DeriveConfidence(std::vector<double>{1.0}), 1.0);
  // Smoothing at least the widest possible spread: confident iff the spread
  // is within it, including exactly at it.
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{0.5, 0.0, 0.5}, 1.0), 1.0);
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{1.0, 0.0, 0.0}, 4.0), 1.0);
  // Unnormalized mass is normalized first.
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{0.0, 0.0, 2.0}, 0.0), 1.0);
  EXPECT_EQ(DeriveScoreConfidence(std::vector<double>{1.0, 0.0, 1.0}, 0.0), 0.0);
}
