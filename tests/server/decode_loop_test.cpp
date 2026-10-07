// The server's decode loop (server/decode_loop.hpp) against a fake engine: speculative cycles must emit exactly
// what ordinary steps emit, record exactly what the engine ran (accepted drafts past a stop or a cancel
// included), never draft past the budget, and stop, cancel and refuse at the same token as without drafting.

#include <hegel/gtest.h>

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "engine/mtp_depth.hpp"
#include "server/decode_loop.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::server;

namespace {

constexpr std::int32_t kStop = 0;

// The model: a deterministic greedy continuation of everything run so far (a small vocabulary, so stops come).
std::int32_t Oracle(const std::vector<std::int32_t>& seq, std::uint32_t vocab, std::uint64_t salt) {
  std::uint64_t h = 1469598103934665603ULL ^ salt;
  for (std::size_t i = seq.size() > 3 ? seq.size() - 3 : 0; i < seq.size(); ++i)
    h = (h ^ static_cast<std::uint64_t>(seq[i] + 7)) * 1099511628211ULL;
  return static_cast<std::int32_t>((h >> 17) % vocab);
}

// A fake engine: runs tokens (its history), verifies drafts against the oracle like SpecStep's greedy cycle, and
// checks the loop's draft cap against the tokens the request may still emit.
struct FakeEngine {
  std::uint32_t vocab = 12;
  std::uint64_t salt = 0;
  double draft_right = 0.8;  // probability a draft equals the oracle's token
  std::mt19937_64 rng{1};
  std::vector<std::int32_t> history;  // the prompt, then every token run
  std::uint32_t budget = 0, emitted = 0;
  bool cap_violated = false;

  bool Step(std::int32_t token, std::uint32_t drafts, DecodeCycle* c) {
    if (1 + drafts > budget - emitted) cap_violated = true;
    history.push_back(token);
    c->tokens.clear();
    c->drafted = drafts;
    c->accepted = 0;
    if (drafts == 0) {
      c->tokens.push_back(Oracle(history, vocab, salt));
      return true;
    }
    // Drafts: the oracle's continuation of history, each wrong with probability 1 - draft_right.
    std::vector<std::int32_t> seq = history, d;
    std::bernoulli_distribution right(draft_right);
    for (std::uint32_t j = 0; j < drafts; ++j) {
      const std::int32_t y = Oracle(seq, vocab, salt);
      const std::int32_t dj = right(rng) ? y : static_cast<std::int32_t>((y + 1 + rng() % (vocab - 1)) % vocab);
      d.push_back(dj);
      seq.push_back(dj);
    }
    // Verify rows token, d1 .. dk: y_i follows history + d1..d_i; accept while d_{i+1} == y_i.
    std::vector<std::int32_t> run = history;
    for (std::uint32_t i = 0; i <= drafts; ++i) {
      const std::int32_t y = Oracle(run, vocab, salt);
      c->tokens.push_back(y);
      if (i == drafts || d[i] != y) break;
      ++c->accepted;
      run.push_back(d[i]);
    }
    history = run;  // the committed rows: token and the accepted drafts
    return true;
  }
};

struct Outcome {
  std::vector<std::int32_t> emitted, ran, engine_ran;
  DecodeOutcome out;
  SpecDecodeStats stats;
  bool cap_violated = false;
};

// `gone_after` / `refuse_at`: the requester leaves once that many tokens were emitted / emit refuses that token
// (1-based); nullopt: never.
Outcome Decode(bool spec, std::uint32_t budget, std::uint64_t salt, double right, std::uint32_t limit,
           std::optional<std::uint32_t> gone_after, std::optional<std::uint32_t> refuse_at, std::uint64_t seed) {
  FakeEngine e;
  e.salt = salt;
  e.draft_right = right;
  e.rng.seed(seed);
  e.budget = budget;
  e.history = {3, 5, 7};  // the prompt minus its last token
  const std::size_t prompt = e.history.size();
  Outcome r;
  DecodeIo io;
  io.step = [&](std::int32_t t, std::uint32_t k, DecodeCycle* c, std::string*) { return e.Step(t, k, c); };
  io.ran = [&](std::int32_t t) { r.ran.push_back(t); };
  io.gone = [&] { return gone_after && r.emitted.size() >= *gone_after; };
  io.is_stop = [](text::TokenId t) { return t == kStop; };
  io.emit = [&](text::TokenId t) {
    r.emitted.push_back(static_cast<std::int32_t>(t));
    ++e.emitted;
    return !(refuse_at && r.emitted.size() == *refuse_at);
  };
  engine::MtpDepthChooser chooser(engine::MtpMeasuredCosts(), {.limit = limit});
  r.out = RunDecode(9, budget, spec ? &chooser : nullptr, io, &r.stats);
  r.engine_ran.assign(e.history.begin() + static_cast<std::ptrdiff_t>(prompt), e.history.end());
  r.cap_violated = e.cap_violated;
  return r;
}

}  // namespace

TEST(DecodeLoop, SpeculativeCyclesEmitWhatStepsEmitAndRecordWhatRan) {
  hegel::test([](hegel::TestCase& tc) {
    const auto budget = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 40}));
    const auto salt = tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = 1000}));
    const double right = tc.draw(gs::sampled_from<double>({0.0, 0.5, 0.9, 1.0}));
    const auto limit = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 7}));
    const auto seed = tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = 1u << 30}));
    std::optional<std::uint32_t> gone_after, refuse_at;
    if (tc.draw(gs::booleans())) gone_after = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 40}));
    if (tc.draw(gs::booleans())) refuse_at = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
    const Outcome ar = Decode(false, budget, salt, right, limit, gone_after, refuse_at, seed);
    const Outcome sp = Decode(true, budget, salt, right, limit, gone_after, refuse_at, seed);
    ASSERT_TRUE(ar.out.error.empty()) << ar.out.error;
    ASSERT_TRUE(sp.out.error.empty()) << sp.out.error;
    // The same tokens, ending for the same reason.
    ASSERT_EQ(sp.emitted, ar.emitted);
    ASSERT_EQ(sp.out.completion, ar.out.completion);
    ASSERT_EQ(sp.out.stopped, ar.out.stopped);
    ASSERT_EQ(sp.out.cancelled, ar.out.cancelled);
    ASSERT_EQ(sp.out.refused, ar.out.refused);
    ASSERT_EQ(sp.out.length_limited, ar.out.length_limited);
    ASSERT_LE(sp.emitted.size(), budget);
    // No cycle drafted past what the request could still emit.
    ASSERT_FALSE(sp.cap_violated);
    // The history is exactly what the engine ran, in both modes.
    ASSERT_EQ(ar.ran, ar.engine_ran);
    ASSERT_EQ(sp.ran, sp.engine_ran);
    // What ran is the prompt's last token, then the emitted tokens (the last one emitted is not run unless it was
    // an accepted draft), then accepted drafts past a stop or a cancel.
    ASSERT_GE(sp.ran.size(), sp.emitted.size());
    if (!sp.ran.empty()) ASSERT_EQ(sp.ran.front(), 9);
    for (std::size_t i = 0; i < sp.emitted.size() && i + 1 < sp.ran.size(); ++i) ASSERT_EQ(sp.ran[i + 1], sp.emitted[i]);
    if (!sp.out.stopped && !sp.out.cancelled && !sp.out.refused) ASSERT_EQ(sp.ran.size(), sp.emitted.size());
    // ran_past_end counts exactly the endings inside an accepted run (drafts that ran but were not emitted).
    ASSERT_EQ(sp.stats.ran_past_end, sp.ran.size() > sp.emitted.size() + 1 ? 1u : 0u);
    ASSERT_LE(sp.stats.capped, sp.stats.depth[0] + sp.stats.cycles);
    // Counters add up.
    std::uint64_t steps = 0, at = 0;
    for (const auto d : sp.stats.depth) steps += d;
    for (const auto a : sp.stats.accepted_at) at += a;
    ASSERT_EQ(at, sp.stats.accepted);
    ASSERT_LE(sp.stats.accepted, sp.stats.drafted);
    ASSERT_EQ(steps - sp.stats.depth[0], sp.stats.cycles);
    ASSERT_EQ(ar.stats.cycles, 0u);
  }, {.test_cases = 300});
}

// Directed cases the property may reach rarely: a stop inside an accepted run (the drafts past it ran and are
// recorded), and a budget that ends inside what a deeper cycle would have emitted.
TEST(DecodeLoop, StopInsideAnAcceptedRunAndBudgetCut) {
  std::uint64_t past = 0, capped = 0;
  for (std::uint64_t salt = 0; salt < 400; ++salt) {
    const Outcome ar = Decode(false, 24, salt, 1.0, 4, std::nullopt, std::nullopt, salt);
    const Outcome sp = Decode(true, 24, salt, 1.0, 4, std::nullopt, std::nullopt, salt);
    ASSERT_EQ(sp.emitted, ar.emitted) << salt;
    ASSERT_EQ(sp.ran, sp.engine_ran) << salt;
    ASSERT_FALSE(sp.cap_violated) << salt;
    past += sp.stats.ran_past_end;
    capped += sp.stats.capped;
    if (sp.out.stopped) {
      // The stop's cycle may have run past it: everything recorded beyond emitted + 1 is accepted drafts.
      ASSERT_GE(sp.ran.size(), sp.emitted.size() + 1) << salt;
      ASSERT_EQ(std::vector<std::int32_t>(sp.ran.begin() + 1, sp.ran.begin() + 1 + static_cast<std::ptrdiff_t>(sp.emitted.size())),
                sp.emitted) << salt;
    }
  }
  // Positive controls: the directed runs reach both cases.
  EXPECT_GT(past, 0u) << "no stop landed inside an accepted run";
  EXPECT_GT(capped, 0u) << "the max_tokens cap never lowered a cycle's drafts";
  // Budget 1..6 with every draft right: cycles must shrink to fit.
  for (std::uint32_t b = 1; b <= 6; ++b) {
    const Outcome sp = Decode(true, b, 7, 1.0, 7, std::nullopt, std::nullopt, 1);
    ASSERT_TRUE(sp.out.error.empty()) << sp.out.error;
    ASSERT_FALSE(sp.cap_violated) << b;
    ASSERT_LE(sp.emitted.size(), b);
  }
}
