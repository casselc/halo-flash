// Score aggregation and the paired run comparison: against reference values
// computed with Python's statistics module, against an independent
// reference implementation on generated data, and the algebraic properties
// a paired difference must have (antisymmetry, shift invariance).

#include <hegel/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "scope/eval_stats.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;

namespace {

ScoreRow Row(const std::string& item, const std::string& name, double v, const std::string& type = "NUMERIC") {
  ScoreRow r;
  r.item_id = item;
  r.name = name;
  r.data_type = type;
  r.value = v;
  return r;
}

// Independent reference: textbook formulas in long double, no compensation.
struct Ref {
  double mean = 0, se = 0;
};
Ref Reference(const std::vector<double>& x) {
  long double s = 0;
  for (double v : x) s += v;
  const long double n = static_cast<long double>(x.size());
  const long double m = s / n;
  long double ss = 0;
  for (double v : x) ss += (v - m) * (v - m);
  return {static_cast<double>(m), x.size() > 1 ? static_cast<double>(std::sqrt(ss / (n - 1)) / std::sqrt(n)) : 0};
}

bool Close(double a, double b, double rel = 1e-9, double abs = 1e-12) {
  return std::fabs(a - b) <= abs + rel * std::max(std::fabs(a), std::fabs(b));
}

}  // namespace

TEST(ScopeEvalStats, MatchesPythonStatistics) {
  // python3: statistics.mean / stdev(d) / sqrt(n) over the paired items
  const std::map<std::string, double> a = {{"i1", 0.2}, {"i2", 0.5}, {"i3", 0.9}, {"i4", 0.4}, {"i5", 0.7}, {"i6", 1.0}};
  const std::map<std::string, double> b = {{"i1", 0.3}, {"i2", 0.5}, {"i3", 0.6}, {"i4", 0.8}, {"i5", 0.9}, {"i7", 0.1}};
  std::vector<ScoreRow> ra, rb;
  for (const auto& [k, v] : a) ra.push_back(Row(k, "q", v));
  for (const auto& [k, v] : b) rb.push_back(Row(k, "q", v));
  ra.push_back(Row("i1", "other", 7));
  ScoreRow failed = Row("i8", "q", 0);
  failed.value.reset();  // a failed evaluation: neither paired nor unpaired
  rb.push_back(failed);
  std::string err;
  auto c = ComparePaired(ra, rb, "q", &err);
  ASSERT_TRUE(c) << err;
  EXPECT_EQ(c->n, 5u);
  EXPECT_EQ(c->only_a, 1u);
  EXPECT_EQ(c->only_b, 1u);
  EXPECT_TRUE(Close(*c->mean_delta, 0.08, 1e-12)) << *c->mean_delta;
  EXPECT_TRUE(Close(*c->se_delta, 0.11575836902790225, 1e-12)) << *c->se_delta;
  EXPECT_TRUE(Close(*c->mean_a, 0.54, 1e-12));
  EXPECT_TRUE(Close(*c->mean_b, 0.62, 1e-12));
  EXPECT_EQ(c->wins, 3u);
  EXPECT_EQ(c->losses, 1u);
  EXPECT_EQ(c->ties, 1u);

  // Large offsets keep their spread (python3: stdev([1e9+1..1e9+4]) / 2).
  const auto ms = MeanAndSe({1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4});
  EXPECT_EQ(*ms.mean, 1000000002.5);
  EXPECT_TRUE(Close(*ms.se, 0.6454972243679028, 1e-12));
  EXPECT_FALSE(MeanAndSe({3.0}).se);
  EXPECT_FALSE(MeanAndSe({}).mean);

  ra.push_back(Row("i1", "cat", 0, "CATEGORICAL"));
  EXPECT_FALSE(ComparePaired(ra, rb, "cat", &err));
}

TEST(ScopeEvalStats, PairedComparisonMatchesReference) {
  hegel::test(
      [](hegel::TestCase& tc) {
        const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 40}));
        const bool boolean = tc.draw(gs::booleans());
        std::vector<ScoreRow> a, b;
        std::map<std::string, std::pair<std::optional<double>, std::optional<double>>> model;
        for (int i = 0; i < n; ++i) {
          const std::string item = "it" + std::to_string(i);
          auto draw = [&]() -> std::optional<double> {
            const auto presence = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}));
            if (presence == 0) return std::nullopt;  // not scored in this run
            if (boolean) return tc.draw(gs::booleans()) ? 1.0 : 0.0;
            return tc.draw(gs::floats<double>(
                {.min_value = -1e3, .max_value = 1e3, .allow_nan = false, .allow_infinity = false}));
          };
          const auto x = draw(), y = draw();
          model[item] = {x, y};
          if (x) a.push_back(Row(item, "s", *x, boolean ? "BOOLEAN" : "NUMERIC"));
          if (y) b.push_back(Row(item, "s", *y, boolean ? "BOOLEAN" : "NUMERIC"));
          if (tc.draw(gs::booleans())) {  // a failed evaluation: no value, never paired
            ScoreRow f = Row(item + "f", "s", 0);
            f.value.reset();
            (tc.draw(gs::booleans()) ? a : b).push_back(f);
          }
        }
        std::string err;
        auto c = ComparePaired(a, b, "s", &err);
        if (!c) throw std::runtime_error(err);
        std::vector<double> d;
        std::size_t only_a = 0, only_b = 0, wins = 0, losses = 0;
        for (const auto& [item, xy] : model) {
          if (xy.first && xy.second) {
            d.push_back(*xy.second - *xy.first);
            wins += *xy.second > *xy.first;
            losses += *xy.second < *xy.first;
          } else if (xy.first) {
            ++only_a;
          } else if (xy.second) {
            ++only_b;
          }
        }
        if (c->n != d.size() || c->only_a != only_a || c->only_b != only_b || c->wins != wins ||
            c->losses != losses || c->ties != d.size() - wins - losses)
          throw std::runtime_error("counts differ from the reference");
        if (d.empty()) {
          if (c->mean_delta || c->se_delta) throw std::runtime_error("statistics on no pairs");
          return;
        }
        const Ref ref = Reference(d);
        if (!Close(*c->mean_delta, ref.mean, 1e-9, 1e-9)) throw std::runtime_error("mean differs from the reference");
        if (d.size() < 2) {
          if (c->se_delta) throw std::runtime_error("SE with one pair");
        } else if (!Close(*c->se_delta, ref.se, 1e-7, 1e-9)) {
          throw std::runtime_error("SE " + std::to_string(*c->se_delta) + " differs from the reference " +
                                   std::to_string(ref.se));
        }
        // Antisymmetry: swapping the runs negates the deltas, keeps the SE.
        auto r = ComparePaired(b, a, "s", &err);
        if (!r || r->n != c->n || r->wins != c->losses || r->losses != c->wins || r->only_a != c->only_b ||
            !Close(*r->mean_delta, -*c->mean_delta, 1e-12, 1e-12) ||
            (c->se_delta && !Close(*r->se_delta, *c->se_delta, 1e-12, 1e-12)))
          throw std::runtime_error("comparison is not antisymmetric");
        // Shifting run b by a constant shifts the mean delta and keeps the SE.
        if (!boolean) {
          auto shifted = b;
          for (auto& s : shifted)
            if (s.value) *s.value += 1000.0;
          auto sc = ComparePaired(a, shifted, "s", &err);
          if (!Close(*sc->mean_delta, *c->mean_delta + 1000.0, 1e-9, 1e-7) ||
              (c->se_delta && !Close(*sc->se_delta, *c->se_delta, 1e-6, 1e-7)))
            throw std::runtime_error("shift changed the SE");
        }
      },
      {.test_cases = 300});
}

TEST(ScopeEvalStats, SummaryMatchesModel) {
  hegel::test(
      [](hegel::TestCase& tc) {
        std::vector<ScoreRow> rows;
        std::map<std::string, std::vector<double>> numeric;
        std::map<std::string, std::size_t> errors;
        std::map<std::string, std::size_t> cats;
        const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 50}));
        for (int i = 0; i < n; ++i) {
          const auto kind = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
          const std::string item = "it" + std::to_string(i);
          if (kind == 0) {
            const double v = tc.draw(gs::integers<int>({.min_value = -5, .max_value = 5}));
            rows.push_back(Row(item, "num", v));
            numeric["num"].push_back(v);
          } else if (kind == 1) {
            const bool v = tc.draw(gs::booleans());
            rows.push_back(Row(item, "ok", v ? 1 : 0, "BOOLEAN"));
            numeric["ok"].push_back(v ? 1 : 0);
          } else if (kind == 2) {
            const auto v = tc.draw(gs::sampled_from<std::string>({"good", "bad", "ugly"}));
            ScoreRow r = Row(item, "cat", 0, "CATEGORICAL");
            r.string_value = v;
            rows.push_back(r);
            ++cats[v];
          } else {
            ScoreRow r = Row(item, "num", 0);
            r.value.reset();
            rows.push_back(r);
            ++errors["num"];
          }
        }
        for (const auto& s : Summarize(rows)) {
          if (s.name == "cat") {
            if (s.counts != cats || s.mean) throw std::runtime_error("categorical summary differs");
            continue;
          }
          const auto& v = numeric[s.name];
          if (s.n != v.size() || s.errors != errors[s.name]) throw std::runtime_error("counts differ for " + s.name);
          if (v.empty()) {
            if (s.mean) throw std::runtime_error("a mean of nothing");
            continue;
          }
          const Ref ref = Reference(v);
          if (!Close(*s.mean, ref.mean, 1e-12, 1e-12)) throw std::runtime_error("mean differs for " + s.name);
          if (v.size() > 1 && !Close(*s.se, ref.se, 1e-9, 1e-12)) throw std::runtime_error("SE differs for " + s.name);
        }
      },
      {.test_cases = 200});
}
