#include "scope/eval_stats.hpp"

#include <cmath>

// HALO_SCOPE_MUTANT selects a deliberately broken variant for the positive
// controls in CMakeLists.txt; 0 is the real code.
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

MeanSe MeanAndSe(const std::vector<double>& x) {
  MeanSe r;
  r.n = x.size();
  if (x.empty()) return r;
  long double sum = 0;
  for (double v : x) sum += v;
  const long double n = static_cast<long double>(x.size());
  const long double mean = sum / n;
  r.mean = static_cast<double>(mean);
  if (x.size() < 2) return r;
  // Corrected two-pass algorithm (Chan, Golub and LeVeque): the second sum
  // removes the rounding error left in the mean.
  long double ss = 0, comp = 0;
  for (double v : x) {
    const long double d = static_cast<long double>(v) - mean;
    ss += d * d;
    comp += d;
  }
#if HALO_SCOPE_MUTANT == 13  // positive control: population variance
  const long double var = (ss - comp * comp / n) / n;
#else
  const long double var = (ss - comp * comp / n) / (n - 1);
#endif
  const long double sd = std::sqrt(var > 0 ? var : 0);
  r.sd = static_cast<double>(sd);
  r.se = static_cast<double>(sd / std::sqrt(n));
  return r;
}

std::vector<ScoreSummary> Summarize(const std::vector<ScoreRow>& scores) {
  std::map<std::string, ScoreSummary> by;
  std::map<std::string, std::vector<double>> values;
  for (const auto& s : scores) {
    auto& sum = by[s.name];
    if (sum.name.empty()) {
      sum.name = s.name;
      sum.data_type = s.data_type;
    } else if (sum.data_type != s.data_type) {
      sum.data_type = "MIXED";
    }
    if (!s.value) {
      ++sum.errors;
      continue;
    }
    ++sum.n;
    if (s.data_type == "CATEGORICAL") ++sum.counts[s.string_value];
    else {
      values[s.name].push_back(*s.value);
      if (s.data_type == "BOOLEAN") ++sum.counts[*s.value != 0 ? "true" : "false"];
    }
  }
  std::vector<ScoreSummary> out;
  for (auto& [name, sum] : by) {
    if (sum.data_type == "NUMERIC" || sum.data_type == "BOOLEAN") {
      const auto ms = MeanAndSe(values[name]);
      sum.mean = ms.mean;
      sum.se = ms.se;
    }
    out.push_back(std::move(sum));
  }
  return out;
}

std::optional<Comparison> ComparePaired(const std::vector<ScoreRow>& a, const std::vector<ScoreRow>& b,
                                        const std::string& score, std::string* err) {
  auto collect = [&](const std::vector<ScoreRow>& rows, std::map<std::string, double>* out) -> bool {
    for (const auto& s : rows) {
      if (s.name != score) continue;
      if (s.data_type == "CATEGORICAL") return *err = "score " + score + " is categorical; no paired mean", false;
      if (s.value) (*out)[s.item_id] = *s.value;
    }
    return true;
  };
  std::map<std::string, double> va, vb;
  if (!collect(a, &va) || !collect(b, &vb)) return std::nullopt;
  Comparison c;
  c.score = score;
  std::vector<double> xa, xb, d;
  for (const auto& [item, x] : va) {
    const auto it = vb.find(item);
    if (it == vb.end()) {
      ++c.only_a;
      continue;
    }
    const double y = it->second;
    c.items.push_back({item, x, y, y - x});
    xa.push_back(x);
    xb.push_back(y);
    d.push_back(y - x);
    if (y > x) ++c.wins;
    else if (y < x) ++c.losses;
    else ++c.ties;
  }
  for (const auto& [item, y] : vb)
    if (!va.count(item)) ++c.only_b;
  c.n = d.size();
  c.mean_a = MeanAndSe(xa).mean;
  c.mean_b = MeanAndSe(xb).mean;
  const auto md = MeanAndSe(d);
  c.mean_delta = md.mean;
  c.se_delta = md.se;
  c.sd_delta = md.sd;
  return c;
}

}  // namespace halo::scope
