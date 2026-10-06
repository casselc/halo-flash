#ifndef HALO_SCOPE_EVAL_STATS_HPP_
#define HALO_SCOPE_EVAL_STATS_HPP_

// Score aggregation for experiment runs and the paired comparison of two runs
// (PLAN.md P12). Pure functions over committed score rows: a run's summary
// per score name (n, mean, standard error; categorical counts; failed
// evaluations counted apart, never as values), and per-item paired deltas
// between two runs on the same dataset with their mean and standard error,
// the statistic the quality gates in docs/BENCHMARKS.md use.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "scope/evals.hpp"

namespace halo::scope {

/// Mean and standard error of the mean of `x` (sample standard deviation,
/// n - 1). se is nullopt for n < 2; mean is nullopt for n == 0. Two-pass
/// in long double with a compensated correction, so shifted data keeps its
/// SE.
struct MeanSe {
  std::size_t n = 0;
  std::optional<double> mean, se, sd;
};
MeanSe MeanAndSe(const std::vector<double>& x);

struct ScoreSummary {
  std::string name, data_type;
  std::size_t n = 0;       // scored (value present)
  std::size_t errors = 0;  // evaluations that failed
  std::optional<double> mean, se;  // NUMERIC and BOOLEAN (as 0/1)
  std::map<std::string, std::size_t> counts;  // CATEGORICAL (string_value), BOOLEAN ("true"/"false")
};

/// One summary per score name, sorted by name. A name whose rows mix data
/// types is reported with data_type "MIXED" and no mean.
std::vector<ScoreSummary> Summarize(const std::vector<ScoreRow>& scores);

struct PairedDelta {
  std::string item_id;
  double a = 0, b = 0, delta = 0;  // delta = b - a
};

struct Comparison {
  std::string score;
  std::size_t n = 0;            // paired items (scored in both)
  std::size_t only_a = 0, only_b = 0;  // scored in one run only (unpaired; excluded)
  std::size_t wins = 0, losses = 0, ties = 0;  // b > a, b < a, equal
  std::optional<double> mean_a, mean_b;  // over the paired items
  std::optional<double> mean_delta, se_delta, sd_delta;
  std::vector<PairedDelta> items;  // by item id
};

/// Pairs the numeric (or boolean, as 0/1) scores named `score` of run a and
/// run b by item id. Returns nullopt with `*err` when the score is
/// categorical in either run.
std::optional<Comparison> ComparePaired(const std::vector<ScoreRow>& a, const std::vector<ScoreRow>& b,
                                        const std::string& score, std::string* err);

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVAL_STATS_HPP_
