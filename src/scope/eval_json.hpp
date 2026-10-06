#ifndef HALO_SCOPE_EVAL_JSON_HPP_
#define HALO_SCOPE_EVAL_JSON_HPP_

// JSON views of datasets, runs, scores and comparisons, shared by the HTTP
// routes (eval_routes.hpp), the C ABI and the CLI.

#include <string>

#include "core/json.hpp"
#include "scope/eval_stats.hpp"
#include "scope/evals.hpp"

namespace halo::scope {

json::Value DatasetJson(const DatasetInfo& d);
json::Value ItemJson(const DatasetItem& it);
json::Value EvaluatorJson(const EvaluatorDef& e);
/// The run record, its progress and (with `summary`) the per-score summary.
json::Value RunJson(EvalStore& store, const RunRecord& r, bool summary);
json::Value RunItemsJson(EvalStore& store, const RunRecord& r);
json::Value SummaryJson(const std::vector<ScoreSummary>& s);
json::Value ComparisonJson(const Comparison& c, const RunRecord& a, const RunRecord& b, bool items);
/// A JSON number, or null when absent.
json::Value OptNumber(const std::optional<double>& v);

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVAL_JSON_HPP_
