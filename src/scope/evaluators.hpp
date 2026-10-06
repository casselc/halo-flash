#ifndef HALO_SCOPE_EVALUATORS_HPP_
#define HALO_SCOPE_EVALUATORS_HPP_

// halo-scope evaluators (PLAN.md P12): each turns one run item's output into
// one score row. Kinds and their configs (JSON objects):
//
//   exact        {trim?=true, collapse_whitespace?, case_insensitive?, first_line?, json?}
//                BOOLEAN: output equals the expected output (a string; with
//                json, both parse and are equal as JSON).
//   regex        {pattern? | from_expected?, full_match?, case_insensitive?, negate?}
//                BOOLEAN: ECMAScript search (or full match) on the output.
//   json_schema  {schema, strip_fences?=true}
//                BOOLEAN: the output parses as JSON and satisfies the schema
//                (json_schema.hpp's subset); the comment names the violation.
//   numeric      {abs_tol?=0, rel_tol?=0, pick?="last"|"first"}
//                BOOLEAN: a number in the output is within
//                max(abs_tol, rel_tol*|expected|) of the expected number.
//   code         {hook}  a function registered in-process (C ABI:
//                halo_scope_evals_register_hook); returns any data type.
//   llm_judge    {prompt: {name, version|label}, model, endpoint?, score: {type:
//                numeric|boolean|categorical, min?, max?, categories?},
//                temperature?=0, max_tokens?=512, request?}
//                The judge prompt version is rendered ({{input}}, {{output}},
//                {{expected}}, {{metadata}}, and the input object's keys) and
//                sent to an OpenAI-compatible /v1/chat/completions (the engine's
//                own by default); the reply must contain a JSON object
//                {"score": ..., "reasoning"?: ...}. The judge's model, prompt
//                version, trace id and raw reply are kept with the score.
//   decider      {question, state?, endpoint?, model?, threshold?}
//                POST /v1/systemone with the question (strands-decider schema)
//                over a state (default {input, output, expected}; a string
//                state is a template). noul: NUMERIC P(true), or BOOLEAN with
//                threshold; score: NUMERIC level; choice: CATEGORICAL.
//
// Outcomes: kScored (a value), kFailed (deterministic: a missing expected
// output, an unparseable judge reply; recorded with its error and no value,
// so it is never averaged), kTransient (the endpoint was unreachable or
// returned 429/5xx: the item is not committed and is retried on resume).

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "core/json.hpp"
#include "scope/evals.hpp"
#include "scope/http_client.hpp"

namespace halo::scope {

struct EvalContext {
  const DatasetItem* item = nullptr;
  json::Value input, expected, metadata;  // parsed from the item
  std::string output;                     // the target's output text
  std::string run_id;
  int attempt = 0;
};

struct EvalOutcome {
  enum class Status { kScored, kFailed, kTransient } status = Status::kFailed;
  ScoreRow row;  // data_type, value, string_value, comment, error, judge_*, raw_output
  std::string transient_error;

  static EvalOutcome Numeric(double v, std::string comment = {});
  static EvalOutcome Boolean(bool v, std::string comment = {});
  static EvalOutcome Categorical(std::string v, std::string comment = {});
  static EvalOutcome Failed(std::string data_type, std::string error);
  static EvalOutcome Transient(std::string error);
};

/// In-process code evaluators, by hook name. Thread-safe.
class CodeHooks {
 public:
  using Fn = std::function<EvalOutcome(const EvalContext&)>;
  void Register(const std::string& name, Fn fn);
  std::optional<Fn> Find(const std::string& name) const;

 private:
  mutable std::mutex mu_;
  std::map<std::string, Fn> hooks_;
};

struct EvalEnv {
  JsonPoster* poster = nullptr;
  std::string default_endpoint;  // OpenAI-compatible base URL for judges without one
  std::string decider_endpoint;  // base URL serving /v1/systemone (default: default_endpoint)
  HttpHeaders headers;           // added to every judge/decider request (e.g. Authorization)
  EvalStore* store = nullptr;    // judge prompts
  const CodeHooks* hooks = nullptr;
};

/// Validates an evaluator `kind` and `config_json` and returns the config to
/// store: canonical JSON, with a judge's prompt label resolved to the version
/// it points at now (an evaluator version never changes meaning).
std::optional<std::string> PrepareEvaluatorConfig(EvalStore& store, const std::string& kind,
                                                  const std::string& config_json, std::string* err);

/// The trace id a judge or decider call of `evaluator` for this attempt uses.
std::string EvaluatorTraceId(std::string_view run_id, std::string_view item_id, int attempt,
                             std::string_view evaluator);

/// Runs one evaluator version on one item. Fills row.name/evaluator/version.
EvalOutcome RunEvaluator(const EvaluatorDef& def, const EvalContext& ctx, const EvalEnv& env);

/// {{name}} substitution (spaces inside the braces allowed). Unknown
/// variables are an error, so a typo cannot render silently.
std::optional<std::string> RenderTemplate(const std::string& tmpl, const std::map<std::string, std::string>& vars,
                                          std::string* err);

/// The variables a dataset item offers to templates: input, expected,
/// metadata (strings as-is, other JSON compact), output when given, and each
/// key of an object input.
std::map<std::string, std::string> TemplateVars(const EvalContext& ctx, bool with_output);

/// Pulls the judge's verdict out of a reply: the last JSON object in the
/// text (after any <think> block) that has a "score" member.
std::optional<json::Value> ExtractVerdict(const std::string& reply);

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVALUATORS_HPP_
