#include "scope/eval_json.hpp"

#include <cmath>
#include <map>

namespace halo::scope {

namespace {

using json::Value;

Value ParsedOr(const std::string& text, Value dflt) {
  std::string err;
  auto v = json::Parse(text, &err);
  return v ? std::move(*v) : std::move(dflt);
}

Value Int(std::int64_t v) { return Value::Int(v); }
Value Uint(std::size_t v) { return Value::Int(static_cast<std::int64_t>(v)); }

}  // namespace

Value OptNumber(const std::optional<double>& v) {
  return v && std::isfinite(*v) ? Value::Float(*v) : Value::Null();
}

Value DatasetJson(const DatasetInfo& d) {
  Value o = Value::MakeObject();
  o.set("name", Value::String(d.name));
  o.set("description", Value::String(d.description));
  o.set("metadata", ParsedOr(d.metadata_json, Value::MakeObject()));
  o.set("created_at", Value::String(d.created_at));
  o.set("archived", Value::Bool(!d.archived_at.empty()));
  if (!d.archived_at.empty()) o.set("archived_at", Value::String(d.archived_at));
  Value vs = Value::MakeArray();
  for (const auto& v : d.versions) {
    Value x = Value::MakeObject();
    x.set("version", Int(v.version));
    x.set("frozen", Value::Bool(v.frozen));
    x.set("item_count", Int(v.item_count));
    x.set("created_at", Value::String(v.created_at));
    if (v.frozen) {
      x.set("content_hash", Value::String(v.content_hash));
      x.set("frozen_at", Value::String(v.frozen_at));
    }
    vs.push(std::move(x));
  }
  o.set("versions", std::move(vs));
  return o;
}

Value ItemJson(const DatasetItem& it) {
  Value o = Value::MakeObject();
  o.set("id", Value::String(it.id));
  o.set("input", ParsedOr(it.input_json, Value()));
  o.set("expected_output", ParsedOr(it.expected_json, Value()));
  o.set("metadata", ParsedOr(it.metadata_json, Value::MakeObject()));
  if (!it.source_trace_id.empty()) o.set("source_trace_id", Value::String(it.source_trace_id));
  if (!it.source_observation_id.empty()) o.set("source_observation_id", Value::String(it.source_observation_id));
  o.set("content_hash", Value::String(it.content_hash));
  return o;
}

Value EvaluatorJson(const EvaluatorDef& e) {
  Value o = Value::MakeObject();
  o.set("name", Value::String(e.name));
  o.set("version", Int(e.version));
  o.set("kind", Value::String(e.kind));
  o.set("config", ParsedOr(e.config_json, Value::MakeObject()));
  return o;
}

Value SummaryJson(const std::vector<ScoreSummary>& summaries) {
  Value arr = Value::MakeArray();
  for (const auto& s : summaries) {
    Value o = Value::MakeObject();
    o.set("name", Value::String(s.name));
    o.set("data_type", Value::String(s.data_type));
    o.set("n", Uint(s.n));
    o.set("errors", Uint(s.errors));
    o.set("mean", OptNumber(s.mean));
    o.set("se", OptNumber(s.se));
    if (!s.counts.empty()) {
      Value c = Value::MakeObject();
      for (const auto& [k, n] : s.counts) c.set(k, Uint(n));
      o.set("counts", std::move(c));
    }
    arr.push(std::move(o));
  }
  return arr;
}

Value RunJson(EvalStore& store, const RunRecord& r, bool summary) {
  Value o = Value::MakeObject();
  o.set("id", Value::String(r.id));
  o.set("name", Value::String(r.name));
  o.set("dataset", Value::String(r.dataset));
  o.set("dataset_version", Int(r.dataset_version));
  o.set("target", ParsedOr(r.target_json, Value::MakeObject()));
  o.set("evaluators", ParsedOr(r.evaluators_json, Value::MakeArray()));
  o.set("binding", Value::String(r.binding));
  o.set("status", Value::String(r.status));
  o.set("max_attempts", Int(r.max_attempts));
  o.set("metadata", ParsedOr(r.metadata_json, Value::MakeObject()));
  o.set("created_at", Value::String(r.created_at));
  o.set("updated_at", Value::String(r.updated_at));
  if (!r.completed_at.empty()) o.set("completed_at", Value::String(r.completed_at));
  const auto p = store.Progress(r);
  Value pj = Value::MakeObject();
  pj.set("total", Int(p.total));
  pj.set("done", Int(p.done));
  pj.set("running", Int(p.running));
  pj.set("error", Int(p.error));
  pj.set("exhausted", Int(p.exhausted));
  pj.set("pending", Int(p.pending));
  o.set("progress", std::move(pj));
  if (summary) o.set("scores", SummaryJson(Summarize(store.RunScores(r.id))));
  return o;
}

Value RunItemsJson(EvalStore& store, const RunRecord& r) {
  std::map<std::string, std::vector<const ScoreRow*>> by_item;
  const auto scores = store.RunScores(r.id);
  for (const auto& s : scores) by_item[s.item_id].push_back(&s);
  Value arr = Value::MakeArray();
  for (const auto& it : store.RunItems(r.id)) {
    Value o = Value::MakeObject();
    o.set("item_id", Value::String(it.item_id));
    o.set("status", Value::String(it.status));
    o.set("attempts", Int(it.attempts));
    o.set("trace_id", Value::String(it.trace_id));
    o.set("observation_id", Value::String(it.observation_id));
    o.set("output", Value::String(it.output));
    o.set("model", Value::String(it.model));
    o.set("finish_reason", Value::String(it.finish_reason));
    o.set("input_tokens", Int(it.input_tokens));
    o.set("output_tokens", Int(it.output_tokens));
    o.set("latency_ms", Value::Float(it.latency_ms));
    if (!it.error.empty()) o.set("error", Value::String(it.error));
    Value ss = Value::MakeArray();
    for (const ScoreRow* s : by_item[it.item_id]) {
      Value x = Value::MakeObject();
      x.set("name", Value::String(s->name));
      x.set("evaluator_version", Int(s->evaluator_version));
      x.set("data_type", Value::String(s->data_type));
      x.set("value", OptNumber(s->value));
      if (!s->string_value.empty()) x.set("string_value", Value::String(s->string_value));
      if (!s->comment.empty()) x.set("comment", Value::String(s->comment));
      if (!s->error.empty()) x.set("error", Value::String(s->error));
      x.set("score_id", Value::String(s->score_id));
      if (!s->judge_model.empty() || !s->judge_trace_id.empty()) {
        Value j = Value::MakeObject();
        j.set("model", Value::String(s->judge_model));
        if (!s->judge_prompt.empty()) {
          j.set("prompt", Value::String(s->judge_prompt));
          j.set("prompt_version", Int(s->judge_prompt_version));
        }
        j.set("trace_id", Value::String(s->judge_trace_id));
        j.set("raw_output", Value::String(s->raw_output));
        x.set("judge", std::move(j));
      }
      ss.push(std::move(x));
    }
    o.set("scores", std::move(ss));
    arr.push(std::move(o));
  }
  return arr;
}

Value ComparisonJson(const Comparison& c, const RunRecord& a, const RunRecord& b, bool items) {
  Value o = Value::MakeObject();
  o.set("score", Value::String(c.score));
  o.set("a", Value::String(a.id));
  o.set("a_name", Value::String(a.name));
  o.set("b", Value::String(b.id));
  o.set("b_name", Value::String(b.name));
  o.set("n", Uint(c.n));
  o.set("only_a", Uint(c.only_a));
  o.set("only_b", Uint(c.only_b));
  o.set("mean_a", OptNumber(c.mean_a));
  o.set("mean_b", OptNumber(c.mean_b));
  o.set("mean_delta", OptNumber(c.mean_delta));
  o.set("se_delta", OptNumber(c.se_delta));
  o.set("sd_delta", OptNumber(c.sd_delta));
  o.set("wins", Uint(c.wins));
  o.set("losses", Uint(c.losses));
  o.set("ties", Uint(c.ties));
  if (items) {
    Value arr = Value::MakeArray();
    for (const auto& d : c.items) {
      Value x = Value::MakeObject();
      x.set("item_id", Value::String(d.item_id));
      x.set("a", Value::Float(d.a));
      x.set("b", Value::Float(d.b));
      x.set("delta", Value::Float(d.delta));
      arr.push(std::move(x));
    }
    o.set("items", std::move(arr));
  }
  return o;
}

}  // namespace halo::scope
