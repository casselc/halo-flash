#include "train/eval_report.hpp"

#include "core/atomic_file.hpp"

namespace halo::train {

json::Value EvalReport::ToJson() const {
  using json::Value;
  Value v = Value::MakeObject();
  v.set("run_id", Value::String(run_id));
  v.set("variant", Value::String(variant));
  v.set("variant_version", Value::Int(variant_version));
  v.set("base_sha256", Value::String(base_sha256));
  v.set("dataset", Value::String(dataset));
  v.set("dataset_sha256", Value::String(dataset_sha256));
  v.set("metric", Value::String(metric));
  v.set("value", Value::Float(value));
  v.set("baseline", baseline ? Value::Float(*baseline) : Value::Null());
  v.set("n", Value::Int(static_cast<std::int64_t>(n)));
  v.set("step", Value::Int(step));
  v.set("engine", Value::String(engine));
  v.set("unix_ms", Value::Int(unix_ms));
  return v;
}

bool JsonlEvalSink::Report(const EvalReport& r, std::string* error) {
  return core::DurableAppend(path_, json::DumpPython(r.ToJson()) + "\n", error);
}

bool FanOutEvalSink::Report(const EvalReport& r, std::string* error) {
  bool ok = true;
  for (auto* s : sinks_) ok = s->Report(r, error) && ok;
  return ok;
}

model::VariantEval ToVariantEval(const EvalReport& r) {
  model::VariantEval e;
  e.name = r.dataset;
  e.metric = r.metric;
  e.dataset_sha256 = r.dataset_sha256;
  e.value = r.value;
  e.baseline = r.baseline;
  e.n = r.n;
  e.step = r.step;
  return e;
}

}  // namespace halo::train
