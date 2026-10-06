#include "scope/runner.hpp"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

// HALO_SCOPE_MUTANT selects a deliberately broken variant for the positive
// controls in CMakeLists.txt; 0 is the real code.
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

namespace {

using json::Value;
using Kind = json::Value::Kind;

std::string Str(const Value& c, const char* key) {
  const Value* v = c.find(key);
  return v && v->is(Kind::kString) ? v->str() : std::string();
}

std::string ScoreSource(const std::string& kind) {
  if (kind == "llm_judge") return "judge";
  if (kind == "decider") return "decider";
  if (kind == "code") return "code";
  return "rule";
}

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

struct Shared {
  EvalStore& store;
  JsonPoster& poster;
  const RunnerOptions& opt;
  const RunRecord& run;
  Value target;
  std::vector<EvaluatorDef> evaluators;
  std::vector<DatasetItem> items;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> crashed{false};
  std::mutex mu;  // error, on_progress
  std::string error;
};

void Fault(Shared& s, std::string_view point, const std::string& item) {
  if (s.opt.fault) s.opt.fault(point, item);
}

// Mirrors an attempt's scores into chDB as llm_scores facts on the
// attempt's trace, with the run context as a trace fact.
void Mirror(Shared& s, const DatasetItem& item, const ItemStart& start, const std::vector<ScoreRow>& rows,
            const ChatReply& reply) {
  const int attempt = start.attempt;
  if (Annotator* a = s.opt.annotator) {
    TraceFact t;
    t.trace_id = start.trace_id;
    t.name = s.run.name;
    t.session_id = s.run.id;
    t.tags = {"experiment"};
    t.metadata = {{"run_id", s.run.id},
                  {"dataset", s.run.dataset},
                  {"dataset_version", std::to_string(s.run.dataset_version)},
                  {"item_id", item.id},
                  {"attempt", std::to_string(attempt)},
                  {"model", Str(s.target, "model")}};
    if (const std::string v = Str(s.target, "variant"); !v.empty()) t.metadata.emplace_back("variant", v);
    (void)a->RecordTrace(std::move(t));
    const std::int64_t now = NowNs();
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& r = rows[i];
      if (!r.value) continue;  // a failed evaluation has no value to assert
      Score f;
      f.score_id = r.score_id;
      f.trace_id = start.trace_id;
      f.observation_id = reply.id;
      f.session_id = s.run.id;
      f.name = r.name;
      f.source = ScoreSource(s.evaluators[i].kind);
      f.data_type = r.data_type;
      f.value = *r.value;
      f.string_value = r.string_value;
      f.comment = r.comment;
      f.event_ns = now;
      (void)a->RecordScore(std::move(f));
    }
  }
}

// One item, start to checkpoint. Throws SimulatedCrash through.
void RunItem(Shared& s, const DatasetItem& item) {
  std::string err;
  Fault(s, "begin", item.id);
  auto start = s.store.BeginItem(s.run, item, &err);
  if (!start) {
    std::lock_guard<std::mutex> lk(s.mu);
    s.error = err;
    return;
  }
  if (start->state != ItemStart::State::kStarted) return;
  const int attempt = start->attempt;
  Fault(s, "started", item.id);
  auto fail = [&](const std::string& why) {
    std::string e2;
    if (!s.store.FailItem(s.run, item.id, attempt, why, &e2)) {
      std::lock_guard<std::mutex> lk(s.mu);
      s.error = e2;
    }
  };

  auto body = BuildTargetRequest(s.store, s.target, item, &err);
  if (!body) return fail("cannot build the request: " + err);
  TargetCall call;
  call.run_id = s.run.id;
  call.item_id = item.id;
  call.trace_id = start->trace_id;
  call.attempt = attempt;
  call.body = std::move(*body);
  call.headers = s.opt.target_headers;
  call.headers.emplace_back("X-Halo-Trace-Id", start->trace_id);
  call.headers.emplace_back("X-Halo-Session-Id", s.run.id);
  if (const Value* p = s.target.find("prompt"); p && p->is(Kind::kObject))
    call.headers.emplace_back("X-Halo-Prompt", Str(*p, "name") + "@" + p->find("version")->int_digits());
  if (const std::string v = Str(s.target, "variant"); !v.empty()) call.headers.emplace_back("X-Halo-Variant-Id", v);
  ChatReply reply;
  if (s.opt.target) {
    reply = s.opt.target(call);
  } else {
    std::string url = Str(s.target, "endpoint");
    if (url.empty()) url = s.opt.target_endpoint;
    if (url.empty()) return fail("no target endpoint");
    reply = CallChat(s.poster, url, call.body, call.headers);
  }
  Fault(s, "generated", item.id);
  if (reply.status != ChatReply::Status::kOk) return fail("target: " + reply.error);

  EvalContext ctx;
  ctx.item = &item;
  ctx.run_id = s.run.id;
  ctx.attempt = attempt;
  ctx.output = reply.text;
  for (auto [text, out] : {std::pair{&item.input_json, &ctx.input}, std::pair{&item.expected_json, &ctx.expected},
                           std::pair{&item.metadata_json, &ctx.metadata}}) {
    auto v = json::Parse(*text, &err);
    *out = v ? std::move(*v) : Value();
  }
  std::vector<ScoreRow> rows;
  for (const auto& def : s.evaluators) {
    EvalOutcome o = RunEvaluator(def, ctx, s.opt.env);
    Fault(s, "evaluated", item.id);
    if (o.status == EvalOutcome::Status::kTransient) return fail(o.transient_error);
    o.row.score_id = ScoreFactId(s.run.id, item.id, attempt, def.name);
    rows.push_back(std::move(o.row));
  }

  // Mirror before the commit: a crash between the two leaves facts on an
  // attempt's trace that the ledger never counted (an honest record of an
  // abandoned attempt), never a counted item without its facts.
#if HALO_SCOPE_MUTANT != 14  // positive control: mirror after the commit
  Mirror(s, item, *start, rows, reply);
  Fault(s, "mirrored", item.id);
#endif

  ItemResult result;
  result.observation_id = reply.id;
  result.output = reply.text;
  result.model = reply.model;
  result.finish_reason = reply.finish_reason;
  result.input_tokens = static_cast<std::int64_t>(reply.prompt_tokens);
  result.output_tokens = static_cast<std::int64_t>(reply.completion_tokens);
  result.latency_ms = reply.latency_ms;
  if (!s.store.CompleteItem(s.run, item.id, attempt, result, rows, &err)) {
    std::lock_guard<std::mutex> lk(s.mu);
    s.error = err;
    return;
  }
  Fault(s, "committed", item.id);
#if HALO_SCOPE_MUTANT == 14
  Mirror(s, item, *start, rows, reply);
  Fault(s, "mirrored", item.id);
#endif
}

void Worker(Shared& s) {
  while (!s.crashed.load()) {
    if (s.opt.cancel && s.opt.cancel->load()) return;
    const std::size_t i = s.next.fetch_add(1);
    if (i >= s.items.size()) return;
    try {
      RunItem(s, s.items[i]);
    } catch (const SimulatedCrash&) {
      s.crashed.store(true);
      return;
    }
    if (s.opt.on_progress) {
      const auto p = s.store.Progress(s.run);
      std::lock_guard<std::mutex> lk(s.mu);
      s.opt.on_progress(p);
    }
  }
}

}  // namespace

std::string ScoreFactId(std::string_view run_id, std::string_view item_id, int attempt, std::string_view name) {
  const std::string a = std::to_string(attempt);
  return FramedHash({"score.v1", run_id, item_id, a, name}).substr(0, 16);
}

std::optional<json::Value> BuildTargetRequest(EvalStore& store, const json::Value& target, const DatasetItem& item,
                                              std::string* err) {
  EvalContext ctx;
  ctx.item = &item;
  for (auto [text, out] : {std::pair{&item.input_json, &ctx.input}, std::pair{&item.expected_json, &ctx.expected},
                           std::pair{&item.metadata_json, &ctx.metadata}}) {
    auto v = json::Parse(*text, err);
    if (!v) return std::nullopt;
    *out = std::move(*v);
  }
  Value body = Value::MakeObject();
  if (const Value* req = target.find("request"); req && req->is(Kind::kObject))
    for (const auto& [k, v] : req->members()) body.set(k, v);
  body.set("model", Value::String(Str(target, "model")));
  Value messages = Value::MakeArray();
  if (const Value* p = target.find("prompt"); p && p->is(Kind::kObject)) {
    const Value* ver = p->find("version");
    auto pv = store.PromptByVersion(Str(*p, "name"), ver ? static_cast<std::int64_t>(ver->number()) : 0);
    if (!pv) return *err = "prompt " + Str(*p, "name") + " not found", std::nullopt;
    auto text = RenderTemplate(pv->template_text, TemplateVars(ctx, false), err);
    if (!text) return std::nullopt;
    auto pc = json::Parse(pv->config_json, err);
    if (pc && pc->is(Kind::kObject))
      if (const Value* sys = pc->find("system"); sys && sys->is(Kind::kString)) {
        Value m = Value::MakeObject();
        m.set("role", Value::String("system"));
        m.set("content", Value::String(sys->str()));
        messages.push(std::move(m));
      }
    Value m = Value::MakeObject();
    m.set("role", Value::String("user"));
    m.set("content", Value::String(*text));
    messages.push(std::move(m));
  } else if (ctx.input.is(Kind::kString)) {
    Value m = Value::MakeObject();
    m.set("role", Value::String("user"));
    m.set("content", ctx.input);
    messages.push(std::move(m));
  } else if (const Value* ms = ctx.input.find("messages"); ms && ms->is(Kind::kArray)) {
    messages = *ms;
  } else {
    return *err = "without a prompt, the item input must be a string or {\"messages\": [...]}", std::nullopt;
  }
  body.set("messages", std::move(messages));
  if (const Value* params = target.find("params"); params && params->is(Kind::kObject)) {
    for (const auto& [k, v] : params->members()) {
      if (k == "seed" && v.is(Kind::kInt)) {
        // A per-item seed, fixed across attempts: a resumed item samples as
        // the first attempt did.
        const std::uint64_t h = std::strtoull(FramedHash({"seed.v1", item.id}).substr(0, 8).c_str(), nullptr, 16);
        const auto base = static_cast<std::uint64_t>(std::llabs(static_cast<long long>(v.number())));
        body.set("seed", Value::Int(static_cast<std::int64_t>((base + h) & 0x7fffffffu)));
      } else {
        body.set(k, v);
      }
    }
  }
  body.set("stream", Value::Bool(false));
  return body;
}

RunOutcome ExecuteRun(EvalStore& store, JsonPoster& poster, const std::string& run_id, const RunnerOptions& opt) {
  RunOutcome out;
  std::string err;
  auto run = store.GetRun(run_id);
  if (!run) return out.status = "refused", out.error = "no run " + run_id, out;
  auto binding = store.RecomputeBinding(*run, &err);
  if (!binding || *binding != run->binding) {
    out.status = "refused";
    out.error = binding ? "run " + run_id + ": stored content no longer matches its binding" : err;
    return out;
  }
  auto target = json::Parse(run->target_json, &err);
  auto evals = json::Parse(run->evaluators_json, &err);
  if (!target || !evals) return out.status = "refused", out.error = err, out;
  Shared s{store, poster, opt, *run, std::move(*target), {}, {}, {}, {}, {}, {}};
  for (const auto& e : evals->items()) {
    auto def = store.GetEvaluator(e.find("name")->str(), static_cast<std::int64_t>(e.find("version")->number()));
    if (!def) return out.status = "refused", out.error = "evaluator " + e.find("name")->str() + " missing", out;
    if (def->kind == "code") {
      auto c = json::Parse(def->config_json, &err);
      const std::string hook = c ? Str(*c, "hook") : "";
      if (!opt.env.hooks || !opt.env.hooks->Find(hook))
        return out.status = "refused", out.error = "code hook " + hook + " is not registered", out;
    }
    s.evaluators.push_back(std::move(*def));
  }
  auto pending = store.PendingItems(*run, &err);
  if (!pending) return out.status = "refused", out.error = err, out;
  s.items = std::move(*pending);
  if (!store.SetRunStatus(run->id, "running", &err)) return out.status = "refused", out.error = err, out;

  const int n = std::max(1, opt.concurrency);
  if (n == 1) {
    Worker(s);
  } else {
    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) threads.emplace_back([&s] { Worker(s); });
    for (auto& t : threads) t.join();
  }
  out.progress = store.Progress(*run);
  out.error = s.error;
  if (s.crashed.load()) return out.status = "crashed", out;  // as a crash would: no final status
  if (opt.cancel && opt.cancel->load()) out.status = "cancelled";
  else if (out.progress.done == out.progress.total) out.status = "completed";
  else out.status = "failed";
  if (out.status == "failed" && out.error.empty())
    out.error = std::to_string(out.progress.error + out.progress.exhausted) + " items failed; " +
                std::to_string(out.progress.error) + " have attempts left (resume to retry)";
  if (!store.SetRunStatus(run->id, out.status, &err) && out.error.empty()) out.error = err;
  return out;
}

}  // namespace halo::scope
