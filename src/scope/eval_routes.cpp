#include "scope/eval_routes.hpp"

#include <charconv>

#include "core/json.hpp"
#include "scope/eval_json.hpp"
#include "scope/eval_stats.hpp"
#include "scope/evaluators.hpp"
#include "server/openai_response.hpp"

namespace halo::scope {

namespace {

using json::Value;
using Kind = json::Value::Kind;
using server::HttpRequest;
using server::HttpResponse;

HttpResponse Fail(int status, const std::string& message) {
  Value err = Value::MakeObject();
  err.set("message", Value::String(message));
  Value body = Value::MakeObject();
  body.set("error", std::move(err));
  return HttpResponse::Json(status, server::CompactJson(body));
}

HttpResponse Ok(const Value& v, int status = 200) { return HttpResponse::Json(status, server::CompactJson(v)); }

std::string PercentDecode(std::string_view s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      out += ' ';
    } else if (s[i] == '%' && i + 2 < s.size()) {
      int v = 0;
      const auto [p, ec] = std::from_chars(s.data() + i + 1, s.data() + i + 3, v, 16);
      if (ec == std::errc() && p == s.data() + i + 3) {
        out += static_cast<char>(v);
        i += 2;
      } else {
        out += s[i];
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

std::string Param(const std::string& query, std::string_view key) {
  std::size_t i = 0;
  while (i <= query.size()) {
    const auto j = std::min(query.find('&', i), query.size());
    const std::string_view kv(query.data() + i, j - i);
    const auto eq = kv.find('=');
    if (PercentDecode(kv.substr(0, eq)) == key)
      return eq == std::string_view::npos ? "1" : PercentDecode(kv.substr(eq + 1));
    i = j + 1;
  }
  return {};
}

std::optional<std::int64_t> PositiveInt(const std::string& s) {
  std::int64_t n = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
  if (ec != std::errc() || p != s.data() + s.size() || n <= 0) return std::nullopt;
  return n;
}

// Splits "/v1/scope/<prefix>/a/b" into {"a", "b"} (percent-decoded).
std::vector<std::string> Segments(const std::string& path, std::string_view prefix) {
  std::vector<std::string> out;
  const std::string rest = path.substr(prefix.size());
  std::size_t i = 0;
  while (i <= rest.size()) {
    const auto j = std::min(rest.find('/', i), rest.size());
    out.push_back(PercentDecode(std::string_view(rest).substr(i, j - i)));
    i = j + 1;
  }
  return out;
}

std::optional<Value> Body(const HttpRequest& r, std::string* err) {
  auto b = json::Parse(r.body, err);
  if (!b || !b->is(Kind::kObject)) {
    *err = "the body must be a JSON object";
    return std::nullopt;
  }
  return b;
}

std::string Str(const Value& o, const char* k) {
  const Value* v = o.find(k);
  return v && v->is(Kind::kString) ? v->str() : std::string();
}

// Store errors that mean "the state forbids it" rather than "bad input".
int StatusFor(const std::string& err) {
  if (err.rfind("no ", 0) == 0) return 404;
  if (err.find("frozen") != std::string::npos || err.find("archived") != std::string::npos ||
      err.find("exists") != std::string::npos || err.find("binding") != std::string::npos)
    return 409;
  return 400;
}

DatasetItem ItemFromJson(const Value& v) {
  DatasetItem it;
  it.id = Str(v, "id");
  if (const Value* x = v.find("input")) it.input_json = CanonicalJson(*x);
  if (const Value* x = v.find("expected_output")) it.expected_json = CanonicalJson(*x);
  if (const Value* x = v.find("metadata"); x && !x->is(Kind::kNull)) it.metadata_json = CanonicalJson(*x);
  it.source_trace_id = Str(v, "source_trace_id");
  it.source_observation_id = Str(v, "source_observation_id");
  return it;
}

}  // namespace

EvalService::EvalService(EvalStore& store, EvalRoutesOptions options) : store_(store), opt_(std::move(options)) {
  if (!opt_.poster) {
    owned_poster_ = std::make_unique<HttpPoster>();
    opt_.poster = owned_poster_.get();
  }
}

EvalService::~EvalService() {
  std::map<std::string, std::unique_ptr<Exec>> all;
  {
    std::lock_guard<std::mutex> lk(mu_);
    all.swap(execs_);
  }
  for (auto& [id, e] : all) e->cancel.store(true);
  for (auto& [id, e] : all)
    if (e->thread.joinable()) e->thread.join();
}

void EvalService::Reap() {  // under mu_
  for (auto it = execs_.begin(); it != execs_.end();) {
    if (it->second->finished.load()) {
      if (it->second->thread.joinable()) it->second->thread.join();
      it = execs_.erase(it);
    } else {
      ++it;
    }
  }
}

bool EvalService::Start(const std::string& run_id, int concurrency) {
  std::lock_guard<std::mutex> lk(mu_);
  Reap();
  if (execs_.count(run_id)) return false;
  auto e = std::make_unique<Exec>();
  Exec* raw = e.get();
  RunnerOptions ro;
  ro.concurrency = std::max(1, concurrency);
  ro.target_endpoint = opt_.endpoint;
  ro.target_headers = opt_.headers;
  ro.env.poster = opt_.poster;
  ro.env.default_endpoint = opt_.endpoint;
  ro.env.decider_endpoint = opt_.decider_endpoint.empty() ? opt_.endpoint : opt_.decider_endpoint;
  ro.env.headers = opt_.headers;
  ro.env.store = &store_;
  ro.env.hooks = opt_.hooks;
  ro.annotator = opt_.annotator;
  ro.cancel = &raw->cancel;
  raw->thread = std::thread([this, raw, run_id, ro = std::move(ro)] {
    (void)ExecuteRun(store_, *opt_.poster, run_id, ro);
    raw->finished.store(true);
  });
  execs_.emplace(run_id, std::move(e));
  return true;
}

void EvalService::Wait(const std::string& run_id) {
  std::unique_ptr<Exec> e;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = execs_.find(run_id);
    if (it == execs_.end()) return;
    e = std::move(it->second);
    execs_.erase(it);
  }
  if (e->thread.joinable()) e->thread.join();
}

bool EvalService::Cancel(const std::string& run_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = execs_.find(run_id);
  if (it == execs_.end() || it->second->finished.load()) return false;
  it->second->cancel.store(true);
  return true;
}

bool EvalService::Active(const std::string& run_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = execs_.find(run_id);
  return it != execs_.end() && !it->second->finished.load();
}

void EvalService::Register(server::HttpServer& srv) {
  EvalStore& store = store_;

  // ---------------------------------------------------------------- datasets
  srv.Route("POST", "/v1/scope/datasets", [&store](const HttpRequest& r) {
    std::string err;
    auto b = Body(r, &err);
    if (!b) return Fail(400, err);
    std::string meta = "{}";
    if (const Value* m = b->find("metadata"); m && !m->is(Kind::kNull)) meta = CanonicalJson(*m);
    if (!store.CreateDataset(Str(*b, "name"), Str(*b, "description"), meta, &err)) return Fail(StatusFor(err), err);
    return Ok(DatasetJson(*store.GetDataset(Str(*b, "name"))));
  });
  srv.Route("GET", "/v1/scope/datasets", [&store](const HttpRequest& r) {
    Value arr = Value::MakeArray();
    for (const auto& d : store.ListDatasets(!Param(r.query, "archived").empty())) arr.push(DatasetJson(d));
    Value body = Value::MakeObject();
    body.set("data", std::move(arr));
    return Ok(body);
  });
  srv.RoutePrefix("GET", "/v1/scope/datasets/", [&store](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/datasets/");
    if (seg[0].empty()) return Fail(404, "no such dataset route");
    if (seg.size() == 1) {
      auto d = store.GetDataset(seg[0]);
      return d ? Ok(DatasetJson(*d)) : Fail(404, "no dataset named " + seg[0]);
    }
    if (seg.size() == 2 && seg[1] == "items") {
      std::int64_t version = 0;
      if (const std::string v = Param(r.query, "version"); !v.empty()) {
        auto n = PositiveInt(v);
        if (!n) return Fail(400, "version must be a positive integer");
        version = *n;
      }
      std::string err;
      auto items = store.Items(seg[0], version, &err);
      if (!items) return Fail(StatusFor(err), err);
      Value arr = Value::MakeArray();
      for (const auto& it : *items) arr.push(ItemJson(it));
      Value body = Value::MakeObject();
      body.set("data", std::move(arr));
      return Ok(body);
    }
    return Fail(404, "no such dataset route");
  });
  srv.RoutePrefix("DELETE", "/v1/scope/datasets/", [&store](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/datasets/");
    if (seg.size() != 3 || seg[1] != "items") return Fail(404, "no such dataset route");
    std::string err;
    if (!store.DeleteItem(seg[0], seg[2], &err)) return Fail(StatusFor(err), err);
    return Ok(Value::MakeObject());
  });
  TraceReader* traces = opt_.traces;
  srv.RoutePrefix("POST", "/v1/scope/datasets/", [&store, traces](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/datasets/");
    if (seg.size() != 2 || seg[0].empty()) return Fail(404, "no such dataset route");
    const std::string& name = seg[0];
    const std::string& op = seg[1];
    std::string err;
    if (op == "import") {
      std::size_t count = 0;
      const auto v = store.ImportJsonl(name, r.body, &count, &err);
      if (v == 0) return Fail(StatusFor(err), err);
      Value o = Value::MakeObject();
      o.set("version", Value::Int(v));
      o.set("count", Value::Int(static_cast<std::int64_t>(count)));
      return Ok(o);
    }
    if (op == "freeze") {
      auto v = store.Freeze(name, &err);
      if (!v) return Fail(StatusFor(err), err);
      Value o = Value::MakeObject();
      o.set("version", Value::Int(v->version));
      o.set("content_hash", Value::String(v->content_hash));
      o.set("item_count", Value::Int(v->item_count));
      o.set("frozen_at", Value::String(v->frozen_at));
      return Ok(o);
    }
    if (op == "archive") {
      if (!store.Archive(name, &err)) return Fail(StatusFor(err), err);
      return Ok(DatasetJson(*store.GetDataset(name)));
    }
    auto b = Body(r, &err);
    if (!b) return Fail(400, err);
    std::vector<DatasetItem> items;
    if (op == "items") {
      if (const Value* arr = b->find("items")) {
        if (!arr->is(Kind::kArray)) return Fail(400, "items must be an array");
        for (const auto& v : arr->items()) {
          if (!v.is(Kind::kObject) || !v.find("input")) return Fail(400, "each item needs an input");
          items.push_back(ItemFromJson(v));
        }
      } else {
        if (!b->find("input")) return Fail(400, "an item needs an input");
        items.push_back(ItemFromJson(*b));
      }
    } else if (op == "from_traces") {
      if (!traces) return Fail(400, "trace reading is not configured on this server");
      std::vector<std::string> tids, oids;
      for (auto [key, out] : {std::pair{"trace_ids", &tids}, std::pair{"observation_ids", &oids}})
        if (const Value* a = b->find(key)) {
          if (!a->is(Kind::kArray)) return Fail(400, std::string(key) + " must be an array of strings");
          for (const auto& x : a->items())
            if (x.is(Kind::kString)) out->push_back(x.str());
        }
      if (tids.empty() && oids.empty()) return Fail(400, "give trace_ids or observation_ids");
      const bool with_expected = Str(*b, "expected") != "none";
      auto contents = traces->Contents(tids, oids, &err);
      if (!contents) return Fail(500, err);
      if (contents->empty()) return Fail(404, "no matching observations");
      for (const auto& c : *contents) {
        if (!c.has_input) return Fail(400, "observation " + c.span_id + " has no recorded input (capture must be full)");
        DatasetItem it;
        it.input_json = CanonicalJson(Value::String(c.input));
        if (with_expected && c.has_output) it.expected_json = CanonicalJson(Value::String(c.output));
        Value meta = Value::MakeObject();
        meta.set("source_type", Value::String(c.type));
        meta.set("source_name", Value::String(c.name));
        it.metadata_json = CanonicalJson(meta);
        it.source_trace_id = c.trace_id;
        it.source_observation_id = c.span_id;
        items.push_back(std::move(it));
      }
    } else {
      return Fail(404, "no such dataset route");
    }
    const auto v = store.UpsertItems(name, items, &err);
    if (v == 0) return Fail(StatusFor(err), err);
    Value ids = Value::MakeArray();
    for (const auto& it : items) ids.push(Value::String(it.id));
    Value o = Value::MakeObject();
    o.set("version", Value::Int(v));
    o.set("ids", std::move(ids));
    return Ok(o);
  });

  // -------------------------------------------------------------- evaluators
  srv.Route("POST", "/v1/scope/evaluators", [&store](const HttpRequest& r) {
    std::string err;
    auto b = Body(r, &err);
    if (!b) return Fail(400, err);
    std::string config = "{}";
    if (const Value* c = b->find("config"); c && !c->is(Kind::kNull)) config = CanonicalJson(*c);
    const std::string name = Str(*b, "name"), kind = Str(*b, "kind");
    if (name.empty() || name.find('/') != std::string::npos || name.find('@') != std::string::npos)
      return Fail(400, "name must be non-empty, without '/' or '@'");
    auto prepared = PrepareEvaluatorConfig(store, kind, config, &err);
    if (!prepared) return Fail(400, err);
    const auto v = store.CreateEvaluatorVersion(name, kind, *prepared, &err);
    if (v == 0) return Fail(StatusFor(err), err);
    return Ok(EvaluatorJson(*store.GetEvaluator(name, v)));
  });
  srv.Route("GET", "/v1/scope/evaluators", [&store](const HttpRequest&) {
    Value arr = Value::MakeArray();
    for (const auto& e : store.ListEvaluators()) arr.push(EvaluatorJson(e));
    Value body = Value::MakeObject();
    body.set("data", std::move(arr));
    return Ok(body);
  });
  srv.RoutePrefix("GET", "/v1/scope/evaluators/", [&store](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/evaluators/");
    if (seg.size() != 1 || seg[0].empty()) return Fail(404, "no such evaluator route");
    std::int64_t version = 0;
    if (const std::string v = Param(r.query, "version"); !v.empty()) {
      auto n = PositiveInt(v);
      if (!n) return Fail(400, "version must be a positive integer");
      version = *n;
    }
    auto e = store.GetEvaluator(seg[0], version);
    return e ? Ok(EvaluatorJson(*e)) : Fail(404, "no evaluator " + seg[0]);
  });

  // -------------------------------------------------------------------- runs
  srv.Route("POST", "/v1/scope/runs", [this, &store](const HttpRequest& r) {
    std::string err;
    auto b = Body(r, &err);
    if (!b) return Fail(400, err);
    RunSpec spec;
    spec.name = Str(*b, "name");
    spec.dataset = Str(*b, "dataset");
    if (const Value* v = b->find("dataset_version"); v && v->is(Kind::kInt)) spec.dataset_version = static_cast<std::int64_t>(v->number());
    if (const Value* t = b->find("target")) {
      if (!t->is(Kind::kObject)) return Fail(400, "target must be an object");
      spec.target_json = CanonicalJson(*t);
    }
    if (const Value* m = b->find("metadata"); m && !m->is(Kind::kNull)) spec.metadata_json = CanonicalJson(*m);
    if (const Value* v = b->find("max_attempts"); v && v->is(Kind::kInt)) spec.max_attempts = static_cast<int>(v->number());
    int concurrency = 1;
    if (const Value* v = b->find("concurrency"); v && v->is(Kind::kInt))
      concurrency = static_cast<int>(std::clamp<double>(v->number(), 1, 64));
    if (const Value* es = b->find("evaluators")) {
      if (!es->is(Kind::kArray)) return Fail(400, "evaluators must be an array");
      for (const auto& e : es->items()) {
        if (e.is(Kind::kString)) {
          const auto at = e.str().find('@');
          std::int64_t v = 0;
          if (at != std::string::npos) {
            auto n = PositiveInt(e.str().substr(at + 1));
            if (!n) return Fail(400, "bad evaluator reference " + e.str());
            v = *n;
          }
          spec.evaluators.emplace_back(e.str().substr(0, at), v);
        } else if (e.is(Kind::kObject)) {
          const Value* v = e.find("version");
          spec.evaluators.emplace_back(Str(e, "name"), v && v->is(Kind::kInt) ? static_cast<std::int64_t>(v->number()) : 0);
        } else {
          return Fail(400, "evaluators must be names or {name, version}");
        }
      }
    }
    bool existed = false;
    auto run = store.CreateRun(spec, &existed, &err);
    if (!run) return Fail(StatusFor(err), err);
    const bool wait = b->find("wait") && b->find("wait")->is(Kind::kBool) && b->find("wait")->boolean();
    if (run->status != "completed") Start(run->id, concurrency);
    if (wait) {
      Wait(run->id);
      return Ok(RunJson(store, *store.GetRun(run->id), true));
    }
    Value o = RunJson(store, *store.GetRun(run->id), false);
    o.set("resumed", Value::Bool(existed));
    return Ok(o, 202);
  });
  srv.Route("GET", "/v1/scope/runs", [&store](const HttpRequest& r) {
    Value arr = Value::MakeArray();
    for (const auto& run : store.ListRuns(Param(r.query, "dataset"))) arr.push(RunJson(store, run, false));
    Value body = Value::MakeObject();
    body.set("data", std::move(arr));
    return Ok(body);
  });
  srv.RoutePrefix("GET", "/v1/scope/runs/", [this, &store](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/runs/");
    auto run = seg[0].empty() ? std::nullopt : store.GetRun(seg[0]);
    if (!run) return Fail(404, "no run " + seg[0]);
    if (seg.size() == 1) {
      Value o = RunJson(store, *run, true);
      o.set("active", Value::Bool(Active(run->id)));
      return Ok(o);
    }
    if (seg.size() == 2 && seg[1] == "items") {
      Value body = Value::MakeObject();
      body.set("data", RunItemsJson(store, *run));
      return Ok(body);
    }
    if (seg.size() == 2 && seg[1] == "scores") {
      Value body = Value::MakeObject();
      body.set("data", SummaryJson(Summarize(store.RunScores(run->id))));
      return Ok(body);
    }
    return Fail(404, "no such run route");
  });
  srv.RoutePrefix("POST", "/v1/scope/runs/", [this, &store](const HttpRequest& r) {
    const auto seg = Segments(r.path, "/v1/scope/runs/");
    auto run = seg[0].empty() ? std::nullopt : store.GetRun(seg[0]);
    if (!run) return Fail(404, "no run " + seg[0]);
    if (seg.size() == 2 && seg[1] == "cancel") {
      if (!Cancel(run->id)) return Fail(409, "run " + run->id + " is not executing");
      return Ok(RunJson(store, *run, false));
    }
    if (seg.size() == 2 && seg[1] == "resume") {
      if (run->status == "completed") return Fail(409, "run " + run->id + " is complete");
      int concurrency = 1;
      std::string err;
      if (auto b = json::Parse(r.body.empty() ? "{}" : r.body, &err); b && b->is(Kind::kObject))
        if (const Value* v = b->find("concurrency"); v && v->is(Kind::kInt))
          concurrency = static_cast<int>(std::clamp<double>(v->number(), 1, 64));
      if (!Start(run->id, concurrency)) return Fail(409, "run " + run->id + " is already executing");
      return Ok(RunJson(store, *store.GetRun(run->id), false), 202);
    }
    return Fail(404, "no such run route");
  });
  srv.Route("GET", "/v1/scope/compare", [&store](const HttpRequest& r) {
    const std::string a_id = Param(r.query, "a"), b_id = Param(r.query, "b"), score = Param(r.query, "score");
    if (a_id.empty() || b_id.empty() || score.empty()) return Fail(400, "give a, b and score");
    auto a = store.GetRun(a_id);
    auto b = store.GetRun(b_id);
    if (!a || !b) return Fail(404, "no such run");
    if (a->dataset_id != b->dataset_id) return Fail(400, "runs are on different datasets");
    std::string err;
    auto c = ComparePaired(store.RunScores(a->id), store.RunScores(b->id), score, &err);
    if (!c) return Fail(400, err);
    return Ok(ComparisonJson(*c, *a, *b, !Param(r.query, "items").empty()));
  });
}

}  // namespace halo::scope
