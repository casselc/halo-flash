#include "scope/http_annotations.hpp"

#include <cmath>
#include <optional>
#include <string>

#include "core/json.hpp"
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

HttpResponse Done(Outcome o, Value body = Value::MakeObject()) {
  switch (o) {
    case Outcome::kOk: return HttpResponse::Json(200, server::CompactJson(body));
    case Outcome::kDropped: return Fail(429, "dropped: a recording ring is full; retry later");
    case Outcome::kInvalid: break;
  }
  return Fail(400, "invalid annotation (missing id, unknown step or kind)");
}

// Reads an object body; nullopt (with *bad set) when it is not one.
std::optional<Value> Body(const HttpRequest& r, HttpResponse* bad) {
  std::string err;
  auto v = json::Parse(r.body.empty() ? std::string_view("{}") : std::string_view(r.body), &err);
  if (!v || !v->is(Kind::kObject)) {
    *bad = Fail(400, "the body must be a JSON object" + (err.empty() ? "" : ": " + err));
    return std::nullopt;
  }
  return v;
}

std::string Str(const Value& o, const char* key) {
  const Value* v = o.find(key);
  return v && v->is(Kind::kString) ? v->str() : std::string();
}

std::optional<Status> StatusOf(const Value& o) {
  const std::string s = Str(o, "status");
  if (s.empty() || s == "ok") return Status::kOk;
  if (s == "error") return Status::kError;
  return std::nullopt;
}

// The path segment after `prefix` and before `suffix`: /v1/scope/steps/{id}/end.
std::optional<std::string> Segment(const std::string& path, std::string_view prefix, std::string_view suffix) {
  if (!path.starts_with(prefix) || !path.ends_with(suffix) || path.size() <= prefix.size() + suffix.size())
    return std::nullopt;
  std::string id = path.substr(prefix.size(), path.size() - prefix.size() - suffix.size());
  if (id.find('/') != std::string::npos) return std::nullopt;
  return id;
}

}  // namespace

void RegisterAnnotationRoutes(server::HttpServer& srv, Annotator& a) {
  srv.Route("POST", "/v1/scope/steps", [&a](const HttpRequest& r) {
    HttpResponse bad;
    const auto b = Body(r, &bad);
    if (!b) return bad;
    static const std::pair<const char*, StepKind> kinds[] = {{"tool", StepKind::kTool}, {"retry", StepKind::kRetry},
                                                             {"plan", StepKind::kPlan}, {"memory", StepKind::kMemory},
                                                             {"custom", StepKind::kCustom}};
    const std::string k = Str(*b, "kind");
    const StepKind* kind = nullptr;
    for (const auto& [name, v] : kinds)
      if (k == name) kind = &v;
    if (!kind) return Fail(400, "kind must be tool, retry, plan, memory or custom");
    std::string id;
    const auto o = a.StepBegin(Str(*b, "trace_id"), Str(*b, "parent"), *kind, Str(*b, "name"), &id);
    Value out = Value::MakeObject();
    out.set("step_id", Value::String(id));
    return Done(o, std::move(out));
  });
  srv.Route("POST", "/v1/scope/tool_results", [&a](const HttpRequest& r) {
    HttpResponse bad;
    const auto b = Body(r, &bad);
    if (!b) return bad;
    const auto status = StatusOf(*b);
    if (!status) return Fail(400, "status must be ok or error");
    std::uint64_t latency_ns = 0;
    if (const Value* l = b->find("latency_ms"); l && (l->is(Kind::kInt) || l->is(Kind::kFloat))) {
      const double ms = l->number();
      if (!(ms >= 0) || !std::isfinite(ms)) return Fail(400, "latency_ms must be a non-negative number");
      latency_ns = static_cast<std::uint64_t>(ms * 1e6);
    }
    return Done(a.ToolResult(Str(*b, "trace_id"), Str(*b, "tool_call_id"), Str(*b, "tool_name"), *status,
                             Str(*b, "output"), latency_ns));
  });
  srv.Route("POST", "/v1/scope/scores", [&a](const HttpRequest& r) {
    HttpResponse bad;
    const auto b = Body(r, &bad);
    if (!b) return bad;
    const ScoreTarget t{Str(*b, "trace_id"), Str(*b, "observation_id"), Str(*b, "session_id")};
    const Value* v = b->find("value");
    if (!v) return Fail(400, "value is required (number, string or boolean)");
    std::string id;
    Outcome o;
    if (v->is(Kind::kBool))
      o = a.ScoreBoolean(t, Str(*b, "name"), v->boolean(), Str(*b, "source"), Str(*b, "comment"),
                         Str(*b, "supersedes"), &id);
    else if (v->is(Kind::kString))
      o = a.ScoreCategorical(t, Str(*b, "name"), v->str(), Str(*b, "source"), Str(*b, "comment"),
                             Str(*b, "supersedes"), &id);
    else if (v->is(Kind::kInt) || v->is(Kind::kFloat))
      o = a.ScoreNumeric(t, Str(*b, "name"), v->number(), Str(*b, "source"), Str(*b, "comment"),
                         Str(*b, "supersedes"), &id);
    else
      return Fail(400, "value must be a number, string or boolean");
    Value out = Value::MakeObject();
    out.set("score_id", Value::String(id));
    return Done(o, std::move(out));
  });
  srv.RoutePrefix("POST", "/v1/scope/", [&a](const HttpRequest& r) {
    HttpResponse bad;
    const auto b = Body(r, &bad);
    if (!b) return bad;
    if (const auto step = Segment(r.path, "/v1/scope/steps/", "/end")) {
      const auto status = StatusOf(*b);
      if (!status) return Fail(400, "status must be ok or error");
      return Done(a.StepEnd(*step, *status, Str(*b, "output")));
    }
    if (const auto score = Segment(r.path, "/v1/scope/scores/", "/retract"))
      return Done(a.Retract(*score, Str(*b, "source"), Str(*b, "comment")));
    if (const auto trace = Segment(r.path, "/v1/scope/traces/", "/meta"))
      return Done(a.TraceMeta(*trace, Str(*b, "key"), Str(*b, "value")));
    return Fail(404, "no annotation route for " + r.path);
  });
}

}  // namespace halo::scope
