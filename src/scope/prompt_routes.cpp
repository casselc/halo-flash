#include "scope/prompt_routes.hpp"

#include <charconv>
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

std::string Param(const std::string& query, std::string_view key) {
  std::size_t i = 0;
  while (i <= query.size()) {
    const auto j = std::min(query.find('&', i), query.size());
    const std::string_view kv(query.data() + i, j - i);
    const auto eq = kv.find('=');
    if (kv.substr(0, eq) == key) return eq == std::string_view::npos ? "" : std::string(kv.substr(eq + 1));
    i = j + 1;
  }
  return {};
}

Value PromptJson(MetadataStore& store, const PromptVersion& p, bool full) {
  Value v = Value::MakeObject();
  v.set("name", Value::String(p.name));
  v.set("version", Value::Int(p.version));
  if (full) {
    v.set("template", Value::String(p.template_text));
    std::string err;
    auto config = json::Parse(p.config_json, &err);
    v.set("config", config ? *config : Value::MakeObject());
  }
  Value labels = Value::MakeArray();
  for (const auto& l : store.LabelsOf(p.name, p.version)) labels.push(Value::String(l));
  v.set("labels", std::move(labels));
  return v;
}

bool ValidName(const std::string& n) {
  return !n.empty() && n.size() <= 200 && n.find('/') == std::string::npos && n.find('?') == std::string::npos;
}

}  // namespace

void RegisterPromptRoutes(server::HttpServer& srv, MetadataStore& store) {
  srv.Route("POST", "/v1/scope/prompts", [&store](const HttpRequest& r) {
    std::string err;
    const auto b = json::Parse(r.body, &err);
    if (!b || !b->is(Kind::kObject)) return Fail(400, "the body must be a JSON object");
    const Value* name = b->find("name");
    const Value* tmpl = b->find("template");
    if (!name || !name->is(Kind::kString) || !ValidName(name->str())) return Fail(400, "name must be a non-empty string without '/' or '?'");
    if (!tmpl || !tmpl->is(Kind::kString)) return Fail(400, "template must be a string");
    std::string config = "{}";
    if (const Value* c = b->find("config"); c && !c->is(Kind::kNull)) {
      if (!c->is(Kind::kObject)) return Fail(400, "config must be an object");
      config = server::CompactJson(*c);
    }
    const auto version = store.CreatePromptVersion(name->str(), tmpl->str(), config, &err);
    if (version == 0) return Fail(500, err);
    return HttpResponse::Json(200, server::CompactJson(PromptJson(store, {name->str(), version, "", ""}, false)));
  });
  srv.RoutePrefix("GET", "/v1/scope/prompts/", [&store](const HttpRequest& r) {
    const std::string name = r.path.substr(std::string_view("/v1/scope/prompts/").size());
    if (!ValidName(name)) return Fail(404, "no such prompt route");
    std::optional<PromptVersion> p;
    if (const std::string v = Param(r.query, "version"); !v.empty()) {
      std::int64_t n = 0;
      const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
      if (ec != std::errc() || end != v.data() + v.size() || n <= 0) return Fail(400, "version must be a positive integer");
      p = store.GetByVersion(name, n);
    } else {
      const std::string label = Param(r.query, "label");
      p = store.GetByLabel(name, label.empty() ? "latest" : label);
    }
    if (!p) return Fail(404, "no such prompt version");
    return HttpResponse::Json(200, server::CompactJson(PromptJson(store, *p, true)));
  });
  srv.RoutePrefix("POST", "/v1/scope/prompts/", [&store](const HttpRequest& r) {
    const std::string rest = r.path.substr(std::string_view("/v1/scope/prompts/").size());
    const auto slash = rest.find('/');
    if (slash == std::string::npos || rest.substr(slash) != "/labels") return Fail(404, "no such prompt route");
    const std::string name = rest.substr(0, slash);
    std::string err;
    const auto b = json::Parse(r.body, &err);
    if (!b || !b->is(Kind::kObject)) return Fail(400, "the body must be a JSON object");
    const Value* label = b->find("label");
    const Value* version = b->find("version");
    if (!label || !label->is(Kind::kString) || label->str().empty()) return Fail(400, "label must be a non-empty string");
    if (!version || !version->is(Kind::kInt)) return Fail(400, "version must be an integer");
    if (!store.SetLabel(name, label->str(), static_cast<std::int64_t>(version->number()), &err))
      return Fail(400, "cannot set the label: " + err);
    return HttpResponse::Json(200, "{}");
  });
}

}  // namespace halo::scope
