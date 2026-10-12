#include "server/contract.hpp"

#include "core/json.hpp"

namespace halo::server {

std::string ContractJson(const Incarnation& inc, const ContractInfo& info) {
  using json::Value;
  Value repeat = Value::MakeObject();
  repeat.set("external_effects", Value::String("none"));
  repeat.set("spend", Value::String("a new generation spends again"));
  repeat.set("exact", Value::String("identical greedy output on the same build and weights"));
  repeat.set("relaxed", Value::String("greedy output may differ with batch participation"));
  Value inference = Value::MakeObject();
  inference.set("mode", Value::String(info.relaxed ? "relaxed" : "exact"));
  inference.set("repeat", std::move(repeat));
  inference.set("cost", Value::String("tokens: prompt, cached, completion"));
  Value doc = Value::MakeObject();
  doc.set("incarnation", Value::String(inc.id()));
  doc.set("time_ms", Value::Int(inc.NowMs()));
  doc.set("inference", std::move(inference));
  doc.set("operations", Value::Null());  // the operation verbs are not served yet
  return json::DumpPython(doc);
}

void RegisterContract(HttpServer& server, const Incarnation& inc, ContractInfo info) {
  server.Route("GET", "/v1/halo/contract",
               [&inc, info](const HttpRequest&) { return HttpResponse::Json(200, ContractJson(inc, info)); });
}

}  // namespace halo::server
