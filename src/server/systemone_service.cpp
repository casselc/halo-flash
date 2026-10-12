#include "server/systemone_service.hpp"

#include <chrono>
#include <cmath>

#include "server/openai_response.hpp"

namespace halo::server {

namespace {

HttpResponse Detail(int status, const std::string& message) {
  json::Value body = json::Value::MakeObject();
  body.set("detail", json::Value::String(message));
  return HttpResponse::Json(status, CompactJson(body));
}

}  // namespace

void RegisterSystemOne(HttpServer& server, DecisionEngine& engine, DeciderGate* gate) {
  auto own = gate ? nullptr : std::make_shared<DeciderGate>();
  DeciderGate* g = gate ? gate : own.get();
  server.Route("POST", "/v1/systemone", [&engine, g, own](const HttpRequest& req) {
    std::string error;
    const auto body = json::Parse(req.body, &error);
    if (!body) return Detail(422, "the request body is not valid JSON: " + error);
    const auto request = decide::ParseRequest(*body, &error);
    if (!request) return Detail(422, error);
    const auto start = std::chrono::steady_clock::now();
    std::optional<json::Value> answer;
    std::string gate_error;
    if (!g->Run([&] { answer = engine.Evaluate(*request, &error); }, &gate_error))
      return Detail(503, "the decider could not run: " + gate_error);
    if (!answer) return Detail(422, error);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    answer->set("latency_ms", json::Value::Float(std::round(ms * 100.0) / 100.0));
    return HttpResponse::Json(200, CompactJson(*answer));
  });
}

OperationApi SystemOneOperationAdapter(DecisionEngine& engine, DeciderGate* gate) {
  auto own = gate ? nullptr : std::make_shared<DeciderGate>();
  DeciderGate* g = gate ? gate : own.get();
  OperationApi a;
  a.prepare = [](const json::Value& body, const HttpRequest&, std::string* error) -> std::optional<OperationApi::Prepared> {
    auto request = decide::ParseRequest(body, error);
    if (!request) return std::nullopt;
    return OperationApi::Prepared{std::make_shared<decide::Request>(std::move(*request)), 0};
  };
  a.run = [&engine, g, own](std::shared_ptr<void> p, const OperationHooks& hooks) -> std::optional<HttpResponse> {
    const auto& request = *std::static_pointer_cast<decide::Request>(p);
    if (hooks.admitted) hooks.admitted({});
    if (hooks.cancelled && hooks.cancelled()) {
      if (hooks.done) hooks.done({json::Value::Null(), "", "cancelled", 0, 0, 0, 0});
      return std::nullopt;
    }
    std::optional<json::Value> answer;
    std::string error, gate_error;
    double ms = 0;
    const bool ran = g->Run(
        [&] {
          const auto start = std::chrono::steady_clock::now();
          answer = engine.Evaluate(request, &error);
          ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        },
        &gate_error);
    OperationHooks::End end;
    if (!ran) {
      end.error = "the decider could not run: " + gate_error, end.finish = "error";
    } else if (!answer) {
      end.error = error, end.finish = "error";
    } else {
      answer->set("latency_ms", json::Value::Float(std::round(ms * 100.0) / 100.0));
      if (const auto* u = answer->find("usage"))
        if (const auto* in = u->find("input_tokens")) end.prompt_tokens = static_cast<std::uint64_t>(in->number());
      end.final = std::move(*answer);
      end.finish = "stop";
      end.engine_ms = ms;
    }
    if (hooks.done) hooks.done(std::move(end));
    return std::nullopt;
  };
  return a;
}

}  // namespace halo::server
