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

void RegisterSystemOne(HttpServer& server, DecisionEngine& engine) {
  auto mu = std::make_shared<std::mutex>();
  server.Route("POST", "/v1/systemone", [&engine, mu](const HttpRequest& req) {
    std::string error;
    const auto body = json::Parse(req.body, &error);
    if (!body) return Detail(422, "the request body is not valid JSON: " + error);
    const auto request = decide::ParseRequest(*body, &error);
    if (!request) return Detail(422, error);
    const auto start = std::chrono::steady_clock::now();
    std::optional<json::Value> answer;
    {
      std::lock_guard<std::mutex> lk(*mu);
      answer = engine.Evaluate(*request, &error);
    }
    if (!answer) return Detail(422, error);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    answer->set("latency_ms", json::Value::Float(std::round(ms * 100.0) / 100.0));
    return HttpResponse::Json(200, CompactJson(*answer));
  });
}

}  // namespace halo::server
