// Test doubles for halo-scope's evaluation tests: an OpenAI-compatible
// target / judge / decider endpoint answered in-process, recording every
// request it receives.

#ifndef HALO_TESTS_SCOPE_EVAL_FAKES_HPP_
#define HALO_TESTS_SCOPE_EVAL_FAKES_HPP_

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "scope/http_client.hpp"

namespace halo::scope::testing {

struct Recorded {
  std::string url, path, body;
  HttpHeaders headers;
  std::string Header(const std::string& k) const {
    for (const auto& [n, v] : headers)
      if (n == k) return v;
    return {};
  }
};

/// Answers each POST with `handler(path, parsed body)`; records them all.
class FakePoster : public JsonPoster {
 public:
  using Handler = std::function<HttpResult(const std::string& path, const json::Value& body, const Recorded& req)>;
  explicit FakePoster(Handler h) : handler_(std::move(h)) {}
  HttpResult Post(const std::string& url, const std::string& path, const std::string& body,
                  const HttpHeaders& headers) override {
    Recorded r{url, path, body, headers};
    {
      std::lock_guard<std::mutex> lk(mu_);
      requests_.push_back(r);
    }
    std::string err;
    auto v = json::Parse(body, &err);
    if (!v) return HttpResult{true, 400, R"({"error": {"message": "bad json"}})", ""};
    return handler_(path, *v, r);
  }
  std::vector<Recorded> requests() const {
    std::lock_guard<std::mutex> lk(mu_);
    return requests_;
  }
  std::size_t count(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::size_t n = 0;
    for (const auto& r : requests_) n += r.path == path;
    return n;
  }

 private:
  Handler handler_;
  mutable std::mutex mu_;
  std::vector<Recorded> requests_;
};

/// A chat completion response body with `text` as the message content.
inline std::string ChatBody(const std::string& text, const std::string& id = "chatcmpl-1",
                            const std::string& model = "fake-model") {
  json::Value msg = json::Value::MakeObject();
  msg.set("role", json::Value::String("assistant"));
  msg.set("content", json::Value::String(text));
  json::Value c0 = json::Value::MakeObject();
  c0.set("index", json::Value::Int(0));
  c0.set("message", std::move(msg));
  c0.set("finish_reason", json::Value::String("stop"));
  json::Value choices = json::Value::MakeArray();
  choices.push(std::move(c0));
  json::Value usage = json::Value::MakeObject();
  usage.set("prompt_tokens", json::Value::Int(7));
  usage.set("completion_tokens", json::Value::Int(3));
  json::Value o = json::Value::MakeObject();
  o.set("id", json::Value::String(id));
  o.set("model", json::Value::String(model));
  o.set("choices", std::move(choices));
  o.set("usage", std::move(usage));
  return json::DumpPython(o);
}

/// The last user message's content of a chat request.
inline std::string UserText(const json::Value& body) {
  const json::Value* ms = body.find("messages");
  std::string out;
  if (ms)
    for (const auto& m : ms->items())
      if (m.find("role") && m.find("role")->str() == "user") out = m.find("content")->str();
  return out;
}

}  // namespace halo::scope::testing

#endif  // HALO_TESTS_SCOPE_EVAL_FAKES_HPP_
