#ifndef HALO_SCOPE_HTTP_CLIENT_HPP_
#define HALO_SCOPE_HTTP_CLIENT_HPP_

// A minimal blocking HTTP/1.1 client for halo-scope's experiment runner and
// judges: JSON POSTs and GETs to an OpenAI-compatible endpoint (the engine's
// own server or any configured one) over plain TCP. One connection per
// request (Connection: close); fixed-length and chunked response bodies.
// No TLS: remote judges go through a local proxy.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"

namespace halo::scope {

struct HttpResult {
  bool transport_ok = false;  // connected, sent and read a complete response
  int status = 0;
  std::string body;
  std::string error;  // transport errors
};

using HttpHeaders = std::vector<std::pair<std::string, std::string>>;

/// `url` is http://host[:port][/base]; `path` is appended to it.
/// `timeout_ms` bounds connect and every read.
HttpResult HttpRequestJson(const std::string& method, const std::string& url, const std::string& path,
                           const std::string& body, const HttpHeaders& headers, int timeout_ms);

/// Interface over the HTTP client so tests and embedders can substitute the
/// target, judge and decider endpoints.
class JsonPoster {
 public:
  virtual ~JsonPoster() = default;
  virtual HttpResult Post(const std::string& url, const std::string& path, const std::string& body,
                          const HttpHeaders& headers) = 0;
};

class HttpPoster : public JsonPoster {
 public:
  explicit HttpPoster(int timeout_ms = 600000) : timeout_ms_(timeout_ms) {}
  HttpResult Post(const std::string& url, const std::string& path, const std::string& body,
                  const HttpHeaders& headers) override {
    return HttpRequestJson("POST", url, path, body, headers, timeout_ms_);
  }

 private:
  int timeout_ms_;
};

/// One OpenAI-compatible chat completion (POST {url}/v1/chat/completions).
struct ChatReply {
  enum class Status { kOk, kTransient, kFailed } status = Status::kFailed;
  std::string text;  // choices[0].message.content
  std::string id, model, finish_reason;
  std::uint64_t prompt_tokens = 0, completion_tokens = 0;
  double latency_ms = 0;
  std::string raw;    // the response body
  std::string error;  // kTransient / kFailed
};

/// Unreachable endpoints, timeouts, 429 and 5xx are kTransient (retry
/// later); other non-200 answers and malformed bodies are kFailed.
ChatReply CallChat(JsonPoster& poster, const std::string& url, const json::Value& body, const HttpHeaders& headers);

}  // namespace halo::scope

#endif  // HALO_SCOPE_HTTP_CLIENT_HPP_
