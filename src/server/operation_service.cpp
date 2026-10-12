#include "server/operation_service.hpp"

#include <chrono>
#include <cstdlib>
#include <thread>
#include <utility>

#include "core/sha256.hpp"
#include "server/openai_response.hpp"

namespace halo::server {

using json::Value;
using Ops = Operations;

namespace {

HttpResponse Error(int status, const std::string& type, const std::string& message) {
  Value inner = Value::MakeObject();
  inner.set("message", Value::String(message));
  inner.set("type", Value::String(type));
  Value e = Value::MakeObject();
  e.set("error", std::move(inner));
  return HttpResponse::Json(status, CompactJson(e));
}

bool ValidOpId(const std::string& s) {
  if (s.empty() || s.size() > 128) return false;
  for (const char c : s)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == ':' || c == '-')) return false;
  return true;
}

bool ParseU64(std::string_view s, std::uint64_t* out) {
  if (s.empty() || s.size() > 19) return false;
  std::uint64_t v = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
    v = v * 10 + static_cast<std::uint64_t>(c - '0');
  }
  return *out = v, true;
}

bool ParseI64(std::string_view s, std::int64_t* out) {
  const bool neg = !s.empty() && s[0] == '-';
  std::uint64_t v = 0;
  if (!ParseU64(neg ? s.substr(1) : s, &v)) return false;
  return *out = neg ? -static_cast<std::int64_t>(v) : static_cast<std::int64_t>(v), true;
}

Value ViewJson(const Ops::Ref& ref, const Ops::View& v) {
  Value o = Value::MakeObject();
  o.set("operation_id", Value::String(ref.id.op));
  o.set("attempt", Value::Int(static_cast<std::int64_t>(ref.id.attempt)));
  o.set("status", Value::String(StatusName(v.status)));
  o.set("digest", Value::String(ref.digest));
  if (v.status == Ops::Status::kRunning) o.set("cancel_requested", Value::Bool(v.cancel_requested));
  if (v.status == Ops::Status::kDone || v.status == Ops::Status::kReleased || v.status == Ops::Status::kExpired) {
    o.set("finish", Value::String(v.finish));
    Value u = Value::MakeObject();
    u.set("prompt_tokens", Value::Int(static_cast<std::int64_t>(v.usage.prompt)));
    u.set("cached_tokens", Value::Int(static_cast<std::int64_t>(v.usage.cached)));
    u.set("completion_tokens", Value::Int(static_cast<std::int64_t>(v.usage.completion)));
    u.set("engine_ms", Value::Float(v.usage.engine_ms));
    o.set("usage", std::move(u));
  }
  if (v.status == Ops::Status::kDone) o.set("expires_at", Value::Int(v.expires_at));
  return o;
}

/// The HTTP answer for a verb's verdict.
HttpResponse Answer(const Ops::Ref& ref, const Ops::Answer& a, const std::string& incarnation, int ok_status = 200) {
  switch (a.verdict) {
    case Ops::Verdict::kOk:
    case Ops::Verdict::kDuplicate: return HttpResponse::Json(a.verdict == Ops::Verdict::kOk ? ok_status : 200,
                                                             CompactJson(ViewJson(ref, a.view)));
    case Ops::Verdict::kStale: {
      Value e = Value::MakeObject();
      e.set("message", Value::String("meant for incarnation " + ref.incarnation + ", this is " + incarnation));
      e.set("type", Value::String("stale_incarnation"));
      Value o = Value::MakeObject();
      o.set("error", std::move(e));
      o.set("incarnation", Value::String(incarnation));
      return HttpResponse::Json(409, CompactJson(o));
    }
    case Ops::Verdict::kConflict:
      return Error(409, "conflict", "the attempt is bound to another request (digest)");
    case Ops::Verdict::kLate: return Error(410, "late", "past its not_after");
    case Ops::Verdict::kTooFarAhead: return Error(400, "invalid_request_error", "not_after is too far ahead");
    case Ops::Verdict::kRefusedRetention:
    case Ops::Verdict::kRefusedRecords: {
      HttpResponse r = Error(503, a.verdict == Ops::Verdict::kRefusedRecords ? "records" : "retention",
                             a.verdict == Ops::Verdict::kRefusedRecords
                                 ? "at the record ceiling (unacknowledged outcomes and tombstones); not accepted"
                                 : "the retention budget is full; not accepted");
      r.headers.emplace_back("Retry-After", std::to_string(std::max<std::int64_t>(1, a.retry_after_s)));
      return r;
    }
  }
  return Error(500, "internal", "unhandled verdict");
}

/// What a submit waits for: the attempt's admission (its answer) or an error before it ran.
struct Admitted {
  std::mutex mu;
  std::condition_variable cv;
  std::optional<HttpResponse> answer;
  void Set(HttpResponse r) {
    std::lock_guard<std::mutex> lk(mu);
    if (!answer) answer = std::move(r);
    cv.notify_all();
  }
};

}  // namespace

OperationService::OperationService(const Incarnation& incarnation, OperationServiceOptions options)
    : inc_(incarnation), opt_(options), ops_(incarnation.id(), options.records) {}

void OperationService::AddApi(std::string api, OperationApi adapter) { apis_[std::move(api)] = std::move(adapter); }

OperationService::~OperationService() {
  stopping_ = true;
  std::unique_lock<std::mutex> lk(workers_mu_);
  workers_cv_.wait(lk, [this] { return workers_ == 0; });
}

void OperationService::Register(HttpServer& server) {
  server.Route("POST", "/v1/operations", [this](const HttpRequest& r) { return Submit(r); });
  server.RoutePrefix("GET", "/v1/operations/", [this](const HttpRequest& r) { return Verb(r); });
  server.RoutePrefix("POST", "/v1/operations/", [this](const HttpRequest& r) { return Verb(r); });
}

Value OperationService::Contract() const {
  const auto& o = ops_.options();
  const auto s = ops_.stats(inc_.NowMs());
  Value v = Value::MakeObject();
  Value apis = Value::MakeArray();
  for (const auto& [name, a] : apis_) apis.push(Value::String(name));
  v.set("apis", std::move(apis));
  v.set("owner", Value::String("X-Halo-Principal"));
  v.set("digest", Value::String("sha256 (hex) of the submit body"));
  Value ret = Value::MakeObject();
  ret.set("capture_s", Value::Int(o.capture_ms / 1000));
  ret.set("until", Value::String("ack or capture window"));
  ret.set("budget_bytes", Value::Int(static_cast<std::int64_t>(o.budget_bytes)));
  ret.set("reserve", Value::String("at acceptance: " + std::to_string(opt_.base_bytes) + " + max_tokens x " +
                                   std::to_string(opt_.bytes_per_token) + " bytes"));
  ret.set("reserved_bytes", Value::Int(static_cast<std::int64_t>(s.reserved_bytes)));
  v.set("retention", std::move(ret));
  Value rec = Value::MakeObject();
  rec.set("max", Value::Int(static_cast<std::int64_t>(o.max_records)));
  rec.set("in_use", Value::Int(static_cast<std::int64_t>(s.records)));
  rec.set("unacked_outcomes", Value::Int(static_cast<std::int64_t>(s.unacked_outcomes)));
  rec.set("oldest_unacked_ms", Value::Int(s.oldest_unacked_ms));
  v.set("records", std::move(rec));
  Value dedup = Value::MakeObject();
  dedup.set("scope", Value::String("owner, incarnation"));
  dedup.set("max_submit_age_s", Value::Int(o.max_submit_age_ms / 1000));
  v.set("dedup", std::move(dedup));
  v.set("cancel", Value::String("next unit boundary (<= one decode unit or one prefill slice); usage reported"));
  v.set("disconnect", Value::String("never cancels"));
  return v;
}

void OperationService::DropReleased() {
  std::lock_guard<std::mutex> lk(mu_);
  const std::int64_t now = inc_.NowMs();
  for (auto it = live_.begin(); it != live_.end();) {
    if (!ops_.Holds(it->first, now)) it = live_.erase(it);
    else ++it;
  }
}

HttpResponse OperationService::Submit(const HttpRequest& http) {
  const std::string* inc = http.Header("halo-incarnation");
  if (!inc) return Error(400, "invalid_request_error", "a submit names its incarnation (Halo-Incarnation)");
  std::string err;
  const auto body = json::Parse(http.body, &err);
  if (!body || !body->is(Value::Kind::kObject)) return Error(400, "invalid_request_error", "the body is not a JSON object");
  const Value* op = body->find("operation_id");
  const Value* attempt = body->find("attempt");
  const Value* not_after = body->find("not_after");
  const Value* api = body->find("api");
  const Value* request = body->find("request");
  const Value* mode = body->find("mode");
  std::uint64_t att = 0;
  std::int64_t nb = 0;
  if (!op || !op->is(Value::Kind::kString) || !ValidOpId(op->str()))
    return Error(400, "invalid_request_error", "operation_id: 1-128 of [A-Za-z0-9._:-]");
  if (!attempt || !attempt->is(Value::Kind::kInt) || !ParseU64(attempt->int_digits(), &att))
    return Error(400, "invalid_request_error", "attempt: a non-negative integer");
  if (!not_after || !not_after->is(Value::Kind::kInt) || !ParseI64(not_after->int_digits(), &nb))
    return Error(400, "invalid_request_error", "not_after: an integer (ms on the incarnation's clock)");
  const auto adapter = api && api->is(Value::Kind::kString) ? apis_.find(api->str()) : apis_.end();
  if (adapter == apis_.end()) {
    std::string names;
    for (const auto& [name, a] : apis_) names += (names.empty() ? "" : ", ") + name;
    return Error(400, "invalid_request_error", "api: one of " + names);
  }
  if (!request || !request->is(Value::Kind::kObject)) return Error(400, "invalid_request_error", "request: an object");
  const std::string ours = opt_.relaxed ? "relaxed" : "exact";
  if (!mode || !mode->is(Value::Kind::kString) || (mode->str() != "relaxed" && mode->str() != "exact"))
    return Error(400, "invalid_request_error", "mode: \"relaxed\" or \"exact\"");
  if (mode->str() != ours)
    return Error(422, "mode_unavailable", "this server runs " + ours + " inference (see GET /v1/halo/contract)");
  const std::string digest = core::Sha256Hex(http.body);
  if (const std::string* d = http.Header("halo-digest"); d && *d != digest)
    return Error(400, "invalid_request_error", "Halo-Digest is not the SHA-256 of the body");
  auto prepared = adapter->second.prepare(*request, http, &err);
  if (!prepared) return Error(400, "invalid_request_error", err);
  const std::string owner = http.Header("x-halo-principal") ? *http.Header("x-halo-principal") : "";
  const Ops::Ref ref{{owner, op->str(), att}, *inc, nb, digest};
  const std::uint64_t reserve = opt_.base_bytes + prepared->max_tokens * opt_.bytes_per_token;
  DropReleased();
  const Ops::Answer a = ops_.Submit(ref, reserve, inc_.NowMs());
  if (a.verdict != Ops::Verdict::kOk) return Answer(ref, a, inc_.id());

  // Accepted: run it on a worker, and answer once it is admitted (or refused, or could not run).
  auto live = std::make_shared<Live>();
  {
    std::lock_guard<std::mutex> lk(mu_);
    live_[ref.id] = live;
  }
  auto admitted = std::make_shared<Admitted>();
  {
    std::lock_guard<std::mutex> lk(workers_mu_);
    ++workers_;
  }
  std::thread([this, ref, run_api = adapter->second.run, run = std::move(prepared->run), live, admitted]() mutable {
    OperationHooks h;
    h.cancelled = [this, &ref] { return stopping_.load() || ops_.CancelRequested(ref.id); };
    h.admitted = [this, &ref, admitted](const OperationHooks::Admission& adm) {
      if (!adm.refused) {
        Ops::View v;
        v.status = Ops::Status::kRunning;
        admitted->Set(HttpResponse::Json(202, CompactJson(ViewJson(ref, v))));
        return;
      }
      // The scheduler refused it: this copy was not accepted.
      ops_.Withdraw(ref.id);
      HttpResponse r = Error(adm.why == "memory" ? 503 : 429, adm.why == "memory" ? "insufficient_memory" : "too_many_requests",
                             "refused by the scheduler (" + adm.why + "); not accepted");
      r.headers.emplace_back("Retry-After", std::to_string(std::max<std::int64_t>(1, adm.retry_after_s)));
      admitted->Set(std::move(r));
    };
    h.events = [live](std::vector<std::string>& frames) {
      std::lock_guard<std::mutex> lk(live->mu);
      for (auto& f : frames) live->frames.push_back(std::move(f));
      live->cv.notify_all();
    };
    h.done = [this, &ref](OperationHooks::End end) {
      const std::string body = end.error.empty() ? CompactJson(end.final) : [&] {
        Value e = Value::MakeObject();
        e.set("message", Value::String(end.error));
        e.set("type", Value::String("generation_failed"));
        Value o = Value::MakeObject();
        o.set("error", std::move(e));
        return CompactJson(o);
      }();
      ops_.Finish(ref.id, body, {end.prompt_tokens, end.cached_tokens, end.completion_tokens, end.engine_ms}, end.finish,
                  inc_.NowMs());
    };
    if (auto error = run_api(std::move(run), h)) {
      ops_.Withdraw(ref.id);
      admitted->Set(std::move(*error));
    }
    {
      std::lock_guard<std::mutex> lk(live->mu);
      live->ended = true;
      live->cv.notify_all();
    }
    std::lock_guard<std::mutex> lk(workers_mu_);
    if (--workers_ == 0) workers_cv_.notify_all();
  }).detach();
  std::unique_lock<std::mutex> lk(admitted->mu);
  admitted->cv.wait(lk, [&] { return admitted->answer.has_value(); });
  return std::move(*admitted->answer);
}

HttpResponse OperationService::Verb(const HttpRequest& http) {
  // /v1/operations/{op}/{attempt}[/result|/stream|/cancel|/ack]
  const std::string_view rest = std::string_view(http.path).substr(std::string_view("/v1/operations/").size());
  std::vector<std::string_view> parts;
  for (std::size_t at = 0; at <= rest.size();) {
    const std::size_t slash = rest.find('/', at);
    parts.push_back(rest.substr(at, slash == std::string_view::npos ? std::string_view::npos : slash - at));
    if (slash == std::string_view::npos) break;
    at = slash + 1;
  }
  std::uint64_t att = 0;
  if (parts.size() < 2 || parts.size() > 3 || !ValidOpId(std::string(parts[0])) || !ParseU64(parts[1], &att))
    return Error(404, "invalid_request_error", "no operation route for " + http.path);
  const std::string_view what = parts.size() == 3 ? parts[2] : "";
  const bool post = http.method == "POST";
  if ((post && what != "cancel" && what != "ack") || (!post && what != "" && what != "result" && what != "stream"))
    return Error(405, "invalid_request_error", "method not allowed");
  const std::string* inc = http.Header("halo-incarnation");
  const std::string* nbh = http.Header("halo-not-after");
  const std::string* digest = http.Header("halo-digest");
  std::int64_t nb = 0;
  if (!inc || !nbh || !digest || !ParseI64(*nbh, &nb))
    return Error(400, "invalid_request_error", "every verb carries Halo-Incarnation, Halo-Not-After and Halo-Digest");
  const std::string owner = http.Header("x-halo-principal") ? *http.Header("x-halo-principal") : "";
  const Ops::Ref ref{{owner, std::string(parts[0]), att}, *inc, nb, *digest};
  DropReleased();
  const std::int64_t now = inc_.NowMs();
  if (what == "cancel") return Answer(ref, ops_.Cancel(ref, now), inc_.id());
  if (what == "ack") {
    HttpResponse r = Answer(ref, ops_.Ack(ref, now), inc_.id());
    DropReleased();
    return r;
  }
  const Ops::Answer a = ops_.StatusOf(ref, now);
  if (what.empty()) {
    if (a.verdict == Ops::Verdict::kStale) return HttpResponse::Json(200, CompactJson(ViewJson(ref, a.view)));
    return Answer(ref, a, inc_.id());
  }
  if (a.verdict != Ops::Verdict::kOk) return Answer(ref, a, inc_.id());
  if (what == "result") {
    if (auto body = ops_.Fetch(ref, now)) return HttpResponse::Json(200, std::move(*body));
    return HttpResponse::Json(409, CompactJson(ViewJson(ref, a.view)));  // not done, or no longer held
  }
  std::uint64_t from = 0;
  if (http.query.starts_with("from=") && !ParseU64(std::string_view(http.query).substr(5), &from))
    return Error(400, "invalid_request_error", "from: an event index");
  return Observe(ref, from);
}

HttpResponse OperationService::Observe(const Ops::Ref& ref, std::size_t from) {
  std::shared_ptr<Live> live;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (const auto it = live_.find(ref.id); it != live_.end()) live = it->second;
  }
  const Ops::View view = ops_.StatusOf(ref, inc_.NowMs()).view;
  if (!live || (view.status != Ops::Status::kRunning && view.status != Ops::Status::kDone))
    return HttpResponse::Json(410, CompactJson(ViewJson(ref, view)));
  // Events from `from`, then each new one until the attempt ends; then its status as the last event. Leaving the
  // stream is an observer departing: the attempt runs on.
  return HttpResponse::EventStream([this, live, ref, from](const BodyWriter& write) {
    std::size_t next = from;
    for (;;) {
      std::vector<std::string> batch;
      bool ended = false;
      {
        std::unique_lock<std::mutex> lk(live->mu);
        live->cv.wait_for(lk, std::chrono::seconds(15),
                          [&] { return live->ended || live->frames.size() > next || stopping_.load(); });
        for (; next < live->frames.size(); ++next) batch.push_back(live->frames[next]);
        ended = live->ended || stopping_.load();
      }
      for (const auto& f : batch)
        if (!write(f)) return;
      if (ended) break;
      if (batch.empty() && !write(": keepalive\n\n")) return;
    }
    Value status = ViewJson(ref, ops_.StatusOf(ref, inc_.NowMs()).view);
    (void)write("event: status\ndata: " + CompactJson(status) + "\n\n");
  });
}

}  // namespace halo::server
