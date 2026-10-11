#include "server/turn.hpp"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include "core/sha256.hpp"

namespace halo::server {

namespace {

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void SetTurnContext(const HttpRequest& http, GenerationRequest& req) {
  if (const std::string* t = http.Header("x-halo-trace-id")) req.trace_id = *t;
  if (const std::string* s = http.Header("x-halo-session-id")) req.session_id = *s;
  if (const std::string* p = http.Header("x-halo-prompt")) {
    const auto at = p->rfind('@');
    req.prompt_name = p->substr(0, at);
    if (at != std::string::npos) req.prompt_version = static_cast<std::uint32_t>(std::strtoul(p->c_str() + at + 1, nullptr, 10));
  }
  if (const std::string* v = http.Header("x-halo-variant-id")) req.variant_id = *v;
  if (const std::string* v = http.Header("x-halo-principal")) req.principal = *v;
  if (const std::string* v = http.Header("x-halo-chain-id")) req.chain_root = *v;
  if (const std::string* v = http.Header("idempotency-key")) req.idempotency_key = *v;
}

void DefaultIdempotencyKey(GenerationRequest& req, std::string_view material, Api api) {
  if (!req.idempotency_key.empty() || req.principal.empty()) return;
  req.idempotency_key = "d5:" + core::Sha256Hex(req.principal + '\n' + std::string(material) + '\n' +
                                                std::to_string(static_cast<int>(api)));
}

TextGenerator::Admission AdmitTurn(TextGenerator& gen, const GenerationRequest& req, const std::string& prompt,
                                   const std::function<bool()>& gone) {
  return gen.Admit({prompt, req.variant, req.sampling, req.max_tokens, gone, req.principal, req.chain_root,
                    req.idempotency_key, nullptr});
}

void AddAnswerHeader(HttpResponse& r, const TextGenerator::Admission& a) {
  if (!a.answer.empty()) r.headers.emplace_back("Halo-Answer", a.answer);
}

TurnOutcome RunTurn(TextGenerator& gen, const ServiceOptions& opt, const GenerationRequest& req,
                    const std::string& prompt, std::shared_ptr<void> ticket, const TurnIo& io, GenerationRecord* rec) {
  TurnOutcome out;
  StopMatcher stops(req.stop);
  // The generator must tell a client gone from a stop string: both end the callback with false, but only a stop
  // completes the answer (G3: a completed answer is stored and replayed to the key's retry). So a departure the
  // callback found is also what Params::cancelled reports.
  const auto cancelled_or_gone = [&out, &io] { return out.client_gone || (io.poll && io.poll()); };
  TextGenerator::Params params{prompt, req.variant, req.sampling, req.max_tokens, cancelled_or_gone, req.principal,
                               req.chain_root, req.idempotency_key, std::move(ticket)};
  if (opt.on_activity) opt.on_activity(true);
  struct Done {
    const std::function<void(bool)>& f;
    ~Done() {
      if (f) f(false);
    }
  } done{opt.on_activity};
  const bool observed = opt.observer && rec;
  if (rec) rec->start_ns = NowNs();
  std::int64_t last_ns = 0;
  out.result = gen.Generate(params, [&](std::string_view piece) {
    if (observed) {
      const std::int64_t now = NowNs();
      if (last_ns == 0) rec->first_text_ns = now;
      else rec->gaps_us.push_back(static_cast<std::uint32_t>(std::min<std::int64_t>((now - last_ns) / 1000, UINT32_MAX)));
      last_ns = now;
      rec->output += piece;
    }
    const std::string text = stops.Push(piece);
    const bool more = text.empty() || io.sink(text);
    if (stops.stopped()) return false;
    if (!more) return out.cut = true, false;
    if (!io.flush() || io.gone()) return out.client_gone = true, false;
    return true;
  });
  if (const std::string rest = stops.Flush(); !rest.empty()) (void)io.sink(rest);
  out.stopped = stops.stopped();
  if (out.stopped) out.matched = stops.matched();
  return out;
}

void ReportTurn(const ServiceOptions& opt, GenerationRecord& rec, std::string api, std::string request_id,
                const GenerationRequest& req, const std::string& prompt, const TurnOutcome& out, bool cancelled,
                std::string finish, const std::vector<std::pair<std::string, std::string>>& tool_calls) {
  if (!opt.observer) return;
  const TextGenerator::Result& r = out.result;
  rec.end_ns = NowNs();
  rec.api = std::move(api);
  rec.request_id = std::move(request_id);
  rec.trace_id = req.trace_id;
  rec.session_id = req.session_id;
  rec.prompt_name = req.prompt_name;
  rec.prompt_version = req.prompt_version;
  // The variant that actually ran (the "variant" field or a "base:variant" alias) is authoritative; the
  // X-Halo-Variant-Id header labels requests that name none.
  rec.variant_id = req.variant.empty() ? req.variant_id : req.variant;
  rec.user = req.user;
  rec.model = req.model;
  rec.prompt_tokens = r.prompt_tokens;
  rec.cached_tokens = r.cached_tokens;
  rec.completion_tokens = r.uncounted ? 0 : r.completion_tokens;
  rec.finish_reason = !r.error.empty() ? "error" : r.detached ? "detached" : cancelled ? "cancelled" : std::move(finish);
  rec.sampling = req.sampling;
  rec.max_tokens = req.max_tokens;
  for (const auto& [call_id, name] : tool_calls) {
    rec.tool_call_ids.push_back(call_id);
    rec.tool_names.push_back(name);
  }
  rec.input = prompt;
  rec.error = r.error;
  rec.spec = r.spec;
  opt.observer->OnGeneration(rec);
}

SsePump::SsePump(const BodyWriter& write, int keepalive_ms, std::function<std::string(const json::Value&)> frame,
                 std::function<void(SsePump&)> keepalive)
    : write_(write), keepalive_ms_(keepalive_ms), frame_(std::move(frame)), keepalive_(std::move(keepalive)) {}

bool SsePump::Flush() {
  for (const auto& e : events)
    if (alive_) alive_ = write_(frame_(e));
  if (!events.empty()) last_ = Clock::now();
  events.clear();
  return alive_;
}

bool SsePump::Write(const std::string& raw) {
  if (alive_) alive_ = write_(raw);
  last_ = Clock::now();
  return alive_;
}

bool SsePump::Poll(const std::function<bool()>& gone) {
  if (gone && gone()) return true;
  if (keepalive_ms_ > 0 && alive_ && Clock::now() - last_ >= std::chrono::milliseconds(keepalive_ms_)) keepalive_(*this);
  return !alive_;
}

}  // namespace halo::server
