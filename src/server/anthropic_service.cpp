#include "server/anthropic_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <utility>

#include "server/anthropic_response.hpp"
#include "server/openai_response.hpp"
#include "text/chat_template.hpp"
#include "text/tool_call_parser.hpp"

namespace halo::server {

namespace {

using json::Value;
using Kind = json::Value::Kind;

std::string RandomId() {
  thread_local std::mt19937_64 rng(std::random_device{}());
  char b[33];
  std::snprintf(b, sizeof b, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return b;
}

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string Join(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "" : ",") + x;
  return s;
}

}  // namespace

HttpResponse MessagesErrorResponse(const MessagesError& e) {
  return HttpResponse::Json(e.status, CompactJson(MessagesErrorBody(e)));
}

struct AnthropicService::Prepared {
  MessagesRequest req;
  std::string served;  // the model that serves it (an alias's target)
  std::string prompt;  // rendered, plus the forced tool-call prefix
  text::ToolCallParserOptions popt;
};

AnthropicService::AnthropicService(TextGenerator& generator, ServiceOptions options, std::vector<ModelAlias> aliases)
    : gen_(generator), opt_(std::move(options)), aliases_(std::move(aliases)) {
  if (!opt_.new_id) opt_.new_id = RandomId;
}

void AnthropicService::Register(HttpServer& server) {
  server.Route("POST", "/v1/messages", [this](const HttpRequest& r) { return Messages(r); });
  server.Route("POST", "/v1/messages/count_tokens", [this](const HttpRequest& r) { return CountTokens(r); });
  server.AuthErrorBody("/v1/messages", CompactJson(MessagesErrorBody({401, "authentication_error", "invalid x-api-key"})));
}

std::shared_ptr<AnthropicService::Prepared> AnthropicService::Prepare(const HttpRequest& http, bool count_tokens,
                                                                      HttpResponse* error) {
  std::string err;
  const auto body = json::Parse(http.body, &err);
  if (!body) return *error = MessagesErrorResponse({400, "invalid_request_error", "the request body is not valid JSON: " + err}), nullptr;
  MessagesError e;
  auto req = ParseMessagesRequest(*body, opt_.defaults, count_tokens, &e);
  if (!req) return *error = MessagesErrorResponse(e), nullptr;
  auto p = std::make_shared<Prepared>();
  p->req = std::move(*req);
  GenerationRequest& g = p->req.gen;
  const auto served = ResolveModel(g.model, opt_.model_id, aliases_);
  if (!served)
    return *error = MessagesErrorResponse({404, "not_found_error",
                                           "model: " + g.model + " is not served here (this server serves " +
                                               opt_.model_id + "; see --anthropic-alias)"}),
           nullptr;
  p->served = *served;
  // An alias may name a P10 variant ("<model id>:<variant>").
  if (p->served != opt_.model_id && p->served.starts_with(opt_.model_id + ":")) {
    g.variant = p->served.substr(opt_.model_id.size() + 1);
    const auto names = opt_.variants ? opt_.variants() : std::vector<std::string>{};
    if (std::find(names.begin(), names.end(), g.variant) == names.end())
      return *error = MessagesErrorResponse({404, "not_found_error", "model " + opt_.model_id + " has no variant \"" + g.variant + "\""}),
             nullptr;
  } else if (p->served != opt_.model_id) {
    return *error = MessagesErrorResponse({404, "not_found_error", "model: alias target " + p->served + " is not served here"}),
           nullptr;
  }
  if (const std::string* t = http.Header("x-halo-trace-id")) g.trace_id = *t;
  if (const std::string* s = http.Header("x-halo-session-id")) g.session_id = *s;
  auto rendered = text::RenderChat(opt_.template_kind, g.messages, g.tools.is(Kind::kNull) ? nullptr : &g.tools,
                                   g.template_options, &err);
  if (!rendered) return *error = MessagesErrorResponse({400, "invalid_request_error", "messages: the conversation cannot be rendered: " + err}), nullptr;
  p->prompt = std::move(*rendered) + p->req.forced_prefix;
  p->popt.start = !p->req.forced_prefix.empty() ? text::StreamStart::kContent
                  : text::GenerationPrompt(opt_.template_kind, g.template_options).ends_with("<think>\n")
                      ? text::StreamStart::kReasoning
                      : text::StreamStart::kContent;
  p->popt.tools = g.tools;
  return p;
}

HttpResponse AnthropicService::CountTokens(const HttpRequest& http) {
  HttpResponse error;
  const auto p = Prepare(http, /*count_tokens=*/true, &error);
  if (!p) return error;
  const auto n = gen_.CountTokens(p->prompt);
  if (!n) return MessagesErrorResponse({500, "api_error", "this server's generator cannot count tokens"});
  Value v = Value::MakeObject();
  v.set("input_tokens", Value::Int(static_cast<std::int64_t>(*n)));
  return HttpResponse::Json(200, CompactJson(v));
}

HttpResponse AnthropicService::Messages(const HttpRequest& http) {
  HttpResponse error;
  const auto run = Prepare(http, /*count_tokens=*/false, &error);
  if (!run) return error;
  const std::string id = "msg_" + opt_.new_id();
  const std::function<bool()> connected = http.connected;
  const auto gone = [connected] { return connected && !connected(); };

  struct Executed {
    std::string error;
    Value final;
    bool cancelled = false;
  };
  // `events` null: not streaming. `flush` writes the events so far (false once the client is gone); `poll` is
  // Params::cancelled (gone, plus the stream's pings); `started` ran message_start already.
  const auto execute = [this, id, run, gone](MessageBuilder& builder, std::vector<Value>* events,
                                             const std::function<bool()>& flush, const std::function<bool()>& poll) {
    const MessagesRequest& mr = run->req;
    const GenerationRequest& req = mr.gen;
    text::ToolCallStreamParser parser(run->popt);
    StopMatcher stops(req.stop);
    std::vector<text::StreamEvent> parsed;
    const auto feed = [&](std::string_view piece) {
      if (piece.empty()) return;
      parsed.clear();
      parser.Push(piece, &parsed);
      builder.Add(parsed, events);
    };
    feed(mr.forced_prefix);  // the call the prompt opened (no event until it completes)
    TextGenerator::Params params{run->prompt, req.variant, req.sampling, req.max_tokens, poll};
    if (opt_.on_activity) opt_.on_activity(true);
    struct Done {
      const std::function<void(bool)>& f;
      ~Done() {
        if (f) f(false);
      }
    } done{opt_.on_activity};
    GenerationRecord rec;
    rec.start_ns = NowNs();
    std::int64_t last_ns = 0;
    bool client_gone = false;
    bool parallel_cut = false;  // disable_parallel_tool_use: ended after the first call
    auto result = gen_.Generate(params, [&](std::string_view piece) {
      if (opt_.observer) {
        const std::int64_t now = NowNs();
        if (last_ns == 0) rec.first_text_ns = now;
        else rec.gaps_us.push_back(static_cast<std::uint32_t>(std::min<std::int64_t>((now - last_ns) / 1000, UINT32_MAX)));
        last_ns = now;
        rec.output += piece;
      }
      feed(stops.Push(piece));
      if (stops.stopped()) return false;
      if (mr.disable_parallel_tool_use && builder.tool_uses() > 0) return parallel_cut = true, false;
      if (!flush() || gone()) return client_gone = true, false;
      return true;
    });
    feed(stops.Flush());
    parsed.clear();
    parser.Finish(&parsed);
    builder.Add(parsed, events);
    const bool stop_seq = stops.stopped() && !parallel_cut;
    const bool length = result.length_limited && !stops.stopped() && !parallel_cut;
    const StopReason reason =
        ChooseStopReason(length, length && result.completion_tokens < req.max_tokens, stop_seq, builder.tool_uses() > 0);
    const MessagesUsage usage{result.prompt_tokens, result.cached_tokens, result.cache_write_tokens,
                              result.completion_tokens};
    if (result.error.empty()) builder.Finish(reason, stops.matched(), usage, events);
    const bool cancelled = client_gone || result.cancelled || !flush() || gone();
    if (cancelled) builder.MarkCancelled();
    if (opt_.observer) {
      rec.end_ns = NowNs();
      rec.api = "messages";
      rec.request_id = id;
      rec.trace_id = req.trace_id;
      rec.session_id = req.session_id;
      rec.variant_id = req.variant;
      rec.user = req.user;
      rec.model = req.model;
      rec.prompt_tokens = result.prompt_tokens;
      rec.cached_tokens = result.cached_tokens;
      rec.completion_tokens = result.completion_tokens;
      rec.finish_reason = !result.error.empty() ? "error"
                          : cancelled           ? "cancelled"
                          : reason == StopReason::kMaxTokens || reason == StopReason::kContextWindow ? "length"
                          : reason == StopReason::kToolUse                                             ? "tool_calls"
                                                                                                       : "stop";
      rec.sampling = req.sampling;
      rec.max_tokens = req.max_tokens;
      for (const auto& [call_id, name] : builder.tool_calls()) {
        rec.tool_call_ids.push_back(call_id);
        rec.tool_names.push_back(name);
      }
      rec.input = run->prompt;
      rec.error = result.error;
      rec.spec = result.spec;
      opt_.observer->OnGeneration(rec);
    }
    return Executed{result.error.empty() ? "" : "generation failed: " + result.error, builder.Final(), cancelled};
  };

  MessageBuilder::Options bo{id, run->served, run->req.omit_thinking,
                             run->req.gen.tool_choice.mode != ToolChoice::Mode::kNone, opt_.new_id};
  std::vector<std::pair<std::string, std::string>> headers = {{"request-id", "req_" + id.substr(4)}};
  if (!run->req.ignored.empty()) headers.emplace_back("x-halo-ignored", Join(run->req.ignored));

  if (!run->req.gen.stream) {
    MessageBuilder builder(bo);
    auto [err, final, cancelled] = execute(builder, nullptr, [] { return true; }, gone);
    HttpResponse r = !err.empty() ? MessagesErrorResponse({500, "api_error", err})
                     : cancelled  ? MessagesErrorResponse({503, "api_error", "generation cancelled: the client left or the server is stopping"})
                                  : HttpResponse::Json(200, CompactJson(final));
    for (auto& h : headers) r.headers.push_back(h);
    return r;
  }
  // The provisional input_tokens of message_start: the whole prompt (the cache is consulted when it runs).
  const std::uint64_t provisional = gen_.CountTokens(run->prompt).value_or(0);
  HttpResponse r = HttpResponse::EventStream([bo, run, execute, gone, provisional,
                                              keepalive = opt_.sse_keepalive_ms](const BodyWriter& write) {
    using Clock = std::chrono::steady_clock;
    MessageBuilder builder(bo);
    std::vector<Value> events;
    bool alive = true;
    auto last = Clock::now();
    const auto flush = [&] {
      for (const auto& e : events)
        if (alive) alive = write(MessagesFrame(e));
      if (!events.empty()) last = Clock::now();
      events.clear();
      return alive;
    };
    builder.Start(provisional, &events);
    flush();
    // Queued or prefilling (minutes at long context), or between tokens: a ping after `keepalive` ms without a
    // write keeps the client's idle timer from firing, and a client that left fails the write.
    const auto poll = [&] {
      if (gone()) return true;
      if (keepalive > 0 && alive && Clock::now() - last >= std::chrono::milliseconds(keepalive)) {
        builder.Ping(&events);
        flush();
      }
      return !alive;
    };
    auto [err, final, cancelled] = execute(builder, &events, flush, poll);
    flush();
    if (!err.empty() && alive) alive = write(MessagesFrame(MessagesErrorBody({500, "api_error", err})));
  });
  for (auto& h : headers) r.headers.push_back(h);
  return r;
}

}  // namespace halo::server
