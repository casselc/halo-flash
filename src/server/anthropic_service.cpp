#include "server/anthropic_service.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>

#include "server/anthropic_response.hpp"
#include "server/openai_response.hpp"
#include "server/turn.hpp"
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
  SetTurnContext(http, g);
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
  GenerationRequest& g = run->req.gen;
  // D5: the key material is the rendered prompt (system, tools and messages).
  DefaultIdempotencyKey(g, run->prompt, Api::kMessages);
  // G3: admitted before any response is sent (a streamed 429 is impossible once message_start went out).
  const auto admission = AdmitTurn(gen_, g, run->prompt, gone);
  if (admission.refused) {
    HttpResponse r = MessagesErrorResponse({429, "rate_limit_error",
                                            "the server is busy (" + admission.why + "); retry after " +
                                                std::to_string(admission.retry_after_s) +
                                                " s with the same Idempotency-Key"});
    r.headers.emplace_back("Retry-After", std::to_string(std::max<std::int64_t>(1, admission.retry_after_s)));
    if (opt_.on_refused) opt_.on_refused(admission.why);
    return r;
  }
  const auto ticket = admission.ticket;

  struct Executed {
    std::string error;
    Value final;
    bool cancelled = false;
  };
  // `events` null: not streaming. `flush` writes the events so far (false once the client is gone); `poll` is
  // Params::cancelled (gone, plus the stream's pings).
  const auto execute = [this, id, run, gone, ticket](MessageBuilder& builder, std::vector<Value>* events,
                                                     const std::function<bool()>& flush,
                                                     const std::function<bool()>& poll) {
    const MessagesRequest& mr = run->req;
    const GenerationRequest& req = mr.gen;
    text::ToolCallStreamParser parser(run->popt);
    std::vector<text::StreamEvent> parsed;
    const auto feed = [&](std::string_view piece) {
      if (piece.empty()) return;
      parsed.clear();
      parser.Push(piece, &parsed);
      builder.Add(parsed, events);
    };
    feed(mr.forced_prefix);  // the call the prompt opened (no event until it completes)
    GenerationRecord rec;
    // disable_parallel_tool_use: the generation ends after the first call (a cut, not a departure).
    const TurnOutcome turn = RunTurn(gen_, opt_, req, run->prompt, ticket,
                                     {[&](std::string_view text) {
                                        feed(text);
                                        return !(mr.disable_parallel_tool_use && builder.tool_uses() > 0);
                                      },
                                      flush, gone, poll},
                                     &rec);
    const TextGenerator::Result& result = turn.result;
    parsed.clear();
    parser.Finish(&parsed);
    builder.Add(parsed, events);
    const bool stop_seq = turn.stopped && !turn.cut;
    const bool length = result.length_limited && !turn.stopped && !turn.cut;
    const StopReason reason =
        ChooseStopReason(length, length && result.completion_tokens < req.max_tokens, stop_seq, builder.tool_uses() > 0);
    const MessagesUsage usage{result.prompt_tokens, result.cached_tokens, result.cache_write_tokens,
                              result.completion_tokens};
    if (result.error.empty()) builder.Finish(reason, turn.matched, usage, events);
    const bool cancelled = turn.client_gone || result.cancelled || !flush() || gone();
    if (cancelled) builder.MarkCancelled();
    ReportTurn(opt_, rec, "messages", id, req, run->prompt, turn, cancelled,
               reason == StopReason::kMaxTokens || reason == StopReason::kContextWindow ? "length"
               : reason == StopReason::kToolUse                                         ? "tool_calls"
                                                                                        : "stop",
               builder.tool_calls());
    return Executed{result.error.empty() ? "" : "generation failed: " + result.error, builder.Final(), cancelled};
  };

  MessageBuilder::Options bo{id, run->served, run->req.omit_thinking,
                             run->req.gen.tool_choice.mode != ToolChoice::Mode::kNone, opt_.new_id};
  std::vector<std::pair<std::string, std::string>> headers = {{"request-id", "req_" + id.substr(4)}};
  if (!run->req.ignored.empty()) headers.emplace_back("x-halo-ignored", Join(run->req.ignored));
  for (auto& h : AnswerHeaders(admission)) headers.push_back(std::move(h));

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
  HttpResponse r = HttpResponse::EventStream([bo, execute, gone, provisional,
                                              keepalive = opt_.sse_keepalive_ms](const BodyWriter& write) {
    MessageBuilder builder(bo);
    // Queued or prefilling (minutes at long context), or between tokens: a ping after `keepalive` ms without a
    // write keeps the client's idle timer from firing, and a client that left fails the write.
    SsePump pump(write, keepalive, [](const Value& e) { return MessagesFrame(e); }, [&builder](SsePump& p) {
      builder.Ping(&p.events);
      p.Flush();
    });
    builder.Start(provisional, &pump.events);
    pump.Flush();
    auto [err, final, cancelled] = execute(builder, &pump.events, [&pump] { return pump.Flush(); },
                                           [&pump, &gone] { return pump.Poll(gone); });
    pump.Flush();
    if (!err.empty() && pump.alive()) (void)pump.Write(MessagesFrame(MessagesErrorBody({500, "api_error", err})));
  });
  for (auto& h : headers) r.headers.push_back(h);
  return r;
}

}  // namespace halo::server
