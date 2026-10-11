#include "server/openai_service.hpp"

#include <algorithm>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <utility>

#include "server/openai_response.hpp"
#include "server/turn.hpp"
#include "text/tool_call_parser.hpp"

namespace halo::server {

namespace {

using json::Value;
using Kind = json::Value::Kind;

HttpResponse Error(int status, const RequestError& e) { return HttpResponse::Json(status, CompactJson(ErrorBody(e))); }

std::string RandomId() {
  thread_local std::mt19937_64 rng(std::random_device{}());
  char b[33];
  std::snprintf(b, sizeof b, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return b;
}

/// The SSE frame for one event; Responses events carry their type.
std::string Frame(Api api, const Value& event) {
  std::string f;
  if (api == Api::kResponses) f = "event: " + event.find("type")->str() + "\n";
  return f + "data: " + CompactJson(event) + "\n\n";
}

}  // namespace

std::string StopMatcher::Push(std::string_view text) {
  if (stopped_) return {};
  const std::size_t old = pending_.size();
  pending_ += text;
  // As if the text arrived a byte at a time: the output ends where the first
  // stop sequence completes (at that byte, the longest one that does), so
  // the result does not depend on how the text was chunked.
  for (std::size_t end = old + 1; end <= pending_.size(); ++end) {
    std::size_t cut = std::string::npos;
    const std::string* which = nullptr;
    for (const auto& s : stops_)
      if (!s.empty() && s.size() <= end && pending_.compare(end - s.size(), s.size(), s) == 0 &&
          end - s.size() < cut) {
        cut = end - s.size();
        which = &s;
      }
    if (cut != std::string::npos) {
      stopped_ = true;
      matched_ = *which;
      std::string out = pending_.substr(0, cut);
      pending_.clear();
      return out;
    }
  }
  // Hold back the longest suffix that could still grow into a stop sequence.
  std::size_t hold = 0;
  for (const auto& s : stops_)
    for (std::size_t n = std::min(s.size() - 1, pending_.size()); n > hold; --n)
      if (pending_.compare(pending_.size() - n, n, s, 0, n) == 0) {
        hold = n;
        break;
      }
  std::string out = pending_.substr(0, pending_.size() - hold);
  pending_.erase(0, pending_.size() - hold);
  return out;
}

std::string StopMatcher::Flush() {
  if (stopped_) return {};
  return std::exchange(pending_, {});
}

OpenAiService::OpenAiService(TextGenerator& generator, ServiceOptions options)
    : gen_(generator), opt_(std::move(options)) {
  if (opt_.store) {
    store_ = opt_.store;
  } else {
    own_store_ = std::make_unique<InMemoryResponseStore>(opt_.max_stored_responses);
    store_ = own_store_.get();
  }
  if (!opt_.new_id) opt_.new_id = RandomId;
  if (!opt_.now_s)
    opt_.now_s = [] {
      return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
}

void OpenAiService::Register(HttpServer& server) {
  server.Route("GET", "/health", [](const HttpRequest&) { return HttpResponse::Json(200, R"({"status":"ok"})"); });
  server.Route("GET", "/v1/models", [this](const HttpRequest& r) { return Models(r); });
  server.Route("POST", "/v1/completions", [this](const HttpRequest& r) { return Completions(r); });
  server.Route("POST", "/v1/chat/completions", [this](const HttpRequest& r) { return Chat(r); });
  server.Route("POST", "/v1/responses", [this](const HttpRequest& r) { return Responses(r); });
  server.RoutePrefix("GET", "/v1/responses/",
                     [this](const HttpRequest& r) { return GetResponse(r.path.substr(std::string("/v1/responses/").size())); });
}

HttpResponse OpenAiService::Models(const HttpRequest&) {
  Value m = Value::MakeObject();
  m.set("id", Value::String(opt_.model_id));
  m.set("object", Value::String("model"));
  m.set("created", Value::Int(0));
  m.set("owned_by", Value::String("halo-flash"));
  Value data = Value::MakeArray();
  data.push(m);
  if (opt_.variants)
    for (const auto& v : opt_.variants()) {
      Value e = m;
      e.set("id", Value::String(opt_.model_id + ":" + v));
      e.set("parent", Value::String(opt_.model_id));
      data.push(std::move(e));
    }
  Value list = Value::MakeObject();
  list.set("object", Value::String("list"));
  list.set("data", std::move(data));
  return HttpResponse::Json(200, CompactJson(list));
}

namespace {

std::optional<Value> Body(const HttpRequest& req, RequestError* e) {
  std::string err;
  auto v = json::Parse(req.body, &err);
  if (!v) *e = {"the request body is not valid JSON: " + err, ""};
  return v;
}

}  // namespace

HttpResponse OpenAiService::Completions(const HttpRequest& http) {
  RequestError e;
  const auto body = Body(http, &e);
  if (!body) return Error(400, e);
  auto req = ParseCompletionsRequest(*body, opt_.defaults, &e);
  if (!req) return Error(400, e);
  SetTurnContext(http, *req);
  return Generate(Api::kCompletions, std::move(*req), nullptr, http.connected);
}

HttpResponse OpenAiService::Chat(const HttpRequest& http) {
  RequestError e;
  const auto body = Body(http, &e);
  if (!body) return Error(400, e);
  auto req = ParseChatRequest(*body, opt_.defaults, &e);
  if (!req) return Error(400, e);
  SetTurnContext(http, *req);
  return Generate(Api::kChat, std::move(*req), nullptr, http.connected);
}

HttpResponse OpenAiService::Responses(const HttpRequest& http) {
  RequestError e;
  const auto body = Body(http, &e);
  if (!body) return Error(400, e);
  auto req = ParseResponsesRequest(*body, opt_.defaults, &e);
  if (!req) return Error(400, e);
  SetTurnContext(http, *req);
  std::optional<Value> prior;
  if (!req->previous_response_id.empty()) {
    auto stored = store_->Get(req->previous_response_id);
    if (!stored)
      return Error(404, {"previous response not found: " + req->previous_response_id, "previous_response_id"});
    if (stored->conversation.is(Kind::kNull))
      return Error(400, {"previous response " + req->previous_response_id +
                             " was cancelled before it was delivered in full (its client left); continue from an "
                             "earlier response or send the conversation again",
                         "previous_response_id"});
    prior = std::move(stored->conversation);
  }
  return Generate(Api::kResponses, std::move(*req), prior ? &*prior : nullptr, http.connected);
}

std::string OpenAiService::Store(const std::string& id, const Value& conversation, const Value& final) {
  // The stored conversation gains this response's turn, mapped back
  // through the request parser (the round trip the tests check).
  Value body = Value::MakeObject();
  Value input = Value::MakeArray();
  Value marker = Value::MakeObject();
  marker.set("role", Value::String("user"));
  marker.set("content", Value::String(""));
  input.push(std::move(marker));
  for (const auto& item : final.find("output")->items()) input.push(item);
  body.set("input", std::move(input));
  RequestError e;
  auto turn = ParseResponsesRequest(body, opt_.defaults, &e);
  if (!turn) return "its turn does not parse back: " + e.message;
  Value conv = conversation;
  for (std::size_t i = 1; i < turn->messages.items().size(); ++i) conv.push(turn->messages.items()[i]);
  std::string err;
  if (!store_->Put(id, {std::move(conv), final}, &err)) return err.empty() ? std::string("the store refused it") : err;
  return {};
}

HttpResponse OpenAiService::GetResponse(const std::string& id) {
  const auto stored = store_->Get(id);
  if (!stored) return Error(404, {"no stored response " + id, ""});
  return HttpResponse::Json(200, CompactJson(stored->response));
}

HttpResponse OpenAiService::Generate(Api api, GenerationRequest req, const Value* prior,
                                     const std::function<bool()>& connected) {
  if (req.model.empty()) req.model = opt_.model_id;
  if (!req.variant.empty()) {
    const auto names = opt_.variants ? opt_.variants() : std::vector<std::string>{};
    if (std::find(names.begin(), names.end(), req.variant) == names.end())
      return Error(404, {"model " + opt_.model_id + " has no variant \"" + req.variant + "\"", "model"});
  }
  // The conversation: instructions (this request's only), stored turns, new input.
  Value conversation = Value::MakeArray();
  Value messages = Value::MakeArray();
  if (api == Api::kResponses) {
    std::size_t first = 0;
    if (req.has_instructions) messages.push(req.messages.items()[first++]);
    if (prior)
      for (const auto& m : prior->items()) conversation.push(m);
    for (std::size_t i = first; i < req.messages.items().size(); ++i) conversation.push(req.messages.items()[i]);
    for (const auto& m : conversation.items()) messages.push(m);
    req.messages = messages;
  }
  std::string prompt;
  text::ToolCallParserOptions popt;
  if (api == Api::kCompletions) {
    prompt = req.prompt;
  } else {
    std::string err;
    auto rendered = text::RenderChat(opt_.template_kind, req.messages,
                                     req.tools.is(Kind::kNull) ? nullptr : &req.tools, req.template_options, &err);
    if (!rendered) return Error(400, {"the conversation cannot be rendered: " + err, "messages"});
    prompt = std::move(*rendered);
    popt.start = text::GenerationPrompt(opt_.template_kind, req.template_options).ends_with("<think>\n")
                     ? text::StreamStart::kReasoning
                     : text::StreamStart::kContent;
    popt.tools = req.tools;
  }
  const char* prefix = api == Api::kChat ? "chatcmpl-" : api == Api::kResponses ? "resp_" : "cmpl-";
  const std::string id = prefix + opt_.new_id();
  // D5: a turn's key material is (previous_response_id, input): a client's retry of the same turn carries the same key.
  DefaultIdempotencyKey(req, req.previous_response_id + '\n' + (api == Api::kCompletions ? prompt : CompactJson(req.messages)),
                        api);
  const auto gone0 = [connected] { return connected && !connected(); };
  const auto admission = AdmitTurn(gen_, req, prompt, gone0);
  if (admission.refused) {
    // Not done, and never will be: the scheduler took it off the queue as it answered (docs/SCHEDULER.md 4).
    Value inner = Value::MakeObject();
    inner.set("message", Value::String("the server is busy (" + admission.why + "); retry after " +
                                       std::to_string(admission.retry_after_s) + " s with the same Idempotency-Key"));
    inner.set("type", Value::String("rate_limit_exceeded"));
    inner.set("param", Value::Null());
    inner.set("code", Value::String(admission.why == "memory" ? "insufficient_memory" : "too_many_requests"));
    Value e = Value::MakeObject();
    e.set("error", std::move(inner));
    HttpResponse r = HttpResponse::Json(429, CompactJson(e));
    r.headers.emplace_back("Retry-After", std::to_string(std::max<std::int64_t>(1, admission.retry_after_s)));
    if (opt_.on_refused) opt_.on_refused(admission.why);
    return r;
  }

  struct Executed {
    std::string error;  // the request failed (generation, or the store write): answered with it
    Value final;
    bool cancelled = false;  // the client left before the answer was delivered
  };
  // Shared by the streaming callback, which runs after this returns.
  struct Run {
    GenerationRequest req;
    std::string prompt;
    text::ToolCallParserOptions popt;
    Value conversation;
  };
  auto run = std::make_shared<Run>(Run{std::move(req), std::move(prompt), std::move(popt), std::move(conversation)});
  // The client closed its end (or the server is stopping): nobody will receive the rest.
  const auto gone = [connected] { return connected && !connected(); };
  // `poll`: Params::cancelled (gone, plus the stream's keepalive).
  const auto ticket = admission.ticket;
  const auto execute = [this, api, id, run, gone, ticket](std::vector<Value>* events, const std::function<bool()>& flush,
                                                          const std::function<bool()>& poll) {
    OutputBuilder builder(run->req, id, opt_.now_s(), opt_.new_id);
    text::ToolCallStreamParser parser(run->popt);
    std::vector<text::StreamEvent> parsed;
    const auto emit = [&](const std::string& piece) {
      if (piece.empty()) return;
      if (api == Api::kCompletions) {
        builder.AddText(piece, events);
      } else {
        parsed.clear();
        parser.Push(piece, &parsed);
        builder.Add(parsed, events);
      }
    };
    GenerationRecord rec;
    TurnOutcome turn = RunTurn(gen_, opt_, run->req, run->prompt, ticket,
                               {[&](std::string_view text) { return emit(std::string(text)), true; }, flush, gone, poll},
                               &rec);
    TextGenerator::Result& result = turn.result;
    if (api != Api::kCompletions) {
      parsed.clear();
      parser.Finish(&parsed);
      builder.Add(parsed, events);
    }
    const Usage usage{result.prompt_tokens, result.cached_tokens, result.completion_tokens, 0};
    builder.Finish(result.length_limited && !turn.stopped ? FinishReason::kLength : FinishReason::kStop, usage,
                   events);
    // A stored response is answered as finished (the terminal event, or the 200 object) only after its store
    // write succeeded: a failed write turns the answer into an error, so an id a client holds as finished always
    // continues, and nothing is kept for one answered with an error (audit F7).
    const bool storing = api == Api::kResponses && run->req.store && result.error.empty();
    std::optional<Value> terminal;  // held back until the store write
    if (storing && events && !events->empty()) {
      terminal = std::move(events->back());
      events->pop_back();
    }
    // Delivered in full only if the client was there for every event, the terminal one included. Otherwise the
    // response was cut short or never received: it is cancelled, never "completed" or "stop" (a truncated
    // generation must not look like a finished one to the store, the trace or a training set).
    bool cancelled = turn.client_gone || result.cancelled || !flush() || gone();
    const std::string store_error = storing && !cancelled ? Store(id, run->conversation, builder.Final()) : "";
    if (terminal && !cancelled && store_error.empty()) {
      events->push_back(std::move(*terminal));
      cancelled = !flush() || gone();
    }
    const std::string error = !store_error.empty() ? "the response was not stored (" + store_error +
                                                         "), so it is not answered: nothing was kept; retry, or "
                                                         "send store: false"
                              : !result.error.empty() ? "generation failed: " + result.error
                                                      : std::string();
    if (!store_error.empty()) result.error = error;  // the observer records the request as failed, and why
    if (cancelled) builder.MarkCancelled();
    Value final = builder.Final();
    ReportTurn(opt_, rec, api == Api::kChat ? "chat" : api == Api::kResponses ? "responses" : "completions", id, run->req,
               run->prompt, turn, cancelled,
               result.length_limited && !turn.stopped ? "length" : builder.has_tool_calls() ? "tool_calls" : "stop",
               builder.tool_calls());
    if (storing && cancelled && run->req.stream) {
      // A streaming client may hold the id (response.created): GET answers with the cancelled object, and
      // continuing from it is refused (no conversation is kept for a turn that was not delivered). This replaces
      // the turn stored above when the terminal event could not be delivered. A non-streaming client never got
      // the id: nothing is kept.
      std::string err;
      if (!store_->Put(id, {Value::Null(), final}, &err))
        std::fprintf(stderr, "halo_serve: storing cancelled response %s failed: %s\n", id.c_str(), err.c_str());
    }
    return Executed{error, std::move(final), cancelled};
  };

  if (!run->req.stream) {
    auto [error, final, cancelled] = execute(nullptr, [] { return true; }, gone);
    if (!error.empty()) return HttpResponse::Json(500, CompactJson(ErrorBody({error, ""})));
    if (cancelled)  // nobody is there to read it, unless the server is stopping
      return HttpResponse::Json(503, CompactJson(ErrorBody({"generation cancelled: the client left or the server is stopping", ""})));
    return HttpResponse::Json(200, CompactJson(final));
  }
  return HttpResponse::EventStream([api, run, execute, gone, keepalive = opt_.sse_keepalive_ms](const BodyWriter& write) {
    // Nothing else is written while the request is queued or prefilling (minutes at long context): with
    // sse_keepalive_ms, an SSE comment after that long without a write keeps a client's idle-read timer from
    // firing, and a client that left fails the write instead of waiting for its first token (audit D3).
    SsePump pump(write, keepalive, [api](const Value& e) { return Frame(api, e); },
                 [](SsePump& p) { p.Write(": keepalive\n\n"); });
    const auto flush = [&pump] { return pump.Flush(); };
    const auto poll = [&pump, &gone] { return pump.Poll(gone); };
    auto [error, final, cancelled] = execute(&pump.events, flush, poll);
    pump.Flush();
    if (!error.empty() && pump.alive()) {
      Value e = ErrorBody({error, ""});
      if (api == Api::kResponses) e.set("type", Value::String("error"));
      (void)pump.Write(Frame(api, e));
    }
    if (pump.alive() && api != Api::kResponses) (void)pump.Write("data: [DONE]\n\n");
  });
}

}  // namespace halo::server
