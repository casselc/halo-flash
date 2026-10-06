#include "server/openai_service.hpp"

#include <algorithm>

#include <chrono>
#include <cstdio>
#include <random>
#include <utility>

#include "server/openai_response.hpp"
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
    for (const auto& s : stops_)
      if (!s.empty() && s.size() <= end && pending_.compare(end - s.size(), s.size(), s) == 0)
        cut = std::min(cut, end - s.size());
    if (cut != std::string::npos) {
      stopped_ = true;
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

void SetContext(const HttpRequest& http, GenerationRequest& req) {
  if (const std::string* t = http.Header("x-halo-trace-id")) req.trace_id = *t;
  if (const std::string* s = http.Header("x-halo-session-id")) req.session_id = *s;
}

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

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
  SetContext(http, *req);
  return Generate(Api::kCompletions, std::move(*req), nullptr);
}

HttpResponse OpenAiService::Chat(const HttpRequest& http) {
  RequestError e;
  const auto body = Body(http, &e);
  if (!body) return Error(400, e);
  auto req = ParseChatRequest(*body, opt_.defaults, &e);
  if (!req) return Error(400, e);
  SetContext(http, *req);
  return Generate(Api::kChat, std::move(*req), nullptr);
}

HttpResponse OpenAiService::Responses(const HttpRequest& http) {
  RequestError e;
  const auto body = Body(http, &e);
  if (!body) return Error(400, e);
  auto req = ParseResponsesRequest(*body, opt_.defaults, &e);
  if (!req) return Error(400, e);
  SetContext(http, *req);
  std::optional<Value> prior;
  if (!req->previous_response_id.empty()) {
    auto stored = store_->Get(req->previous_response_id);
    if (!stored)
      return Error(404, {"previous response not found: " + req->previous_response_id, "previous_response_id"});
    prior = std::move(stored->conversation);
  }
  return Generate(Api::kResponses, std::move(*req), prior ? &*prior : nullptr);
}

HttpResponse OpenAiService::GetResponse(const std::string& id) {
  const auto stored = store_->Get(id);
  if (!stored) return Error(404, {"no stored response " + id, ""});
  return HttpResponse::Json(200, CompactJson(stored->response));
}

HttpResponse OpenAiService::Generate(Api api, GenerationRequest req, const Value* prior) {
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

  // Shared by the streaming callback, which runs after this returns.
  struct Run {
    GenerationRequest req;
    std::string prompt;
    text::ToolCallParserOptions popt;
    Value conversation;
  };
  auto run = std::make_shared<Run>(Run{std::move(req), std::move(prompt), std::move(popt), std::move(conversation)});
  const auto execute = [this, api, id, run](std::vector<Value>* events, const std::function<bool()>& flush) {
    OutputBuilder builder(run->req, id, opt_.now_s(), opt_.new_id);
    text::ToolCallStreamParser parser(run->popt);
    StopMatcher stops(run->req.stop);
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
    TextGenerator::Params params{run->prompt, run->req.variant, run->req.sampling, run->req.max_tokens};
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
    const auto result = gen_.Generate(params, [&](std::string_view piece) {
      if (opt_.observer) {
        const std::int64_t now = NowNs();
        if (last_ns == 0) rec.first_text_ns = now;
        else rec.gaps_us.push_back(static_cast<std::uint32_t>(std::min<std::int64_t>((now - last_ns) / 1000, UINT32_MAX)));
        last_ns = now;
        rec.output += piece;
      }
      emit(stops.Push(piece));
      return !stops.stopped() && flush();
    });
    emit(stops.Flush());
    if (api != Api::kCompletions) {
      parsed.clear();
      parser.Finish(&parsed);
      builder.Add(parsed, events);
    }
    const Usage usage{result.prompt_tokens, result.cached_tokens, result.completion_tokens, 0};
    builder.Finish(result.length_limited && !stops.stopped() ? FinishReason::kLength : FinishReason::kStop, usage,
                   events);
    Value final = builder.Final();
    if (opt_.observer) {
      rec.end_ns = NowNs();
      rec.api = api == Api::kChat ? "chat" : api == Api::kResponses ? "responses" : "completions";
      rec.request_id = id;
      rec.trace_id = run->req.trace_id;
      rec.session_id = run->req.session_id;
      rec.user = run->req.user;
      rec.model = run->req.model;
      rec.prompt_tokens = result.prompt_tokens;
      rec.cached_tokens = result.cached_tokens;
      rec.completion_tokens = result.completion_tokens;
      rec.finish_reason = !result.error.empty()                                ? "error"
                          : result.length_limited && !stops.stopped()          ? "length"
                          : builder.has_tool_calls()                           ? "tool_calls"
                                                                               : "stop";
      rec.sampling = run->req.sampling;
      rec.max_tokens = run->req.max_tokens;
      for (const auto& [call_id, name] : builder.tool_calls()) {
        rec.tool_call_ids.push_back(call_id);
        rec.tool_names.push_back(name);
      }
      rec.input = run->prompt;
      rec.error = result.error;
      opt_.observer->OnGeneration(rec);
    }
    if (api == Api::kResponses && run->req.store && result.error.empty()) {
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
      if (auto turn = ParseResponsesRequest(body, opt_.defaults, &e)) {
        Value conv = run->conversation;
        for (std::size_t i = 1; i < turn->messages.items().size(); ++i) conv.push(turn->messages.items()[i]);
        store_->Put(id, {std::move(conv), final});
      }
    }
    return std::make_pair(result.error, std::move(final));
  };

  if (!run->req.stream) {
    auto [error, final] = execute(nullptr, [] { return true; });
    if (!error.empty()) return HttpResponse::Json(500, CompactJson(ErrorBody({"generation failed: " + error, ""})));
    return HttpResponse::Json(200, CompactJson(final));
  }
  return HttpResponse::EventStream([api, run, execute](const BodyWriter& write) {
    std::vector<Value> events;
    bool alive = true;
    const auto flush = [&] {
      for (const auto& e : events)
        if (alive) alive = write(Frame(api, e));
      events.clear();
      return alive;
    };
    auto [error, final] = execute(&events, flush);
    flush();
    if (!error.empty() && alive) {
      Value e = ErrorBody({"generation failed: " + error, ""});
      if (api == Api::kResponses) e.set("type", Value::String("error"));
      alive = write(Frame(api, e));
    }
    if (alive && api != Api::kResponses) (void)write("data: [DONE]\n\n");
  });
}

}  // namespace halo::server
