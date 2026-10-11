#include "server/anthropic_response.hpp"

#include <utility>

#include "server/anthropic_request.hpp"
#include "server/openai_response.hpp"

namespace halo::server {

namespace {

using json::Value;

Value Str(std::string s) { return Value::String(std::move(s)); }
Value U(std::uint64_t n) { return Value::Int(static_cast<std::int64_t>(n)); }

Value Event(const char* type) {
  Value e = Value::MakeObject();
  e.set("type", Str(type));
  return e;
}

}  // namespace

const char* StopReasonName(StopReason r) {
  switch (r) {
    case StopReason::kMaxTokens: return "max_tokens";
    case StopReason::kStopSequence: return "stop_sequence";
    case StopReason::kToolUse: return "tool_use";
    case StopReason::kContextWindow: return "model_context_window_exceeded";
    case StopReason::kEndTurn: break;
  }
  return "end_turn";
}

StopReason ChooseStopReason(bool length_limited, bool context_limited, bool stop_sequence, bool tool_use) {
  if (length_limited) return context_limited ? StopReason::kContextWindow : StopReason::kMaxTokens;
  if (stop_sequence) return StopReason::kStopSequence;
  if (tool_use) return StopReason::kToolUse;
  return StopReason::kEndTurn;
}

std::string MessagesFrame(const json::Value& event) {
  return "event: " + event.find("type")->str() + "\ndata: " + CompactJson(event) + "\n\n";
}

MessageBuilder::MessageBuilder(Options options) : opt_(std::move(options)) {}

Value MessageBuilder::UsageJson() const {
  const std::uint64_t cached = usage_.cache_read + usage_.cache_write;
  Value u = Value::MakeObject();
  u.set("input_tokens", U(usage_.prompt_tokens > cached ? usage_.prompt_tokens - cached : 0));
  u.set("cache_creation_input_tokens", U(usage_.cache_write));
  u.set("cache_read_input_tokens", U(usage_.cache_read));
  u.set("output_tokens", U(usage_.output_tokens));
  return u;
}

void MessageBuilder::Start(std::uint64_t prompt_tokens, std::vector<Value>* out) {
  if (started_ || !out) return;
  started_ = true;
  Value m = Value::MakeObject();
  m.set("id", Str(opt_.id));
  m.set("type", Str("message"));
  m.set("role", Str("assistant"));
  m.set("content", Value::MakeArray());
  m.set("model", Str(opt_.model));
  m.set("stop_reason", Value::Null());
  m.set("stop_sequence", Value::Null());
  Value u = Value::MakeObject();
  u.set("input_tokens", U(prompt_tokens));
  u.set("output_tokens", U(0));
  m.set("usage", std::move(u));
  Value e = Event("message_start");
  e.set("message", std::move(m));
  out->push_back(std::move(e));
}

void MessageBuilder::Ping(std::vector<Value>* out) {
  if (out && started_ && !finished_) out->push_back(Event("ping"));
}

Value MessageBuilder::Delta(std::size_t index, Value delta) const {
  Value e = Event("content_block_delta");
  e.set("index", U(index));
  e.set("delta", std::move(delta));
  return e;
}

Value MessageBuilder::BlockJson(const Block& b, bool final) const {
  Value v = Value::MakeObject();
  switch (b.kind) {
    case BlockKind::kText:
      v.set("type", Str("text"));
      v.set("text", Str(final ? b.text : ""));
      break;
    case BlockKind::kThinking:
      v.set("type", Str("thinking"));
      v.set("thinking", Str(final && !opt_.omit_thinking ? b.text : ""));
      v.set("signature", Str(final ? EncodeThinkingSignature(b.text) : ""));
      break;
    case BlockKind::kToolUse:
      v.set("type", Str("tool_use"));
      v.set("id", Str(b.id));
      v.set("name", Str(b.name));
      v.set("input", final ? b.input : Value::MakeObject());
      break;
  }
  return v;
}

void MessageBuilder::Text(BlockKind kind, std::string_view text, std::vector<Value>* out) {
  if (text.empty()) return;
  if (open_ && blocks_.back().kind != kind) Close(out);
  if (!open_) {
    blocks_.push_back({kind, "", "", "", Value::MakeObject()});
    open_ = true;
    if (out) {
      Value e = Event("content_block_start");
      e.set("index", U(blocks_.size() - 1));
      e.set("content_block", BlockJson(blocks_.back(), false));
      out->push_back(std::move(e));
    }
  }
  blocks_.back().text += text;
  if (!out) return;
  const bool thinking = kind == BlockKind::kThinking;
  if (thinking && opt_.omit_thinking) return;  // the signature carries it
  Value d = Value::MakeObject();
  d.set("type", Str(thinking ? "thinking_delta" : "text_delta"));
  d.set(thinking ? "thinking" : "text", Str(std::string(text)));
  out->push_back(Delta(blocks_.size() - 1, std::move(d)));
}

void MessageBuilder::Close(std::vector<Value>* out) {
  if (!open_) return;
  open_ = false;
  if (!out) return;
  const std::size_t index = blocks_.size() - 1;
  const Block& b = blocks_.back();
  if (b.kind == BlockKind::kThinking) {
    if (opt_.omit_thinking) {  // as the API streams an omitted block: one empty thinking_delta
      Value d = Value::MakeObject();
      d.set("type", Str("thinking_delta"));
      d.set("thinking", Str(""));
      out->push_back(Delta(index, std::move(d)));
    }
    Value d = Value::MakeObject();
    d.set("type", Str("signature_delta"));
    d.set("signature", Str(EncodeThinkingSignature(b.text)));
    out->push_back(Delta(index, std::move(d)));
  }
  Value e = Event("content_block_stop");
  e.set("index", U(index));
  out->push_back(std::move(e));
}

void MessageBuilder::ToolUse(const text::ParsedToolCall& call, std::vector<Value>* out) {
  Close(out);
  Block b{BlockKind::kToolUse, CompactJson(call.arguments), "toolu_" + opt_.ids(), call.name, call.arguments};
  calls_.emplace_back(b.id, b.name);
  blocks_.push_back(std::move(b));
  if (!out) return;
  const std::size_t index = blocks_.size() - 1;
  Value e = Event("content_block_start");
  e.set("index", U(index));
  e.set("content_block", BlockJson(blocks_.back(), false));
  out->push_back(std::move(e));
  Value d = Value::MakeObject();
  d.set("type", Str("input_json_delta"));
  d.set("partial_json", Str(blocks_.back().text));
  out->push_back(Delta(index, std::move(d)));
  e = Event("content_block_stop");
  e.set("index", U(index));
  out->push_back(std::move(e));
}

void MessageBuilder::Add(const std::vector<text::StreamEvent>& events, std::vector<Value>* out) {
  for (const auto& e : events) {
    switch (e.kind) {
      case text::StreamEventKind::kContent:
      case text::StreamEventKind::kError:  // a malformed call stays visible as text
        Text(BlockKind::kText, e.text, out);
        break;
      case text::StreamEventKind::kReasoning: Text(BlockKind::kThinking, e.text, out); break;
      case text::StreamEventKind::kToolCall:
        if (opt_.tools_allowed) ToolUse(e.call, out);
        else Text(BlockKind::kText, e.text, out);
        break;
      case text::StreamEventKind::kFraming: break;
    }
  }
}

void MessageBuilder::Finish(StopReason reason, const std::string& stop_sequence, const MessagesUsage& usage,
                            std::vector<Value>* out) {
  if (finished_) return;
  reason_ = reason;
  stop_sequence_ = reason == StopReason::kStopSequence ? stop_sequence : "";
  usage_ = usage;
  if (out) Start(usage.prompt_tokens, out);
  Close(out);
  finished_ = true;
  if (!out) return;
  Value d = Value::MakeObject();
  d.set("stop_reason", Str(StopReasonName(reason_)));
  d.set("stop_sequence", stop_sequence_.empty() ? Value::Null() : Str(stop_sequence_));
  Value e = Event("message_delta");
  e.set("delta", std::move(d));
  e.set("usage", UsageJson());
  out->push_back(std::move(e));
  out->push_back(Event("message_stop"));
}

Value MessageBuilder::Final() const {
  Value m = Value::MakeObject();
  m.set("id", Str(opt_.id));
  m.set("type", Str("message"));
  m.set("role", Str("assistant"));
  Value content = Value::MakeArray();
  for (const auto& b : blocks_) content.push(BlockJson(b, true));
  m.set("content", std::move(content));
  m.set("model", Str(opt_.model));
  m.set("stop_reason", Str(StopReasonName(reason_)));
  m.set("stop_sequence", stop_sequence_.empty() ? Value::Null() : Str(stop_sequence_));
  m.set("usage", UsageJson());
  return m;
}

}  // namespace halo::server
