#include "server/openai_response.hpp"

#include <utility>

namespace halo::server {

namespace {

using json::Value;
using Kind = json::Value::Kind;

Value Str(std::string s) { return Value::String(std::move(s)); }
Value U(std::uint64_t n) { return Value::Int(static_cast<std::int64_t>(n)); }

void Compact(const Value& v, std::string& out) {
  switch (v.kind()) {
    case Kind::kArray: {
      out += '[';
      bool first = true;
      for (const auto& x : v.items()) {
        if (!first) out += ',';
        first = false;
        Compact(x, out);
      }
      out += ']';
      break;
    }
    case Kind::kObject: {
      out += '{';
      bool first = true;
      for (const auto& [k, x] : v.members()) {
        if (!first) out += ',';
        first = false;
        out += json::DumpPython(Value::String(k));
        out += ':';
        Compact(x, out);
      }
      out += '}';
      break;
    }
    default: out += json::DumpPython(v);  // scalars: the same text either way
  }
}

const char* ToolChoiceName(ToolChoice::Mode m) {
  switch (m) {
    case ToolChoice::Mode::kNone: return "none";
    case ToolChoice::Mode::kRequired: return "required";
    default: return "auto";
  }
}

}  // namespace

std::string CompactJson(const json::Value& v) {
  std::string out;
  Compact(v, out);
  return out;
}

OutputBuilder::OutputBuilder(const GenerationRequest& request, std::string id, std::int64_t created_s, IdSource ids)
    : req_(request), id_(std::move(id)), created_(created_s), ids_(std::move(ids)) {}

Value OutputBuilder::Event(const char* type) {
  Value e = Value::MakeObject();
  e.set("type", Str(type));
  e.set("sequence_number", U(sequence_++));
  return e;
}

Value OutputBuilder::ChatChunk(Value delta, const char* finish) const {
  Value choice = Value::MakeObject();
  choice.set("index", Value::Int(0));
  if (req_.api == Api::kCompletions) {
    const Value* t = delta.find("text");
    choice.set("text", t ? *t : Str(""));
  } else {
    choice.set("delta", std::move(delta));
  }
  choice.set("logprobs", Value::Null());
  choice.set("finish_reason", finish ? Str(finish) : Value::Null());
  Value c = Value::MakeObject();
  c.set("id", Str(id_));
  c.set("object", Str(req_.api == Api::kCompletions ? "text_completion" : "chat.completion.chunk"));
  c.set("created", Value::Int(created_));
  c.set("model", Str(req_.model));
  Value choices = Value::MakeArray();
  choices.push(std::move(choice));
  c.set("choices", std::move(choices));
  return c;
}

void OutputBuilder::StartStream(std::vector<Value>* out) {
  if (started_ || !out) return;
  started_ = true;
  if (req_.api == Api::kChat) {
    Value d = Value::MakeObject();
    d.set("role", Str("assistant"));
    d.set("content", Str(""));
    out->push_back(ChatChunk(std::move(d), nullptr));
  } else if (req_.api == Api::kResponses) {
    Value e = Event("response.created");
    e.set("response", ResponseJson("in_progress"));
    out->push_back(std::move(e));
    e = Event("response.in_progress");
    e.set("response", ResponseJson("in_progress"));
    out->push_back(std::move(e));
  }
}

Value OutputBuilder::ItemJson(const Item& item, bool done) const {
  Value v = Value::MakeObject();
  switch (item.kind) {
    case Item::Kind::kReasoning: {
      v.set("type", Str("reasoning"));
      v.set("id", Str(item.id));
      v.set("summary", Value::MakeArray());
      Value content = Value::MakeArray();
      if (done) {
        Value part = Value::MakeObject();
        part.set("type", Str("reasoning_text"));
        part.set("text", Str(item.text));
        content.push(std::move(part));
      }
      v.set("content", std::move(content));
      break;
    }
    case Item::Kind::kMessage: {
      v.set("type", Str("message"));
      v.set("id", Str(item.id));
      v.set("status", Str(done ? "completed" : "in_progress"));
      v.set("role", Str("assistant"));
      Value content = Value::MakeArray();
      if (done) {
        Value part = Value::MakeObject();
        part.set("type", Str("output_text"));
        part.set("text", Str(item.text));
        part.set("annotations", Value::MakeArray());
        content.push(std::move(part));
      }
      v.set("content", std::move(content));
      break;
    }
    case Item::Kind::kFunctionCall: {
      const Call& c = calls_[item.call];
      v.set("type", Str("function_call"));
      v.set("id", Str(c.item_id));
      v.set("call_id", Str(c.call_id));
      v.set("name", Str(c.name));
      if (!c.ns.empty()) v.set("namespace", Str(c.ns));
      v.set("arguments", Str(done ? c.arguments : ""));
      v.set("status", Str(done ? "completed" : "in_progress"));
      break;
    }
  }
  return v;
}

Value OutputBuilder::ResponseJson(const char* status) const {
  const bool final = std::string_view(status) != "in_progress";
  Value r = Value::MakeObject();
  r.set("id", Str(id_));
  r.set("object", Str("response"));
  r.set("created_at", Value::Int(created_));
  r.set("status", Str(status));
  r.set("error", Value::Null());
  if (std::string_view(status) == "incomplete") {
    Value d = Value::MakeObject();
    d.set("reason", Str("max_output_tokens"));
    r.set("incomplete_details", std::move(d));
  } else {
    r.set("incomplete_details", Value::Null());
  }
  r.set("max_output_tokens", U(req_.max_tokens));
  r.set("model", Str(req_.model));
  Value output = Value::MakeArray();
  if (final)
    for (const auto& item : items_) output.push(ItemJson(item, true));
  r.set("output", std::move(output));
  r.set("parallel_tool_calls", Value::Bool(req_.parallel_tool_calls));
  r.set("previous_response_id", req_.previous_response_id.empty() ? Value::Null() : Str(req_.previous_response_id));
  Value reasoning = Value::MakeObject();
  reasoning.set("effort", req_.template_options.enable_thinking == false ? Str("none")
                          : req_.template_options.reasoning_effort    ? Str(*req_.template_options.reasoning_effort)
                                                                      : Value::Null());
  r.set("reasoning", std::move(reasoning));
  r.set("store", Value::Bool(req_.store));
  r.set("temperature", Value::Float(req_.sampling.temperature));
  Value format = Value::MakeObject();
  switch (req_.format.kind) {
    case ResponseFormat::Kind::kText: format.set("type", Str("text")); break;
    case ResponseFormat::Kind::kJsonObject: format.set("type", Str("json_object")); break;
    case ResponseFormat::Kind::kJsonSchema:
      format.set("type", Str("json_schema"));
      format.set("name", Str(req_.format.name));
      format.set("schema", req_.format.schema);
      format.set("strict", Value::Bool(req_.format.strict));
      break;
  }
  Value text = Value::MakeObject();
  text.set("format", std::move(format));
  r.set("text", std::move(text));
  if (req_.tool_choice.mode == ToolChoice::Mode::kFunction) {
    Value c = Value::MakeObject();
    c.set("type", Str("function"));
    c.set("name", Str(req_.tool_choice.function));
    r.set("tool_choice", std::move(c));
  } else {
    r.set("tool_choice", Str(ToolChoiceName(req_.tool_choice.mode)));
  }
  // Tools back in the Responses (flat) shape.
  Value tools = Value::MakeArray();
  if (req_.tools.is(Kind::kArray))
    for (const auto& t : req_.tools.items()) {
      Value flat = Value::MakeObject();
      flat.set("type", Str("function"));
      if (const Value* f = t.find("function"))
        for (const auto& [k, v] : f->members()) flat.set(k, v);
      tools.push(std::move(flat));
    }
  r.set("tools", std::move(tools));
  r.set("top_p", Value::Float(req_.sampling.top_p));
  r.set("truncation", Str(req_.truncation_auto ? "auto" : "disabled"));
  if (final) {
    Value u = Value::MakeObject();
    u.set("input_tokens", U(usage_.prompt_tokens));
    Value in = Value::MakeObject();
    in.set("cached_tokens", U(usage_.cached_tokens));
    u.set("input_tokens_details", std::move(in));
    u.set("output_tokens", U(usage_.completion_tokens));
    Value outd = Value::MakeObject();
    outd.set("reasoning_tokens", U(usage_.reasoning_tokens));
    u.set("output_tokens_details", std::move(outd));
    u.set("total_tokens", U(usage_.prompt_tokens + usage_.completion_tokens));
    r.set("usage", std::move(u));
  } else {
    r.set("usage", Value::Null());
  }
  r.set("user", req_.user.empty() ? Value::Null() : Str(req_.user));
  r.set("metadata", req_.metadata.is(Kind::kObject) ? req_.metadata : Value::MakeObject());
  return r;
}

void OutputBuilder::Text(Item::Kind kind, std::string_view text, std::vector<Value>* out) {
  if (text.empty()) return;
  StartStream(out);
  const bool reasoning = kind == Item::Kind::kReasoning;
  if (req_.api != Api::kResponses) {
    (reasoning ? reasoning_ : content_) += text;
    if (out) {
      Value d = Value::MakeObject();
      d.set(req_.api == Api::kCompletions ? "text" : reasoning ? "reasoning_content" : "content", Str(std::string(text)));
      out->push_back(ChatChunk(std::move(d), nullptr));
    }
    return;
  }
  if (item_open_ && items_.back().kind != kind) CloseItem(out);
  const std::size_t index = items_.size() - (item_open_ ? 1 : 0);
  if (!item_open_) {
    items_.push_back({kind, std::string(reasoning ? "rs_" : "msg_") + ids_(), "", 0});
    item_open_ = true;
    if (out) {
      Value e = Event("response.output_item.added");
      e.set("output_index", U(index));
      e.set("item", ItemJson(items_.back(), false));
      out->push_back(std::move(e));
      if (!reasoning) {
        e = Event("response.content_part.added");
        e.set("item_id", Str(items_.back().id));
        e.set("output_index", U(index));
        e.set("content_index", Value::Int(0));
        Value part = Value::MakeObject();
        part.set("type", Str("output_text"));
        part.set("text", Str(""));
        part.set("annotations", Value::MakeArray());
        e.set("part", std::move(part));
        out->push_back(std::move(e));
      }
    }
  }
  items_.back().text += text;
  if (out) {
    Value e = Event(reasoning ? "response.reasoning_text.delta" : "response.output_text.delta");
    e.set("item_id", Str(items_.back().id));
    e.set("output_index", U(index));
    e.set("content_index", Value::Int(0));
    e.set("delta", Str(std::string(text)));
    if (!reasoning) e.set("logprobs", Value::MakeArray());
    out->push_back(std::move(e));
  }
}

void OutputBuilder::CloseItem(std::vector<Value>* out) {
  if (!item_open_) return;
  item_open_ = false;
  if (!out) return;
  const Item& item = items_.back();
  const std::size_t index = items_.size() - 1;
  const bool reasoning = item.kind == Item::Kind::kReasoning;
  Value e = Event(reasoning ? "response.reasoning_text.done" : "response.output_text.done");
  e.set("item_id", Str(item.id));
  e.set("output_index", U(index));
  e.set("content_index", Value::Int(0));
  e.set("text", Str(item.text));
  if (!reasoning) e.set("logprobs", Value::MakeArray());
  out->push_back(std::move(e));
  if (!reasoning) {
    e = Event("response.content_part.done");
    e.set("item_id", Str(item.id));
    e.set("output_index", U(index));
    e.set("content_index", Value::Int(0));
    Value part = Value::MakeObject();
    part.set("type", Str("output_text"));
    part.set("text", Str(item.text));
    part.set("annotations", Value::MakeArray());
    e.set("part", std::move(part));
    out->push_back(std::move(e));
  }
  e = Event("response.output_item.done");
  e.set("output_index", U(index));
  e.set("item", ItemJson(item, true));
  out->push_back(std::move(e));
}

void OutputBuilder::ToolCall(const text::ParsedToolCall& parsed, std::vector<Value>* out) {
  StartStream(out);
  Call c{"fc_" + ids_(), "call_" + ids_(), parsed.name, CompactJson(parsed.arguments)};
  if (req_.api == Api::kResponses)
    if (const auto* t = req_.FindNamespaced(parsed.name)) {
      c.name = t->name;
      c.ns = t->ns;
    }
  calls_.push_back(c);
  if (req_.api == Api::kCompletions) {
    // No parsing for completions; never reached (AddText), kept total.
    return;
  }
  if (req_.api == Api::kChat) {
    if (out) {
      Value fn = Value::MakeObject();
      fn.set("name", Str(c.name));
      fn.set("arguments", Str(c.arguments));
      Value call = Value::MakeObject();
      call.set("index", U(calls_.size() - 1));
      call.set("id", Str(c.call_id));
      call.set("type", Str("function"));
      call.set("function", std::move(fn));
      Value list = Value::MakeArray();
      list.push(std::move(call));
      Value d = Value::MakeObject();
      d.set("tool_calls", std::move(list));
      out->push_back(ChatChunk(std::move(d), nullptr));
    }
    return;
  }
  CloseItem(out);
  items_.push_back({Item::Kind::kFunctionCall, c.item_id, "", calls_.size() - 1});
  if (!out) return;
  const std::size_t index = items_.size() - 1;
  Value e = Event("response.output_item.added");
  e.set("output_index", U(index));
  e.set("item", ItemJson(items_.back(), false));
  out->push_back(std::move(e));
  e = Event("response.function_call_arguments.delta");
  e.set("item_id", Str(c.item_id));
  e.set("output_index", U(index));
  e.set("delta", Str(c.arguments));
  out->push_back(std::move(e));
  e = Event("response.function_call_arguments.done");
  e.set("item_id", Str(c.item_id));
  e.set("output_index", U(index));
  e.set("name", Str(c.name));
  e.set("arguments", Str(c.arguments));
  out->push_back(std::move(e));
  e = Event("response.output_item.done");
  e.set("output_index", U(index));
  e.set("item", ItemJson(items_.back(), true));
  out->push_back(std::move(e));
}

void OutputBuilder::Add(const std::vector<text::StreamEvent>& events, std::vector<Value>* out) {
  for (const auto& e : events) {
    switch (e.kind) {
      case text::StreamEventKind::kContent:
      case text::StreamEventKind::kError:  // a malformed call stays visible as content
        Text(Item::Kind::kMessage, e.text, out);
        break;
      case text::StreamEventKind::kReasoning: Text(Item::Kind::kReasoning, e.text, out); break;
      case text::StreamEventKind::kToolCall: ToolCall(e.call, out); break;
      case text::StreamEventKind::kFraming: break;
    }
  }
}

void OutputBuilder::AddText(std::string_view text, std::vector<Value>* out) { Text(Item::Kind::kMessage, text, out); }

void OutputBuilder::Finish(FinishReason reason, const Usage& usage, std::vector<Value>* out) {
  if (finished_) return;
  finished_ = true;
  reason_ = reason;
  usage_ = usage;
  StartStream(out);
  if (req_.api == Api::kResponses) {
    CloseItem(out);
    if (out) {
      const bool complete = reason == FinishReason::kStop;
      Value e = Event(complete ? "response.completed" : "response.incomplete");
      e.set("response", ResponseJson(complete ? "completed" : "incomplete"));
      out->push_back(std::move(e));
    }
    return;
  }
  if (!out) return;
  const char* finish = reason == FinishReason::kLength ? "length" : !calls_.empty() ? "tool_calls" : "stop";
  out->push_back(ChatChunk(Value::MakeObject(), finish));
  if (req_.api == Api::kChat && req_.include_usage) {
    Value c = ChatChunk(Value::MakeObject(), nullptr);
    c.set("choices", Value::MakeArray());
    Value u = Value::MakeObject();
    u.set("prompt_tokens", U(usage.prompt_tokens));
    u.set("completion_tokens", U(usage.completion_tokens));
    u.set("total_tokens", U(usage.prompt_tokens + usage.completion_tokens));
    c.set("usage", std::move(u));
    out->push_back(std::move(c));
  }
}

Value OutputBuilder::Final() const {
  if (req_.api == Api::kResponses) return ResponseJson(reason_ == FinishReason::kStop ? "completed" : "incomplete");
  const char* finish = reason_ == FinishReason::kLength ? "length" : !calls_.empty() ? "tool_calls" : "stop";
  Value choice = Value::MakeObject();
  choice.set("index", Value::Int(0));
  if (req_.api == Api::kCompletions) {
    choice.set("text", Str(content_));
  } else {
    Value m = Value::MakeObject();
    m.set("role", Str("assistant"));
    m.set("content", content_.empty() && !calls_.empty() ? Value::Null() : Str(content_));
    if (!reasoning_.empty()) m.set("reasoning_content", Str(reasoning_));
    if (!calls_.empty()) {
      Value list = Value::MakeArray();
      for (const auto& c : calls_) {
        Value fn = Value::MakeObject();
        fn.set("name", Str(c.name));
        fn.set("arguments", Str(c.arguments));
        Value call = Value::MakeObject();
        call.set("id", Str(c.call_id));
        call.set("type", Str("function"));
        call.set("function", std::move(fn));
        list.push(std::move(call));
      }
      m.set("tool_calls", std::move(list));
    }
    choice.set("message", std::move(m));
  }
  choice.set("logprobs", Value::Null());
  choice.set("finish_reason", Str(finish));
  Value r = Value::MakeObject();
  r.set("id", Str(id_));
  r.set("object", Str(req_.api == Api::kCompletions ? "text_completion" : "chat.completion"));
  r.set("created", Value::Int(created_));
  r.set("model", Str(req_.model));
  Value choices = Value::MakeArray();
  choices.push(std::move(choice));
  r.set("choices", std::move(choices));
  Value u = Value::MakeObject();
  u.set("prompt_tokens", U(usage_.prompt_tokens));
  u.set("completion_tokens", U(usage_.completion_tokens));
  u.set("total_tokens", U(usage_.prompt_tokens + usage_.completion_tokens));
  Value pd = Value::MakeObject();
  pd.set("cached_tokens", U(usage_.cached_tokens));
  u.set("prompt_tokens_details", std::move(pd));
  Value cd = Value::MakeObject();
  cd.set("reasoning_tokens", U(usage_.reasoning_tokens));
  u.set("completion_tokens_details", std::move(cd));
  r.set("usage", std::move(u));
  return r;
}

}  // namespace halo::server
