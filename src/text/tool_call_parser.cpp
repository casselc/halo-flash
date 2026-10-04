// Streaming Qwen output parser; see tool_call_parser.hpp for the contract.
#include "text/tool_call_parser.hpp"

#include <algorithm>
#include <optional>

namespace halo::text {

namespace {

using json::Value;
using Kind = Value::Kind;

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kCallOpen = "<tool_call>";
constexpr std::string_view kCallClose = "</tool_call>";
constexpr std::string_view kFunctionOpen = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kParamClose = "</parameter>";

bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

std::size_t LeadingSpace(std::string_view s) {
  std::size_t n = 0;
  while (n < s.size() && IsSpace(s[n])) ++n;
  return n;
}

std::size_t TrailingSpace(std::string_view s) {
  std::size_t n = 0;
  while (n < s.size() && IsSpace(s[s.size() - 1 - n])) ++n;
  return n;
}

std::string_view TrimSpace(std::string_view s) {
  s.remove_prefix(LeadingSpace(s));
  s.remove_suffix(TrailingSpace(s));
  return s;
}

// Length of the longest suffix of `s` that is a proper prefix of `tag`.
std::size_t PartialSuffix(std::string_view s, std::string_view tag) {
  for (std::size_t n = std::min(s.size(), tag.size() - 1); n > 0; --n)
    if (tag.starts_with(s.substr(s.size() - n))) return n;
  return 0;
}

// Length of a trailing UTF-8 sequence that is still incomplete (0 if none).
std::size_t IncompleteUtf8Tail(std::string_view s) {
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, s.size()); ++back) {
    const auto c = static_cast<unsigned char>(s[s.size() - back]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
    const std::size_t need = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xC2 && c <= 0xDF ? 2 : 1;
    return need > back ? back : 0;
  }
  return 0;
}

// ---- one tool call -----------------------------------------------------------

struct CallParse {
  enum Status { kNeedMore, kComplete, kMalformed } status = kNeedMore;
  std::size_t length = 0;  // kComplete: bytes of the call, from "<tool_call>"
  ParsedToolCall call;
  std::string error;  // kMalformed, or a complete call that is not valid
};

bool TypeMatches(std::string_view type, const Value& v) {
  if (type == "integer") return v.is(Kind::kInt);
  if (type == "number") return v.is(Kind::kInt) || v.is(Kind::kFloat);
  if (type == "boolean") return v.is(Kind::kBool);
  if (type == "null") return v.is(Kind::kNull);
  if (type == "array") return v.is(Kind::kArray);
  if (type == "object") return v.is(Kind::kObject);
  if (type == "string") return v.is(Kind::kString);
  return false;
}

// The template prints a string argument raw and anything else as JSON.
Value ArgumentValue(std::string_view raw, const std::vector<std::string>* types) {
  std::string ignored;
  json::ParseLimits limits;
  limits.max_depth = 64;
  std::optional<Value> parsed;
  const auto parse = [&]() -> const std::optional<Value>& {
    if (!parsed) parsed = json::Parse(raw, &ignored, limits);
    return parsed;
  };
  if (types && !types->empty()) {
    const bool string_ok = std::ranges::find(*types, "string") != types->end();
    if (string_ok && types->size() == 1) return Value::String(std::string(raw));
    if (const auto& v = parse(); v && !v->is(Kind::kString))
      for (const auto& t : *types)
        if (TypeMatches(t, *v)) return *v;
    // The Qwen3.5 template prints scalars with Python str(): True, False, None.
    const Value literal = raw == "True" ? Value::Bool(true) : raw == "False" ? Value::Bool(false) : Value::Null();
    if (raw == "True" || raw == "False" || raw == "None")
      for (const auto& t : *types)
        if (TypeMatches(t, literal)) return literal;
    if (string_ok) return Value::String(std::string(raw));
  }
  if (const auto& v = parse(); v && !v->is(Kind::kString)) return *v;
  return Value::String(std::string(raw));
}

using ParamTypes = std::map<std::string, std::vector<std::string>, std::less<>>;
using SchemaMap = std::map<std::string, ParamTypes, std::less<>>;

class CallParser {
 public:
  CallParser(std::string_view text, bool final, const SchemaMap& schemas)
      : s_(text), final_(final), schemas_(schemas) {}

  CallParse Run() {
    p_ = kCallOpen.size();
    SkipSpace();
    if (p_ == s_.size()) return More("truncated tool call");
    if (s_[p_] == '{') return Json();
    if (!Expect(kFunctionOpen)) return Bad("expected <function=NAME> or a JSON object");
    if (pending_) return More("truncated tool call");
    std::string name;
    if (!TagName(&name)) return Result();
    out_.call.name = name;
    const auto found = schemas_.find(name);
    const ParamTypes* params = found == schemas_.end() ? nullptr : &found->second;
    for (;;) {
      SkipSpace();
      const std::string_view rest = s_.substr(p_);
      if (rest.starts_with(kParamOpen)) {
        p_ += kParamOpen.size();
        std::string key;
        if (!TagName(&key)) return Result();
        if (!ParamValue(params, key)) return Result();
        continue;
      }
      if (rest.starts_with(kFunctionClose)) {
        p_ += kFunctionClose.size();
        SkipSpace();
        if (!Expect(kCallClose)) return pending_ ? More("truncated tool call") : Bad("expected </tool_call>");
        if (pending_) return More("truncated tool call");
        return Complete();
      }
      // Lenient: a call closed without </function>.
      if (rest.starts_with(kCallClose)) {
        p_ += kCallClose.size();
        return Complete();
      }
      if (rest.empty() || kParamOpen.starts_with(rest) || kFunctionClose.starts_with(rest) ||
          kCallClose.starts_with(rest))
        return More("truncated tool call");
      return Bad("expected <parameter=KEY>, </function> or </tool_call>");
    }
  }

 private:
  CallParse More(std::string why) { return final_ ? Bad(std::move(why)) : CallParse{}; }
  CallParse Bad(std::string why) {
    CallParse r;
    r.status = CallParse::kMalformed;
    r.error = std::move(why);
    return r;
  }
  CallParse Complete() {
    out_.status = CallParse::kComplete;
    out_.length = p_;
    return std::move(out_);
  }
  // Result of a helper that already decided (need more or malformed).
  CallParse Result() { return pending_ ? More("truncated tool call") : Bad(why_); }

  void SkipSpace() { p_ += LeadingSpace(s_.substr(p_)); }

  // Consumes `tag` at p_. A proper prefix of it at the end sets pending_.
  bool Expect(std::string_view tag) {
    const std::string_view rest = s_.substr(p_);
    if (rest.starts_with(tag)) {
      p_ += tag.size();
      return true;
    }
    pending_ = tag.starts_with(rest);
    return pending_;
  }

  // NAME> after "<function=" or "<parameter=".
  bool TagName(std::string* name) {
    const std::size_t close = s_.find('>', p_);
    const std::size_t stop = s_.find_first_of("<\n", p_);
    if (stop < close) return Fail("unterminated tag name");
    if (close == std::string_view::npos) return Pending();
    *name = std::string(TrimSpace(s_.substr(p_, close - p_)));
    if (name->empty()) return Fail("empty tag name");
    p_ = close + 1;
    return true;
  }

  // "\nVALUE\n</parameter>": one newline each side is framing.
  bool ParamValue(const ParamTypes* params, const std::string& key) {
    const std::string_view rest = s_.substr(p_);
    if (rest.empty()) return Pending();
    std::string_view raw;
    if (rest[0] == '\n') {
      const std::size_t close = rest.find("\n</parameter>");
      if (close == std::string_view::npos) return Pending();
      raw = close == 0 ? std::string_view{} : rest.substr(1, close - 1);
      p_ += close + 1 + kParamClose.size();
    } else {
      const std::size_t close = rest.find(kParamClose);
      if (close == std::string_view::npos) return Pending();
      raw = rest.substr(0, close);
      if (raw.ends_with('\n')) raw.remove_suffix(1);
      p_ += close + kParamClose.size();
    }
    const std::vector<std::string>* types = nullptr;
    if (params)
      if (auto it = params->find(key); it != params->end()) types = &it->second;
    json::Value v = ArgumentValue(raw, types);
    if (const json::Value* prev = out_.call.arguments.find(key)) {
      // A repeated identical parameter is harmless; conflicting copies leave
      // no safe choice, and keeping either would silently drop the other.
      if (!(*prev == v) && out_.error.empty()) out_.error = "conflicting values for parameter '" + key + "'";
      return true;
    }
    out_.call.arguments.set(key, std::move(v));
    return true;
  }

  CallParse Json() {
    // Find the end of the JSON object, ignoring braces inside strings.
    int depth = 0;
    bool in_string = false, escaped = false;
    std::size_t end = std::string_view::npos;
    for (std::size_t i = p_; i < s_.size(); ++i) {
      const char c = s_[i];
      if (in_string) {
        if (escaped) escaped = false;
        else if (c == '\\') escaped = true;
        else if (c == '"') in_string = false;
      } else if (c == '"') {
        in_string = true;
      } else if (c == '{') {
        ++depth;
      } else if (c == '}' && --depth == 0) {
        end = i + 1;
        break;
      }
    }
    if (end == std::string_view::npos) return More("truncated tool call");
    const std::string_view body = s_.substr(p_, end - p_);
    p_ = end;
    SkipSpace();
    if (!Expect(kCallClose)) return pending_ ? More("truncated tool call") : Bad("expected </tool_call>");
    if (pending_) return More("truncated tool call");
    std::string err;
    const auto v = json::Parse(body, &err);
    const json::Value* name = v ? v->find("name") : nullptr;
    const json::Value* args = v ? v->find("arguments") : nullptr;
    if (!v) out_.error = "tool call JSON: " + err;
    else if (!name || !name->is(Kind::kString) || name->str().empty()) out_.error = "tool call JSON has no \"name\"";
    else {
      out_.call.name = name->str();
      if (args && args->is(Kind::kString)) {
        // OpenAI-style arguments as a JSON string.
        const auto inner = json::Parse(args->str(), &err);
        if (inner && inner->is(Kind::kObject)) out_.call.arguments = *inner;
        else out_.error = "tool call \"arguments\" is not a JSON object";
      } else if (args && args->is(Kind::kObject)) {
        out_.call.arguments = *args;
      } else if (args && !args->is(Kind::kNull)) {
        out_.error = "tool call \"arguments\" is not a JSON object";
      }
    }
    return Complete();
  }

  bool Fail(std::string why) {
    why_ = std::move(why);
    return false;
  }
  bool Pending() {
    pending_ = true;
    return false;
  }

  std::string_view s_;
  bool final_;
  const SchemaMap& schemas_;
  std::size_t p_ = 0;
  bool pending_ = false;
  std::string why_;
  CallParse out_;
};

}  // namespace

ToolCallStreamParser::ToolCallStreamParser(ToolCallParserOptions options) : options_(std::move(options)) {
  if (options_.max_tool_call_bytes < kCallClose.size() + 1) options_.max_tool_call_bytes = kCallClose.size() + 1;
  if (options_.tools.is(Kind::kArray)) {
    for (const auto& tool : options_.tools.items()) {
      const json::Value* fn = tool.find("function");
      if (!fn) fn = &tool;
      const json::Value* name = fn->find("name");
      if (!name || !name->is(Kind::kString)) continue;
      auto& params = schemas_[name->str()];
      const json::Value* parameters = fn->find("parameters");
      const json::Value* props = parameters ? parameters->find("properties") : nullptr;
      if (!props || !props->is(Kind::kObject)) continue;
      for (const auto& [key, schema] : props->members()) {
        const json::Value* type = schema.find("type");
        auto& types = params[key];
        if (type && type->is(Kind::kString)) types.push_back(type->str());
        if (type && type->is(Kind::kArray))
          for (const auto& t : type->items())
            if (t.is(Kind::kString)) types.push_back(t.str());
      }
    }
  }
  state_ = options_.start == StreamStart::kReasoning ? State::kReasoning
           : options_.start == StreamStart::kContent ? State::kContent
                                                     : State::kDetect;
}

void ToolCallStreamParser::Push(std::string_view chunk, std::vector<StreamEvent>* events) {
  pending_.append(chunk);
  Run(false, events);
}

void ToolCallStreamParser::Finish(std::vector<StreamEvent>* events) {
  Run(true, events);
  // Run(final) consumes everything; start over for the next stream.
  pending_.clear();
  segment_start_ = true;
  bad_reason_.clear();
  state_ = options_.start == StreamStart::kReasoning ? State::kReasoning
           : options_.start == StreamStart::kContent ? State::kContent
                                                     : State::kDetect;
}

void ToolCallStreamParser::Emit(std::vector<StreamEvent>* events, StreamEventKind kind, std::size_t n) {
  if (n == 0) return;
  StreamEvent e;
  e.kind = kind;
  e.text = pending_.substr(0, n);
  pending_.erase(0, n);
  events->push_back(std::move(e));
}

void ToolCallStreamParser::EnterContent() {
  state_ = State::kContent;
  segment_start_ = true;
}

void ToolCallStreamParser::Run(bool final, std::vector<StreamEvent>* events) {
  for (;;) {
    bool progressed = false;
    switch (state_) {
      case State::kDetect: progressed = StepDetect(final, events); break;
      case State::kReasoning:
      case State::kContent: progressed = StepText(final, events); break;
      case State::kCall: progressed = StepCall(final, events); break;
      case State::kBadCall: progressed = StepBadCall(final, events); break;
    }
    if (!progressed) return;
  }
}

bool ToolCallStreamParser::StepDetect(bool final, std::vector<StreamEvent>* events) {
  const std::size_t ws = LeadingSpace(pending_);
  const std::string_view rest = std::string_view(pending_).substr(ws);
  if (rest.starts_with(kThinkOpen)) {
    Emit(events, StreamEventKind::kFraming, ws + kThinkOpen.size());
    state_ = State::kReasoning;
    segment_start_ = true;
    return true;
  }
  if (!final && kThinkOpen.starts_with(rest)) return false;  // "" or a prefix of <think>
  EnterContent();
  return true;
}

bool ToolCallStreamParser::StepText(bool final, std::vector<StreamEvent>* events) {
  const bool reasoning = state_ == State::kReasoning;
  const StreamEventKind kind = reasoning ? StreamEventKind::kReasoning : StreamEventKind::kContent;
  if (segment_start_) {
    Emit(events, StreamEventKind::kFraming, LeadingSpace(pending_));
    if (pending_.empty()) {
      // Wait for the segment's first non-space byte.
      return false;
    }
    segment_start_ = false;
  }
  const std::string_view text = pending_;
  std::size_t at = std::string_view::npos;
  std::string_view tag;
  const auto consider = [&](std::string_view t) {
    const std::size_t pos = text.find(t);
    if (pos < at) {
      at = pos;
      tag = t;
    }
  };
  if (reasoning) consider(kThinkClose);
  if (!reasoning || options_.tool_calls_in_reasoning) consider(kCallOpen);
  if (at != std::string_view::npos) {
    const std::size_t trailing = TrailingSpace(text.substr(0, at));
    Emit(events, kind, at - trailing);
    Emit(events, StreamEventKind::kFraming, trailing);
    if (tag == kThinkClose) {
      Emit(events, StreamEventKind::kFraming, kThinkClose.size());
      EnterContent();
    } else {
      state_ = State::kCall;
    }
    return true;
  }
  if (final) {
    const std::size_t trailing = TrailingSpace(text);
    Emit(events, kind, text.size() - trailing);
    Emit(events, StreamEventKind::kFraming, trailing);
    return false;
  }
  // Hold back what may still become markup: a partial tag, the whitespace
  // before it (framing if a tag follows) and an incomplete UTF-8 sequence.
  std::size_t hold = 0;
  if (reasoning) hold = std::max(hold, PartialSuffix(text, kThinkClose));
  if (!reasoning || options_.tool_calls_in_reasoning) hold = std::max(hold, PartialSuffix(text, kCallOpen));
  std::size_t ready = text.size() - hold;
  if (hold == 0) ready -= IncompleteUtf8Tail(text);
  ready -= TrailingSpace(text.substr(0, ready));
  Emit(events, kind, ready);
  return false;
}

bool ToolCallStreamParser::StepCall(bool final, std::vector<StreamEvent>* events) {
  const std::size_t limit = options_.max_tool_call_bytes;
  const std::string_view view = std::string_view(pending_).substr(0, limit);
  const bool view_final = final || pending_.size() >= limit;
  CallParse r = CallParser(view, view_final, schemas_).Run();
  switch (r.status) {
    case CallParse::kNeedMore:
      return false;
    case CallParse::kComplete: {
      StreamEvent e;
      e.text = pending_.substr(0, r.length);
      if (r.error.empty()) {
        e.kind = StreamEventKind::kToolCall;
        e.call = std::move(r.call);
      } else {
        e.kind = StreamEventKind::kError;
        e.error = std::move(r.error);
      }
      pending_.erase(0, r.length);
      events->push_back(std::move(e));
      EnterContent();
      return true;
    }
    case CallParse::kMalformed:
      bad_reason_ = std::move(r.error);
      state_ = State::kBadCall;
      return true;
  }
  return false;
}

// A malformed call runs to its "</tool_call>", or up to the next "<tool_call>"
// (so a valid call after it still parses), or to the end of the stream.
bool ToolCallStreamParser::StepBadCall(bool final, std::vector<StreamEvent>* events) {
  const std::size_t limit = options_.max_tool_call_bytes;
  const std::string_view view = std::string_view(pending_).substr(0, limit);
  std::size_t end = std::string_view::npos;
  if (const std::size_t close = view.find(kCallClose); close != std::string_view::npos) end = close + kCallClose.size();
  if (const std::size_t open = view.find(kCallOpen, kCallOpen.size()); open < end) end = open;
  if (end == std::string_view::npos) {
    if (final || pending_.size() >= limit) end = view.size();
    else return false;
  }
  StreamEvent e;
  e.kind = StreamEventKind::kError;
  e.text = pending_.substr(0, end);
  e.error = std::move(bad_reason_);
  bad_reason_.clear();
  pending_.erase(0, end);
  events->push_back(std::move(e));
  EnterContent();
  return true;
}

std::vector<StreamEvent> ParseAssistantText(std::string_view text, const ToolCallParserOptions& options) {
  ToolCallStreamParser parser(options);
  std::vector<StreamEvent> events;
  parser.Push(text, &events);
  parser.Finish(&events);
  return events;
}

}  // namespace halo::text
