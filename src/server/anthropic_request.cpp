#include "server/anthropic_request.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "text/chat_template.hpp"

namespace halo::server {

namespace {

using json::Value;
using Kind = json::Value::Kind;

constexpr std::string_view kBillingHeader = "x-anthropic-billing-header:";
constexpr std::string_view kSignatureTag = "halo1.";

// The first error wins; every helper returns false after it.
class Parser {
 public:
  explicit Parser(MessagesError* error) : error_(error) {}
  bool Fail(const std::string& param, const std::string& message) {
    if (ok_) {
      ok_ = false;
      if (error_) *error_ = {400, "invalid_request_error", param.empty() ? message : param + ": " + message};
    }
    return false;
  }
  bool ok() const { return ok_; }

 private:
  MessagesError* error_;
  bool ok_ = true;
};

const Value* Get(const Value& obj, std::string_view key) {
  const Value* v = obj.find(key);
  return v && !v->is(Kind::kNull) ? v : nullptr;
}

std::string TypeOf(const Value& block) {
  const Value* t = block.is(Kind::kObject) ? block.find("type") : nullptr;
  return t && t->is(Kind::kString) ? t->str() : std::string();
}

/// cache_control on a block or tool: {"type": "ephemeral", "ttl"?}; counted, not acted on.
bool CacheControl(Parser& p, const Value& owner, const std::string& param, std::uint32_t* breakpoints) {
  const Value* cc = Get(owner, "cache_control");
  if (!cc) return true;
  if (!cc->is(Kind::kObject) || TypeOf(*cc) != "ephemeral")
    return p.Fail(param + ".cache_control", "cache_control must be {\"type\": \"ephemeral\"}");
  if (const Value* ttl = Get(*cc, "ttl"); ttl && (!ttl->is(Kind::kString) || (ttl->str() != "5m" && ttl->str() != "1h")))
    return p.Fail(param + ".cache_control.ttl", "ttl must be \"5m\" or \"1h\"");
  ++*breakpoints;
  return true;
}

/// Why a block type that carries input this model cannot read is refused.
std::string Unsupported(const std::string& type) {
  if (type == "image") return "image input is not supported: this model reads text only";
  if (type == "document") return "document input is not supported: this model reads text only";
  return "content blocks of type \"" + type + "\" are not supported";
}

/// A text block's text ({"type": "text", "text"}); citations are refused.
bool TextBlock(Parser& p, const Value& block, const std::string& param, std::uint32_t* breakpoints, std::string* out) {
  const Value* text = block.find("text");
  if (!text || !text->is(Kind::kString)) return p.Fail(param + ".text", "a text block needs text");
  if (const Value* c = Get(block, "citations"); c && !(c->is(Kind::kArray) && c->items().empty()))
    return p.Fail(param + ".citations", "citations are not supported");
  if (!CacheControl(p, block, param, breakpoints)) return false;
  *out = text->str();
  return true;
}

class Conversation {
 public:
  Conversation(Parser& p, std::uint32_t* breakpoints) : p_(p), breakpoints_(breakpoints) {}

  bool System(const Value& v) {
    std::string text;
    if (v.is(Kind::kString)) {
      text = v.str();
    } else if (v.is(Kind::kArray)) {
      for (std::size_t i = 0; i < v.items().size(); ++i) {
        const std::string param = "system." + std::to_string(i);
        const Value& b = v.items()[i];
        if (TypeOf(b) != "text") return p_.Fail(param + ".type", "system blocks must be text blocks");
        std::string t;
        if (!TextBlock(p_, b, param, breakpoints_, &t)) return false;
        if (std::string_view(t).starts_with(kBillingHeader)) continue;  // Claude Code's billing attribution
        if (!text.empty()) text += "\n\n";
        text += t;
      }
    } else {
      return p_.Fail("system", "system must be a string or an array of text blocks");
    }
    if (text.empty()) return true;
    Push("system", Value::String(std::move(text)));
    return true;
  }

  bool Message(const Value& m, const std::string& param) {
    if (!m.is(Kind::kObject)) return p_.Fail(param, "a message must be an object");
    const Value* role = m.find("role");
    if (!role || !role->is(Kind::kString)) return p_.Fail(param + ".role", "role is required");
    const Value* content = m.find("content");
    if (!content || !(content->is(Kind::kString) || content->is(Kind::kArray)))
      return p_.Fail(param + ".content", "content must be a string or an array of content blocks");
    const std::string& r = role->str();
    if (r == "user") return User(*content, param);
    if (r == "assistant") return Assistant(*content, param);
    if (r == "system") return MidSystem(*content, param);
    return p_.Fail(param + ".role", "role must be user, assistant or system");
  }

  Value Finish() { return std::move(messages_); }
  bool last_is_assistant() const { return last_assistant_; }

 private:
  Value& Push(const char* role, Value content) {
    Value m = Value::MakeObject();
    m.set("role", Value::String(role));
    m.set("content", std::move(content));
    messages_.push(std::move(m));
    last_assistant_ = std::string_view(role) == "assistant";
    return messages_.items().back();
  }

  bool User(const Value& content, const std::string& param) {
    if (content.is(Kind::kString)) {
      Push("user", content);
      return true;
    }
    Value parts = Value::MakeArray();  // text blocks not yet emitted
    const auto flush = [&] {
      if (parts.items().empty()) return;
      Push("user", std::move(parts));
      parts = Value::MakeArray();
    };
    if (content.items().empty()) return p_.Fail(param + ".content", "content must not be empty");
    for (std::size_t i = 0; i < content.items().size(); ++i) {
      const std::string bp = param + ".content." + std::to_string(i);
      const Value& b = content.items()[i];
      const std::string type = TypeOf(b);
      if (type == "text") {
        std::string t;
        if (!TextBlock(p_, b, bp, breakpoints_, &t)) return false;
        Value part = Value::MakeObject();
        part.set("type", Value::String("text"));
        part.set("text", Value::String(std::move(t)));
        parts.push(std::move(part));
      } else if (type == "tool_result") {
        flush();
        if (!ToolResult(b, bp)) return false;
      } else {
        return p_.Fail(bp + ".type", type.empty() ? "a content block needs a type" : Unsupported(type));
      }
    }
    flush();
    return true;
  }

  bool ToolResult(const Value& b, const std::string& param) {
    const Value* id = b.find("tool_use_id");
    if (!id || !id->is(Kind::kString) || id->str().empty()) return p_.Fail(param + ".tool_use_id", "tool_use_id is required");
    if (!CacheControl(p_, b, param, breakpoints_)) return false;
    bool is_error = false;
    if (const Value* e = Get(b, "is_error")) {
      if (!e->is(Kind::kBool)) return p_.Fail(param + ".is_error", "is_error must be a boolean");
      is_error = e->boolean();
    }
    std::string text = is_error ? "[error] " : "";
    if (const Value* c = Get(b, "content")) {
      if (c->is(Kind::kString)) {
        text += c->str();
      } else if (c->is(Kind::kArray)) {
        for (std::size_t i = 0; i < c->items().size(); ++i) {
          const std::string cp = param + ".content." + std::to_string(i);
          const Value& part = c->items()[i];
          const std::string type = TypeOf(part);
          if (type != "text") return p_.Fail(cp + ".type", type.empty() ? "a content block needs a type" : Unsupported(type));
          std::string t;
          if (!TextBlock(p_, part, cp, breakpoints_, &t)) return false;
          text += t;
        }
      } else {
        return p_.Fail(param + ".content", "tool_result content must be a string or an array of text blocks");
      }
    }
    Value& m = Push("tool", Value::String(std::move(text)));
    m.set("tool_call_id", *id);
    return true;
  }

  bool Assistant(const Value& content, const std::string& param) {
    Value m = Value::MakeObject();
    m.set("role", Value::String("assistant"));
    std::string text, reasoning;
    bool any_reasoning = false;
    Value calls = Value::MakeArray();
    if (content.is(Kind::kString)) {
      text = content.str();
    } else {
      for (std::size_t i = 0; i < content.items().size(); ++i) {
        const std::string bp = param + ".content." + std::to_string(i);
        const Value& b = content.items()[i];
        const std::string type = TypeOf(b);
        if (type == "text") {
          std::string t;
          if (!TextBlock(p_, b, bp, breakpoints_, &t)) return false;
          text += t;
        } else if (type == "thinking") {
          const Value* t = b.find("thinking");
          if (!t || !t->is(Kind::kString)) return p_.Fail(bp + ".thinking", "a thinking block needs thinking text");
          std::string r = t->str();
          if (const Value* sig = Get(b, "signature"); sig && sig->is(Kind::kString))
            if (auto ours = DecodeThinkingSignature(sig->str())) r = std::move(*ours);
          if (any_reasoning && !r.empty() && !reasoning.empty()) reasoning += "\n\n";
          reasoning += r;
          any_reasoning = true;
        } else if (type == "redacted_thinking") {
          continue;  // opaque, and never ours
        } else if (type == "tool_use") {
          const Value* id = b.find("id");
          const Value* name = b.find("name");
          if (!name || !name->is(Kind::kString) || name->str().empty()) return p_.Fail(bp + ".name", "a tool_use block needs a name");
          const Value* input = b.find("input");
          if (input && !input->is(Kind::kObject) && !input->is(Kind::kNull))
            return p_.Fail(bp + ".input", "tool_use input must be an object");
          if (!CacheControl(p_, b, bp, breakpoints_)) return false;
          Value fn = Value::MakeObject();
          fn.set("name", *name);
          fn.set("arguments", input && input->is(Kind::kObject) ? *input : Value::MakeObject());
          Value call = Value::MakeObject();
          if (id && id->is(Kind::kString)) call.set("id", *id);
          call.set("type", Value::String("function"));
          call.set("function", std::move(fn));
          calls.push(std::move(call));
        } else {
          return p_.Fail(bp + ".type", type.empty() ? "a content block needs a type" : Unsupported(type));
        }
      }
    }
    if (any_reasoning) m.set("reasoning_content", Value::String(std::move(reasoning)));
    m.set("content", Value::String(std::move(text)));
    if (!calls.items().empty()) m.set("tool_calls", std::move(calls));
    messages_.push(std::move(m));
    last_assistant_ = true;
    return true;
  }

  /// A mid-conversation system message (Claude Code sends them after a user turn): the template allows system
  /// messages only at the start, so its text joins the user turn before it, or follows tool results as a user
  /// turn of its own.
  bool MidSystem(const Value& content, const std::string& param) {
    std::string text;
    if (content.is(Kind::kString)) {
      text = content.str();
    } else {
      for (std::size_t i = 0; i < content.items().size(); ++i) {
        const std::string bp = param + ".content." + std::to_string(i);
        if (TypeOf(content.items()[i]) != "text") return p_.Fail(bp + ".type", "system message blocks must be text blocks");
        std::string t;
        if (!TextBlock(p_, content.items()[i], bp, breakpoints_, &t)) return false;
        if (!text.empty()) text += "\n\n";
        text += t;
      }
    }
    if (text.empty()) return true;
    auto& items = messages_.items();
    const bool only_system = std::all_of(items.begin(), items.end(), [](const Value& m) {
      return m.find("role")->str() == "system";
    });
    if (only_system) {
      Push("system", Value::String(std::move(text)));
      return true;
    }
    Value& last = items.back();
    if (last.find("role")->str() == "user") {
      const Value& c = *last.find("content");
      if (c.is(Kind::kString)) {
        last.set("content", Value::String(c.str() + "\n\n" + text));
      } else {
        Value parts = c;
        Value part = Value::MakeObject();
        part.set("type", Value::String("text"));
        part.set("text", Value::String("\n\n" + text));
        parts.push(std::move(part));
        last.set("content", std::move(parts));
      }
      last_assistant_ = false;
      return true;
    }
    Push("user", Value::String(std::move(text)));
    return true;
  }

  Parser& p_;
  std::uint32_t* breakpoints_;
  Value messages_ = Value::MakeArray();
  bool last_assistant_ = false;
};

bool Tools(Parser& p, const Value& v, GenerationRequest& r, std::uint32_t* breakpoints) {
  if (!v.is(Kind::kArray)) return p.Fail("tools", "tools must be an array");
  if (v.items().empty()) return true;
  r.tools = Value::MakeArray();
  std::set<std::string> names;
  for (std::size_t i = 0; i < v.items().size(); ++i) {
    const std::string param = "tools." + std::to_string(i);
    const Value& t = v.items()[i];
    if (!t.is(Kind::kObject)) return p.Fail(param, "a tool must be an object");
    const std::string type = TypeOf(t);
    if (!type.empty() && type != "custom")
      return p.Fail(param + ".type", "Anthropic-defined tool type \"" + type +
                                         "\" is not supported: define it as a custom tool (name, input_schema)");
    const Value* name = t.find("name");
    if (!name || !name->is(Kind::kString) || name->str().empty()) return p.Fail(param + ".name", "a tool needs a name");
    if (!names.insert(name->str()).second) return p.Fail(param + ".name", "duplicate tool name: " + name->str());
    const Value* schema = t.find("input_schema");
    if (!schema || !schema->is(Kind::kObject)) return p.Fail(param + ".input_schema", "input_schema must be an object");
    if (!CacheControl(p, t, param, breakpoints)) return false;
    Value fn = Value::MakeObject();
    fn.set("name", *name);
    if (const Value* d = Get(t, "description")) {
      if (!d->is(Kind::kString)) return p.Fail(param + ".description", "description must be a string");
      fn.set("description", *d);
    }
    fn.set("parameters", *schema);
    Value tool = Value::MakeObject();
    tool.set("type", Value::String("function"));
    tool.set("function", std::move(fn));
    r.tools.push(std::move(tool));
  }
  return true;
}

bool Number(Parser& p, const Value& v, const char* param, double lo, double hi, double* out) {
  if (!v.is(Kind::kInt) && !v.is(Kind::kFloat)) return p.Fail(param, "must be a number");
  const double x = v.number();
  if (!std::isfinite(x) || x < lo || x > hi)
    return p.Fail(param, "must be between " + json::PythonFloatRepr(lo) + " and " + json::PythonFloatRepr(hi));
  *out = x;
  return true;
}

bool Integer(Parser& p, const Value& v, const char* param, std::int64_t lo, std::int64_t hi, std::int64_t* out) {
  if (!v.is(Kind::kInt)) return p.Fail(param, "must be an integer");
  const std::string& d = v.int_digits();
  if (d.size() > 18) return p.Fail(param, "is out of range");
  const std::int64_t x = std::stoll(d);
  if (x < lo || x > hi) return p.Fail(param, "must be between " + std::to_string(lo) + " and " + std::to_string(hi));
  *out = x;
  return true;
}

}  // namespace

std::optional<MessagesRequest> ParseMessagesRequest(const json::Value& body, const RequestDefaults& d,
                                                    bool count_tokens, MessagesError* error) {
  Parser p(error);
  if (!body.is(Kind::kObject)) return p.Fail("", "the request body must be a JSON object"), std::nullopt;
  MessagesRequest out;
  GenerationRequest& r = out.gen;
  r.api = Api::kMessages;
  r.template_options.add_generation_prompt = true;
  r.sampling = d.sampling;

  static const std::set<std::string, std::less<>> kActed = {
      "model", "messages", "max_tokens", "system", "stop_sequences", "stream", "temperature", "top_p", "top_k",
      "tools", "tool_choice", "thinking", "output_config", "metadata", "cache_control"};
  // Steer only Anthropic's own service: accepted, reported as ignored. Anything else unknown is too (Claude Code
  // adds fields often); the response lists them.
  static const std::set<std::string, std::less<>> kRefused = {"mcp_servers", "container", "output_format"};
  for (const auto& [k, v] : body.members()) {
    if (kActed.count(k) || v.is(Kind::kNull)) continue;
    if (kRefused.count(k)) {
      p.Fail(k, k == "output_format" ? "structured output formats are not supported"
                                     : k + " is not supported by this server");
      return std::nullopt;
    }
    out.ignored.push_back(k);
  }

  const Value* model = Get(body, "model");
  if (!model || !model->is(Kind::kString) || model->str().empty()) return p.Fail("model", "model is required"), std::nullopt;
  r.model = model->str();

  if (!count_tokens) {
    const Value* mt = Get(body, "max_tokens");
    if (!mt) return p.Fail("max_tokens", "max_tokens is required"), std::nullopt;
    std::int64_t n = 0;
    if (!Integer(p, *mt, "max_tokens", 1, d.max_tokens_limit, &n)) return std::nullopt;
    r.max_tokens = static_cast<std::uint32_t>(n);
    if (const Value* s = Get(body, "stream")) {
      if (!s->is(Kind::kBool)) return p.Fail("stream", "must be a boolean"), std::nullopt;
      r.stream = s->boolean();
    }
    if (const Value* s = Get(body, "stop_sequences")) {
      if (!s->is(Kind::kArray)) return p.Fail("stop_sequences", "must be an array of strings"), std::nullopt;
      for (std::size_t i = 0; i < s->items().size(); ++i) {
        const Value& x = s->items()[i];
        if (!x.is(Kind::kString) || x.str().empty())
          return p.Fail("stop_sequences." + std::to_string(i), "stop sequences must be non-empty strings"), std::nullopt;
        r.stop.push_back(x.str());
      }
    }
    double x = 0;
    if (const Value* v = Get(body, "temperature")) {
      if (!Number(p, *v, "temperature", 0, 1, &x)) return std::nullopt;
      r.sampling.temperature = static_cast<float>(x);
    }
    if (const Value* v = Get(body, "top_p")) {
      if (!Number(p, *v, "top_p", 0, 1, &x)) return std::nullopt;
      r.sampling.top_p = static_cast<float>(x);
    }
    if (const Value* v = Get(body, "top_k")) {
      std::int64_t k = 0;
      if (!Integer(p, *v, "top_k", 0, 1 << 30, &k)) return std::nullopt;
      r.sampling.top_k = static_cast<std::int32_t>(k);
    }
    try {
      r.sampling.Validate();
    } catch (const std::exception& e) {
      return p.Fail("", e.what()), std::nullopt;
    }
  }
  if (const Value* m = Get(body, "metadata")) {
    if (!m->is(Kind::kObject)) return p.Fail("metadata", "must be an object"), std::nullopt;
    if (const Value* u = Get(*m, "user_id")) {
      if (!u->is(Kind::kString)) return p.Fail("metadata.user_id", "must be a string"), std::nullopt;
      r.user = u->str();
    }
  }
  if (const Value* cc = Get(body, "cache_control"); cc && !CacheControl(p, body, "", &out.cache_breakpoints))
    return std::nullopt;

  // Thinking: on (adaptive, enabled) or off (disabled, between_tools); unset keeps the template's default (on).
  bool thinking_explicit_on = false;
  if (const Value* t = Get(body, "thinking")) {
    const std::string type = TypeOf(*t);
    if (type == "adaptive" || type == "enabled") {
      r.template_options.enable_thinking = true;
      thinking_explicit_on = true;
      if (type == "enabled") {
        const Value* b = Get(*t, "budget_tokens");
        std::int64_t n = 0;
        if (!b) return p.Fail("thinking.budget_tokens", "budget_tokens is required"), std::nullopt;
        if (!Integer(p, *b, "thinking.budget_tokens", 1, d.max_tokens_limit, &n)) return std::nullopt;
        out.ignored.push_back("thinking.budget_tokens");
      }
      if (const Value* disp = Get(*t, "display")) {
        if (!disp->is(Kind::kString) ||
            (disp->str() != "summarized" && disp->str() != "omitted" && disp->str() != "updates"))
          return p.Fail("thinking.display", "display must be summarized, omitted or updates"), std::nullopt;
        // No summarizer: "summarized" returns the full reasoning; "updates" has no progress notes to show.
        out.omit_thinking = disp->str() != "summarized";
      }
    } else if (type == "disabled" || type == "between_tools") {
      r.template_options.enable_thinking = false;
    } else {
      return p.Fail("thinking.type", "thinking.type must be adaptive, enabled, disabled or between_tools"), std::nullopt;
    }
  }
  if (const Value* oc = Get(body, "output_config")) {
    if (!oc->is(Kind::kObject)) return p.Fail("output_config", "must be an object"), std::nullopt;
    for (const auto& [k, v] : oc->members()) {
      if (v.is(Kind::kNull)) continue;
      if (k == "effort") {
        if (!v.is(Kind::kString)) return p.Fail("output_config.effort", "must be a string"), std::nullopt;
        const std::string& e = v.str();
        if (e != "low" && e != "medium" && e != "high" && e != "xhigh" && e != "max")
          return p.Fail("output_config.effort", "effort must be low, medium, high, xhigh or max"), std::nullopt;
        r.template_options.reasoning_effort = e == "max" ? "xhigh" : e;  // Flash-Next's highest
      } else if (k == "format") {
        return p.Fail("output_config.format", "structured output formats are not supported"), std::nullopt;
      } else {
        out.ignored.push_back("output_config." + k);
      }
    }
  }

  std::uint32_t* bp = &out.cache_breakpoints;
  if (const Value* t = Get(body, "tools"); t && !Tools(p, *t, r, bp)) return std::nullopt;

  // tool_choice: auto (default); none: no tool calls (tools stay in the prompt; a call the model writes anyway is
  // returned as text); any / tool: the prompt opens the call (thinking off, as the API requires).
  if (const Value* tc = Get(body, "tool_choice")) {
    const std::string type = TypeOf(*tc);
    if (const Value* dp = Get(*tc, "disable_parallel_tool_use")) {
      if (!dp->is(Kind::kBool)) return p.Fail("tool_choice.disable_parallel_tool_use", "must be a boolean"), std::nullopt;
      out.disable_parallel_tool_use = dp->boolean();
    }
    const bool has_tools = r.tools.is(Kind::kArray) && !r.tools.items().empty();
    if (type == "auto") {
      r.tool_choice.mode = ToolChoice::Mode::kAuto;
    } else if (type == "none") {
      r.tool_choice.mode = ToolChoice::Mode::kNone;
    } else if (type == "any" || type == "tool") {
      if (!has_tools) return p.Fail("tool_choice", "tool_choice " + type + " needs tools"), std::nullopt;
      if (thinking_explicit_on)
        return p.Fail("tool_choice", "Thinking may not be enabled when tool_choice forces tool use."), std::nullopt;
      r.template_options.enable_thinking = false;
      out.forced_prefix = "<tool_call>\n<function=";
      r.tool_choice.mode = ToolChoice::Mode::kRequired;
      if (type == "tool") {
        const Value* name = Get(*tc, "name");
        if (!name || !name->is(Kind::kString) || name->str().empty())
          return p.Fail("tool_choice.name", "tool_choice tool needs a name"), std::nullopt;
        bool found = false;
        for (const auto& t : r.tools.items()) found |= t.find("function")->find("name")->str() == name->str();
        if (!found) return p.Fail("tool_choice.name", "no tool named " + name->str()), std::nullopt;
        r.tool_choice.mode = ToolChoice::Mode::kFunction;
        r.tool_choice.function = name->str();
        out.forced_prefix += name->str() + ">\n";
      }
    } else {
      return p.Fail("tool_choice.type", "tool_choice.type must be auto, any, tool or none"), std::nullopt;
    }
  }
  r.parallel_tool_calls = !out.disable_parallel_tool_use;

  Conversation conv(p, bp);
  if (const Value* s = Get(body, "system"); s && !conv.System(*s)) return std::nullopt;
  const Value* messages = Get(body, "messages");
  if (!messages || !messages->is(Kind::kArray) || messages->items().empty())
    return p.Fail("messages", "messages must be a non-empty array"), std::nullopt;
  for (std::size_t i = 0; i < messages->items().size(); ++i)
    if (!conv.Message(messages->items()[i], "messages." + std::to_string(i))) return std::nullopt;
  if (conv.last_is_assistant())
    return p.Fail("messages", "assistant message prefill is not supported: the conversation must end with a user "
                              "message"),
           std::nullopt;
  r.messages = conv.Finish();
  std::sort(out.ignored.begin(), out.ignored.end());
  return out;
}

json::Value MessagesErrorBody(const MessagesError& e) {
  Value err = Value::MakeObject();
  err.set("type", Value::String(e.type));
  err.set("message", Value::String(e.message));
  Value body = Value::MakeObject();
  body.set("type", Value::String("error"));
  body.set("error", std::move(err));
  return body;
}

namespace {
constexpr char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}

std::string EncodeThinkingSignature(std::string_view s) {
  std::string out(kSignatureTag);
  std::size_t i = 0;
  for (; i + 2 < s.size(); i += 3) {
    const unsigned v = (static_cast<unsigned char>(s[i]) << 16) | (static_cast<unsigned char>(s[i + 1]) << 8) |
                       static_cast<unsigned char>(s[i + 2]);
    for (int k = 18; k >= 0; k -= 6) out += kB64[(v >> k) & 63];
  }
  if (i < s.size()) {
    unsigned v = static_cast<unsigned char>(s[i]) << 16;
    if (i + 1 < s.size()) v |= static_cast<unsigned char>(s[i + 1]) << 8;
    out += kB64[(v >> 18) & 63];
    out += kB64[(v >> 12) & 63];
    out += i + 1 < s.size() ? kB64[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

std::optional<std::string> DecodeThinkingSignature(std::string_view sig) {
  if (!sig.starts_with(kSignatureTag)) return std::nullopt;
  sig.remove_prefix(kSignatureTag.size());
  if (sig.size() % 4 != 0) return std::nullopt;
  std::string out;
  for (std::size_t i = 0; i < sig.size(); i += 4) {
    unsigned v = 0;
    int pad = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = sig[i + k];
      unsigned x = 0;
      if (c == '=') {
        if (i + 4 != sig.size() || k < 2) return std::nullopt;
        ++pad;
      } else {
        if (pad) return std::nullopt;
        const char* f = std::char_traits<char>::find(kB64, 64, c);
        if (!f) return std::nullopt;
        x = static_cast<unsigned>(f - kB64);
      }
      v = (v << 6) | x;
    }
    out += static_cast<char>((v >> 16) & 255);
    if (pad < 2) out += static_cast<char>((v >> 8) & 255);
    if (pad < 1) out += static_cast<char>(v & 255);
  }
  return out;
}

std::optional<ModelAlias> ParseModelAliasSpec(std::string_view spec) {
  const auto eq = spec.find('=');
  if (eq == std::string_view::npos || eq == 0) return std::nullopt;
  ModelAlias a{std::string(spec.substr(0, eq)), std::string(spec.substr(eq + 1))};
  // '*' only as the last character of a pattern.
  if (a.pattern.find('*') != std::string::npos && a.pattern.find('*') != a.pattern.size() - 1) return std::nullopt;
  return a;
}

std::optional<std::string> ResolveModel(std::string_view requested, const std::string& model_id,
                                        const std::vector<ModelAlias>& aliases) {
  if (requested == model_id) return model_id;
  for (const auto& a : aliases) {
    const bool match = a.pattern.ends_with('*')
                           ? requested.starts_with(std::string_view(a.pattern).substr(0, a.pattern.size() - 1))
                           : requested == a.pattern;
    if (match) return a.target.empty() ? model_id : a.target;
  }
  return std::nullopt;
}

}  // namespace halo::server
